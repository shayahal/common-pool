"""Langfuse logging integration for CPR game tracing and metrics.

Provides hierarchical tracing of games, rounds, and LLM generations
with custom metrics for research analysis.

Logging Destinations:
    - Langfuse Cloud: Structured traces sent to Langfuse service
      (requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
    - In-memory storage: Round metrics, generation data, API metrics
      (for dashboard display and analysis)
    
This module handles high-level game tracing for research purposes.
For application-level logging, see logger_setup.py (writes to logs/cpr_game.log).
For API call metrics, see api_logger.py (writes to logs/api_calls.log).
"""

from typing import Dict, List, Optional, Any
import time
import logging
import sys
import io
from contextlib import redirect_stderr
from datetime import datetime
from packaging import version
import importlib.metadata

from langfuse import Langfuse

from .config import CONFIG
from .logger_setup import get_logger

logger = get_logger(__name__)

# Suppress Langfuse context warnings about missing spans
# These are expected when logging at trace level without round spans
langfuse_logger = logging.getLogger("langfuse")
langfuse_logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings about span context

# Also suppress langfuse.decorators logger if it exists
decorators_logger = logging.getLogger("langfuse.decorators")
decorators_logger.setLevel(logging.ERROR)

# Required Langfuse version - must match exactly
REQUIRED_LANGFUSE_VERSION = "3.11.0"

# Check Langfuse version at import time
try:
    installed_version = importlib.metadata.version("langfuse")
    if version.parse(installed_version) != version.parse(REQUIRED_LANGFUSE_VERSION):
        raise RuntimeError(
            f"Incompatible langfuse version: {installed_version}. "
            f"Required version: {REQUIRED_LANGFUSE_VERSION}. "
            f"Please install the correct version: pip install langfuse=={REQUIRED_LANGFUSE_VERSION}"
        )
except importlib.metadata.PackageNotFoundError:
    raise RuntimeError(
        f"langfuse package is not installed. "
        f"Please install: pip install langfuse=={REQUIRED_LANGFUSE_VERSION}"
    )
except Exception as e:
    raise RuntimeError(
        f"Error checking langfuse version: {e}. "
        f"Please ensure langfuse=={REQUIRED_LANGFUSE_VERSION} is installed: pip install langfuse=={REQUIRED_LANGFUSE_VERSION}"
    ) from e


class LoggingManager:
    """Manager for Langfuse tracing and metrics logging.

    Tracing structure:
        Each player action in each round gets its own trace containing:
        - Prompt sent to LLM
        - Response received from LLM
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize logging manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else CONFIG

        # Initialize client as None - will be set below if initialization succeeds
        self.client = None

        # Langfuse is required - initialize client and raise error if it fails
        public_key = self.config.get("langfuse_public_key", "")
        secret_key = self.config.get("langfuse_secret_key", "")

        if not public_key or not secret_key:
            raise ValueError(
                "Langfuse API keys are required. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables, "
                "or provide langfuse_public_key and langfuse_secret_key in config."
            )

        try:
            langfuse_host = self.config.get("langfuse_host", "https://cloud.langfuse.com")
            logger.info(f"Initializing Langfuse client with host: {langfuse_host}")

            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=langfuse_host
            )

            # Try a simple operation to verify the client works
            logger.info("Langfuse client created, verifying connection...")
            # Note: We don't check for methods here - if they don't exist,
            # they'll raise AttributeError when we try to use them, which we handle below

            logger.info("✓ Langfuse client initialized successfully")
        except (ValueError, AttributeError, ConnectionError) as e:
            error_msg = (
                f"Failed to initialize Langfuse client: {e}. "
                "Please check your Langfuse API keys and network connection."
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = (
                f"Unexpected error initializing Langfuse client: {e}. "
                "Please check your Langfuse configuration."
            )
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        # Game tracking
        self.game_id = None
        self.current_round = 0

        # Metrics accumulation (for dashboard/analysis only)
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []
        self.api_metrics_data: List[Dict] = []

    def start_game_trace(self, game_id: str, config: Dict) -> None:
        """Initialize game tracking (no trace created).

        Args:
            game_id: Unique identifier for this game
            config: Game configuration
        """
        self.game_id = game_id
        self.current_round = 0
        logger.info(f"Starting game tracking for game_id: {game_id}")

    def set_current_round(self, round_num: int):
        """Set the current round number for trace naming.

        Args:
            round_num: Round number
        """
        self.current_round = round_num

    def log_generation(
        self,
        player_id: int,
        prompt: str,
        response: str,
        action: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict] = None,
        api_metrics: Optional[Dict] = None,
        system_prompt: Optional[str] = None
    ):
        """Log an LLM generation as a separate trace.

        Args:
            player_id: Player identifier
            prompt: User input prompt to LLM
            response: LLM response text
            action: Parsed extraction action
            reasoning: Extracted reasoning text
            metadata: Additional metadata
            api_metrics: API call metrics (latency, tokens, cost, etc.)
            system_prompt: System prompt/instructions (optional)
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot log generation.")

        try:
            # Create a unique trace name for this player action
            trace_name = f"{self.game_id}_round_{self.current_round}_player_{player_id}"

            # Format input as structured messages if system prompt is provided
            # This allows Langfuse to properly display both system and user prompts
            if self.config["log_llm_prompts"]:
                if system_prompt:
                    # Use structured message format for better display in Langfuse
                    input_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    # Fallback to plain string if no system prompt
                    input_messages = prompt
            else:
                input_messages = "[prompt hidden]"

            # Use Langfuse 3.11.0 API: start_generation creates a generation
            # that automatically creates its own parent trace
            generation_params = {
                "name": trace_name,
                "model": self.config["llm_model"],
                "input": input_messages,
                "output": response if self.config["log_llm_responses"] else "[response hidden]",
                "metadata": {
                    "game_id": self.game_id,
                    "round": self.current_round,
                    "player_id": player_id,
                    "action": action,
                    "reasoning": reasoning or "",
                }
            }

            # Add token usage if available
            if api_metrics:
                prompt_tokens = api_metrics.get("prompt_tokens")
                completion_tokens = api_metrics.get("completion_tokens")
                total_tokens = api_metrics.get("total_tokens")

                if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
                    usage_details = {}
                    if prompt_tokens is not None:
                        usage_details["promptTokens"] = int(prompt_tokens)
                    if completion_tokens is not None:
                        usage_details["completionTokens"] = int(completion_tokens)
                    if total_tokens is not None:
                        usage_details["totalTokens"] = int(total_tokens)

                    if usage_details:
                        generation_params["usage_details"] = usage_details

            # Create generation using start_generation (automatically creates trace)
            generation_obs = self.client.start_generation(**generation_params)

            # End the observation immediately since we have all the data
            if generation_obs and hasattr(generation_obs, 'end'):
                generation_obs.end()

            # Debug log
            if api_metrics:
                logger.info(
                    f"✅ Logged trace to Langfuse: {trace_name} | "
                    f"Tokens: {api_metrics.get('total_tokens', 'N/A')} | "
                    f"Latency: {api_metrics.get('latency', 0):.2f}s | "
                    f"Cost: ${api_metrics.get('cost', 0):.4f}"
                )
            else:
                logger.debug(f"Logged trace to Langfuse: {trace_name}")

            # Store for later analysis (dashboard/metrics)
            self.generation_data.append({
                "player_id": player_id,
                "prompt": prompt,
                "response": response,
                "action": action,
                "reasoning": reasoning,
                "api_metrics": api_metrics,
            })

            # Store API metrics separately
            if api_metrics:
                api_record = {
                    "player_id": player_id,
                    "timestamp": datetime.now().isoformat(),
                    **api_metrics
                }
                self.api_metrics_data.append(api_record)

        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error logging generation: {e}", exc_info=True)
            raise RuntimeError(f"Failed to log generation: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error logging generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error logging generation: {e}") from e

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Store metrics for a completed round (for dashboard only).

        Args:
            round_num: Round number
            metrics: Dictionary of metric values
        """
        # Store for aggregation and dashboard display
        metrics["round"] = round_num
        self.round_metrics.append(metrics)

    def end_game_trace(self, summary: Dict):
        """Finalize game and flush traces to Langfuse.

        Args:
            summary: Game summary statistics (stored locally only, not sent to Langfuse)
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot end game trace.")

        try:
            # Flush all pending traces to Langfuse
            logger.info("Flushing data to Langfuse...")
            self.client.flush()
            logger.info("✓ Successfully flushed all traces to Langfuse")

        except Exception as e:
            logger.error(f"Error flushing to Langfuse: {e}", exc_info=True)
            # Don't raise - we want to continue even if flush fails

    def get_round_metrics(self) -> List[Dict]:
        """Get all collected round metrics.

        Returns:
            List of metric dictionaries
        """
        return self.round_metrics.copy()

    def get_generation_data(self) -> List[Dict]:
        """Get all LLM generation data.

        Returns:
            List of generation dictionaries
        """
        return self.generation_data.copy()
    
    def get_api_metrics_data(self) -> List[Dict]:
        """Get all API metrics data.

        Returns:
            List of API metrics dictionaries
        """
        return self.api_metrics_data.copy()

    def reset(self):
        """Reset manager state for new game."""
        self.game_id = None
        self.current_round = 0
        self.round_metrics = []
        self.generation_data = []
        self.api_metrics_data = []

    def __del__(self):
        """Cleanup: flush any pending traces."""
        if self.client:
            try:
                self.client.flush()
            except Exception as e:
                # Log error but don't raise during cleanup (destructor)
                logger.warning(f"Error flushing logs during cleanup: {e}", exc_info=True)


class MockLoggingManager(LoggingManager):
    """Mock logging manager for testing without Langfuse.

    Stores all logs in memory for inspection.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize mock manager."""
        self.config = config if config is not None else CONFIG
        self.client = None  # Mock doesn't use Langfuse client

        # Mock storage
        self.traces = []
        self.generations = []

        self.round_metrics = []
        self.generation_data = []
        self.api_metrics_data = []
        self.game_id = None
        self.current_round = 0

    def start_game_trace(self, game_id: str, config: Dict) -> None:
        """Start mock game tracking."""
        self.game_id = game_id
        self.current_round = 0
        logger.info(f"Mock: Starting game tracking for game_id: {game_id}")

    def set_current_round(self, round_num: int):
        """Set the current round number."""
        self.current_round = round_num

    def log_generation(
        self,
        player_id: int,
        prompt: str,
        response: str,
        action: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict] = None,
        api_metrics: Optional[Dict] = None
    ):
        """Log mock generation."""
        trace_name = f"{self.game_id}_round_{self.current_round}_player_{player_id}"

        gen = {
            "trace_name": trace_name,
            "game_id": self.game_id,
            "round": self.current_round,
            "player_id": player_id,
            "prompt": prompt,
            "response": response,
            "action": action,
            "reasoning": reasoning,
            "metadata": metadata,
            "api_metrics": api_metrics,
        }
        self.generations.append(gen)
        self.traces.append(gen)  # Each generation is its own trace now
        self.generation_data.append(gen)

        # Store API metrics separately
        if api_metrics:
            api_record = {
                "player_id": player_id,
                "timestamp": datetime.now().isoformat(),
                **api_metrics
            }
            self.api_metrics_data.append(api_record)

        logger.debug(f"Mock: Logged trace {trace_name}")

    def end_game_trace(self, summary: Dict):
        """End mock game tracking."""
        logger.info(f"Mock: Game {self.game_id} completed with {len(self.traces)} traces")

    def get_all_traces(self) -> List[Dict]:
        """Get all collected traces."""
        return self.traces.copy()

    def reset(self):
        """Reset mock manager."""
        self.game_id = None
        self.current_round = 0
        self.round_metrics = []
        self.generation_data = []
        self.api_metrics_data = []
        self.traces = []
        self.generations = []

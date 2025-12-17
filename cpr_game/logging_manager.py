"""Langfuse logging integration for CPR game tracing and metrics.

Provides hierarchical tracing of games, rounds, and LLM generations
with custom metrics for research analysis.
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

    Tracing hierarchy:
        game_trace (top level)
        ├── round_0_span
        │   ├── player_0_generation
        │   ├── player_1_generation
        │   └── round_0_scores
        ├── round_1_span
        │   ├── player_0_generation
        │   ├── player_1_generation
        │   └── round_1_scores
        └── game_summary
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
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=self.config.get("langfuse_host", "https://cloud.langfuse.com")
            )
            # Note: We don't check for methods here - if they don't exist,
            # they'll raise AttributeError when we try to use them, which we handle below
        except (ValueError, AttributeError, ConnectionError) as e:
            raise RuntimeError(
                f"Failed to initialize Langfuse client: {e}. "
                "Please check your Langfuse API keys and network connection."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error initializing Langfuse client: {e}. "
                "Please check your Langfuse configuration."
            ) from e

        # Current trace and span tracking
        # In Langfuse 3.x, we track trace/span IDs, not objects
        self.current_trace_id = None
        self.current_round_span_id = None
        self.game_id = None

        # Metrics accumulation
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []

    def start_game_trace(self, game_id: str, config: Dict) -> Any:
        """Initialize top-level trace for a game.

        Args:
            game_id: Unique identifier for this game
            config: Game configuration

        Returns:
            Trace object

        Raises:
            RuntimeError: If Langfuse client is not initialized
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot start game trace.")

        self.game_id = game_id

        try:
            # Langfuse 3.x API: Use start_as_current_observation with as_type="trace" to create a trace
            # Note: tags parameter is not supported in Langfuse 3.11.0, so we include tags in metadata instead
            trace_observation = self.client.start_as_current_observation(
                as_type="trace",
                name=f"CPR_Game_{game_id}",
                metadata={
                    "game_id": game_id,
                    "timestamp": datetime.now().isoformat(),
                    "n_players": config["n_players"],
                    "max_steps": config["max_steps"],
                    "personas": config["player_personas"][:config["n_players"]],
                    "llm_model": config["llm_model"],
                    "tags": ["cpr_game", "multi_agent", "llm"],  # Store tags in metadata
                }
            )
            # Store trace ID for reference
            # get_current_trace_id() might return None if trace isn't ready yet
            trace_id = self.client.get_current_trace_id()
            if trace_id is None:
                # If we can't get trace ID immediately, we'll still allow logging
                # The trace was created, so we mark it as started
                # Use a placeholder to indicate trace is active
                self.current_trace_id = "active"
            else:
                self.current_trace_id = trace_id
            return trace_observation
        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error starting game trace: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start game trace: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error starting game trace: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error starting game trace: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error starting game trace: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error starting game trace: {e}") from e

    def start_round_span(self, round_num: int, game_state: Dict) -> Any:
        """Start a span for a single round.

        Args:
            round_num: Round number
            game_state: Current game state

        Returns:
            Span object

        Raises:
            RuntimeError: If Langfuse client or trace is not initialized
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot start round span.")
        if self.current_trace_id is None:
            raise RuntimeError("Game trace is not started. Call start_game_trace() first.")

        try:
            # Langfuse 3.x API: Use start_as_current_span to create a span
            span_observation = self.client.start_as_current_span(
                name=f"round_{round_num}",
                metadata={
                    "round": round_num,
                    "resource_level": game_state.get("resource", 0),
                    "step": game_state.get("step", 0),
                }
            )
            # Store span ID for reference
            self.current_round_span_id = self.client.get_current_observation_id()
            return span_observation
        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error starting round span: {e}", exc_info=True)
            raise RuntimeError(f"Failed to start round span: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error starting round span: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error starting round span: {e}") from e

    def log_generation(
        self,
        player_id: int,
        prompt: str,
        response: str,
        action: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log an LLM generation (player decision).

        Args:
            player_id: Player identifier
            prompt: Input prompt to LLM
            response: LLM response text
            action: Parsed extraction action
            reasoning: Extracted reasoning text
            metadata: Additional metadata
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot log generation.")
        if self.current_trace_id is None:
            raise RuntimeError("Game trace is not started. Call start_game_trace() first.")

        try:
            generation_metadata = {
                "player_id": player_id,
                "action": action,
                "reasoning": reasoning or "",
            }

            if metadata:
                generation_metadata.update(metadata)

            # Langfuse 3.x API: Use start_generation to log a generation
            # Note: This may log warnings if no active span context exists, but that's okay
            # Generations will be logged at the trace level instead
            # Suppress stderr warnings about missing span context
            stderr_buffer = io.StringIO()
            try:
                with redirect_stderr(stderr_buffer):
                    self.client.start_generation(
                        name=f"player_{player_id}_decision",
                        model=self.config["llm_model"],
                        input=prompt if self.config["log_llm_prompts"] else "[prompt hidden]",
                        output=response if self.config["log_llm_responses"] else "[response hidden]",
                        metadata=generation_metadata,
                    )
            except (RuntimeError, AttributeError, ValueError) as gen_error:
                # If generation fails due to span context issues, log warning but continue
                error_msg = str(gen_error).lower()
                if "no active span" in error_msg or "active span" in error_msg or "span context" in error_msg:
                    logger.warning(f"Generation logged at trace level (no active span): {gen_error}")
                    # Generation will be logged at trace level automatically by Langfuse
                else:
                    # Re-raise if it's a different error
                    raise

            # Store for later analysis
            self.generation_data.append({
                "player_id": player_id,
                "prompt": prompt,
                "response": response,
                "action": action,
                "reasoning": reasoning,
            })

        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error logging generation: {e}", exc_info=True)
            raise RuntimeError(f"Failed to log generation: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error logging generation: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error logging generation: {e}") from e

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Log metrics for a completed round.

        Args:
            round_num: Round number
            metrics: Dictionary of metric values
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot log round metrics.")
        if self.current_trace_id is None:
            raise RuntimeError("Game trace is not started. Call start_game_trace() first.")

        try:
            # Langfuse 3.x API: Use score_current_trace if no span is active,
            # otherwise use score_current_span
            # Note: game_runner doesn't call start_round_span, so we score at trace level
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    # If no span is active, score at trace level
                    if self.current_round_span_id is None:
                        self.client.score_current_trace(
                            name=f"round_{round_num}_{metric_name}",
                            value=float(value)
                        )
                    else:
                        # Try to score at span level, but fall back to trace if no active span context
                        # Suppress stderr warnings about missing span context
                        stderr_buffer = io.StringIO()
                        try:
                            with redirect_stderr(stderr_buffer):
                                self.client.score_current_span(
                                    name=f"round_{round_num}_{metric_name}",
                                    value=float(value)
                                )
                        except (RuntimeError, AttributeError, ValueError) as span_error:
                            # If span context is not active, fall back to trace level
                            error_msg = str(span_error).lower()
                            if "no active span" in error_msg or "active span" in error_msg:
                                logger.warning(f"No active span context, falling back to trace level scoring: {span_error}")
                                self.client.score_current_trace(
                                    name=f"round_{round_num}_{metric_name}",
                                    value=float(value)
                                )
                            else:
                                # Re-raise if it's a different error
                                raise

            # Store for aggregation
            metrics["round"] = round_num
            self.round_metrics.append(metrics)

        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error logging round metrics: {e}", exc_info=True)
            raise RuntimeError(f"Failed to log round metrics: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error logging round metrics: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error logging round metrics: {e}") from e

    def end_round_span(self):
        """End the current round span.
        
        Raises:
            RuntimeError: If Langfuse client or round span is not initialized
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot end round span.")
        if self.current_round_span_id is None:
            raise RuntimeError("Round span is not started. Call start_round_span() first.")

        try:
            # Langfuse 3.x API: Span ends automatically when context exits
            # Just clear our reference
            self.current_round_span_id = None
        except Exception as e:
            logger.error(f"Error ending round span: {e}", exc_info=True)

    def end_game_trace(self, summary: Dict):
        """Finalize game trace with summary metrics.

        Args:
            summary: Game summary statistics
        """
        if not self.client:
            raise RuntimeError("Langfuse client is not initialized. Cannot end game trace.")
        if self.current_trace_id is None:
            raise RuntimeError("Game trace is not started. Call start_game_trace() first.")

        try:
            # Log game-level scores
            game_scores = {
                "total_rounds": summary.get("total_rounds", 0),
                "final_resource_level": summary.get("final_resource_level", 0),
                "tragedy_occurred": 1.0 if summary.get("tragedy_occurred") else 0.0,
                "avg_cooperation_index": summary.get("avg_cooperation_index", 0),
                "gini_coefficient": summary.get("gini_coefficient", 0),
                "sustainability_score": summary.get("sustainability_score", 0),
                "payoff_fairness": 1.0 - summary.get("gini_coefficient", 0),
            }

            # Langfuse 3.x API: Use score_current_trace to log game-level scores
            for score_name, value in game_scores.items():
                if isinstance(value, (int, float)):
                    self.client.score_current_trace(
                        name=score_name,
                        value=float(value)
                    )

            # Langfuse 3.x API: Use update_current_trace to add summary metadata
            current_metadata = {
                "game_id": self.game_id,
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "end_time": datetime.now().isoformat(),
            }
            self.client.update_current_trace(
                metadata=current_metadata
            )

            # Flush to Langfuse
            if self.client:
                self.client.flush()

        except (AttributeError, ValueError, ConnectionError) as e:
            logger.error(f"Error ending game trace: {e}", exc_info=True)
            raise RuntimeError(f"Failed to end game trace: {e}") from e
        except Exception as e:
            # Catch any other unexpected exceptions
            logger.error(f"Unexpected error ending game trace: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error ending game trace: {e}") from e
        finally:
            self.current_trace_id = None
            self.game_id = None

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

    def reset(self):
        """Reset manager state for new game."""
        self.current_trace_id = None
        self.current_round_span_id = None
        self.game_id = None
        self.round_metrics = []
        self.generation_data = []

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
        self.current_trace_data = None
        self.spans = []
        self.generations = []
        self.scores = []

        self.round_metrics = []
        self.generation_data = []
        self.game_id = None

    def start_game_trace(self, game_id: str, config: Dict) -> Dict:
        """Start mock game trace."""
        self.game_id = game_id
        self.current_trace_data = {
            "game_id": game_id,
            "config": config,
            "start_time": time.time(),
            "spans": [],
            "generations": [],
            "scores": [],
        }
        return self.current_trace_data

    def start_round_span(self, round_num: int, game_state: Dict) -> Dict:
        """Start mock round span."""
        span = {
            "round": round_num,
            "game_state": game_state,
        }
        if self.current_trace_data:
            self.current_trace_data["spans"].append(span)
        return span

    def log_generation(
        self,
        player_id: int,
        prompt: str,
        response: str,
        action: float,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log mock generation."""
        gen = {
            "player_id": player_id,
            "prompt": prompt,
            "response": response,
            "action": action,
            "reasoning": reasoning,
            "metadata": metadata,
        }
        self.generations.append(gen)
        if self.current_trace_data:
            self.current_trace_data["generations"].append(gen)

        self.generation_data.append(gen)

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Log mock round metrics."""
        metrics["round"] = round_num
        self.round_metrics.append(metrics)
        if self.current_trace_data:
            self.current_trace_data["scores"].append(metrics)

    def end_game_trace(self, summary: Dict):
        """End mock game trace."""
        if self.current_trace_data:
            self.current_trace_data["summary"] = summary
            self.current_trace_data["end_time"] = time.time()
            self.traces.append(self.current_trace_data)
        self.current_trace_data = None

    def get_all_traces(self) -> List[Dict]:
        """Get all collected traces."""
        return self.traces.copy()

    def reset(self):
        """Reset mock manager."""
        self.current_trace_data = None
        self.game_id = None
        self.round_metrics = []
        self.generation_data = []

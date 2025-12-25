"""OpenTelemetry-based distributed tracing for CPR game observability.

This module provides hierarchical tracing of games, rounds, and LLM generations
using OpenTelemetry as the single source of truth. Traces are exported via OTLP
to configured receivers (Langfuse, LangSmith, etc.) based on OTEL_RECEIVER
configuration.

Purpose: Distributed tracing for research and observability
Output: OpenTelemetry traces → Selected receiver(s) (Langfuse/LangSmith)
Note: API metrics (tokens, costs, latency) are captured via OpenTelemetry spans
      and can be viewed in the configured receiver dashboards.

For application-level logging (errors, info, debug), see logger_setup.py
(writes to logs/cpr_game.log and console).
"""

from typing import Dict, List, Optional, Any
import time
import logging
from datetime import datetime
from opentelemetry import trace

from .config import CONFIG
from .logger_setup import get_logger
from .otel_manager import OTelManager

logger = get_logger(__name__)


class LoggingManager:
    """Manager for OpenTelemetry tracing and metrics logging.

    Uses OpenTelemetry as single source of truth. Traces are exported
    via OTLP to configured receivers (Langfuse, LangSmith, etc.).

    Tracing structure (Thread model - works for both Langfuse and LangSmith):
        Thread: game_{game_id} (via session_id metadata)
        ├── Trace: player_{player_id}_action_round_{round_num}
        │   ├── Span: llm_generation (auto-instrumented)
        │   └── Span: action_processing
        ├── Trace: player_{player_id}_action_round_{round_num}
        │   └── ...
        └── ...
    
    Each player action is a separate trace, grouped into a thread using
    session_id metadata. This allows viewing all player actions in a game
    as a conversation thread in both Langfuse and LangSmith.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize logging manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config if config is not None else CONFIG

        # Initialize OpenTelemetry manager
        try:
            self.otel_manager = OTelManager(self.config)
            self.tracer = self.otel_manager.get_tracer()
            if self.tracer is None:
                logger.warning("OpenTelemetry is disabled - tracing will be no-op")
        except Exception as e:
            # Check if it's a configuration error (missing API key, etc.) - don't print stack trace
            error_str = str(e).lower()
            if "api_key" in error_str or "api key" in error_str or "missing" in error_str:
                logger.warning(f"Failed to initialize OpenTelemetry: {e}. Tracing will be disabled.")
                self.otel_manager = None
                self.tracer = None
            else:
                # For unexpected errors, log with stack trace
                logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize OpenTelemetry: {e}") from e

        # Game tracking (thread identifier)
        self.game_id = None
        self.current_round = 0
        # No root trace - each player action will be its own trace

        # Metrics accumulation (for dashboard/analysis only)
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []
        self.api_metrics_data: List[Dict] = []

    def start_game_trace(self, game_id: str, config: Dict) -> None:
        """Initialize game tracking (thread).

        Args:
            game_id: Unique identifier for this game (used as thread/session_id)
            config: Game configuration
        """
        self.game_id = game_id
        self.current_round = 0

        if self.tracer is None:
            logger.debug(f"OTel disabled - skipping thread initialization for game_id: {game_id}")
            return

        # No root trace needed - each player action will be its own trace
        # The game_id will be used as session_id to group traces into a thread
        # This works for both Langfuse and LangSmith
        logger.info(f"Starting game thread for game_id: {game_id} (n_players={config.get('n_players', 2)})")

    def set_current_round(self, round_num: int):
        """Set the current round number.

        Args:
            round_num: Round number
        """
        self.current_round = round_num
        # No round span needed - round info will be in trace attributes

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
        """Log an LLM generation as a separate trace (threaded by game).

        Each player action becomes its own trace, grouped into a thread
        using session_id metadata (game_id). This follows LangSmith's
        thread model for multi-turn conversations.

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
        if self.tracer is None:
            # Still store data for dashboard/metrics even if tracing is disabled
            self._store_generation_data(player_id, prompt, response, action, reasoning, api_metrics)
            return

        try:
            # Each player action is its own trace (root span)
            trace_name = f"player_{player_id}_action_round_{self.current_round}"
            
            # Create trace attributes
            attributes = {
                "player.id": player_id,
                "action.extraction": float(action),
                "game.id": self.game_id,
                "round.number": self.current_round,
            }
            
            if reasoning:
                attributes["reasoning"] = reasoning[:500]  # Limit length
            
            # Add persona if available
            if metadata and "persona" in metadata:
                attributes["player.persona"] = metadata["persona"]
            
            # Add LLM model info
            attributes["llm.model"] = self.config.get("llm_model", "unknown")
            attributes["llm.temperature"] = self.config.get("llm_temperature", 0.7)
            
            # Add token usage if available
            if api_metrics:
                if api_metrics.get("prompt_tokens") is not None:
                    attributes["llm.prompt_tokens"] = int(api_metrics["prompt_tokens"])
                if api_metrics.get("completion_tokens") is not None:
                    attributes["llm.completion_tokens"] = int(api_metrics["completion_tokens"])
                if api_metrics.get("total_tokens") is not None:
                    attributes["llm.total_tokens"] = int(api_metrics["total_tokens"])
                if api_metrics.get("latency") is not None:
                    attributes["llm.latency_seconds"] = float(api_metrics["latency"])
                if api_metrics.get("cost") is not None:
                    attributes["llm.cost"] = float(api_metrics["cost"])

            # Add input/output as attributes for Langfuse compatibility
            # Langfuse expects 'input' and 'output' attributes to display them in the UI
            if self.config.get("log_llm_prompts", True):
                # Build input from system prompt and user prompt
                input_parts = []
                if system_prompt:
                    input_parts.append(f"System: {system_prompt}")
                if prompt:
                    input_parts.append(f"User: {prompt}")
                if input_parts:
                    attributes["input"] = "\n\n".join(input_parts)
            
            if self.config.get("log_llm_responses", True) and response:
                attributes["output"] = response

            # CRITICAL: Add session_id to attributes for thread grouping
            # Both Langfuse and LangSmith use session_id, thread_id, or conversation_id
            # in span attributes to group traces into threads
            attributes["session_id"] = self.game_id
            
            # Create a new trace (root span) for this player action
            # Each trace is independent but grouped by session_id into a thread
            # This works for both Langfuse and LangSmith
            # start_as_current_span creates a root span if no parent context exists
            with self.tracer.start_as_current_span(
                trace_name,
                attributes=attributes
            ) as root_span:
                # Add events for prompt/response (for backward compatibility and detailed logging)
                if self.config.get("log_llm_prompts", True):
                    if system_prompt:
                        root_span.add_event("prompt.system", {"content": system_prompt[:1000]})
                    root_span.add_event("prompt.user", {"content": prompt[:1000]})
                
                if self.config.get("log_llm_responses", True):
                    root_span.add_event("response.received", {"content": response[:1000]})
                
                root_span.add_event("action.extracted", {"action": float(action)})
                
                # LLM generation spans will be auto-instrumented as child spans

            # Debug log for API calls
            if api_metrics:
                logger.debug(
                    f"✅ Logged trace (thread: {self.game_id}): {trace_name} | "
                    f"Tokens: {api_metrics.get('total_tokens', 'N/A')} | "
                    f"Latency: {api_metrics.get('latency', 0):.2f}s | "
                    f"Cost: ${api_metrics.get('cost', 0):.4f}"
                )
            else:
                logger.debug(f"Logged trace (thread: {self.game_id}): {trace_name}")

            # Store for later analysis (dashboard/metrics)
            self._store_generation_data(player_id, prompt, response, action, reasoning, api_metrics)

        except Exception as e:
            error_msg = f"Error logging generation: {e}"
            logger.error(error_msg, exc_info=True)
            # Still store data for dashboard/metrics before raising
            self._store_generation_data(player_id, prompt, response, action, reasoning, api_metrics)
            raise RuntimeError(error_msg) from e

    def _store_generation_data(
        self,
        player_id: int,
        prompt: str,
        response: str,
        action: float,
        reasoning: Optional[str],
        api_metrics: Optional[Dict]
    ):
        """Store generation data for dashboard/metrics (internal helper)."""
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

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Store metrics for a completed round and create metrics span.

        Args:
            round_num: Round number
            metrics: Dictionary of metric values
        """
        # Store for aggregation and dashboard display
        metrics["round"] = round_num
        self.round_metrics.append(metrics)

        if self.tracer is None:
            return

        try:
            # Create round_metrics trace (part of the same thread)
            with self.tracer.start_as_current_span(
                f"round_{round_num}_metrics",
                attributes={
                    "session_id": self.game_id,  # Part of the game thread
                    "round.number": round_num,
                    "game.id": self.game_id,
                    "resource.level": float(metrics.get("resource_level", 0)),
                    "total.extraction": float(metrics.get("total_extraction", 0)),
                    "cooperation.index": float(metrics.get("cooperation_index", 0)),
                }
            ) as metrics_span:
                # Add individual extractions and payoffs as events
                if "individual_extractions" in metrics:
                    for i, extraction in enumerate(metrics["individual_extractions"]):
                        metrics_span.add_event(
                            "player.extraction",
                            {"player_id": i, "extraction": float(extraction)}
                        )
                if "individual_payoffs" in metrics:
                    for i, payoff in enumerate(metrics["individual_payoffs"]):
                        metrics_span.add_event(
                            "player.payoff",
                            {"player_id": i, "payoff": float(payoff)}
                        )
        except Exception as e:
            error_msg = f"Error logging round metrics: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def end_game_trace(self, summary: Dict):
        """Finalize game thread and flush traces.

        Args:
            summary: Game summary statistics
        """
        if self.tracer is None:
            logger.debug("OTel disabled - skipping thread finalization")
            return

        try:
            # Optionally create a summary trace for the game completion
            # This will also be part of the same thread (same session_id)
            with self.tracer.start_as_current_span(
                "game_summary",
                attributes={
                    "session_id": self.game_id,  # Keep in same thread
                    "game.id": self.game_id,
                    "game.total_rounds": int(summary.get("total_rounds", 0)),
                    "game.final_resource": float(summary.get("final_resource_level", 0)),
                    "game.tragedy_occurred": bool(summary.get("tragedy_occurred", False)),
                    "game.avg_cooperation_index": float(summary.get("avg_cooperation_index", 0)),
                    "game.gini_coefficient": float(summary.get("gini_coefficient", 0)),
                }
            ) as summary_span:
                summary_span.add_event("game.completed")

            # Flush all pending traces
            logger.info(f"Flushing OTel traces for thread: {self.game_id}...")
            self.otel_manager.flush()
            logger.info(f"✓ Successfully flushed all traces for thread: {self.game_id}")

        except Exception as e:
            error_msg = f"Error ending game thread: {e}"
            # Don't raise - just log warning and continue
            # This allows games to complete even if tracing has issues
            logger.warning(error_msg)

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
        if hasattr(self, 'otel_manager') and self.otel_manager is not None:
            try:
                self.otel_manager.flush()
            except Exception as e:
                # In __del__, we can't raise exceptions, but we log the error
                # Use ERROR level to ensure it's visible
                logger.error(f"Error flushing traces during cleanup: {e}", exc_info=True)


class MockLoggingManager(LoggingManager):
    """Mock logging manager for testing without OpenTelemetry.

    Stores all logs in memory for inspection.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize mock manager."""
        self.config = config if config is not None else CONFIG
        self.otel_manager = None
        self.tracer = None

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
        self.traces.append(gen)
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
        self.current_trace = None
        self.current_round_span = None

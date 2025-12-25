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

    Tracing structure:
        Trace: game_{game_id}
        ├── Span: game_setup
        ├── Span: round_{round_num}
        │   ├── Span: player_{player_id}_action
        │   │   └── Span: llm_generation (auto-instrumented)
        │   └── Span: round_metrics
        └── Span: game_summary
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

        # Game tracking
        self.game_id = None
        self.current_round = 0
        self.current_trace = None
        self.current_round_span = None

        # Metrics accumulation (for dashboard/analysis only)
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []
        self.api_metrics_data: List[Dict] = []

    def start_game_trace(self, game_id: str, config: Dict) -> None:
        """Initialize game tracking and create trace.

        Args:
            game_id: Unique identifier for this game
            config: Game configuration
        """
        self.game_id = game_id
        self.current_round = 0

        if self.tracer is None:
            logger.debug(f"OTel disabled - skipping trace creation for game_id: {game_id}")
            return

        try:
            # Start root trace for the game
            self.current_trace = self.tracer.start_as_current_span(
                f"game_{game_id}",
                attributes={
                    "game.id": game_id,
                    "game.n_players": config.get("n_players", 2),
                    "game.max_steps": config.get("max_steps", 20),
                    "game.initial_resource": config.get("initial_resource", 1000),
                }
            )
            # Enter the context manager to activate the span
            self.current_trace.__enter__()
            
            # Create game_setup span
            with self.tracer.start_as_current_span(
                "game_setup",
                attributes={
                    "game.id": game_id,
                    "game.config": str(config),
                }
            ) as setup_span:
                setup_span.add_event("game.started")
            
            logger.info(f"Starting game tracking for game_id: {game_id}")
        except Exception as e:
            error_msg = f"Error starting game trace: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def set_current_round(self, round_num: int):
        """Set the current round number and create round span.

        Args:
            round_num: Round number
        """
        self.current_round = round_num

        if self.tracer is None:
            return

        try:
            # End previous round span if exists
            # The context manager automatically ends when exited, but we need to exit it properly
            if self.current_round_span is not None:
                try:
                    # Exit the previous context manager
                    self.current_round_span.__exit__(None, None, None)
                except Exception as e:
                    error_msg = f"Error exiting previous round span: {e}"
                    logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
                self.current_round_span = None

            # Start new round span (returns a context manager)
            self.current_round_span = self.tracer.start_as_current_span(
                f"round_{round_num}",
                attributes={
                    "round.number": round_num,
                    "game.id": self.game_id,
                }
            )
            # Enter the context manager to activate the span
            self.current_round_span.__enter__()
        except Exception as e:
            error_msg = f"Error creating round span: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

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
        """Log an LLM generation as a span.

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
            span_name = f"player_{player_id}_action"
            
            # Create span attributes
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

            # Create player action span
            with self.tracer.start_as_current_span(
                span_name,
                attributes=attributes
            ) as player_span:
                # Add events for prompt/response
                if self.config.get("log_llm_prompts", True):
                    if system_prompt:
                        player_span.add_event("prompt.system", {"content": system_prompt[:1000]})
                    player_span.add_event("prompt.user", {"content": prompt[:1000]})
                
                if self.config.get("log_llm_responses", True):
                    player_span.add_event("response.received", {"content": response[:1000]})
                
                player_span.add_event("action.extracted", {"action": float(action)})

            # Debug log for API calls
            if api_metrics:
                logger.debug(
                    f"✅ Logged trace via OTel: {self.game_id}_round_{self.current_round}_player_{player_id} | "
                    f"Tokens: {api_metrics.get('total_tokens', 'N/A')} | "
                    f"Latency: {api_metrics.get('latency', 0):.2f}s | "
                    f"Cost: ${api_metrics.get('cost', 0):.4f}"
                )
            else:
                logger.debug(f"Logged trace via OTel: {self.game_id}_round_{self.current_round}_player_{player_id}")

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
            # Create round_metrics span
            with self.tracer.start_as_current_span(
                "round_metrics",
                attributes={
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
        """Finalize game and flush traces.

        Args:
            summary: Game summary statistics
        """
        if self.tracer is None:
            logger.debug("OTel disabled - skipping trace finalization")
            return

        try:
            # End current round span if exists
            if self.current_round_span is not None:
                try:
                    # Exit the context manager properly
                    self.current_round_span.__exit__(None, None, None)
                except RuntimeError as e:
                    # "generator didn't stop" means the context wasn't properly entered
                    if "generator didn't stop" in str(e):
                        logger.warning(f"Round span context already closed or not properly entered")
                    else:
                        raise
                except Exception as e:
                    error_msg = f"Error exiting current round span: {e}"
                    logger.warning(error_msg)
                finally:
                    self.current_round_span = None

            # Create game_summary span
            with self.tracer.start_as_current_span(
                "game_summary",
                attributes={
                    "game.id": self.game_id,
                    "game.total_rounds": int(summary.get("total_rounds", 0)),
                    "game.final_resource": float(summary.get("final_resource_level", 0)),
                    "game.tragedy_occurred": bool(summary.get("tragedy_occurred", False)),
                    "game.avg_cooperation_index": float(summary.get("avg_cooperation_index", 0)),
                    "game.gini_coefficient": float(summary.get("gini_coefficient", 0)),
                }
            ) as summary_span:
                summary_span.add_event("game.completed")

            # End root trace
            if self.current_trace is not None:
                try:
                    # Exit the context manager properly
                    # Check if it's already exited by trying to get the span
                    try:
                        self.current_trace.__exit__(None, None, None)
                    except RuntimeError as e:
                        # "generator didn't stop" means the context wasn't properly entered
                        # This can happen if the span was never entered or already exited
                        if "generator didn't stop" in str(e):
                            logger.warning(f"Trace context already closed or not properly entered for game {self.game_id}")
                        else:
                            raise
                except Exception as e:
                    error_msg = f"Error exiting root trace: {e}"
                    # Don't raise for context manager errors - just log and continue
                    logger.warning(error_msg)
                finally:
                    self.current_trace = None

            # Flush all pending spans
            logger.info("Flushing OTel traces...")
            self.otel_manager.flush()
            logger.info("✓ Successfully flushed all traces via OTel")

        except Exception as e:
            error_msg = f"Error ending game trace: {e}"
            # Don't raise - just log warning and continue
            # This allows games to complete even if tracing has issues
            logger.warning(error_msg)
            # Clean up trace references
            self.current_trace = None
            self.current_round_span = None

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
        self.current_trace = None
        self.current_round_span = None
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
        self.current_trace = None
        self.current_round_span = None

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

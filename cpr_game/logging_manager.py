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
import json
import uuid
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
        self.user_id: Optional[str] = None
        # No root trace - each player action will be its own trace

        # Metrics accumulation (for dashboard/analysis only)
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []
        self.api_metrics_data: List[Dict] = []
        
        # Track trace start times for duration calculation
        self.trace_start_times: Dict[str, datetime] = {}
        
        # Track current trace/span IDs for score logging
        self.current_trace_ids: Dict[int, str] = {}  # player_id -> trace_id
        self.current_span_ids: Dict[int, str] = {}  # player_id -> span_id

    def start_game_trace(self, game_id: str, config: Dict) -> None:
        """Initialize game tracking (thread).

        Args:
            game_id: Unique identifier for this game (used as thread/session_id)
            config: Game configuration
        """
        self.game_id = game_id
        self.current_round = 0
        self.user_id = config.get("user_id") or config.get("experiment_id")
        
        # Log session start for GraphRAG
        self.log_session_start(game_id, config)

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
            # Generate unique trace_id for GraphRAG
            trace_timestamp = datetime.utcnow()
            trace_id = f"{self.game_id}_{player_id}_round_{self.current_round}_{trace_timestamp.strftime('%Y%m%d%H%M%S%f')}"
            
            # Each player action is its own trace (root span)
            trace_name = f"player_{player_id}_action_round_{self.current_round}"
            
            # Record trace start time for duration calculation
            trace_start_time = trace_timestamp
            self.trace_start_times[trace_id] = trace_start_time
            
            # Create trace attributes with all required fields for GraphRAG
            attributes = {
                "trace_id": trace_id,  # Explicit trace_id for GraphRAG
                "name": trace_name,
                "player.id": player_id,
                "action.extraction": float(action),
                "game.id": self.game_id,
                "round.number": self.current_round,
                "timestamp": trace_start_time.isoformat(),  # ISO format timestamp
            }
            
            # Add user_id if available
            if self.user_id:
                attributes["user_id"] = self.user_id
            
            # Build trace metadata as JSON
            trace_metadata = {
                "player_id": player_id,
                "round": self.current_round,
                "game_id": self.game_id,
            }
            if metadata:
                trace_metadata.update(metadata)
            if reasoning:
                trace_metadata["reasoning"] = reasoning
            attributes["metadata"] = json.dumps(trace_metadata)
            
            if reasoning:
                attributes["reasoning"] = reasoning  # Keep for backward compatibility
            
            # Add persona if available
            if metadata and "persona" in metadata:
                attributes["player.persona"] = metadata["persona"]
            
            # Add LLM model info
            llm_model = self.config.get("llm_model", "unknown")
            attributes["llm.model"] = llm_model
            attributes["llm.temperature"] = self.config.get("llm_temperature", 0.7)

            # Add input/output as attributes for Langfuse compatibility
            # Langfuse expects 'input' and 'output' attributes to display them in the UI
            # Use full text (not truncated) for GraphRAG semantic extraction
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
            
            # Langfuse-specific attributes for proper OTel integration
            # These ensure Langfuse correctly interprets the span type and metadata
            attributes["langfuse.session.id"] = self.game_id
            attributes["langfuse.trace.name"] = trace_name
            attributes["langfuse.user.id"] = self.user_id or ""
            attributes["langfuse.observation.type"] = "span"
            
            # Create a new trace (root span) for this player action
            # Each trace is independent but grouped by session_id into a thread
            # This works for both Langfuse and LangSmith
            # start_as_current_span creates a root span if no parent context exists
            with self.tracer.start_as_current_span(
                trace_name,
                attributes=attributes
            ) as root_span:
                # Record trace end time and calculate duration
                trace_end_time = datetime.utcnow()
                duration_ms = (trace_end_time - trace_start_time).total_seconds() * 1000
                
                # Update root span with duration
                root_span.set_attribute("duration_ms", duration_ms)
                root_span.set_attribute("end_time", trace_end_time.isoformat())
                
                # Create explicit LLM generation span for GraphRAG
                span_id = str(uuid.uuid4())
                generation_id = str(uuid.uuid4())
                
                # Store trace/span IDs for score logging
                self.current_trace_ids[player_id] = trace_id
                self.current_span_ids[player_id] = span_id
                
                # Create child span for LLM generation
                # Use Langfuse-specific attributes for proper generation tracking
                with self.tracer.start_as_current_span(
                    "llm_generation",
                    attributes={
                        "span_id": span_id,
                        "trace_id": trace_id,
                        "type": "llm",
                        "generation_id": generation_id,
                        "start_time": trace_start_time.isoformat(),
                        "model": llm_model,
                        "temperature": self.config.get("llm_temperature", 0.7),
                        # Langfuse-specific attributes
                        "langfuse.observation.type": "generation",
                        "langfuse.observation.name": f"player_{player_id}_generation",
                        "langfuse.observation.model.name": llm_model,
                        "gen_ai.system": "openai",
                        "gen_ai.request.model": llm_model,
                    }
                ) as llm_span:
                    llm_span_start = datetime.utcnow()
                    
                    # Add all generation fields to span attributes
                    # Use both standard and Langfuse-specific attribute names
                    if prompt:
                        llm_span.set_attribute("prompt", prompt)
                        llm_span.set_attribute("langfuse.observation.input", prompt)
                        llm_span.set_attribute("gen_ai.prompt", prompt)
                    if response:
                        llm_span.set_attribute("response", response)
                        llm_span.set_attribute("langfuse.observation.output", response)
                        llm_span.set_attribute("gen_ai.completion", response)
                    if system_prompt:
                        llm_span.set_attribute("system_prompt", system_prompt)
                        llm_span.set_attribute("gen_ai.system_prompt", system_prompt)
                    if reasoning:
                        llm_span.set_attribute("reasoning", reasoning)
                        llm_span.set_attribute("langfuse.observation.metadata.reasoning", reasoning)
                    
                    # Add API metrics to span attributes with GraphRAG field names
                    if api_metrics:
                        if api_metrics.get("prompt_tokens") is not None:
                            tokens_input = int(api_metrics["prompt_tokens"])
                            llm_span.set_attribute("tokens_input", tokens_input)
                            llm_span.set_attribute("llm.prompt_tokens", tokens_input)  # Keep for backward compatibility
                        if api_metrics.get("completion_tokens") is not None:
                            tokens_output = int(api_metrics["completion_tokens"])
                            llm_span.set_attribute("tokens_output", tokens_output)
                            llm_span.set_attribute("llm.completion_tokens", tokens_output)  # Keep for backward compatibility
                        if api_metrics.get("total_tokens") is not None:
                            llm_span.set_attribute("llm.total_tokens", int(api_metrics["total_tokens"]))
                        if api_metrics.get("latency") is not None:
                            latency_ms = float(api_metrics["latency"]) * 1000  # Convert seconds to ms
                            llm_span.set_attribute("latency_ms", latency_ms)
                            llm_span.set_attribute("llm.latency_seconds", float(api_metrics["latency"]))  # Keep for backward compatibility
                        if api_metrics.get("cost") is not None:
                            cost = float(api_metrics["cost"])
                            llm_span.set_attribute("cost", cost)
                            llm_span.set_attribute("llm.cost", cost)  # Keep for backward compatibility
                            llm_span.set_attribute("langfuse.observation.usage.cost", cost)
                    
                    # Record span end time and calculate duration
                    llm_span_end = datetime.utcnow()
                    llm_duration_ms = (llm_span_end - llm_span_start).total_seconds() * 1000
                    llm_span.set_attribute("end_time", llm_span_end.isoformat())
                    llm_span.set_attribute("duration_ms", llm_duration_ms)
                    llm_span.set_attribute("status", "success")
                
                # Add events for prompt/response (for backward compatibility and detailed logging)
                if self.config.get("log_llm_prompts", True):
                    if system_prompt:
                        root_span.add_event("prompt.system", {"content": system_prompt[:1000]})
                    root_span.add_event("prompt.user", {"content": prompt[:1000]})
                
                if self.config.get("log_llm_responses", True):
                    root_span.add_event("response.received", {"content": response[:1000]})
                
                root_span.add_event("action.extracted", {"action": float(action)})
                
                # Clean up trace start time tracking
                if trace_id in self.trace_start_times:
                    del self.trace_start_times[trace_id]

            # Debug log for API calls
            if api_metrics:
                logger.debug(
                    f"Logged trace (thread: {self.game_id}): {trace_name} | "
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
            
            # Log error using enhanced error logging if possible
            try:
                trace_id = self.current_trace_ids.get(player_id, f"{self.game_id}_{player_id}_round_{self.current_round}_error")
                span_id = self.current_span_ids.get(player_id)
                self.log_error(
                    trace_id=trace_id,
                    span_id=span_id,
                    error_type=type(e).__name__,
                    message=str(e),
                    stack_trace=None
                )
            except Exception as log_error:
                logger.warning(f"Failed to log error: {log_error}")
            
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
                # Log scores for GraphRAG
                # Use the most recent trace_id for each player if available
                for player_id in range(len(metrics.get("individual_extractions", []))):
                    trace_id = self.current_trace_ids.get(player_id)
                    span_id = self.current_span_ids.get(player_id)
                    
                    if trace_id:
                        # Log cooperation index as score
                        if "cooperation_index" in metrics:
                            self.log_score(
                                trace_id=trace_id,
                                span_id=span_id,
                                name="cooperation_index",
                                value=float(metrics["cooperation_index"]),
                                comment=f"Round {round_num} cooperation index"
                            )
                        
                        # Log resource level as score
                        if "resource_level" in metrics:
                            self.log_score(
                                trace_id=trace_id,
                                span_id=span_id,
                                name="resource_level",
                                value=float(metrics["resource_level"]),
                                comment=f"Round {round_num} resource level"
                            )
                        
                        # Log individual extraction as score
                        if "individual_extractions" in metrics and player_id < len(metrics["individual_extractions"]):
                            extraction = metrics["individual_extractions"][player_id]
                            self.log_score(
                                trace_id=trace_id,
                                span_id=span_id,
                                name="extraction",
                                value=float(extraction),
                                comment=f"Round {round_num} extraction"
                            )
                        
                        # Log individual payoff as score
                        if "individual_payoffs" in metrics and player_id < len(metrics["individual_payoffs"]):
                            payoff = metrics["individual_payoffs"][player_id]
                            self.log_score(
                                trace_id=trace_id,
                                span_id=span_id,
                                name="payoff",
                                value=float(payoff),
                                comment=f"Round {round_num} payoff"
                            )
                
                # Add individual extractions and payoffs as events (for backward compatibility)
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
                    # Langfuse-specific attributes
                    "langfuse.session.id": self.game_id,
                    "langfuse.observation.type": "span",
                    "langfuse.trace.name": "game_summary",
                    # Store summary as output for semantic extraction
                    "output": json.dumps({
                        "total_rounds": summary.get("total_rounds", 0),
                        "final_resource": summary.get("final_resource_level", 0),
                        "tragedy_occurred": summary.get("tragedy_occurred", False),
                        "avg_cooperation_index": summary.get("avg_cooperation_index", 0),
                        "gini_coefficient": summary.get("gini_coefficient", 0),
                    }),
                }
            ) as summary_span:
                summary_span.add_event("game.completed")

            # Flush all pending traces
            logger.info(f"Flushing OTel traces for thread: {self.game_id}...")
            self.otel_manager.flush()
            logger.info(f"[OK] Successfully flushed all traces for thread: {self.game_id}")

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

    def log_session_start(
        self,
        session_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log session start for GraphRAG.
        
        Args:
            session_id: Session identifier (game_id)
            name: Optional session name
            user_id: Optional user identifier
            metadata: Optional session metadata
        """
        if self.tracer is None:
            return
        
        try:
            session_attributes = {
                "session_id": session_id,
                "created_at": datetime.utcnow().isoformat(),
                # Langfuse-specific session attributes
                "langfuse.session.id": session_id,
                "langfuse.observation.type": "span",
            }
            
            if name:
                session_attributes["name"] = name
            else:
                session_attributes["name"] = f"Game Session {session_id}"
            
            if user_id:
                session_attributes["user_id"] = user_id
                session_attributes["langfuse.user.id"] = user_id
            elif self.user_id:
                session_attributes["user_id"] = self.user_id
                session_attributes["langfuse.user.id"] = self.user_id
            
            if metadata:
                session_attributes["metadata"] = json.dumps(metadata)
                # Add game config as Langfuse metadata
                if isinstance(metadata, dict):
                    if "n_players" in metadata:
                        session_attributes["game.n_players"] = metadata["n_players"]
                    if "max_steps" in metadata:
                        session_attributes["game.max_steps"] = metadata["max_steps"]
                    if "llm_model" in metadata:
                        session_attributes["game.llm_model"] = metadata["llm_model"]
            
            # Create a session span to ensure session entity is created
            with self.tracer.start_as_current_span(
                "session_start",
                attributes=session_attributes
            ) as session_span:
                session_span.add_event("session.started")
        except Exception as e:
            logger.warning(f"Error logging session start: {e}")
    
    def log_score(
        self,
        trace_id: str,
        span_id: Optional[str],
        name: str,
        value: float,
        comment: Optional[str] = None
    ):
        """Log a score/metric for a trace or span.
        
        Args:
            trace_id: Trace identifier
            span_id: Optional span identifier
            name: Score name (e.g., "cooperation_index", "resource_level")
            value: Score value
            comment: Optional comment/description
        """
        if self.tracer is None:
            return
        
        try:
            score_id = str(uuid.uuid4())
            score_attributes = {
                "score_id": score_id,
                "trace_id": trace_id,
                "name": name,
                "value": float(value),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if span_id:
                score_attributes["span_id"] = span_id
            
            if comment:
                score_attributes["comment"] = comment
            
            # Create score as span event
            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event("score.recorded", score_attributes)
        except Exception as e:
            logger.warning(f"Error logging score: {e}")
    
    def log_error(
        self,
        trace_id: str,
        span_id: Optional[str],
        error_type: str,
        message: str,
        stack_trace: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log an error for a trace or span.
        
        Args:
            trace_id: Trace identifier
            span_id: Optional span identifier
            error_type: Error type (e.g., "ValueError", "APIError")
            message: Error message
            stack_trace: Optional stack trace
            metadata: Optional error metadata
        """
        if self.tracer is None:
            return
        
        try:
            error_id = str(uuid.uuid4())
            error_attributes = {
                "error_id": error_id,
                "trace_id": trace_id,
                "type": error_type,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if span_id:
                error_attributes["span_id"] = span_id
            
            if stack_trace:
                error_attributes["stack_trace"] = stack_trace
            
            if metadata:
                error_attributes["metadata"] = json.dumps(metadata)
            
            # Create error span or event
            current_span = trace.get_current_span()
            if current_span:
                current_span.record_exception(Exception(message))
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, message))
                current_span.add_event("error.occurred", error_attributes)
        except Exception as e:
            logger.warning(f"Error logging error: {e}")

    def reset(self):
        """Reset manager state for new game."""
        self.game_id = None
        self.current_round = 0
        self.user_id = None
        self.round_metrics = []
        self.generation_data = []
        self.api_metrics_data = []
        self.trace_start_times = {}
        self.current_trace_ids = {}
        self.current_span_ids = {}

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

"""Langfuse logging integration for CPR game tracing and metrics.

Provides hierarchical tracing of games, rounds, and LLM generations
with custom metrics for research analysis.
"""

from typing import Dict, List, Optional, Any
import time
from datetime import datetime

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("Warning: Langfuse not installed. Logging will be disabled.")

from .config import CONFIG


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
        self.enabled = self.config.get("langfuse_enabled", False) and LANGFUSE_AVAILABLE

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=self.config["langfuse_public_key"],
                    secret_key=self.config["langfuse_secret_key"],
                    host=self.config["langfuse_host"]
                )
            except Exception as e:
                print(f"Failed to initialize Langfuse: {e}")
                self.enabled = False
                self.client = None
        else:
            self.client = None

        # Current trace and span tracking
        self.current_trace = None
        self.current_round_span = None
        self.game_id = None

        # Metrics accumulation
        self.round_metrics: List[Dict] = []
        self.generation_data: List[Dict] = []

    def start_game_trace(self, game_id: str, config: Dict) -> Optional[Any]:
        """Initialize top-level trace for a game.

        Args:
            game_id: Unique identifier for this game
            config: Game configuration

        Returns:
            Trace object or None if disabled
        """
        if not self.enabled:
            return None

        self.game_id = game_id

        try:
            self.current_trace = self.client.trace(
                name=f"CPR_Game_{game_id}",
                metadata={
                    "game_id": game_id,
                    "timestamp": datetime.now().isoformat(),
                    "n_players": config["n_players"],
                    "max_steps": config["max_steps"],
                    "personas": config["player_personas"][:config["n_players"]],
                    "llm_model": config["llm_model"],
                },
                tags=["cpr_game", "multi_agent", "llm"]
            )
            return self.current_trace
        except Exception as e:
            print(f"Error starting game trace: {e}")
            return None

    def start_round_span(self, round_num: int, game_state: Dict) -> Optional[Any]:
        """Start a span for a single round.

        Args:
            round_num: Round number
            game_state: Current game state

        Returns:
            Span object or None if disabled
        """
        if not self.enabled or self.current_trace is None:
            return None

        try:
            self.current_round_span = self.current_trace.span(
                name=f"round_{round_num}",
                metadata={
                    "round": round_num,
                    "resource_level": game_state.get("resource", 0),
                    "step": game_state.get("step", 0),
                }
            )
            return self.current_round_span
        except Exception as e:
            print(f"Error starting round span: {e}")
            return None

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
        if not self.enabled or self.current_trace is None:
            return

        try:
            generation_metadata = {
                "player_id": player_id,
                "action": action,
                "reasoning": reasoning or "",
            }

            if metadata:
                generation_metadata.update(metadata)

            # Log as generation
            self.current_trace.generation(
                name=f"player_{player_id}_decision",
                model=self.config["llm_model"],
                input=prompt if self.config["log_llm_prompts"] else "[prompt hidden]",
                output=response if self.config["log_llm_responses"] else "[response hidden]",
                metadata=generation_metadata,
            )

            # Store for later analysis
            self.generation_data.append({
                "player_id": player_id,
                "prompt": prompt,
                "response": response,
                "action": action,
                "reasoning": reasoning,
            })

        except Exception as e:
            print(f"Error logging generation: {e}")

    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Log metrics for a completed round.

        Args:
            round_num: Round number
            metrics: Dictionary of metric values
        """
        if not self.enabled or self.current_trace is None:
            return

        try:
            # Log as scores on the current trace
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.current_trace.score(
                        name=f"round_{round_num}_{metric_name}",
                        value=float(value)
                    )

            # Store for aggregation
            metrics["round"] = round_num
            self.round_metrics.append(metrics)

        except Exception as e:
            print(f"Error logging round metrics: {e}")

    def end_round_span(self):
        """End the current round span."""
        if not self.enabled or self.current_round_span is None:
            return

        try:
            # Span automatically ends when context exits
            self.current_round_span = None
        except Exception as e:
            print(f"Error ending round span: {e}")

    def end_game_trace(self, summary: Dict):
        """Finalize game trace with summary metrics.

        Args:
            summary: Game summary statistics
        """
        if not self.enabled or self.current_trace is None:
            return

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

            for score_name, value in game_scores.items():
                if isinstance(value, (int, float)):
                    self.current_trace.score(
                        name=score_name,
                        value=float(value)
                    )

            # Add summary metadata
            self.current_trace.update(
                metadata={
                    **self.current_trace.metadata,
                    "summary": summary,
                    "end_time": datetime.now().isoformat(),
                }
            )

            # Flush to Langfuse
            if self.client:
                self.client.flush()

        except Exception as e:
            print(f"Error ending game trace: {e}")
        finally:
            self.current_trace = None
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
        self.current_trace = None
        self.current_round_span = None
        self.game_id = None
        self.round_metrics = []
        self.generation_data = []

    def __del__(self):
        """Cleanup: flush any pending traces."""
        if self.enabled and hasattr(self, 'client') and self.client:
            try:
                self.client.flush()
            except:
                pass


class MockLoggingManager(LoggingManager):
    """Mock logging manager for testing without Langfuse.

    Stores all logs in memory for inspection.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize mock manager."""
        self.config = config if config is not None else CONFIG
        self.enabled = True  # Always "enabled" for mock

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

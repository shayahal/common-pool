"""API call logging utility for tracking LLM API metrics.

Provides structured logging of API calls with timing, token usage, and cost estimation.
"""

from typing import Dict, List, Optional, Any
import time
import json
from datetime import datetime
from pathlib import Path


# OpenAI pricing per 1K tokens (as of 2024, update as needed)
# Format: {model_name: {"input": price_per_1k_tokens, "output": price_per_1k_tokens}}
MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}


def get_token_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost for API call based on model and token usage.
    
    Args:
        model: Model name (e.g., "gpt-3.5-turbo")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    # Normalize model name (remove version suffixes if present)
    model_base = model.split("-")[0:3]  # e.g., ["gpt", "3", "5"]
    if len(model_base) >= 2:
        model_key = "-".join(model_base[:2])  # e.g., "gpt-3"
        if model_key == "gpt-3":
            model_key = "gpt-3.5-turbo"  # Default to turbo for gpt-3.5
        elif model_key == "gpt-4":
            # Check for specific gpt-4 variants
            if "turbo" in model.lower():
                if "mini" in model.lower() or "o-mini" in model.lower():
                    model_key = "gpt-4o-mini"
                elif "o" in model.lower():
                    model_key = "gpt-4o"
                else:
                    model_key = "gpt-4-turbo"
            else:
                model_key = "gpt-4"
    else:
        model_key = model
    
    # Get pricing for this model
    pricing = MODEL_PRICING.get(model_key, MODEL_PRICING.get("gpt-3.5-turbo"))  # Default to gpt-3.5-turbo
    
    input_cost = (prompt_tokens / 1000.0) * pricing["input"]
    output_cost = (completion_tokens / 1000.0) * pricing["output"]
    
    return input_cost + output_cost


class APILogger:
    """Logger for tracking API call metrics.
    
    Tracks timing, token usage, costs, and errors for LLM API calls.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize API logger.
        
        Args:
            log_dir: Directory to store API call logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # In-memory storage for metrics
        self.api_calls: List[Dict] = []
        self.errors: List[Dict] = []
        
        # Aggregated metrics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.error_count = 0
    
    def log_api_call(
        self,
        player_id: int,
        model: str,
        prompt: str,
        response: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Log an API call with all metrics.
        
        Args:
            player_id: Player identifier
            model: Model name used
            prompt: Input prompt text
            response: Response text (if successful)
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            total_tokens: Total tokens used
            latency: API call latency in seconds
            success: Whether call was successful
            error: Error message if call failed
            metadata: Additional metadata
            
        Returns:
            Dictionary with logged call data
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate cost if tokens available
        cost = 0.0
        if prompt_tokens is not None and completion_tokens is not None:
            cost = get_token_cost(model, prompt_tokens, completion_tokens)
        elif total_tokens is not None:
            # Estimate split if only total available (assume 80/20 input/output)
            estimated_prompt = int(total_tokens * 0.8)
            estimated_completion = int(total_tokens * 0.2)
            cost = get_token_cost(model, estimated_prompt, estimated_completion)
        
        # Build call record
        call_record = {
            "timestamp": timestamp,
            "player_id": player_id,
            "model": model,
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens or (prompt_tokens + completion_tokens if prompt_tokens and completion_tokens else None),
            "latency": latency,
            "cost": cost,
            "success": success,
            "error": error,
            "metadata": metadata or {},
        }
        
        # Store in memory
        self.api_calls.append(call_record)
        
        # Update aggregated metrics
        self.total_calls += 1
        if total_tokens:
            self.total_tokens += total_tokens
        elif prompt_tokens and completion_tokens:
            self.total_tokens += (prompt_tokens + completion_tokens)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
        
        if latency:
            self.total_latency += latency
        
        self.total_cost += cost
        
        if not success:
            self.error_count += 1
            error_record = {
                "timestamp": timestamp,
                "player_id": player_id,
                "model": model,
                "error": error,
                "metadata": metadata or {},
            }
            self.errors.append(error_record)
        
        # Log to file (structured JSON)
        # Note: Python logger calls are handled in llm_agent.py to avoid duplication
        self._write_to_file(call_record)
        
        return call_record
    
    def _write_to_file(self, call_record: Dict):
        """Write API call record to JSON log file.
        
        Args:
            call_record: Call record to write
        """
        log_file = self.log_dir / "api_calls.log"
        
        # Append JSON line to file
        with open(log_file, "a") as f:
            json.dump(call_record, f)
            f.write("\n")
    
    def get_metrics_summary(self) -> Dict:
        """Get aggregated metrics summary.
        
        Returns:
            Dictionary with summary statistics
        """
        avg_latency = self.total_latency / self.total_calls if self.total_calls > 0 else 0.0
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost": self.total_cost,
            "avg_latency": avg_latency,
            "error_count": self.error_count,
            "success_rate": (self.total_calls - self.error_count) / self.total_calls if self.total_calls > 0 else 0.0,
        }
    
    def get_api_calls(self) -> List[Dict]:
        """Get all logged API calls.
        
        Returns:
            List of API call records
        """
        return self.api_calls.copy()
    
    def get_errors(self) -> List[Dict]:
        """Get all error records.
        
        Returns:
            List of error records
        """
        return self.errors.copy()
    
    def reset(self):
        """Reset logger state."""
        self.api_calls = []
        self.errors = []
        self.total_calls = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.error_count = 0


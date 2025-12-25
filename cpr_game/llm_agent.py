"""LLM-based agent for Common Pool Resource game.

Uses Large Language Models to make extraction decisions based on
game state, history, and persona-based reasoning.
"""

from typing import Dict, List, Optional, Tuple
import time
import numpy as np
import random
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError, AuthenticationError
try:
    import httpx
except ImportError:
    httpx = None
from pydantic import BaseModel, Field, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import CONFIG
from .utils import validate_action
from .logger_setup import get_logger
from .agent_prompts import build_game_prompt, build_delta_prompt
from .persona_prompts import GAME_RULES_PROMPT

logger = get_logger(__name__)


class AgentResponse(BaseModel):
    """Structured output from LLM agent."""
    reasoning: str = Field(description="The agent's reasoning for the chosen action")
    action: int = Field(description="The extraction amount as an integer (0 to max_extraction)")


class LLMAgent:
    """Agent that uses LLM reasoning to decide extraction amounts.

    The agent maintains memory of past actions and observations,
    constructs prompts with game context, and parses LLM responses
    to extract numerical actions.
    """

    def __init__(
        self,
        player_id: int,
        persona: str,
        config: Optional[Dict] = None,
        api_key: Optional[str] = None
    ):
        """Initialize LLM agent.

        Args:
            player_id: Unique identifier for this player
            persona: Persona type (e.g., "rational_selfish", "cooperative")
            config: Configuration dictionary
            api_key: OpenRouter API key (if None, reads from config)
        """
        self.player_id = player_id
        self.persona = persona
        self.config = config if config is not None else CONFIG

        # LLM settings
        self.llm_model = self.config["llm_model"]
        self.temperature = self.config["llm_temperature"]
        self.max_tokens = self.config["llm_max_tokens"]
        self.timeout = self.config["llm_timeout"]

        # Game parameters
        self.min_extraction = self.config["min_extraction"]
        self.max_extraction = self.config["max_extraction"]
        self.history_rounds = self.config["include_history_rounds"]

        # Get persona prompt and combine with game rules
        # Validate persona is not empty and exists in persona_prompts
        if not persona or persona.strip() == "":
            persona = "null"
        
        persona_prompt = self.config["persona_prompts"].get(
            persona,
            self.config["persona_prompts"].get("null", "")
        )
        
        # Final fallback if persona_prompt is still empty
        if not persona_prompt or persona_prompt.strip() == "":
            error_msg = f"Player {self.player_id}: Invalid persona '{persona}' - no valid persona prompt found"
            logger.error(error_msg)
            raise ValueError(error_msg)
        # Combine persona + game rules for system prompt
        # Game rules are static and don't change between rounds
        game_rules = self.config.get("game_rules_prompt", GAME_RULES_PROMPT)
        self.system_prompt = f"{persona_prompt}\n\n{game_rules}"

        # Initialize OpenRouter client (for API logging)
        api_key = api_key or self.config.get("openrouter_api_key")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        # OpenRouter uses OpenAI-compatible API with custom base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Initialize LangChain ChatOpenAI with structured output (using OpenRouter)
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-repo",  # Optional: for OpenRouter analytics
                "X-Title": "CPR Game"  # Optional: for OpenRouter analytics
            }
        ).with_structured_output(AgentResponse)

        # Memory
        self.observation_history: List[Dict] = []
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.reasoning_history: List[str] = []
        
        # For incremental prompts: store previous round state
        self.use_incremental_prompts = self.config.get("use_incremental_prompts", False)
        self.previous_observation: Optional[Dict] = None
        self.conversation_messages: List = []  # Store conversation history for incremental mode
        
        # API call metrics storage
        self.last_api_metrics: Optional[Dict] = None

    def act(
        self,
        observation: Dict,
        return_reasoning: bool = True
    ) -> Tuple[int, Optional[str]]:
        """Decide extraction amount based on current observation.

        Args:
            observation: Current game state observation
            return_reasoning: Whether to return LLM reasoning text

        Returns:
            Tuple of (extraction_amount, reasoning_text)
            extraction_amount is an integer
        """
        # Build prompt
        prompt = self._build_prompt(observation)

        # Get LLM response with timing and logging
        start_time = time.time()
        structured_response = None
        response_text = None
        reasoning = None
        action = None
        api_metrics = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency": None,
            "success": False,
            "error": None,
        }
        
        # Retry logic with exponential backoff for rate limits
        max_retries = 10  # Increased for rate limits
        base_delay = 5.0  # Start with 5 seconds (increased for rate limits)
        max_delay = 300.0  # Cap at 5 minutes (increased for rate limits)
        structured_response = None
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Player {self.player_id}: Making API call to {self.llm_model} (attempt {attempt + 1}/{max_retries})")
                
                # Use LangChain structured output
                # Enable prompt caching for system message (static content, cached after first round)
                # OpenRouter's cache_control can be set via additional_kwargs on the message
                # Note: This requires OpenRouter API version that supports caching (gpt-3.5-turbo, gpt-4, etc.)
                try:
                    system_message = SystemMessage(
                        content=self.system_prompt,
                        additional_kwargs={"cache_control": {"type": "ephemeral"}}
                    )
                except (TypeError, ValueError):
                    # Fallback if cache_control not supported in this LangChain version
                    # System message will still be cached by OpenRouter if model supports it
                    system_message = SystemMessage(content=self.system_prompt)
                
                messages = [
                    system_message,
                    HumanMessage(content=prompt)
                ]
                
                # Use LangChain callback to get token usage
                from langchain_community.callbacks import get_openai_callback
                with get_openai_callback() as cb:
                    structured_response = self.llm.invoke(messages)
                    api_metrics["prompt_tokens"] = cb.prompt_tokens
                    api_metrics["completion_tokens"] = cb.completion_tokens
                    api_metrics["total_tokens"] = cb.total_tokens
                
                # Success - break out of retry loop
                break
                
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    # For rate limits, use longer delays
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    # Rate limits often need more time - double the delay
                    delay = min(delay * 2, max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    total_delay = delay + jitter
                    
                    logger.info(
                        f"Player {self.player_id}: Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    # Max retries reached, re-raise the error
                    logger.warning(
                        f"Player {self.player_id}: Rate limit error after {max_retries} attempts"
                    )
                    raise
            except Exception as e:
                # Check if it's an httpx error (if httpx is available)
                if httpx and isinstance(e, (httpx.HTTPStatusError, httpx.HTTPError)):
                    # Catch HTTP errors including 429 from underlying HTTP client
                    status_code = None
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                    elif hasattr(e, 'status_code'):
                        status_code = e.status_code
                    
                    if status_code == 429:
                        if attempt < max_retries - 1:
                            # Calculate exponential backoff with jitter
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                            total_delay = delay + jitter
                            
                            logger.info(
                                f"Player {self.player_id}: Rate limit hit (HTTP 429, attempt {attempt + 1}/{max_retries}). "
                                f"Retrying in {total_delay:.2f}s..."
                            )
                            time.sleep(total_delay)
                            continue
                        else:
                            # Max retries reached, re-raise the error
                            logger.warning(
                                f"Player {self.player_id}: Rate limit error (HTTP 429) after {max_retries} attempts"
                            )
                            raise
                    else:
                        # Not a rate limit error, re-raise
                        raise
                
                # Check if error message indicates rate limit
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        # Calculate exponential backoff with jitter
                        # Increased delays for rate limits
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        # For rate limits, add extra delay (rate limits often need more time)
                        if "rate limit" in error_str or "429" in error_str:
                            delay = min(delay * 2, max_delay)  # Double the delay for rate limits
                        jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                        total_delay = delay + jitter
                        
                        logger.info(
                            f"Player {self.player_id}: Rate limit detected (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {total_delay:.2f}s..."
                        )
                        time.sleep(total_delay)
                        continue
                    else:
                        # Max retries reached, re-raise the error
                        logger.warning(
                            f"Player {self.player_id}: Rate limit error after {max_retries} attempts"
                        )
                        raise
                else:
                    # Not a rate limit error, re-raise
                    raise
        
        # Extract reasoning and action from structured response (after successful API call)
        try:
            reasoning = structured_response.reasoning
            action = structured_response.action
            
            # Format response text for logging
            response_text = f"Reasoning: {reasoning}\nAction: {action}"
            
            api_metrics["success"] = True
            api_metrics["latency"] = time.time() - start_time
            
            logger.debug(
                f"Player {self.player_id}: API call successful | "
                f"Tokens: {api_metrics['total_tokens']} | "
                f"Latency: {api_metrics['latency']:.2f}s"
            )

        except (APIError, APIConnectionError, APITimeoutError, AuthenticationError) as e:
            api_metrics["latency"] = time.time() - start_time
            api_metrics["error"] = f"{type(e).__name__}: {str(e)}"
            
            # Don't print stack trace for rate limit errors
            if isinstance(e, RateLimitError) or "429" in str(e) or "rate limit" in str(e).lower():
                logger.warning(
                    f"Player {self.player_id}: Rate limit error - {type(e).__name__}: {e}"
                )
            else:
                logger.error(
                    f"Player {self.player_id}: API error - {type(e).__name__}: {e}",
                    exc_info=True
                )
            
            # Store API metrics for retrieval
            self.last_api_metrics = api_metrics
            
            # Re-raise the error instead of falling back
            raise
            
        except (AttributeError, IndexError, KeyError, ValueError, ValidationError) as e:
            api_metrics["latency"] = time.time() - start_time
            api_metrics["error"] = f"{type(e).__name__}: {str(e)}"
            
            logger.error(
                f"Player {self.player_id}: Error parsing LLM response - {type(e).__name__}: {e}",
                exc_info=True
            )
            
            # Store API metrics for retrieval
            self.last_api_metrics = api_metrics
            
            # Re-raise the error instead of falling back
            raise
            
            
        except Exception as e:
            api_metrics["latency"] = time.time() - start_time
            api_metrics["error"] = f"{type(e).__name__}: {str(e)}"
            
            logger.error(
                f"Player {self.player_id}: Unexpected error calling LLM - {type(e).__name__}: {e}",
                exc_info=True
            )
            
            # Store API metrics for retrieval
            self.last_api_metrics = api_metrics
            
            # Re-raise the error instead of falling back
            raise
        
        # Store API metrics for retrieval
        self.last_api_metrics = api_metrics

        # Validate action
        action = validate_action(action, self.min_extraction, self.max_extraction)

        # Store action and reasoning
        self.action_history.append(action)
        if reasoning:
            self.reasoning_history.append(reasoning)

        if return_reasoning:
            return action, reasoning
        else:
            return action, None

    def _build_prompt(self, observation: Dict) -> str:
        """Construct LLM prompt from observation and history.

        Args:
            observation: Current game state observation

        Returns:
            str: Formatted prompt for LLM
        """
        # Use incremental/delta prompts if enabled
        if self.use_incremental_prompts:
            prompt = build_delta_prompt(
                observation=observation,
                previous_observation=self.previous_observation,
                player_id=self.player_id,
                action_history=self.action_history,
                config=self.config,
                history_rounds=self.history_rounds
            )
            # Store current observation for next round
            self.previous_observation = observation.copy()
            return prompt
        else:
            # Standard prompt building
            return build_game_prompt(
                observation=observation,
                player_id=self.player_id,
                action_history=self.action_history,
                config=self.config,
                history_rounds=self.history_rounds
            )

    def update_memory(self, observation: Dict, action: float, reward: float):
        """Update agent's memory with new experience.

        Args:
            observation: Observation received
            action: Action taken
            reward: Reward received
        """
        self.observation_history.append(observation)

        # action is already stored in act(), but update reward
        if len(self.reward_history) < len(self.action_history):
            self.reward_history.append(reward)

    def get_last_reasoning(self) -> Optional[str]:
        """Get the most recent reasoning text.

        Returns:
            str: Last reasoning, or None if no reasoning available
        """
        if len(self.reasoning_history) > 0:
            return self.reasoning_history[-1]
        return None

    def get_last_api_metrics(self) -> Optional[Dict]:
        """Get metrics from the last API call.
        
        Returns:
            Dictionary with API call metrics or None
        """
        return self.last_api_metrics

    def reset(self):
        """Reset agent's memory for new episode."""
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.reasoning_history = []
        self.previous_observation = None
        self.conversation_messages = []
        self.last_api_metrics = None

    def __repr__(self) -> str:
        """String representation of agent."""
        return f"LLMAgent(player_id={self.player_id}, persona='{self.persona}')"


class MockLLMAgent(LLMAgent):
    """Mock LLM agent for testing without API calls.

    Uses simple heuristics instead of actual LLM calls.
    """

    def __init__(self, player_id: int, persona: str, config: Optional[Dict] = None):
        """Initialize mock agent.

        Args:
            player_id: Unique identifier for this player
            persona: Persona type
            config: Configuration dictionary
        """
        self.player_id = player_id
        self.persona = persona
        self.config = config if config is not None else CONFIG

        # Game parameters
        self.min_extraction = self.config["min_extraction"]
        self.max_extraction = self.config["max_extraction"]
        self.history_rounds = self.config["include_history_rounds"]

        # Memory
        self.observation_history: List[Dict] = []
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.reasoning_history: List[str] = []
        
        # For incremental prompts: store previous round state (needed for _build_prompt)
        self.use_incremental_prompts = self.config.get("use_incremental_prompts", False)
        self.previous_observation: Optional[Dict] = None
        self.conversation_messages: List = []
        
        # API call metrics storage (None for mock agents since no API calls are made)
        self.last_api_metrics: Optional[Dict] = None

    def act(
        self,
        observation: Dict,
        return_reasoning: bool = True
    ) -> Tuple[int, Optional[str]]:
        """Decide extraction using simple heuristics.

        Args:
            observation: Current game state observation
            return_reasoning: Whether to return reasoning text

        Returns:
            Tuple of (extraction_amount, reasoning_text)
            extraction_amount is an integer
        """
        resource = observation["resource_level"][0]

        # Simple heuristic based on persona
        if self.persona == "rational_selfish":
            # Extract more aggressively (30% of resource)
            action = min(resource * 0.30, self.max_extraction)
            reasoning = "Maximizing personal gain by extracting aggressively."
        elif self.persona == "cooperative":
            # Extract sustainably (5% of resource)
            action = min(resource * 0.05, self.max_extraction)
            reasoning = "Extracting conservatively to maintain resource for everyone."
        else:
            # Moderate extraction
            action = min(resource * 0.10, self.max_extraction)
            reasoning = "Extracting a moderate amount."

        # Add some randomness (reduced range for more consistent behavior)
        action = action * np.random.uniform(0.1, 3.0)
        action = int(np.clip(action, self.min_extraction, self.max_extraction))

        # Store action and reasoning
        self.action_history.append(action)
        self.reasoning_history.append(reasoning)

        if return_reasoning:
            return action, reasoning
        else:
            return action, None

    def update_memory(self, observation: Dict, action: float, reward: float):
        """Update agent's memory."""
        self.observation_history.append(observation)
        if len(self.reward_history) < len(self.action_history):
            self.reward_history.append(reward)

    def get_last_reasoning(self) -> Optional[str]:
        """Get the most recent reasoning text."""
        if len(self.reasoning_history) > 0:
            return self.reasoning_history[-1]
        return None

    def reset(self):
        """Reset agent's memory for new episode."""
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.reasoning_history = []
        self.previous_observation = None
        self.conversation_messages = []
        self.last_api_metrics = None

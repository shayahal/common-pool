"""LLM-based agent for Common Pool Resource game.

Uses Large Language Models to make extraction decisions based on
game state, history, and persona-based reasoning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError, AuthenticationError

from .config import CONFIG
from .utils import parse_extraction_from_text, validate_action


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
            api_key: OpenAI API key (if None, reads from config)
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

        # Get persona prompt
        self.system_prompt = self.config["persona_prompts"].get(
            persona,
            self.config["persona_prompts"][""]
        )

        # Initialize OpenAI client
        api_key = api_key or self.config.get("openai_api_key")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=api_key)

        # Memory
        self.observation_history: List[Dict] = []
        self.action_history: List[int] = []
        self.reward_history: List[float] = []
        self.reasoning_history: List[str] = []

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

        # Get LLM response
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            response_text = response.choices[0].message.content

        except (APIError, APIConnectionError, RateLimitError, APITimeoutError, AuthenticationError) as e:
            print(f"Error calling LLM API: {type(e).__name__}: {e}")
            # Fallback to random action (integer)
            response_text = f"EXTRACT: {int(np.random.uniform(self.min_extraction, self.max_extraction))}"
        except (AttributeError, IndexError, KeyError) as e:
            print(f"Error parsing LLM response: {type(e).__name__}: {e}")
            # Fallback to random action (integer)
            response_text = f"EXTRACT: {int(np.random.uniform(self.min_extraction, self.max_extraction))}"
        except Exception as e:
            # Catch any other unexpected exceptions
            print(f"Unexpected error calling LLM: {type(e).__name__}: {e}")
            # Fallback to random action (integer)
            response_text = f"EXTRACT: {int(np.random.uniform(self.min_extraction, self.max_extraction))}"

        # Parse action from response
        action, reasoning = parse_extraction_from_text(
            response_text,
            self.min_extraction,
            self.max_extraction
        )

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
        resource = observation["resource_level"][0]
        step = observation["step"][0]
        my_extractions = observation["my_recent_extractions"]
        other_extractions = observation["other_players_recent_extractions"]
        my_payoff = observation["my_cumulative_payoff"][0]
        other_payoffs = observation["other_players_cumulative_payoffs"]

        prompt = f"""=== Common Pool Resource Game - Round {step + 1} ===

Resource Status:
- Current resource level: {resource:.2f}

"""

        # Add history if available
        if len(self.action_history) > 0:
            prompt += "Your History (last few rounds):\n"
            history_length = min(len(self.action_history), self.history_rounds)
            for i in range(history_length):
                idx = -(history_length - i)
                round_num = len(self.action_history) + idx
                extraction = self.action_history[idx]
                # Check if reward_history has enough elements for this negative index
                reward = self.reward_history[idx] if abs(idx) <= len(self.reward_history) else 0.0
                prompt += f"- Round {round_num}: Extracted {extraction}, Earned {reward:.2f}\n"
            prompt += "\n"

        # Add other players' history if enabled
        if self.config["show_other_players_actions"] and len(self.action_history) > 0 and other_extractions.shape[0] > 0:
            prompt += "Other Players' History:\n"

            # Get the number of other players
            n_other_players = other_extractions.shape[1]

            # Get recent rounds (non-zero entries)
            for player_idx in range(n_other_players):
                prompt += f"- Player {player_idx if player_idx < self.player_id else player_idx + 1}:\n"

                # Show last few rounds
                history_length = min(len(self.action_history), self.history_rounds)
                for i in range(history_length):
                    idx = -(history_length - i)
                    round_num = len(self.action_history) + idx
                    # Get extraction from history (other_extractions is padded with zeros)
                    # Use positive indexing relative to available history
                    obs_idx = i
                    if obs_idx < other_extractions.shape[0]:
                        extraction = other_extractions[obs_idx, player_idx]
                        if extraction > 0 or round_num >= 0:  # Only show if not padding
                            prompt += f"  Round {round_num}: Extracted {int(extraction)}\n"

            prompt += "\n"

        # Add current standings
        prompt += "Current Standings:\n"
        prompt += f"- Your total earnings: {my_payoff:.2f}\n"

        for i, payoff in enumerate(other_payoffs):
            player_num = i if i < self.player_id else i + 1
            prompt += f"- Player {player_num}'s total earnings: {payoff:.2f}\n"

        prompt += "\n"

        # Add instructions
        prompt += f"""Instructions:
1. Analyze the current situation carefully
2. Consider the resource level and available extraction
3. Think about what other players might do
4. Decide how much to extract this round (0-{int(self.max_extraction)})
5. Your extraction amount must be a whole number (integer)
6. Explain your reasoning briefly
7. State your action clearly as "EXTRACT: <integer>"

Your response:"""

        return prompt

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

    def reset(self):
        """Reset agent's memory for new episode."""
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.reasoning_history = []

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
        action = action * np.random.uniform(0.9, 1.1)
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

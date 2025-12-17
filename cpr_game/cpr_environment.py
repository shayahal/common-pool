"""Common Pool Resource (CPR) Gymnasium Environment.

Implements a multi-agent resource extraction game with regeneration dynamics.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import CONFIG
from .utils import compute_cooperation_index, compute_gini_coefficient


class CPREnvironment(gym.Env):
    """Common Pool Resource game environment.

    A multi-agent environment where players simultaneously extract from a
    shared resource pool that regenerates each round.

    The resource dynamics follow:
        R(t+1) = max(min(R(t) * regeneration_rate, max_fishes) - sum(extractions), min_resource)

    Players receive rewards based on their extraction.
    """

    metadata = {"render_modes": ["human", "dict"], "name": "CPREnvironment-v0"}

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the CPR environment.

        Args:
            config: Configuration dictionary. If None, uses default CONFIG.
        """
        super().__init__()

        # Load configuration
        self.config = config if config is not None else CONFIG

        # Game parameters
        self.n_players = self.config["n_players"]
        self.max_steps = self.config["max_steps"]
        self.initial_resource = self.config["initial_resource"]
        self.regeneration_rate = float(self.config["regeneration_rate"])
        self.min_resource = self.config["min_resource"]
        max_fishes_val = self.config.get("max_fishes", float("inf"))
        self.max_fishes = float(max_fishes_val) if max_fishes_val != float("inf") else max_fishes_val
        
        # Reward parameters
        self.sustainability_threshold = float(self.config.get("sustainability_threshold", self.n_players))
        self.sustainability_bonus = float(self.config.get("sustainability_bonus", 0.0))
        self.depletion_penalty = float(self.config.get("depletion_penalty", 0.0))

        # Action space
        self.min_extraction = self.config["min_extraction"]
        self.max_extraction = self.config["max_extraction"]

        # Define action and observation spaces
        # Action space: continuous extraction amount for each player
        self.action_space = spaces.Box(
            low=self.min_extraction,
            high=self.max_extraction,
            shape=(self.n_players,),
            dtype=np.float32
        )

        # Observation space: each player observes resource, step, and history
        # We'll use a dict observation space
        self.observation_space = spaces.Dict({
            "resource_level": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "step": spaces.Box(low=0, high=self.max_steps, shape=(1,), dtype=np.int32),
            "my_recent_extractions": spaces.Box(
                low=0, high=self.max_extraction,
                shape=(self.config["include_history_rounds"],), dtype=np.float32
            ),
            "other_players_recent_extractions": spaces.Box(
                low=0, high=self.max_extraction,
                shape=(self.n_players - 1, self.config["include_history_rounds"]),
                dtype=np.float32
            ),
            "my_cumulative_payoff": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "other_players_cumulative_payoffs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_players - 1,), dtype=np.float32
            ),
        })

        # Game state
        self.current_resource: int = 0
        self.current_step: int = 0
        self.extraction_history: List[np.ndarray] = []
        self.payoff_history: List[np.ndarray] = []
        self.player_cumulative_payoffs: np.ndarray = np.zeros(self.n_players)
        self.resource_history: List[int] = []
        self.cooperation_history: List[float] = []

        # Game status
        self.done: bool = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset game state
        self.current_resource = self.initial_resource
        self.current_step = 0
        self.extraction_history = []
        self.payoff_history = []
        self.player_cumulative_payoffs = np.zeros(self.n_players)
        self.resource_history = [self.initial_resource]
        self.cooperation_history = []
        self.done = False

        # Get initial observation
        observation = self._get_observations()

        info = {
            "resource": self.current_resource,
            "step": self.current_step,
        }

        return observation, info

    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict, np.ndarray, bool, bool, Dict]:
        """Execute one step of the environment.

        Args:
            actions: Array of extraction amounts (one per player)

        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")

        # Validate and clip actions
        actions = np.clip(actions, self.min_extraction, self.max_extraction)

        # Store actions
        self.extraction_history.append(actions.copy())

        # Calculate total extraction
        total_extraction = np.sum(actions)

        # Store resource before step for reward calculation
        resource_before_step = self.current_resource

        # Update resource with regeneration and extraction
        # R(t+1) = max(min(R(t) * regeneration_rate, max_fishes) - sum(extractions), min_resource)
        regenerated_resource = self.current_resource * self.regeneration_rate
        # Cap regeneration at max_fishes (resource capacity limit)
        if self.max_fishes != float("inf"):
            regenerated_resource = min(regenerated_resource, self.max_fishes)
        # Calculate new resource and ensure it's an integer
        new_resource = regenerated_resource - total_extraction
        self.current_resource = max(
            int(new_resource),
            self.min_resource
        )

        # Store resource level
        self.resource_history.append(self.current_resource)

        # Calculate rewards
        rewards = self._compute_rewards(actions, resource_before_step)

        # Store rewards and update cumulative payoffs
        self.payoff_history.append(rewards.copy())
        self.player_cumulative_payoffs += rewards

        # Calculate cooperation index
        cooperation = compute_cooperation_index(actions, self.max_extraction)
        self.cooperation_history.append(cooperation)

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Resource depletion
        if self.current_resource <= self.min_resource:
            terminated = True

        # Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        self.done = terminated or truncated

        # Get observations
        observations = self._get_observations()

        # Build info dict
        info = {
            "resource": self.current_resource,
            "step": self.current_step,
            "total_extraction": float(total_extraction),
            "cooperation_index": cooperation,
            "cumulative_payoffs": self.player_cumulative_payoffs.copy(),
            "tragedy_occurred": self.current_resource <= self.min_resource,
        }

        return observations, rewards, terminated, truncated, info

    def _compute_rewards(self, actions: np.ndarray, resource_before_step: int) -> np.ndarray:
        """Calculate rewards for each player based on their actions.

        Reward function:
            reward_i = extraction_i (value per unit is 1)
            + sustainability_bonus if resource_after >= sustainability_threshold

        Args:
            actions: Array of player extractions
            resource_before_step: Resource level before this step (for bonus calculation)

        Returns:
            Array of rewards (one per player)
        """
        # Base reward: extraction amount (value is 1 per unit)
        # Convert to float to allow adding float bonuses
        rewards = actions.astype(float).copy()

        # Sustainability bonus (based on resource AFTER step)
        if self.current_resource >= self.sustainability_threshold:
            rewards += self.sustainability_bonus

        # Convert rewards to int (round down any fractional parts)
        return np.floor(rewards).astype(int)

    def _compute_cooperation_index(self) -> float:
        """Compute cooperation index for current state.

        Returns:
            float: Cooperation index (0-1)
        """
        if len(self.cooperation_history) == 0:
            return 0.0
        return float(np.mean(self.cooperation_history))

    def _get_observations(self) -> Dict:
        """Build observation dictionary for all agents.

        Each agent gets:
        - Current resource level
        - Current step
        - Their own recent extraction history
        - Other players' recent extraction history (full observability)
        - Their own cumulative payoff
        - Other players' cumulative payoffs

        Returns:
            Dict: Observation dictionary with one entry per player
        """
        history_rounds = self.config["include_history_rounds"]

        observations = {}

        for player_id in range(self.n_players):
            # Get recent extraction history
            recent_extractions = self._get_recent_extractions(history_rounds)

            # Player's own extractions
            if len(recent_extractions) > 0:
                my_extractions = recent_extractions[:, player_id]
            else:
                my_extractions = np.zeros(history_rounds)

            # Pad if necessary
            if len(my_extractions) < history_rounds:
                padding = np.zeros(history_rounds - len(my_extractions))
                my_extractions = np.concatenate([padding, my_extractions])

            # Other players' extractions
            other_player_indices = [i for i in range(self.n_players) if i != player_id]
            if len(recent_extractions) > 0:
                other_extractions = recent_extractions[:, other_player_indices]
            else:
                other_extractions = np.zeros((history_rounds, self.n_players - 1))

            # Pad if necessary
            if other_extractions.shape[0] < history_rounds:
                padding_rows = history_rounds - other_extractions.shape[0]
                padding = np.zeros((padding_rows, self.n_players - 1))
                other_extractions = np.vstack([padding, other_extractions])

            # Other players' cumulative payoffs
            other_payoffs = self.player_cumulative_payoffs[other_player_indices]

            observations[f"player_{player_id}"] = {
                "resource_level": np.array([self.current_resource], dtype=np.float32),
                "step": np.array([self.current_step], dtype=np.int32),
                "my_recent_extractions": my_extractions.astype(np.float32),
                "other_players_recent_extractions": other_extractions.astype(np.float32),
                "my_cumulative_payoff": np.array([self.player_cumulative_payoffs[player_id]], dtype=np.float32),
                "other_players_cumulative_payoffs": other_payoffs.astype(np.float32),
            }

        return observations

    def _get_recent_extractions(self, n_rounds: int) -> np.ndarray:
        """Get extraction history for the last n rounds.

        Args:
            n_rounds: Number of recent rounds to retrieve

        Returns:
            Array of shape (rounds, n_players)
        """
        if len(self.extraction_history) == 0:
            return np.array([])

        recent = self.extraction_history[-n_rounds:]
        return np.array(recent)

    def render(self, mode: str = "dict") -> Any:
        """Render the current game state.

        Args:
            mode: Render mode ("human" for text, "dict" for data)

        Returns:
            Rendering output (string or dict)
        """
        if mode == "human":
            output = f"\n{'='*60}\n"
            output += f"CPR Game - Round {self.current_step}/{self.max_steps}\n"
            output += f"{'='*60}\n"
            output += f"Resource Level: {self.current_resource:.2f} / {self.initial_resource:.2f}\n"

            if len(self.extraction_history) > 0:
                output += f"\nLast Round Extractions:\n"
                for i, extraction in enumerate(self.extraction_history[-1]):
                    output += f"  Player {i}: {extraction:.2f}\n"

            output += f"\nCumulative Payoffs:\n"
            for i, payoff in enumerate(self.player_cumulative_payoffs):
                output += f"  Player {i}: {payoff:.2f}\n"

            if len(self.cooperation_history) > 0:
                avg_coop = np.mean(self.cooperation_history)
                output += f"\nAverage Cooperation Index: {avg_coop:.3f}\n"

            return output

        elif mode == "dict":
            return {
                "resource": self.current_resource,
                "step": self.current_step,
                "max_steps": self.max_steps,
                "resource_history": self.resource_history.copy(),
                "extraction_history": [e.copy() for e in self.extraction_history],
                "payoff_history": [p.copy() for p in self.payoff_history],
                "cumulative_payoffs": self.player_cumulative_payoffs.copy(),
                "cooperation_history": self.cooperation_history.copy(),
                "done": self.done,
            }

        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics for the completed episode.

        Returns:
            Dict: Summary metrics
        """
        from .utils import compute_sustainability_score
        
        sustainability_score = compute_sustainability_score(
            self.resource_history,
            self.sustainability_threshold
        )
        
        return {
            "total_rounds": self.current_step,
            "final_resource_level": self.current_resource,
            "cumulative_payoffs": self.player_cumulative_payoffs.tolist(),
            "sustainability_score": float(sustainability_score),
            "tragedy_occurred": bool(self.current_resource <= self.min_resource),
            "avg_cooperation_index": np.mean(self.cooperation_history) if self.cooperation_history else 0.0,
            "gini_coefficient": compute_gini_coefficient(self.player_cumulative_payoffs),
            "total_extracted": sum(np.sum(e) for e in self.extraction_history),
            "avg_extraction_per_player": [
                np.mean([e[i] for e in self.extraction_history])
                for i in range(self.n_players)
            ] if self.extraction_history else [],
        }

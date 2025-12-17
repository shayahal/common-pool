"""Main game runner for CPR environment.

Orchestrates environment, agents, logging, and visualization.
"""

from typing import Dict, List, Optional, Union
import time
import uuid
import numpy as np
import json
from pathlib import Path

from .config import CONFIG, validate_config
from .cpr_environment import CPREnvironment
from .llm_agent import LLMAgent, MockLLMAgent
from .logging_manager import LoggingManager, MockLoggingManager
from .dashboard import Dashboard
from .utils import format_round_summary


class GameRunner:
    """Main runner for executing CPR games.

    Coordinates environment, agents, logging, and visualization.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_mock_agents: bool = False,
        use_mock_logging: bool = False
    ):
        """Initialize game runner.

        Args:
            config: Configuration dictionary
            use_mock_agents: Use MockLLMAgent instead of real LLM calls
            use_mock_logging: Use MockLoggingManager instead of Langfuse
        """
        self.config = config if config is not None else CONFIG.copy()

        # Validate configuration
        validate_config(self.config)

        # Flags
        self.use_mock_agents = use_mock_agents
        self.use_mock_logging = use_mock_logging

        # Components (initialized in setup)
        self.env: Optional[CPREnvironment] = None
        self.agents: List[Union[LLMAgent, MockLLMAgent]] = []
        self.logger: Optional[Union[LoggingManager, MockLoggingManager]] = None
        self.dashboard: Optional[Dashboard] = None

        # Game state
        self.game_id: Optional[str] = None
        self.current_observations: Optional[Dict] = None
        self.game_history: List[Dict] = []

    def setup_game(self, game_id: Optional[str] = None) -> str:
        """Setup environment, agents, and logging for a new game.

        Args:
            game_id: Unique identifier for this game (auto-generated if None)

        Returns:
            str: Game ID
        """
        # Generate game ID
        self.game_id = game_id or f"game_{uuid.uuid4().hex[:8]}"

        # Initialize environment
        print(f"Initializing CPR environment...")
        self.env = CPREnvironment(self.config)

        # Initialize agents
        print(f"Creating {self.config['n_players']} agents...")
        self.agents = []

        for i in range(self.config['n_players']):
            persona = self.config['player_personas'][i]

            if self.use_mock_agents:
                agent = MockLLMAgent(
                    player_id=i,
                    persona=persona,
                    config=self.config
                )
            else:
                try:
                    agent = LLMAgent(
                        player_id=i,
                        persona=persona,
                        config=self.config
                    )
                except ValueError as e:
                    print(f"Warning: {e}")
                    print("Falling back to MockLLMAgent")
                    agent = MockLLMAgent(
                        player_id=i,
                        persona=persona,
                        config=self.config
                    )

            self.agents.append(agent)
            print(f"  Player {i}: {persona}")

        # Initialize logging
        if self.use_mock_logging or not self.config.get("langfuse_enabled", False):
            print("Using mock logging (Langfuse disabled)")
            self.logger = MockLoggingManager(self.config)
        else:
            print("Initializing Langfuse logging...")
            self.logger = LoggingManager(self.config)

        # Initialize dashboard (optional)
        self.dashboard = Dashboard(self.config)

        print(f"\n✓ Game setup complete: {self.game_id}\n")

        return self.game_id

    def run_episode(
        self,
        visualize: bool = False,
        verbose: bool = True,
        step_delay: float = 0.0
    ) -> Dict:
        """Run a single game episode.

        Args:
            visualize: Whether to use Streamlit dashboard
            verbose: Print round summaries
            step_delay: Delay between steps (seconds)

        Returns:
            Dict: Game summary statistics
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call setup_game() first.")

        # Start logging trace
        self.logger.start_game_trace(self.game_id, self.config)

        # Reset environment and agents
        observations, info = self.env.reset()
        for agent in self.agents:
            agent.reset()

        # Main game loop
        done = False
        step = 0

        while not done:
            # Get actions from all agents
            actions = np.zeros(self.config['n_players'])
            reasonings = []

            for i, agent in enumerate(self.agents):
                obs = observations[f"player_{i}"]
                action, reasoning = agent.act(obs, return_reasoning=True)
                actions[i] = action
                reasonings.append(reasoning)

                # Log generation
                prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                self.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning
                )

                # Add to dashboard
                if self.dashboard and reasoning:
                    self.dashboard.add_reasoning(i, reasoning)

            # Execute step
            observations, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            # Update agent memories
            for i, agent in enumerate(self.agents):
                obs = observations[f"player_{i}"]
                agent.update_memory(obs, actions[i], rewards[i])

            # Log round metrics
            round_metrics = {
                "resource_level": info["resource"],
                "total_extraction": info["total_extraction"],
                "cooperation_index": info["cooperation_index"],
                "individual_extractions": actions.tolist(),
                "individual_payoffs": rewards.tolist(),
            }
            self.logger.log_round_metrics(step, round_metrics)

            # Print summary
            if verbose:
                print(format_round_summary(
                    step + 1,
                    info["resource"],
                    actions,
                    rewards,
                    info["cumulative_payoffs"]
                ))

            # Update dashboard (only if visualization is enabled)
            if self.dashboard and visualize:
                game_state = self.env.render(mode="dict")
                self.dashboard.update(game_state)

            # Step delay
            if step_delay > 0:
                time.sleep(step_delay)

            step += 1

        # Get summary statistics
        summary = self.env.get_summary_stats()

        # End logging trace
        self.logger.end_game_trace(summary)

        # Show dashboard summary
        if self.dashboard:
            self.dashboard.show_summary(summary)

        # Print final summary
        if verbose:
            print("\n" + "=" * 60)
            print("GAME SUMMARY")
            print("=" * 60)
            for key, value in summary.items():
                print(f"{key}: {value}")
            print("=" * 60 + "\n")

        # Store in history
        self.game_history.append({
            "game_id": self.game_id,
            "summary": summary,
            "config": self.config,
        })

        return summary

    def run_tournament(
        self,
        n_games: int,
        verbose: bool = False
    ) -> List[Dict]:
        """Run multiple games and collect results.

        Args:
            n_games: Number of games to run
            verbose: Print detailed output

        Returns:
            List of game summaries
        """
        print(f"\n{'=' * 60}")
        print(f"RUNNING TOURNAMENT: {n_games} games")
        print(f"{'=' * 60}\n")

        results = []

        for i in range(n_games):
            print(f"\n--- Game {i + 1}/{n_games} ---")

            # Setup new game
            game_id = f"tournament_game_{i + 1}"
            self.setup_game(game_id)

            # Run episode
            summary = self.run_episode(visualize=False, verbose=verbose)
            results.append(summary)

            # Brief summary
            print(f"Result: {summary['total_rounds']} rounds, "
                  f"{'DEPLETED' if summary['tragedy_occurred'] else 'SUSTAINED'}, "
                  f"Cooperation: {summary['avg_cooperation_index']:.3f}")

        # Aggregate statistics
        print(f"\n{'=' * 60}")
        print("TOURNAMENT RESULTS")
        print(f"{'=' * 60}")

        tragedy_rate = sum(1 for r in results if r['tragedy_occurred']) / n_games
        avg_rounds = np.mean([r['total_rounds'] for r in results])
        avg_cooperation = np.mean([r['avg_cooperation_index'] for r in results])
        avg_gini = np.mean([r['gini_coefficient'] for r in results])

        print(f"Tragedy Rate: {tragedy_rate:.1%}")
        print(f"Average Rounds: {avg_rounds:.1f}")
        print(f"Average Cooperation: {avg_cooperation:.3f}")
        print(f"Average Gini Coefficient: {avg_gini:.3f}")
        print(f"{'=' * 60}\n")

        return results

    def export_results(self, output_path: str, include_config: bool = True):
        """Export game history to JSON file.

        Args:
            output_path: Path to output file
            include_config: Include configuration in export
        """
        output_data = {
            "games": self.game_history,
        }

        if include_config:
            # Convert config to serializable format
            config_copy = self.config.copy()
            # Remove non-serializable items
            config_copy.pop("persona_prompts", None)
            output_data["config"] = config_copy

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"Results exported to {output_path}")

    def get_latest_summary(self) -> Optional[Dict]:
        """Get summary from most recent game.

        Returns:
            Dict: Game summary or None if no games played
        """
        if len(self.game_history) == 0:
            return None
        return self.game_history[-1]["summary"]


def quick_game(
    n_players: int = 2,
    max_steps: int = 50,
    use_mock: bool = True,
    verbose: bool = True
) -> Dict:
    """Run a quick game with simplified setup.

    Args:
        n_players: Number of players
        max_steps: Maximum rounds
        use_mock: Use mock agents (no API calls)
        verbose: Print output

    Returns:
        Dict: Game summary
    """
    config = CONFIG.copy()
    config["n_players"] = n_players
    config["max_steps"] = max_steps

    runner = GameRunner(config, use_mock_agents=use_mock, use_mock_logging=True)
    runner.setup_game()
    summary = runner.run_episode(visualize=False, verbose=verbose)

    return summary


if __name__ == "__main__":
    # Example usage
    print("Running example CPR game...\n")

    # Create runner with mock agents (no API calls required)
    runner = GameRunner(use_mock_agents=True, use_mock_logging=True)

    # Setup game
    runner.setup_game()

    # Run single game
    summary = runner.run_episode(visualize=False, verbose=True)

    print("\nGame complete!")

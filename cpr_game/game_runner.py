"""Main game runner for CPR environment.

Orchestrates environment, agents, logging, and visualization.
"""

from typing import Dict, List, Optional, Union, Tuple
import time
import uuid
import numpy as np
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CONFIG, validate_config
from .cpr_environment import CPREnvironment
from .llm_agent import LLMAgent, MockLLMAgent
from .logging_manager import LoggingManager
from .dashboard import Dashboard
from .db_manager import DatabaseManager
from .utils import format_round_summary
from .logger_setup import setup_logging, get_logger

logger = get_logger(__name__)


class GameRunner:
    """Main runner for executing CPR games.

    Coordinates environment, agents, logging, and visualization.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_mock_agents: bool = False
    ):
        """Initialize game runner.

        Args:
            config: Configuration dictionary
            use_mock_agents: Use MockLLMAgent instead of real LLM calls
        """
        self.config = config if config is not None else CONFIG.copy()

        # Validate configuration
        validate_config(self.config)

        # Flags
        self.use_mock_agents = use_mock_agents

        # Components (initialized in setup)
        self.env: Optional[CPREnvironment] = None
        self.agents: List[Union[LLMAgent, MockLLMAgent]] = []
        self.logger: Optional[LoggingManager] = None
        self.dashboard: Optional[Dashboard] = None
        self.db_manager: Optional[DatabaseManager] = None

        # Game state
        self.game_id: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.current_observations: Optional[Dict] = None
        self.game_history: List[Dict] = []

    def setup_game(
        self, 
        game_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> str:
        """Setup environment, agents, and logging for a new game.

        Args:
            game_id: Unique identifier for this game (auto-generated if None)
            experiment_id: Optional experiment identifier (auto-generated if None)

        Returns:
            str: Game ID
        """
        # Generate game ID
        self.game_id = game_id or f"game_{uuid.uuid4().hex[:8]}"
        
        # Generate experiment ID if not provided
        if experiment_id is None:
            # Use config experiment_id if available, otherwise generate one
            self.experiment_id = self.config.get("experiment_id") or f"exp_{uuid.uuid4().hex[:8]}"
        else:
            self.experiment_id = experiment_id

        # Initialize environment
        logger.info("Initializing CPR environment...")
        self.env = CPREnvironment(self.config)

        # Initialize agents
        logger.info(f"Creating {self.config['n_players']} agents...")
        self.agents = []

        for i in range(self.config['n_players']):
            # Get persona with validation
            if i >= len(self.config['player_personas']):
                raise ValueError(
                    f"Not enough personas defined for {self.config['n_players']} players. "
                    f"Need {self.config['n_players']} personas, but only {len(self.config['player_personas'])} provided."
                )
            
            persona = self.config['player_personas'][i]
            
            # Validate persona is not empty
            if not persona or persona.strip() == "":
                # Try to use a fallback
                available_personas = list(self.config.get("persona_prompts", {}).keys())
                if available_personas:
                    persona = available_personas[0]
                    logger.warning(f"Player {i} had empty persona, using '{persona}' instead.")
                else:
                    raise ValueError(f"Player {i} has empty persona and no fallback personas available.")

            if self.use_mock_agents:
                agent = MockLLMAgent(
                    player_id=i,
                    persona=persona,
                    config=self.config
                )
            else:
                agent = LLMAgent(
                    player_id=i,
                    persona=persona,
                    config=self.config
                )
            
            # Assign player UUID if available (for experiments)
            player_uuids = self.config.get("player_uuids", [])
            if i < len(player_uuids) and player_uuids[i]:
                agent.player_uuid = player_uuids[i]

            self.agents.append(agent)
            logger.debug(f"Player {i}: {persona}")

        # Initialize file logging first (so we can see any errors)
        log_dir = self.config.get("log_dir", "logs")
        setup_logging(log_dir=log_dir)
        
        # Initialize logging - Langfuse is required
        logger.info("Initializing Langfuse logging...")
        try:
            self.logger = LoggingManager(self.config)
            logger.info("✓ Langfuse client initialized successfully")
        except Exception as e:
            logger.error(f"❌ ERROR: Failed to initialize Langfuse: {e}")
            raise

        # Initialize dashboard (optional)
        self.dashboard = Dashboard(self.config)

        # Initialize database manager
        db_path = self.config.get("db_path", "data/game_results.duckdb")
        db_enabled = self.config.get("db_enabled", True)
        try:
            self.db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
            if self.db_manager.enabled and self.db_manager.conn is not None:
                logger.info(f"Database manager initialized: {db_path}")
            elif not db_enabled:
                logger.info("Database manager disabled by configuration")
            else:
                logger.warning("Database manager initialized but connection is None")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}", exc_info=True)
            self.db_manager = None

        logger.info(f"✓ Game setup complete: {self.game_id}")

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
            # Set current round for logging
            self.logger.set_current_round(step)

            # Get actions from all agents in parallel
            actions = np.zeros(self.config['n_players'])
            reasonings = [None] * self.config['n_players']
            
            # Helper function to run agent action
            def run_agent_action(i: int, agent, obs: Dict) -> Tuple[int, int, str, Optional[Dict], str, Optional[str]]:
                """Run agent action and return results."""
                try:
                    action, reasoning = agent.act(obs, return_reasoning=True)
                    api_metrics = None
                    if hasattr(agent, 'get_last_api_metrics'):
                        api_metrics = agent.get_last_api_metrics()
                    prompt = agent._build_prompt(obs) if hasattr(agent, '_build_prompt') else ""
                    system_prompt = None
                    if hasattr(agent, 'system_prompt'):
                        system_prompt = agent.system_prompt
                    return (i, action, reasoning, api_metrics, prompt, system_prompt)
                except Exception as e:
                    # Log error and return default action
                    logger.error(f"Error in agent {i}: {e}", exc_info=True)
                    return (i, self.config['min_extraction'], f"Error: {str(e)}", None, "", None)
            
            # Run all agents in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config['n_players']) as executor:
                # Submit all agent actions
                future_to_player = {
                    executor.submit(run_agent_action, i, agent, observations[f"player_{i}"]): i
                    for i, agent in enumerate(self.agents)
                }
                
                # Collect results as they complete (maintain order by player_id)
                results = {}
                for future in as_completed(future_to_player):
                    try:
                        player_id, action, reasoning, api_metrics, prompt, system_prompt = future.result()
                        results[player_id] = (action, reasoning, api_metrics, prompt, system_prompt)
                    except Exception as e:
                        player_id = future_to_player[future]
                        logger.error(f"Error getting result from agent {player_id}: {e}", exc_info=True)
                        results[player_id] = (
                            self.config['min_extraction'],
                            f"Error: {str(e)}",
                            None,
                            "",
                            None
                        )
            
            # Process results in order
            for i in range(self.config['n_players']):
                if i not in results:
                    # Fallback if result missing
                    action = self.config['min_extraction']
                    reasoning = "Error: No result from agent"
                    api_metrics = None
                    prompt = ""
                    system_prompt = None
                else:
                    action, reasoning, api_metrics, prompt, system_prompt = results[i]
                
                actions[i] = action
                reasonings[i] = reasoning

                # Log generation
                self.logger.log_generation(
                    player_id=i,
                    prompt=prompt,
                    response=reasoning or "",
                    action=action,
                    reasoning=reasoning,
                    api_metrics=api_metrics,
                    system_prompt=system_prompt
                )

                # Add to dashboard with full context
                if self.dashboard:
                    obs = observations[f"player_{i}"]
                    # Prepare game state context from observation
                    game_state_context = {
                        "resource_level": obs.get("resource_level", [0])[0] if isinstance(obs.get("resource_level"), np.ndarray) else obs.get("resource_level", 0),
                        "step": obs.get("step", [0])[0] if isinstance(obs.get("step"), np.ndarray) else obs.get("step", step),
                    }
                    # Add cumulative payoff if available
                    if "my_cumulative_payoff" in obs:
                        my_payoff = obs["my_cumulative_payoff"]
                        if isinstance(my_payoff, np.ndarray):
                            game_state_context["my_cumulative_payoff"] = my_payoff[0] if len(my_payoff) > 0 else 0
                        else:
                            game_state_context["my_cumulative_payoff"] = my_payoff
                    
                    self.dashboard.add_reasoning(
                        player_id=i,
                        reasoning=reasoning or "",
                        prompt=prompt if prompt else None,
                        action=float(action) if action is not None else None,
                        game_state=game_state_context,
                        round_num=step + 1
                    )

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

            # Log summary
            if verbose:
                logger.info(format_round_summary(
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

        # Add total cost and per-player costs to summary if available from logger
        if hasattr(self.logger, 'get_api_metrics_data'):
            api_metrics = self.logger.get_api_metrics_data()
            total_cost = sum(m.get("cost", 0) or 0 for m in api_metrics)
            summary["total_cost"] = total_cost
            
            # Calculate cost per player
            player_costs = {}
            for metric in api_metrics:
                player_id = metric.get("player_id")
                if player_id is not None:
                    cost = metric.get("cost", 0) or 0
                    if player_id not in player_costs:
                        player_costs[player_id] = 0.0
                    player_costs[player_id] += cost
            
            summary["player_costs"] = player_costs
            summary["api_metrics_data"] = api_metrics  # Store full API metrics for detailed analysis

        # End logging trace
        self.logger.end_game_trace(summary)

        # Show dashboard summary
        if self.dashboard:
            self.dashboard.show_summary(summary)

        # Log final summary
        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("GAME SUMMARY")
            logger.info("=" * 60)
            for key, value in summary.items():
                logger.info(f"{key}: {value}")
            logger.info("=" * 60 + "\n")

        # Store in history
        self.game_history.append({
            "game_id": self.game_id,
            "summary": summary,
            "config": self.config,
        })

        # Save results to database
        if self.db_manager and self.db_manager.enabled:
            try:
                from datetime import datetime
                timestamp = datetime.now().isoformat()
                
                self.db_manager.save_game_results(
                    game_id=self.game_id,
                    agents=self.agents,
                    summary=summary,
                    config=self.config,
                    experiment_id=self.experiment_id,
                    timestamp=timestamp
                )
                # Verify the save
                if self.db_manager.verify_game_saved(self.game_id):
                    logger.info(f"✓ Game results saved and verified for game {self.game_id}")
                else:
                    logger.warning(f"Game results saved but verification failed for {self.game_id}")
            except Exception as e:
                logger.error(f"Failed to save game results to database: {e}", exc_info=True)
        elif not self.db_manager:
            logger.warning("Database manager not initialized - results not saved")
        elif not self.db_manager.enabled:
            logger.debug("Database saving is disabled")

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
        logger.info(f"\n{'=' * 60}")
        logger.info(f"RUNNING TOURNAMENT: {n_games} games")
        logger.info(f"{'=' * 60}\n")

        results = []

        for i in range(n_games):
            logger.info(f"\n--- Game {i + 1}/{n_games} ---")

            # Setup new game
            game_id = f"tournament_game_{i + 1}"
            self.setup_game(game_id)

            # Run episode
            summary = self.run_episode(visualize=False, verbose=verbose)
            results.append(summary)

            # Brief summary
            logger.info(f"Result: {summary['total_rounds']} rounds, "
                  f"{'DEPLETED' if summary['tragedy_occurred'] else 'SUSTAINED'}, "
                  f"Cooperation: {summary['avg_cooperation_index']:.3f}")

        # Aggregate statistics
        logger.info(f"\n{'=' * 60}")
        logger.info("TOURNAMENT RESULTS")
        logger.info(f"{'=' * 60}")

        tragedy_rate = sum(1 for r in results if r['tragedy_occurred']) / n_games
        avg_rounds = np.mean([r['total_rounds'] for r in results])
        avg_cooperation = np.mean([r['avg_cooperation_index'] for r in results])
        avg_gini = np.mean([r['gini_coefficient'] for r in results])

        logger.info(f"Tragedy Rate: {tragedy_rate:.1%}")
        logger.info(f"Average Rounds: {avg_rounds:.1f}")
        logger.info(f"Average Cooperation: {avg_cooperation:.3f}")
        logger.info(f"Average Gini Coefficient: {avg_gini:.3f}")
        logger.info(f"{'=' * 60}\n")

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

        logger.info(f"Results exported to {output_path}")

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

    runner = GameRunner(config, use_mock_agents=use_mock)
    runner.setup_game()
    summary = runner.run_episode(visualize=False, verbose=verbose)

    return summary


if __name__ == "__main__":
    # Example usage
    logger.info("Running example CPR game...\n")

    # Create runner with mock agents (no API calls required)
    runner = GameRunner(use_mock_agents=True)

    # Setup game
    runner.setup_game()

    # Run single game
    summary = runner.run_episode(visualize=False, verbose=True)

    logger.info("\nGame complete!")

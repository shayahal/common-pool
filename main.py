"""Main experiment runner for CPR Game experiments.

Runs experiments defined in the database with parallel game execution.

Usage:
    python main.py --experiment-id exp_123
    python main.py --experiment-id exp_123 --use-mock
    python main.py --experiment-id exp_123 --max-workers 5
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.game_runner import GameRunner
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)


def run_single_game(
    experiment_id: str,
    game_index: int,
    selected_players: List[Dict[str, str]],
    parameters: Dict,
    use_mock_agents: bool = False
) -> Optional[Dict]:
    """Run a single game with selected players.
    
    Args:
        experiment_id: Experiment identifier
        game_index: Index of this game in the experiment
        selected_players: List of player dicts with 'persona' and 'model'
        parameters: Experiment parameters
        use_mock_agents: Whether to use mock agents (no API calls)
        
    Returns:
        Dictionary with game_id and summary, or None if failed
    """
    try:
        # Create game config from parameters
        config = CONFIG.copy()
        config["n_players"] = len(selected_players)
        config["max_steps"] = parameters["max_steps"]
        config["initial_resource"] = parameters["initial_resource"]
        config["regeneration_rate"] = parameters["regeneration_rate"]
        config["max_extraction"] = parameters["max_extraction"]
        config["max_fishes"] = parameters["max_fishes"]
        
        # Set player personas from selected players
        config["player_personas"] = [p["persona"] for p in selected_players]
        
        # Set model for all agents
        # Note: Currently all players in a game use the same model (first selected player's model)
        # This works for experiments where all players use the same model (e.g., all gpt-3.5-turbo)
        # For mixed-model experiments, we'd need to extend GameRunner to support per-agent models
        models_in_game = set(p["model"] for p in selected_players)
        if len(models_in_game) > 1:
            logger.warning(
                f"Game {game_index}: Multiple models detected {models_in_game}. "
                f"Using {selected_players[0]['model']} for all players."
            )
        config["llm_model"] = selected_players[0]["model"]
        config["experiment_id"] = experiment_id
        
        # Create game runner
        runner = GameRunner(config=config, use_mock_agents=use_mock_agents)
        
        # Generate game ID
        game_id = f"{experiment_id}_game_{game_index:04d}"
        
        # Setup game
        runner.setup_game(game_id=game_id, experiment_id=experiment_id)
        
        # Run episode
        summary = runner.run_episode(visualize=False, verbose=False)
        
        return {
            "game_id": game_id,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error running game {game_index} for experiment {experiment_id}: {e}", exc_info=True)
        return None


def run_experiment(
    experiment_id: str,
    use_mock_agents: bool = False,
    max_workers: Optional[int] = None
) -> bool:
    """Run an experiment by executing all games in parallel.
    
    Args:
        experiment_id: Experiment identifier
        use_mock_agents: Whether to use mock agents (no API calls)
        max_workers: Maximum number of parallel workers (default: min(10, number_of_games))
        
    Returns:
        True if experiment completed successfully, False otherwise
    """
    # Initialize database manager
    db_path = CONFIG.get("db_path", "data/game_results.duckdb")
    db_enabled = CONFIG.get("db_enabled", True)
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        logger.error("Database is not enabled. Cannot run experiment.")
        return False
    
    # Load experiment
    experiment = db_manager.load_experiment(experiment_id)
    
    if not experiment:
        logger.error(f"Experiment {experiment_id} not found in database")
        return False
    
    logger.info(f"Starting experiment: {experiment['name']} ({experiment_id})")
    logger.info(f"Players: {len(experiment['players'])}")
    logger.info(f"Games: {experiment['parameters']['number_of_games']}")
    logger.info(f"Players per game: {experiment['parameters']['number_of_players_per_game']}")
    
    # Update status to running
    db_manager.update_experiment_status(experiment_id, "running")
    
    # Get parameters
    parameters = experiment["parameters"]
    players = experiment["players"]
    number_of_games = parameters["number_of_games"]
    number_of_players_per_game = parameters["number_of_players_per_game"]
    
    # Validate
    if number_of_players_per_game > len(players):
        logger.warning(
            f"Number of players per game ({number_of_players_per_game}) "
            f"exceeds pool size ({len(players)}). Using all players for each game."
        )
        number_of_players_per_game = len(players)
    
    # Set max workers
    if max_workers is None:
        max_workers = min(10, number_of_games)
    
    # Prepare game configurations
    game_configs = []
    for i in range(number_of_games):
        # Randomly sample players for this game
        if number_of_players_per_game == len(players):
            selected_players = players.copy()
        else:
            selected_players = random.sample(players, number_of_players_per_game)
        
        game_configs.append((i, selected_players))
    
    # Run games in parallel
    logger.info(f"Running {number_of_games} games with {max_workers} workers...")
    
    successful_games = 0
    failed_games = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all games
        future_to_game = {
            executor.submit(
                run_single_game,
                experiment_id,
                game_index,
                selected_players,
                parameters,
                use_mock_agents
            ): game_index
            for game_index, selected_players in game_configs
        }
        
        # Process results with progress bar
        with tqdm(total=number_of_games, desc="Running games", unit="game") as pbar:
            for future in as_completed(future_to_game):
                game_index = future_to_game[future]
                try:
                    result = future.result()
                    
                    if result:
                        # Save result to database
                        db_manager.save_experiment_result(
                            experiment_id=experiment_id,
                            game_id=result["game_id"],
                            summary=result["summary"]
                        )
                        successful_games += 1
                        pbar.set_postfix({
                            "success": successful_games,
                            "failed": failed_games
                        })
                    else:
                        failed_games += 1
                        logger.warning(f"Game {game_index} failed")
                        pbar.set_postfix({
                            "success": successful_games,
                            "failed": failed_games
                        })
                    
                except Exception as e:
                    failed_games += 1
                    logger.error(f"Error processing game {game_index}: {e}", exc_info=True)
                    pbar.set_postfix({
                        "success": successful_games,
                        "failed": failed_games
                    })
                
                pbar.update(1)
    
    # Update experiment status
    if successful_games == 0:
        status = "failed"
        logger.error(f"Experiment {experiment_id} failed: all games failed")
    elif failed_games > 0:
        status = "completed"
        logger.warning(
            f"Experiment {experiment_id} completed with {failed_games} failed games "
            f"out of {number_of_games} total"
        )
    else:
        status = "completed"
        logger.info(f"Experiment {experiment_id} completed successfully: {successful_games} games")
    
    db_manager.update_experiment_status(experiment_id, status)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Experiment: {experiment['name']} ({experiment_id})")
    print(f"Total Games: {number_of_games}")
    print(f"Successful: {successful_games}")
    print(f"Failed: {failed_games}")
    print(f"Status: {status}")
    print("=" * 60 + "\n")
    
    return successful_games > 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CPR Game experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiment-id exp_123
  python main.py --experiment-id exp_123 --use-mock
  python main.py --experiment-id exp_123 --max-workers 5
        """
    )
    
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment ID to run"
    )
    
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock agents (no API calls)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: min(10, number_of_games))"
    )
    
    args = parser.parse_args()
    
    # Run experiment
    success = run_experiment(
        experiment_id=args.experiment_id,
        use_mock_agents=args.use_mock,
        max_workers=args.max_workers
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


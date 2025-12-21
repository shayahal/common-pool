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
from cpr_game.logger_setup import get_logger, setup_logging

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
    game_id = f"{experiment_id}_game_{game_index:04d}"
    logger.info(f"[Game {game_index:04d}] Starting game {game_id}")
    
    try:
        # Create game config from parameters
        config = CONFIG.copy()
        config["n_players"] = len(selected_players)
        config["max_steps"] = parameters["max_steps"]
        config["initial_resource"] = parameters["initial_resource"]
        config["regeneration_rate"] = parameters["regeneration_rate"]
        config["max_extraction"] = parameters["max_extraction"]
        # Use Max_fish (new name) or max_fishes (old name for backward compatibility)
        config["max_fishes"] = parameters.get("Max_fish") or parameters.get("max_fishes", 1000)
        
        logger.debug(f"[Game {game_index:04d}] Game config: {len(selected_players)} players, "
                    f"max_steps={config['max_steps']}, initial_resource={config['initial_resource']}, "
                    f"max_extraction={config['max_extraction']}")
        
        # Set player personas from selected players
        config["player_personas"] = [p["persona"] for p in selected_players]
        
        # Store player UUIDs mapping (player_index -> player_uuid)
        # This allows agents to have their UUIDs assigned
        config["player_uuids"] = [p.get("player_uuid") for p in selected_players]
        
        # Log selected players
        player_info = ", ".join([f"{p['persona']} ({p.get('player_uuid', 'no-uuid')[:8]})" 
                                 for p in selected_players])
        logger.info(f"[Game {game_index:04d}] Selected players: {player_info}")
        
        # Set model for all agents
        # Note: Currently all players in a game use the same model (first selected player's model)
        # This works for experiments where all players use the same model (e.g., all gpt-3.5-turbo)
        # For mixed-model experiments, we'd need to extend GameRunner to support per-agent models
        models_in_game = set(p["model"] for p in selected_players)
        if len(models_in_game) > 1:
            logger.warning(
                f"[Game {game_index:04d}] Multiple models detected {models_in_game}. "
                f"Using {selected_players[0]['model']} for all players."
            )
        config["llm_model"] = selected_players[0]["model"]
        config["experiment_id"] = experiment_id
        
        logger.debug(f"[Game {game_index:04d}] Using model: {config['llm_model']} "
                    f"({'mock' if use_mock_agents else 'real'} agents)")
        
        # Create game runner
        logger.debug(f"[Game {game_index:04d}] Creating GameRunner...")
        runner = GameRunner(config=config, use_mock_agents=use_mock_agents)
        
        # Setup game
        logger.debug(f"[Game {game_index:04d}] Setting up game environment...")
        runner.setup_game(game_id=game_id, experiment_id=experiment_id)
        
        # Run episode
        logger.info(f"[Game {game_index:04d}] Running game episode...")
        summary = runner.run_episode(visualize=False, verbose=False)
        
        # Log game results
        rounds = summary.get("total_rounds", 0)
        tragedy = summary.get("tragedy_occurred", False)
        final_resource = summary.get("final_resource_level", 0)
        logger.info(f"[Game {game_index:04d}] Completed: {rounds} rounds, "
                   f"final_resource={final_resource:.1f}, "
                   f"tragedy={'YES' if tragedy else 'NO'}")
        
        return {
            "game_id": game_id,
            "summary": summary,
            "selected_players": selected_players  # Include selected players with UUIDs
        }
        
    except Exception as e:
        logger.error(f"[Game {game_index:04d}] Error running game {game_id}: {e}", exc_info=True)
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
    
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {experiment['name']} ({experiment_id})")
    logger.info("=" * 60)
    
    # Get parameters
    parameters = experiment["parameters"]
    players = experiment["players"]
    number_of_games = parameters["number_of_games"]
    number_of_players_per_game = parameters["number_of_players_per_game"]
    
    logger.info(f"Experiment Configuration:")
    logger.info(f"  - Total players in pool: {len(players)}")
    logger.info(f"  - Players per game: {number_of_players_per_game}")
    logger.info(f"  - Number of games: {number_of_games}")
    logger.info(f"  - Max steps per game: {parameters['max_steps']}")
    logger.info(f"  - Initial resource: {parameters['initial_resource']}")
    logger.info(f"  - Max extraction: {parameters['max_extraction']}")
    logger.info(f"  - Regeneration rate: {parameters['regeneration_rate']}")
    logger.info(f"  - Max fish (capacity): {parameters.get('Max_fish') or parameters.get('max_fishes', 'N/A')}")
    
    # Log player pool
    logger.info(f"Player Pool ({len(players)} players):")
    for i, player in enumerate(players):
        uuid_short = player.get('player_uuid', 'no-uuid')[:8] if player.get('player_uuid') else 'no-uuid'
        logger.info(f"  [{i}] {player['persona']:20s} | {player['model']:20s} | UUID: {uuid_short}")
    
    # Update status to running
    logger.info(f"Updating experiment status to 'running'...")
    db_manager.update_experiment_status(experiment_id, "running")
    
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
    logger.info(f"Parallel execution: {max_workers} workers")
    logger.info(f"Agent mode: {'MOCK (no API calls)' if use_mock_agents else 'REAL (API calls enabled)'}")
    
    # Prepare game configurations
    logger.info(f"Preparing {number_of_games} game configurations...")
    game_configs = []
    for i in range(number_of_games):
        # Randomly sample players for this game
        if number_of_players_per_game == len(players):
            selected_players = players.copy()
        else:
            selected_players = random.sample(players, number_of_players_per_game)
        
        game_configs.append((i, selected_players))
    
    logger.info(f"Game configurations prepared. Starting parallel execution...")
    logger.info("=" * 60)
    
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
                        logger.debug(f"[Game {game_index:04d}] Saving results to database...")
                        db_manager.save_experiment_result(
                            experiment_id=experiment_id,
                            game_id=result["game_id"],
                            summary=result["summary"],
                            selected_players=result.get("selected_players", [])
                        )
                        successful_games += 1
                        logger.info(f"[Game {game_index:04d}] ✓ Successfully completed and saved")
                        pbar.set_postfix({
                            "success": successful_games,
                            "failed": failed_games
                        })
                    else:
                        failed_games += 1
                        logger.warning(f"[Game {game_index:04d}] ✗ Game failed (returned None)")
                        pbar.set_postfix({
                            "success": successful_games,
                            "failed": failed_games
                        })
                    
                except Exception as e:
                    failed_games += 1
                    logger.error(f"[Game {game_index:04d}] ✗ Error processing game: {e}", exc_info=True)
                    pbar.set_postfix({
                        "success": successful_games,
                        "failed": failed_games
                    })
                
                pbar.update(1)
    
    # Update experiment status
    logger.info("=" * 60)
    logger.info("Experiment execution completed")
    logger.info(f"  Successful games: {successful_games}/{number_of_games}")
    logger.info(f"  Failed games: {failed_games}/{number_of_games}")
    
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
    
    logger.info(f"Updating experiment status to '{status}'...")
    db_manager.update_experiment_status(experiment_id, status)
    logger.info("=" * 60)
    
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
    
    # Setup logging
    log_dir = CONFIG.get("log_dir", "logs")
    setup_logging(log_dir=log_dir)
    logger.info(f"Logging initialized (logs directory: {log_dir})")
    
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


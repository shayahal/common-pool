"""Quick test to generate a trace and verify it appears in LangSmith."""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger
from main import run_experiment

logger = get_logger(__name__)


def main():
    """Run a quick test game to generate traces."""
    logger.info("=" * 70)
    logger.info("LangSmith Trace Test")
    logger.info("=" * 70)
    
    # Create a minimal experiment with just 1 game
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    experiment_id = "langsmith_test_" + str(int(time.time()))
    experiment_name = "LangSmith Trace Test"
    
    # Create 2 players with gpt-3.5-turbo (cheapest model)
    players = [
        {"persona": "rational_selfish", "model": "gpt-3.5-turbo"},
        {"persona": "cooperative", "model": "gpt-3.5-turbo"},
    ]
    
    parameters = {
        "n_players": 2,
        "max_steps": 5,  # Very short game
        "initial_resource": 1000,
        "regeneration_rate": 2.0,
        "max_extraction": 35,
        "max_fishes": 1000,
        "number_of_games": 1,  # Just 1 game
        "number_of_players_per_game": 2,
    }
    
    logger.info(f"Creating test experiment: {experiment_id}")
    success = db_manager.save_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        players=players,
        parameters=parameters
    )
    
    if not success:
        logger.error("Failed to create experiment")
        db_manager.close()
        return 1
    
    db_manager.close()
    
    logger.info("Running test game...")
    logger.info("This should generate traces that appear in LangSmith")
    logger.info("=" * 70)
    
    # Run the experiment
    success = run_experiment(
        experiment_id=experiment_id,
        use_mock_agents=False,
        max_workers=1
    )
    
    if success:
        logger.info("=" * 70)
        logger.info("✅ Test completed!")
        logger.info(f"Check LangSmith for traces from experiment: {experiment_id}")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("Test failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


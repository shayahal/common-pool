"""Run an experiment comparing 4 different LLM models.

This script creates an experiment that tests 4 different models:
- gpt-3.5-turbo
- gpt-4o-mini
- gpt-4o
- gpt-4-turbo-preview

Each model will be tested with the same set of personas to ensure fair comparison.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)


def create_model_comparison_experiment():
    """Create an experiment comparing 4 different models."""
    
    # Initialize database manager
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    logger.info("Initializing database manager...")
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        logger.error("Database is not enabled!")
        return None
    
    # Models to compare
    models = [
        "gpt-3.5-turbo",
        "gpt-4o-mini", 
        "gpt-4o",
        "gpt-4-turbo-preview"
    ]
    
    # Personas to use for each model (same set for fair comparison)
    personas = [
        "rational_selfish",
        "cooperative",
        "aggressive",
        "conservative"
    ]
    
    # Create experiment ID with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"model_comparison_{timestamp}"
    experiment_name = f"Model Comparison: {', '.join(models)}"
    
    logger.info(f"Creating experiment: {experiment_name}")
    logger.info(f"Experiment ID: {experiment_id}")
    
    # Create players: 4 personas × 4 models = 16 players total
    players = []
    for model in models:
        for persona in personas:
            players.append({
                "persona": persona,
                "model": model
            })
    
    logger.info(f"Created {len(players)} players:")
    for i, player in enumerate(players):
        logger.info(f"  [{i:2d}] {player['persona']:20s} | {player['model']}")
    
    # Experiment parameters
    parameters = {
        "n_players": len(players),  # Total pool size
        "max_steps": 15,  # Reasonable game length
        "initial_resource": 1000,
        "regeneration_rate": 2.0,
        "max_extraction": 35,
        "max_fishes": 1000,
        "number_of_games": 16,  # 4 games per model (4 models × 4 games = 16 total)
        "number_of_players_per_game": 4,  # 4 players per game (all from same model)
    }
    
    # Save experiment
    logger.info("Saving experiment to database...")
    success = db_manager.save_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        players=players,
        parameters=parameters
    )
    
    if not success:
        logger.error("Failed to create experiment")
        db_manager.close()
        return None
    
    logger.info("✅ Experiment created successfully!")
    logger.info(f"   ID: {experiment_id}")
    logger.info(f"   Name: {experiment_name}")
    logger.info(f"   Players: {len(players)}")
    logger.info(f"   Games: {parameters['number_of_games']}")
    logger.info(f"   Players per game: {parameters['number_of_players_per_game']}")
    
    # Verify experiment was saved
    loaded = db_manager.load_experiment(experiment_id)
    if not loaded:
        logger.error("Failed to load experiment after creation")
        db_manager.close()
        return None
    
    logger.info("✅ Experiment verified in database")
    db_manager.close()
    
    return experiment_id


def main():
    """Main function to create and run the experiment."""
    logger.info("=" * 70)
    logger.info("Model Comparison Experiment Setup")
    logger.info("=" * 70)
    
    # Create experiment
    experiment_id = create_model_comparison_experiment()
    
    if not experiment_id:
        logger.error("Failed to create experiment")
        return 1
    
    logger.info("=" * 70)
    logger.info("Experiment created successfully!")
    logger.info("=" * 70)
    
    # Ask if user wants to run it now
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        logger.info("Running experiment now...")
        from main import run_experiment
        
        success = run_experiment(
            experiment_id=experiment_id,
            use_mock_agents=False,  # Use real API calls
            max_workers=4  # Run 4 games in parallel
        )
        
        if success:
            logger.info("=" * 70)
            logger.info("✅ Experiment completed successfully!")
            logger.info("=" * 70)
            logger.info(f"\nCheck Langfuse/LangSmith for traces:")
            logger.info(f"  - Experiment ID: {experiment_id}")
            logger.info(f"  - Models tested: gpt-3.5-turbo, gpt-4o-mini, gpt-4o, gpt-4-turbo-preview")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("Experiment failed")
            return 1
    else:
        logger.info(f"\nTo run this experiment, use:")
        logger.info(f"  python run_model_comparison.py --run")
        logger.info(f"\nOr use the experiment worker:")
        logger.info(f"  python experiment_worker.py")
        logger.info("=" * 70)
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


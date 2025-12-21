"""Example script showing how to create and run an experiment using the new system.

This is the correct way to create experiments that will show up in query_experiments.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
import uuid

def create_and_run_experiment():
    """Create an experiment and run it."""
    
    # Initialize database
    db_manager = DatabaseManager(
        db_path=CONFIG.get("db_path", "data/game_results.duckdb"),
        enabled=True
    )
    
    # Step 1: Create experiment definition
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    experiment_name = "Simple Test Experiment"
    
    # Define players (persona + model combinations)
    players = [
        {"persona": "rational_selfish", "model": "gpt-3.5-turbo"},
        {"persona": "cooperative", "model": "gpt-3.5-turbo"},
    ]
    
    # Define experiment parameters
    # Fixed values per constraints
    INITIAL_RESOURCE = 1000
    Max_fish = 1000
    REGENERATION_RATE = 2.0
    number_of_players_per_game = 2  # Use all players
    # Calculate MAX_EXTRACTION: int(4 * INITIAL_RESOURCE / 7 * Number of Players per Game)
    max_extraction = int(4 * INITIAL_RESOURCE / (7 * number_of_players_per_game))
    
    parameters = {
        "n_players": 2,
        "max_steps": 20,  # Default: 20
        "initial_resource": INITIAL_RESOURCE,  # Fixed: 1000
        "regeneration_rate": REGENERATION_RATE,  # Fixed: 2.0
        "max_extraction": max_extraction,  # Calculated
        "Max_fish": Max_fish,  # Fixed: 1000 (renamed from max_fishes)
        "number_of_games": 10,  # Default: 10
        "number_of_players_per_game": number_of_players_per_game,
    }
    
    print(f"Creating experiment: {experiment_name}")
    print(f"Experiment ID: {experiment_id}")
    
    # Save experiment to database
    success = db_manager.save_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        players=players,
        parameters=parameters
    )
    
    if not success:
        print("❌ Failed to create experiment")
        return
    
    print("✅ Experiment created successfully!")
    print(f"\nTo run this experiment, use:")
    print(f"  python main.py --experiment-id {experiment_id} --use-mock")
    print(f"\nOr view it in the Streamlit app:")
    print(f"  streamlit run experiment_app.py")
    
    db_manager.close()
    
    return experiment_id


if __name__ == "__main__":
    create_and_run_experiment()


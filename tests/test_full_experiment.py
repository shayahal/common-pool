"""End-to-end test: Create experiment and run it with mock agents."""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
from cpr_game.persona_prompts import PERSONA_PROMPTS

def test_full_experiment():
    """Test the complete experiment workflow."""
    print("=" * 60)
    print("End-to-End Experiment Test")
    print("=" * 60)
    
    # Initialize database manager
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    print(f"\n1. Initializing database manager...")
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        print("❌ Database is not enabled!")
        return False
    
    print("✅ Database manager initialized")
    
    # Step 1: Create an experiment
    print("\n2. Creating test experiment...")
    experiment_id = "test_full_exp_001"
    experiment_name = "6 Players GPT-3.5 Mock Test"
    
    # Create 6 players with different personas, all using gpt-3.5-turbo
    players = [
        {"persona": "rational_selfish", "model": "gpt-3.5-turbo"},
        {"persona": "cooperative", "model": "gpt-3.5-turbo"},
        {"persona": "aggressive", "model": "gpt-3.5-turbo"},
        {"persona": "conservative", "model": "gpt-3.5-turbo"},
        {"persona": "opportunistic", "model": "gpt-3.5-turbo"},
        {"persona": "altruistic", "model": "gpt-3.5-turbo"},
    ]
    
    parameters = {
        "n_players": 6,  # Pool size
        "max_steps": 20,  # Short game for testing
        "initial_resource": 100,
        "regeneration_rate": 2.0,
        "max_extraction": 35,
        "max_fishes": 100,
        "number_of_games": 5,  # Small number for quick test
        "number_of_players_per_game": 4,  # Randomly select 4 from pool of 6
    }
    
    success = db_manager.save_experiment(
        experiment_id=experiment_id,
        name=experiment_name,
        players=players,
        parameters=parameters
    )
    
    if not success:
        print("❌ Failed to create experiment")
        return False
    
    print(f"✅ Experiment created: {experiment_name}")
    print(f"   ID: {experiment_id}")
    print(f"   Players: {len(players)}")
    print(f"   Games: {parameters['number_of_games']}")
    print(f"   Players per game: {parameters['number_of_players_per_game']}")
    
    # Verify experiment was saved
    loaded = db_manager.load_experiment(experiment_id)
    if not loaded:
        print("❌ Failed to load experiment after creation")
        return False
    
    print("✅ Experiment verified in database")
    
    # Close database connection before running subprocess
    # (SQLite with WAL mode allows multiple connections)
    db_manager.close()
    
    # Step 2: Run the experiment using main.py
    print("\n3. Running experiment with mock agents...")
    print("   (This may take a moment)")
    
    try:
        # Run main.py with the experiment ID and --use-mock flag
        result = subprocess.run(
            [sys.executable, "main.py", "--experiment-id", experiment_id, "--use-mock"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("\n--- main.py output ---")
        print(result.stdout)
        
        if result.stderr:
            print("\n--- main.py errors ---")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Experiment runner failed with exit code {result.returncode}")
            return False
        
        print("✅ Experiment runner completed successfully")
        
    except subprocess.TimeoutExpired:
        print("❌ Experiment runner timed out")
        return False
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify results
    print("\n4. Verifying experiment results...")
    
    # Reopen database connection
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    # Check experiment status
    loaded = db_manager.load_experiment(experiment_id)
    if not loaded:
        print("❌ Failed to load experiment after run")
        return False
    
    print(f"   Status: {loaded['status']}")
    
    if loaded['status'] not in ['completed', 'running']:
        print(f"⚠️  Unexpected status: {loaded['status']}")
    
    # Get results
    results = db_manager.get_experiment_results(experiment_id)
    
    if not results:
        print("❌ No results found")
        return False
    
    print(f"✅ Found {len(results)} game result(s)")
    
    # Verify each result has required fields
    for i, result in enumerate(results):
        summary = result['summary']
        required_fields = ['total_rounds', 'final_resource_level', 'tragedy_occurred', 
                         'avg_cooperation_index', 'cumulative_payoffs']
        
        missing = [f for f in required_fields if f not in summary]
        if missing:
            print(f"❌ Result {i} missing fields: {missing}")
            return False
        
        print(f"   Game {i+1}: {result['game_id']}")
        print(f"      Rounds: {summary['total_rounds']}")
        print(f"      Final Resource: {summary['final_resource_level']:.1f}")
        print(f"      Tragedy: {'Yes' if summary['tragedy_occurred'] else 'No'}")
        print(f"      Cooperation: {summary['avg_cooperation_index']:.3f}")
    
    # Check that we have the expected number of results
    expected_games = parameters['number_of_games']
    if len(results) < expected_games:
        print(f"⚠️  Expected {expected_games} games, but only {len(results)} completed")
        print("   (This is okay if some games failed)")
    else:
        print(f"✅ All {expected_games} games completed")
    
    # Step 4: Check experiment list
    print("\n5. Verifying experiment appears in list...")
    experiments = db_manager.list_experiments()
    
    found = any(exp['experiment_id'] == experiment_id for exp in experiments)
    if not found:
        print("❌ Experiment not found in list")
        return False
    
    exp_info = next(exp for exp in experiments if exp['experiment_id'] == experiment_id)
    print(f"✅ Experiment in list:")
    print(f"   Name: {exp_info['name']}")
    print(f"   Status: {exp_info['status']}")
    print(f"   Games: {exp_info['game_count']}")
    print(f"   Players: {exp_info['player_count']}")
    
    # Step 5: Test cleanup (optional - comment out if you want to keep the test data)
    print("\n6. Cleaning up test experiment...")
    success = db_manager.delete_experiment(experiment_id)
    if success:
        print("✅ Test experiment deleted")
    else:
        print("⚠️  Failed to delete test experiment (not critical)")
    
    db_manager.close()
    
    print("\n" + "=" * 60)
    print("✅ End-to-end test completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_full_experiment()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


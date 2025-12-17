"""Test script for experiment database functionality."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG
import json

def test_database():
    """Test all database operations for experiments."""
    print("=" * 60)
    print("Testing Experiment Database Functionality")
    print("=" * 60)
    
    # Initialize database manager
    db_path = CONFIG.get("db_path", "data/game_results.duckdb")
    db_enabled = CONFIG.get("db_enabled", True)
    
    print(f"\n1. Initializing database manager...")
    print(f"   DB Path: {db_path}")
    print(f"   Enabled: {db_enabled}")
    
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        print("❌ Database is not enabled!")
        return False
    
    if db_manager.conn is None:
        print("❌ Database connection is None!")
        return False
    
    print("✅ Database manager initialized successfully")
    
    # Test 1: Save experiment
    print("\n2. Testing save_experiment()...")
    test_experiment_id = "test_exp_001"
    test_name = "Test Experiment"
    test_players = [
        {"persona": "rational_selfish", "model": "gpt-3.5-turbo"},
        {"persona": "cooperative", "model": "gpt-3.5-turbo"},
        {"persona": "aggressive", "model": "gpt-3.5-turbo"},
    ]
    test_parameters = {
        "n_players": 3,
        "max_steps": 50,
        "initial_resource": 100,
        "regeneration_rate": 2.0,
        "max_extraction": 35,
        "max_fishes": 100,
        "number_of_games": 5,
        "number_of_players_per_game": 2,
    }
    
    success = db_manager.save_experiment(
        experiment_id=test_experiment_id,
        name=test_name,
        players=test_players,
        parameters=test_parameters
    )
    
    if success:
        print("✅ Experiment saved successfully")
    else:
        print("❌ Failed to save experiment")
        return False
    
    # Test 2: Load experiment
    print("\n3. Testing load_experiment()...")
    loaded_experiment = db_manager.load_experiment(test_experiment_id)
    
    if loaded_experiment is None:
        print("❌ Failed to load experiment")
        return False
    
    print(f"✅ Experiment loaded: {loaded_experiment['name']}")
    print(f"   Players: {len(loaded_experiment['players'])}")
    print(f"   Status: {loaded_experiment['status']}")
    
    # Verify data integrity
    if loaded_experiment['name'] != test_name:
        print(f"❌ Name mismatch: expected '{test_name}', got '{loaded_experiment['name']}'")
        return False
    
    if len(loaded_experiment['players']) != len(test_players):
        print(f"❌ Player count mismatch: expected {len(test_players)}, got {len(loaded_experiment['players'])}")
        return False
    
    for i, player in enumerate(test_players):
        if loaded_experiment['players'][i]['persona'] != player['persona']:
            print(f"❌ Player {i} persona mismatch")
            return False
        if loaded_experiment['players'][i]['model'] != player['model']:
            print(f"❌ Player {i} model mismatch")
            return False
    
    print("✅ Data integrity verified")
    
    # Test 3: List experiments
    print("\n4. Testing list_experiments()...")
    experiments = db_manager.list_experiments()
    
    if not experiments:
        print("❌ No experiments found")
        return False
    
    print(f"✅ Found {len(experiments)} experiment(s)")
    
    # Check if our test experiment is in the list
    found = any(exp['experiment_id'] == test_experiment_id for exp in experiments)
    if not found:
        print(f"❌ Test experiment not found in list")
        return False
    
    print("✅ Test experiment found in list")
    
    # Test 4: Update status
    print("\n5. Testing update_experiment_status()...")
    success = db_manager.update_experiment_status(test_experiment_id, "running")
    
    if not success:
        print("❌ Failed to update status")
        return False
    
    # Verify status update
    loaded = db_manager.load_experiment(test_experiment_id)
    if loaded['status'] != "running":
        print(f"❌ Status not updated: expected 'running', got '{loaded['status']}'")
        return False
    
    print("✅ Status updated successfully")
    
    # Test 5: Save experiment result
    print("\n6. Testing save_experiment_result()...")
    test_game_id = f"{test_experiment_id}_game_0001"
    test_summary = {
        "total_rounds": 25,
        "final_resource_level": 50.5,
        "tragedy_occurred": False,
        "avg_cooperation_index": 0.75,
        "cumulative_payoffs": [100.0, 95.0, 110.0],
        "gini_coefficient": 0.15
    }
    
    success = db_manager.save_experiment_result(
        experiment_id=test_experiment_id,
        game_id=test_game_id,
        summary=test_summary
    )
    
    if not success:
        print("❌ Failed to save experiment result")
        return False
    
    print("✅ Experiment result saved successfully")
    
    # Test 6: Get experiment results
    print("\n7. Testing get_experiment_results()...")
    results = db_manager.get_experiment_results(test_experiment_id)
    
    if not results:
        print("❌ No results found")
        return False
    
    print(f"✅ Found {len(results)} result(s)")
    
    # Verify result data
    found_result = next((r for r in results if r['game_id'] == test_game_id), None)
    if not found_result:
        print("❌ Test result not found")
        return False
    
    if found_result['summary']['total_rounds'] != test_summary['total_rounds']:
        print("❌ Result data mismatch")
        return False
    
    print("✅ Result data verified")
    
    # Test 7: Update status to completed
    print("\n8. Testing status update to 'completed'...")
    success = db_manager.update_experiment_status(test_experiment_id, "completed")
    
    if not success:
        print("❌ Failed to update status to completed")
        return False
    
    loaded = db_manager.load_experiment(test_experiment_id)
    if loaded['status'] != "completed":
        print(f"❌ Status not updated: expected 'completed', got '{loaded['status']}'")
        return False
    
    print("✅ Status updated to completed")
    
    # Test 8: List experiments again (should show updated status)
    print("\n9. Testing list_experiments() with updated data...")
    experiments = db_manager.list_experiments()
    test_exp = next((exp for exp in experiments if exp['experiment_id'] == test_experiment_id), None)
    
    if test_exp is None:
        print("❌ Test experiment not found in list")
        return False
    
    if test_exp['status'] != "completed":
        print(f"❌ Status mismatch in list: expected 'completed', got '{test_exp['status']}'")
        return False
    
    if test_exp['game_count'] != 1:
        print(f"❌ Game count mismatch: expected 1, got {test_exp['game_count']}")
        return False
    
    print("✅ List shows updated status and game count")
    
    # Test 9: Delete experiment
    print("\n10. Testing delete_experiment()...")
    success = db_manager.delete_experiment(test_experiment_id)
    
    if not success:
        print("❌ Failed to delete experiment")
        return False
    
    # Verify deletion
    loaded = db_manager.load_experiment(test_experiment_id)
    if loaded is not None:
        print("❌ Experiment still exists after deletion")
        return False
    
    results = db_manager.get_experiment_results(test_experiment_id)
    if results:
        print("❌ Results still exist after deletion")
        return False
    
    print("✅ Experiment deleted successfully (cascade delete verified)")
    
    # Cleanup
    db_manager.close()
    
    print("\n" + "=" * 60)
    print("✅ All database tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


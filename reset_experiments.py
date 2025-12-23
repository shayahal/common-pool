"""Reset all experiments to pending status.

This script resets all experiments (regardless of current status) to 'pending' status,
allowing them to be reprocessed by the experiment worker.

Usage:
    python reset_experiments.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.db_manager import DatabaseManager
from cpr_game.config import CONFIG

def main():
    """Reset all experiments to pending status."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    print("=" * 60)
    print("RESET EXPERIMENTS TO PENDING")
    print("=" * 60)
    
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        print("❌ Database is not enabled")
        return 1
    
    try:
        # Get current status counts
        conn = db_manager.get_read_connection()
        if conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM experiments GROUP BY status"
            )
            status_counts = dict(cursor.fetchall())
            print("\nCurrent experiment statuses:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        # Reset all experiments
        print("\nResetting all experiments to 'pending'...")
        count = db_manager.reset_all_experiments_to_pending()
        
        if count >= 0:
            print(f"\n✅ Successfully reset {count} experiment(s) to 'pending' status")
            
            # Verify
            conn = db_manager.get_read_connection()
            if conn:
                cursor = conn.execute(
                    "SELECT status, COUNT(*) FROM experiments GROUP BY status"
                )
                status_counts = dict(cursor.fetchall())
                print("\nUpdated experiment statuses:")
                for status, count in status_counts.items():
                    print(f"  {status}: {count}")
            
            print("\n" + "=" * 60)
            return 0
        else:
            print("\n❌ Failed to reset experiments")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db_manager.close()


if __name__ == "__main__":
    sys.exit(main())


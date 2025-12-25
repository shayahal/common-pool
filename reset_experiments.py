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
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

def main():
    """Reset all experiments to pending status."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    logger.info("=" * 60)
    logger.info("RESET EXPERIMENTS TO PENDING")
    logger.info("=" * 60)
    
    db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled)
    
    if not db_manager.enabled:
        logger.error("Database is not enabled")
        return 1
    
    try:
        # Get current status counts
        conn = db_manager.get_read_connection()
        if conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*) FROM experiments GROUP BY status"
            )
            status_counts = dict(cursor.fetchall())
            logger.info("Current experiment statuses:")
            for status, count in status_counts.items():
                logger.info(f"  {status}: {count}")
        
        # Reset all experiments
        logger.info("Resetting all experiments to 'pending'...")
        count = db_manager.reset_all_experiments_to_pending()
        
        if count >= 0:
            logger.info(f"Successfully reset {count} experiment(s) to 'pending' status")
            
            # Verify
            conn = db_manager.get_read_connection()
            if conn:
                cursor = conn.execute(
                    "SELECT status, COUNT(*) FROM experiments GROUP BY status"
                )
                status_counts = dict(cursor.fetchall())
                logger.info("Updated experiment statuses:")
                for status, count in status_counts.items():
                    logger.info(f"  {status}: {count}")
            
            logger.info("=" * 60)
            return 0
        else:
            logger.error("Failed to reset experiments")
            return 1
            
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        db_manager.close()


if __name__ == "__main__":
    sys.exit(main())


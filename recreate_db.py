"""Script to recreate an empty database with all tables."""

import os
import sys
from pathlib import Path

# Import directly to avoid dependency issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config first (minimal dependencies)
from cpr_game.config import CONFIG

# Import DatabaseManager (SQLite only, no external deps)
from cpr_game.db_manager import DatabaseManager
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

def recreate_database():
    """Delete existing database and create a new empty one."""
    db_path = CONFIG.get("db_path", "data/game_results.db")
    db_enabled = CONFIG.get("db_enabled", True)
    
    logger.info(f"Recreating database at: {db_path}")
    
    # Delete existing database file if it exists
    db_file = Path(db_path)
    if db_file.exists():
        logger.debug("Deleting existing database file...")
        try:
            # Close any existing connections first
            db_file.unlink()
            logger.debug("Deleted existing database file")
        except Exception as e:
            logger.warning(f"Could not delete existing file: {e} (This might be because it's in use. Continuing anyway...)")
    
    # Also delete WAL and SHM files if they exist
    wal_file = db_file.with_suffix('.db-wal')
    shm_file = db_file.with_suffix('.db-shm')
    for extra_file in [wal_file, shm_file]:
        if extra_file.exists():
            try:
                extra_file.unlink()
                logger.debug(f"Deleted {extra_file.name}")
            except Exception:
                pass
    
    # Create new database by initializing DatabaseManager
    # This will automatically create all tables
    logger.debug("Creating new empty database...")
    try:
        db_manager = DatabaseManager(db_path=db_path, enabled=db_enabled, access_mode='READ_WRITE')
        
        if db_manager.enabled and db_manager.conn is not None:
            # Verify tables were created
            cursor = db_manager.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"Database created successfully!")
            logger.info(f"Created {len(tables)} tables:")
            for table in tables:
                logger.info(f"  - {table}")
            
            db_manager.close()
            logger.info("Database recreation complete!")
            return True
        else:
            logger.error("Database manager is disabled or connection failed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to create database: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    recreate_database()


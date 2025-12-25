#!/usr/bin/env python3
"""CLI script for managing FalkorDB container.

Usage:
    python manage_falkordb.py [start|stop|status|restart]

Commands:
    start   - Start FalkorDB container (automatically starts if not running)
    stop    - Stop FalkorDB container
    status  - Check if FalkorDB is running
    restart - Restart FalkorDB container
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cpr_game.falkordb_manager import (
    start_falkordb,
    stop_falkordb,
    is_falkordb_running,
    get_falkordb_status,
)
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        command = "status"
    else:
        command = sys.argv[1].lower()
    
    if command == "start":
        logger.info("Starting FalkorDB...")
        if start_falkordb():
            logger.info("✓ FalkorDB started successfully")
            sys.exit(0)
        else:
            logger.error("✗ Failed to start FalkorDB")
            sys.exit(1)
    
    elif command == "stop":
        logger.info("Stopping FalkorDB...")
        if stop_falkordb():
            logger.info("✓ FalkorDB stopped successfully")
            sys.exit(0)
        else:
            logger.error("✗ Failed to stop FalkorDB")
            sys.exit(1)
    
    elif command == "status":
        status = get_falkordb_status()
        is_running = is_falkordb_running()
        
        if status == "running" and is_running:
            logger.info("✓ FalkorDB is running")
            sys.exit(0)
        elif status == "stopped":
            logger.info("○ FalkorDB is stopped")
            sys.exit(0)
        elif status == "not_found":
            logger.info("○ FalkorDB container not found")
            sys.exit(0)
        else:
            logger.warning(f"○ FalkorDB status: {status}")
            sys.exit(1)
    
    elif command == "restart":
        logger.info("Restarting FalkorDB...")
        stop_falkordb()
        if start_falkordb():
            logger.info("✓ FalkorDB restarted successfully")
            sys.exit(0)
        else:
            logger.error("✗ Failed to restart FalkorDB")
            sys.exit(1)
    
    else:
        logger.error(f"Unknown command: {command}")
        logger.info("Usage: python manage_falkordb.py [start|stop|status|restart]")
        sys.exit(1)


if __name__ == "__main__":
    main()


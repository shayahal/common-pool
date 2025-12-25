"""Test script to verify FalkorDB integration with OpenTelemetry traces.

This script:
1. Checks if FalkorDB is running
2. Runs a simple game to generate traces
3. Verifies traces are stored in FalkorDB via Graphiti
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cpr_game.config import CONFIG
from cpr_game.game_runner import GameRunner
from cpr_game.falkordb_exporter import FalkorDBExporter
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    logger.error("Graphiti not available. Install with: pip install graphiti-core[falkordb]")
    sys.exit(1)


async def check_falkordb_connection():
    """Check if FalkorDB is accessible."""
    logger.info("Checking FalkorDB connection...")
    try:
        falkor_driver = FalkorDriver(
            host=CONFIG.get("falkordb_host", "localhost"),
            port=str(CONFIG.get("falkordb_port", 6379)),
            username=CONFIG.get("falkordb_username"),
            password=CONFIG.get("falkordb_password"),
        )
        graphiti = Graphiti(graph_driver=falkor_driver)
        await graphiti.build_indices_and_constraints()
        logger.info("FalkorDB connection successful!")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to FalkorDB: {e}", exc_info=True)
        logger.warning("Make sure FalkorDB is running:")
        logger.warning("  docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest")
        return False


async def query_traces_from_falkordb(group_id: str = "cpr-game-traces"):
    """Query traces from FalkorDB using Graphiti."""
    logger.info(f"Querying traces from FalkorDB (group_id: {group_id})...")
    try:
        falkor_driver = FalkorDriver(
            host=CONFIG.get("falkordb_host", "localhost"),
            port=str(CONFIG.get("falkordb_port", 6379)),
            username=CONFIG.get("falkordb_username"),
            password=CONFIG.get("falkordb_password"),
        )
        graphiti = Graphiti(graph_driver=falkor_driver)
        
        # Search for game traces
        results = await graphiti.search("game trace span")
        logger.info(f"Found {len(results)} results")
        
        if results:
            logger.debug("Sample traces:")
            for i, result in enumerate(results[:5]):  # Show first 5
                logger.debug(f"  {i+1}. {result.fact}")
                if hasattr(result, 'source_node_uuid'):
                    logger.debug(f"     Node UUID: {result.source_node_uuid}")
        
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error querying FalkorDB: {e}", exc_info=True)
        return False


def run_test_game():
    """Run a simple test game to generate traces."""
    logger.info("Running test game to generate traces...")
    
    # Ensure FalkorDB is enabled
    config = CONFIG.copy()
    config["falkordb_enabled"] = True
    config["otel_enabled"] = True
    
    try:
        runner = GameRunner(config=config)
        runner.setup_game(
            n_players=2,
            max_steps=3,  # Short game for testing
            initial_resource=1000
        )
        
        # Run game
        result = runner.run_game()
        logger.info(f"Game completed: {result['summary']['total_rounds']} rounds")
        
        # Flush traces
        if hasattr(runner, 'logging_manager') and runner.logging_manager:
            runner.logging_manager.otel_manager.flush()
            logger.debug("Traces flushed to FalkorDB")
        
        return True
    except Exception as e:
        logger.error(f"Error running game: {e}", exc_info=True)
        return False


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("FalkorDB Integration Test")
    logger.info("=" * 60)
    
    # Step 1: Check connection
    if not await check_falkordb_connection():
        return False
    
    # Step 2: Run test game
    if not run_test_game():
        return False
    
    # Wait a bit for traces to be processed
    logger.debug("Waiting for traces to be processed...")
    await asyncio.sleep(2)
    
    # Step 3: Query traces
    traces_found = await query_traces_from_falkordb()
    
    if traces_found:
        logger.info("=" * 60)
        logger.info("Traces are being stored in FalkorDB!")
        logger.info("=" * 60)
        return True
    else:
        logger.info("=" * 60)
        logger.warning("No traces found in FalkorDB")
        logger.warning("This might be normal if traces are still being processed.")
        logger.warning("Try running the test again or check the logs.")
        logger.info("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)


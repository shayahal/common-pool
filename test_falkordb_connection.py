"""Simple test to verify FalkorDB connection and basic trace storage."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    from graphiti_core.nodes import EpisodeType
    from datetime import datetime, timezone
    import json
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    logger.error("Graphiti not available. Install with: pip install graphiti-core[falkordb]")
    sys.exit(1)


async def test_connection():
    """Test FalkorDB connection."""
    logger.info("Testing FalkorDB connection...")
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
        return graphiti
    except Exception as e:
        logger.error(f"Failed to connect to FalkorDB: {e}", exc_info=True)
        return None


async def test_trace_storage(graphiti):
    """Test storing a sample trace."""
    logger.info("Testing trace storage...")
    try:
        # Create a sample trace episode
        trace_data = {
            "span_name": "test_span",
            "trace_id": "test_trace_123",
            "span_id": "test_span_456",
            "start_time": datetime.now(timezone.utc).timestamp(),
            "attributes": {
                "test.attribute": "test_value",
                "game.id": "test_game"
            }
        }
        
        episode_content = json.dumps(trace_data)
        
        await graphiti.add_episode(
            name="test_span_test_span_456",
            episode_body=episode_content,
            source=EpisodeType.json,
            source_description="Test span | Game: test_game",
            reference_time=datetime.now(timezone.utc),
            group_id=CONFIG.get("falkordb_group_id", "cpr-game-traces")
        )
        logger.info("Trace stored successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to store trace: {e}", exc_info=True)
        return False


async def test_trace_query(graphiti):
    """Test querying traces."""
    logger.info("Testing trace query...")
    try:
        results = await graphiti.search("test span")
        logger.info(f"Found {len(results)} results")
        
        if results:
            logger.debug("Sample results:")
            for i, result in enumerate(results[:3]):
                logger.debug(f"  {i+1}. {result.fact}")
        
        return len(results) > 0
    except Exception as e:
        logger.error(f"Failed to query traces: {e}", exc_info=True)
        return False


async def main():
    """Main test function."""
    logger.info("=" * 60)
    logger.info("FalkorDB Connection Test")
    logger.info("=" * 60)
    
    # Test connection
    graphiti = await test_connection()
    if not graphiti:
        return False
    
    # Test storage
    if not await test_trace_storage(graphiti):
        return False
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Test query
    found = await test_trace_query(graphiti)
    
    logger.info("=" * 60)
    if found:
        logger.info("FalkorDB integration is working!")
    else:
        logger.warning("Traces stored but not found in query")
        logger.warning("This might be normal - Graphiti may need more time to index")
    logger.info("=" * 60)
    
    return found


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)


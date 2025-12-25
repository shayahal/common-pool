"""Check if FalkorDB exporter is actually enabled and working."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cpr_game.config import CONFIG
from cpr_game.falkordb_exporter import FalkorDBExporter
from cpr_game.logger_setup import get_logger, setup_logging
import os

# Setup logging
setup_logging()
logger = get_logger(__name__)

logger.info("=" * 60)
logger.info("FalkorDB Exporter Status Check")
logger.info("=" * 60)

# Check config
logger.info("Config settings:")
logger.info(f"  falkordb_enabled: {CONFIG.get('falkordb_enabled')}")
logger.info(f"  falkordb_host: {CONFIG.get('falkordb_host')}")
logger.info(f"  falkordb_port: {CONFIG.get('falkordb_port')}")

# Check OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")
exporter = None
if openai_key:
    logger.info(f"OPENAI_API_KEY is set (length: {len(openai_key)})")
else:
    logger.error("OPENAI_API_KEY is NOT set!")
    logger.error("This is REQUIRED for Graphiti to work.")
    logger.error("Set it in your .env file or environment variables.")

# Try to create exporter
logger.info("Attempting to create FalkorDB exporter...")
try:
    exporter = FalkorDBExporter(
        host=CONFIG.get("falkordb_host", "localhost"),
        port=CONFIG.get("falkordb_port", 6379),
        username=CONFIG.get("falkordb_username"),
        password=CONFIG.get("falkordb_password"),
        group_id=CONFIG.get("falkordb_group_id", "cpr-game-traces"),
        enabled=CONFIG.get("falkordb_enabled", True)
    )
    
    logger.debug(f"Exporter created")
    logger.debug(f"Enabled: {exporter.enabled}")
    logger.debug(f"Graphiti initialized: {exporter.graphiti is not None}")
    
    if not exporter.enabled:
        logger.warning("Exporter is DISABLED!")
        if not openai_key:
            logger.warning("Reason: OPENAI_API_KEY is missing")
            logger.warning("Solution: Set OPENAI_API_KEY in your .env file")
        else:
            logger.warning("Reason: Unknown (check logs for details)")
    else:
        logger.info("Exporter is ENABLED and ready to export traces!")
        
except Exception as e:
    logger.error(f"Failed to create exporter: {e}", exc_info=True)

logger.info("=" * 60)
logger.info("Summary:")
logger.info("=" * 60)
if exporter and openai_key and exporter.enabled:
    logger.info("FalkorDB exporter is ready!")
    logger.info("Traces from your experiments will be stored in FalkorDB.")
elif not openai_key:
    logger.warning("ACTION REQUIRED: Set OPENAI_API_KEY to enable FalkorDB export")
    logger.warning("Add to .env file: OPENAI_API_KEY=sk-...")
else:
    logger.warning("Exporter is not enabled. Check logs for details.")


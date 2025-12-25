"""Test script to generate a simple trace and verify it appears in LangSmith.

This script creates a trace without making any API calls, so it works even
when API limits are hit.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cpr_game.otel_manager import OTelManager
from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger

logger = get_logger(__name__)


def main():
    """Create a simple test trace."""
    logger.info("=" * 70)
    logger.info("LangSmith Trace Export Test")
    logger.info("=" * 70)
    
    # Initialize OpenTelemetry
    logger.info("Initializing OpenTelemetry...")
    otel_manager = OTelManager(CONFIG)
    tracer = otel_manager.get_tracer()
    
    if tracer is None:
        logger.error("OpenTelemetry is disabled!")
        return 1
    
    logger.info("Creating test trace...")
    
    # Create a root trace
    with tracer.start_as_current_span(
        "test_trace_export",
        attributes={
            "test.type": "langsmith_export_verification",
            "test.timestamp": str(int(time.time())),
        }
    ) as root_span:
        root_span.set_attribute("test.description", "Simple trace to verify LangSmith export")
        
        # Create a child span
        with tracer.start_as_current_span(
            "test_span_1",
            attributes={
                "test.step": 1,
                "test.message": "This is a test span"
            }
        ) as child_span:
            child_span.add_event("test.event", {"message": "Test event in span"})
            time.sleep(0.1)  # Small delay to give it some duration
        
        # Create another child span
        with tracer.start_as_current_span(
            "test_span_2",
            attributes={
                "test.step": 2,
                "test.message": "Another test span"
            }
        ) as child_span2:
            child_span2.add_event("test.event", {"message": "Another test event"})
            time.sleep(0.1)
    
    # Flush to ensure traces are sent
    logger.info("Flushing traces...")
    otel_manager.flush()
    
    logger.info("=" * 70)
    logger.info("✅ Test trace created!")
    logger.info("=" * 70)
    logger.info("Check LangSmith for a trace named 'test_trace_export'")
    logger.info("It should have 2 child spans: 'test_span_1' and 'test_span_2'")
    logger.info("=" * 70)
    
    # Give collector time to export
    logger.info("Waiting 3 seconds for collector to export...")
    time.sleep(3)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


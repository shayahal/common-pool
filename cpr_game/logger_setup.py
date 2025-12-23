"""Application-level logging setup for CPR game.

This module configures Python's standard logging system for application-level
logs (errors, info messages, debug output). All application logs from
logger.info(), logger.error(), etc. go to both the log file and console.

Purpose: Development and debugging
Output:
    - logs/cpr_game.log: Main application log file (DEBUG level and above)
    - Console/stdout: Application logs (INFO level and above)

For distributed tracing and observability (including API metrics), see
logging_manager.py which uses OpenTelemetry to send traces to configured
receivers (Langfuse/LangSmith).
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
import warnings

# Suppress Streamlit warnings when running outside Streamlit context (e.g., in tests)
# These warnings occur when Streamlit code runs outside the Streamlit runtime
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*", category=UserWarning)

# Suppress Streamlit logger warnings about missing ScriptRunContext
# Set to CRITICAL to completely suppress these warnings
streamlit_loggers = [
    "streamlit.runtime.scriptrunner_utils.script_run_context",
    "streamlit.runtime.state.session_state_proxy",
]

for logger_name in streamlit_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    # Also add a filter to catch any remaining warnings
    class ScriptRunContextFilter(logging.Filter):
        def filter(self, record):
            return "ScriptRunContext" not in record.getMessage()
    logger.addFilter(ScriptRunContextFilter())


def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration.
    
    Configures Python logging to write to:
    - File: {log_dir}/cpr_game.log (DEBUG level and above)
    - Console: stdout (INFO level and above)
    
    Args:
        log_dir: Directory to store log files (default: "logs")
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_path / "cpr_game.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    stdout_handler.setFormatter(stdout_formatter)
    root_logger.addHandler(stdout_handler)


class JSONFileHandler(logging.Handler):
    """Custom handler for structured JSON logging of API calls."""
    
    def __init__(self, filename: str):
        """Initialize JSON file handler.
        
        Args:
            filename: Path to JSON log file
        """
        super().__init__()
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record as JSON.
        
        Args:
            record: Log record to emit
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.format(record)
            
            # Write as JSON line
            with open(self.filename, "a") as f:
                json.dump(log_entry, f)
                f.write("\n")
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


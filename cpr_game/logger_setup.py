"""Application-level logging setup for CPR game.

This module configures Python's standard logging system for application-level
logs (errors, info messages, debug output). All application logs from
logger.info(), logger.error(), etc. go to both the log files and console.

Purpose: Development and debugging
Output:
    - logs/debug.log: DEBUG level and above
    - logs/info.log: INFO level and above
    - logs/warning.log: WARNING level and above
    - logs/error.log: ERROR level and above
    - Console/stdout: All levels (DEBUG and above)

Each file handler captures its level and all levels above it.

For distributed tracing and observability (including API metrics), see
logging_manager.py which uses OpenTelemetry to send traces to configured
receivers (Langfuse/LangSmith).
"""

import logging
import sys
import json
import os
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

# Configure third-party library loggers to use WARNING level
# These libraries log HTTP requests/responses at INFO level by default,
# but these are implementation details, not application-level events
third_party_loggers = [
    "httpx",                    # HTTP client library (used by OpenAI, LangChain)
    "httpcore",                 # Low-level HTTP library (used by httpx)
    "urllib3",                  # HTTP library (used by requests and others)
    "requests",                 # HTTP library
    "openai",                   # OpenAI client library
    "openai._client",           # OpenAI internal client
    "openai.resources",         # OpenAI resources
    "langchain",                # LangChain library
    "langchain_openai",         # LangChain OpenAI integration
    "langchain_community",      # LangChain community integrations
]
for logger_name in third_party_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)  # Only show warnings and errors, suppress INFO/DEBUG


def setup_logging(log_dir: str = "logs") -> None:
    """Setup logging configuration.
    
    Configures Python logging to write to:
    - Files: {log_dir}/debug.log (DEBUG+), info.log (INFO+), warning.log (WARNING+), error.log (ERROR+)
    - Console: stdout (all levels)
    
    Each file handler captures its level and all levels above it.
    
    Third-party library loggers (httpx, urllib3, etc.) are configured to WARNING
    level to suppress verbose HTTP request/response logging.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Re-apply third-party logger configuration after clearing handlers
    # (in case setup_logging is called multiple times)
    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Formatter for file handlers
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Formatter for stdout handler
    stdout_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    # File handler for DEBUG level and above
    debug_handler = logging.FileHandler(log_path / "debug.log", encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(file_formatter)
    root_logger.addHandler(debug_handler)
    
    # File handler for INFO level and above
    info_handler = logging.FileHandler(log_path / "info.log", encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(file_formatter)
    root_logger.addHandler(info_handler)
    
    # File handler for WARNING level and above
    warning_handler = logging.FileHandler(log_path / "warning.log", encoding='utf-8')
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(file_formatter)
    root_logger.addHandler(warning_handler)
    
    # File handler for ERROR level and above
    error_handler = logging.FileHandler(log_path / "error.log", encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)
    
    # Console handler - use a wrapper that handles encoding errors gracefully
    class SafeStreamHandler(logging.StreamHandler):
        """StreamHandler that safely handles Unicode encoding errors."""
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Fallback: encode message as ASCII with error replacement
                try:
                    msg = self.format(record)
                    stream = self.stream
                    stream.write(msg.encode('ascii', errors='replace').decode('ascii') + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
    
    # Stdout handler - use STDOUT_LOG_LEVEL from environment if provided, otherwise DEBUG
    stdout_log_level = os.getenv("STDOUT_LOG_LEVEL", "DEBUG").upper()
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    stdout_level = level_map.get(stdout_log_level, logging.DEBUG)
    
    stdout_handler = SafeStreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
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


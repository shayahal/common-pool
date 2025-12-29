"""Utility functions for Langfuse GraphRAG system."""

import json
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"Saved JSON to {file_path}")


def load_json(file_path: Path) -> Any:
    """Load data from JSON file.
    
    Args:
        file_path: Input file path
    
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_datetime(dt: Any) -> Optional[str]:
    """Format datetime to ISO string.
    
    Args:
        dt: Datetime object or string
    
    Returns:
        ISO format string or None
    """
    if dt is None:
        return None
    
    if isinstance(dt, datetime):
        return dt.isoformat()
    
    if isinstance(dt, str):
        return dt
    
    return str(dt)


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
    
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size.
    
    Args:
        items: List to chunk
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def safe_get(dictionary: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested dictionary value.
    
    Args:
        dictionary: Dictionary to query
        keys: Nested keys
        default: Default value if key not found
    
    Returns:
        Value or default
    """
    result = dictionary
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        else:
            return default
    return result if result is not None else default


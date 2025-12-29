"""Unit tests for langfuse_graphrag.utils module."""

import pytest
import json
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory
from langfuse_graphrag.utils import (
    ensure_directory,
    save_json,
    load_json,
    format_datetime,
    truncate_text,
    chunk_list,
    safe_get,
)


class TestDirectoryUtils:
    """Test directory utility functions."""
    
    def test_ensure_directory(self, tmp_path):
        """Test ensuring directory exists."""
        dir_path = tmp_path / "test_dir"
        ensure_directory(dir_path)
        assert dir_path.exists()
        assert dir_path.is_dir()
    
    def test_ensure_directory_nested(self, tmp_path):
        """Test ensuring nested directory exists."""
        dir_path = tmp_path / "level1" / "level2" / "level3"
        ensure_directory(dir_path)
        assert dir_path.exists()


class TestJSONUtils:
    """Test JSON utility functions."""
    
    def test_save_json(self, tmp_path):
        """Test saving JSON file."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        
        save_json(data, file_path)
        
        assert file_path.exists()
        with open(file_path, 'r') as f:
            loaded = json.load(f)
            assert loaded == data
    
    def test_load_json(self, tmp_path):
        """Test loading JSON file."""
        file_path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        
        with open(file_path, 'w') as f:
            json.dump(data, f)
        
        loaded = load_json(file_path)
        assert loaded == data
    
    def test_save_json_creates_parent_dirs(self, tmp_path):
        """Test saving JSON creates parent directories."""
        file_path = tmp_path / "nested" / "path" / "test.json"
        data = {"test": "data"}
        
        save_json(data, file_path)
        assert file_path.exists()


class TestDatetimeUtils:
    """Test datetime utility functions."""
    
    def test_format_datetime_object(self):
        """Test formatting datetime object."""
        dt = datetime(2024, 1, 1, 10, 0, 0)
        result = format_datetime(dt)
        assert result == "2024-01-01T10:00:00"
    
    def test_format_datetime_string(self):
        """Test formatting datetime string."""
        dt_str = "2024-01-01T10:00:00"
        result = format_datetime(dt_str)
        assert result == dt_str
    
    def test_format_datetime_none(self):
        """Test formatting None datetime."""
        result = format_datetime(None)
        assert result is None


class TestTextUtils:
    """Test text utility functions."""
    
    def test_truncate_text_short(self):
        """Test truncating short text."""
        text = "short"
        result = truncate_text(text, max_length=100)
        assert result == "short"
    
    def test_truncate_text_long(self):
        """Test truncating long text."""
        text = "a" * 200
        result = truncate_text(text, max_length=100)
        assert len(result) == 100
        assert result.endswith("...")
    
    def test_truncate_text_empty(self):
        """Test truncating empty text."""
        result = truncate_text("")
        assert result == ""
    
    def test_truncate_text_none(self):
        """Test truncating None."""
        result = truncate_text(None)
        assert result == ""


class TestListUtils:
    """Test list utility functions."""
    
    def test_chunk_list_exact(self):
        """Test chunking list with exact size."""
        items = list(range(10))
        chunks = chunk_list(items, chunk_size=5)
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2, 3, 4]
        assert chunks[1] == [5, 6, 7, 8, 9]
    
    def test_chunk_list_remainder(self):
        """Test chunking list with remainder."""
        items = list(range(10))
        chunks = chunk_list(items, chunk_size=3)
        assert len(chunks) == 4
        assert len(chunks[-1]) == 1
    
    def test_chunk_list_empty(self):
        """Test chunking empty list."""
        chunks = chunk_list([], chunk_size=5)
        assert chunks == []
    
    def test_chunk_list_single(self):
        """Test chunking single item."""
        chunks = chunk_list([1], chunk_size=5)
        assert chunks == [[1]]


class TestDictUtils:
    """Test dictionary utility functions."""
    
    def test_safe_get_existing(self):
        """Test getting existing nested key."""
        data = {"level1": {"level2": {"level3": "value"}}}
        result = safe_get(data, "level1", "level2", "level3")
        assert result == "value"
    
    def test_safe_get_missing(self):
        """Test getting missing key."""
        data = {"level1": {"level2": {}}}
        result = safe_get(data, "level1", "level2", "level3")
        assert result is None
    
    def test_safe_get_with_default(self):
        """Test getting with default value."""
        data = {"level1": {}}
        result = safe_get(data, "level1", "level2", default="default")
        assert result == "default"
    
    def test_safe_get_top_level(self):
        """Test getting top-level key."""
        data = {"key": "value"}
        result = safe_get(data, "key")
        assert result == "value"
    
    def test_safe_get_non_dict(self):
        """Test getting from non-dict value."""
        data = {"key": "string"}
        result = safe_get(data, "key", "nested")
        assert result is None


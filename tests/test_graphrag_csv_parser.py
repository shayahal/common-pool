"""Unit tests for langfuse_graphrag.csv_parser module."""

import pytest
import csv
import tempfile
import os
from pathlib import Path
from langfuse_graphrag.csv_parser import LangfuseCSVParser


@pytest.fixture
def sample_trace_csv(tmp_path):
    """Create a sample trace CSV file."""
    csv_file = tmp_path / "traces.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "session_id", "trace_id", "timestamp", "user_id"])
        writer.writerow(["trace1", "Test Trace", "session1", "trace1", "2024-01-01T10:00:00", "user1"])
        writer.writerow(["trace2", "Another Trace", "session1", "trace2", "2024-01-01T11:00:00", "user1"])
    return str(csv_file)


@pytest.fixture
def sample_generation_csv(tmp_path):
    """Create a sample generation CSV file."""
    csv_file = tmp_path / "generations.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "trace_id", "model", "prompt", "response", "tokens_input", "tokens_output"])
        writer.writerow(["gen1", "trace1", "gpt-4", "What is 2+2?", "4", "10", "5"])
        writer.writerow(["gen2", "trace1", "gpt-4", "What is 3+3?", "6", "10", "5"])
    return str(csv_file)


class TestCSVParser:
    """Test CSV parser functionality."""
    
    def test_parser_initialization(self, sample_trace_csv):
        """Test parser initialization."""
        parser = LangfuseCSVParser(sample_trace_csv)
        assert parser.csv_path == Path(sample_trace_csv)
    
    def test_parser_file_not_found(self):
        """Test parser raises error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            LangfuseCSVParser("nonexistent.csv")
    
    def test_detect_csv_type_trace(self, sample_trace_csv):
        """Test CSV type detection for trace."""
        parser = LangfuseCSVParser(sample_trace_csv)
        csv_type = parser.detect_csv_type()
        assert csv_type == "trace"
    
    def test_detect_csv_type_generation(self, sample_generation_csv):
        """Test CSV type detection for generation."""
        parser = LangfuseCSVParser(sample_generation_csv)
        csv_type = parser.detect_csv_type()
        assert csv_type == "generation"
    
    def test_parse_trace_csv(self, sample_trace_csv):
        """Test parsing trace CSV."""
        parser = LangfuseCSVParser(sample_trace_csv)
        records = parser.parse()
        
        assert len(records) == 2
        assert records[0]["_csv_type"] == "trace"
        assert "id" in records[0]
        assert records[0]["id"] == "trace1"
        assert records[0]["name"] == "Test Trace"
    
    def test_parse_generation_csv(self, sample_generation_csv):
        """Test parsing generation CSV."""
        parser = LangfuseCSVParser(sample_generation_csv)
        records = parser.parse()
        
        assert len(records) == 2
        assert records[0]["_csv_type"] == "generation"
        assert records[0]["id"] == "gen1"
        assert records[0]["model"] == "gpt-4"
    
    def test_column_normalization(self, sample_trace_csv):
        """Test column name normalization."""
        parser = LangfuseCSVParser(sample_trace_csv)
        records = parser.parse()
        
        # Check that standard column names are used
        assert "session_id" in records[0]
        assert "user_id" in records[0]
    
    def test_get_summary(self, sample_trace_csv):
        """Test getting CSV summary."""
        parser = LangfuseCSVParser(sample_trace_csv)
        summary = parser.get_summary()
        
        assert "csv_type" in summary
        assert "row_count" in summary
        assert summary["row_count"] == 2
        assert "column_count" in summary
        assert "columns" in summary
    
    def test_parse_streaming(self, sample_trace_csv):
        """Test streaming CSV parsing."""
        parser = LangfuseCSVParser(sample_trace_csv)
        records = list(parser.parse_streaming())
        
        assert len(records) == 2
        assert records[0]["_csv_type"] == "trace"
    
    def test_parse_empty_csv(self, tmp_path):
        """Test parsing empty CSV."""
        csv_file = tmp_path / "empty.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name"])
        
        parser = LangfuseCSVParser(str(csv_file))
        records = parser.parse()
        assert len(records) == 0
    
    def test_parse_metadata_json(self, tmp_path):
        """Test parsing JSON metadata field."""
        csv_file = tmp_path / "metadata.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "metadata"])
            writer.writerow(["trace1", '{"key": "value", "number": 42}'])
        
        parser = LangfuseCSVParser(str(csv_file))
        records = parser.parse()
        
        assert len(records) == 1
        assert records[0]["metadata"] is not None
        # Metadata should be valid JSON string
        import json
        metadata = json.loads(records[0]["metadata"])
        assert metadata["key"] == "value"
    
    def test_parse_datetime_formats(self, tmp_path):
        """Test parsing various datetime formats."""
        csv_file = tmp_path / "datetime.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp"])
            writer.writerow(["trace1", "2024-01-01T10:00:00"])
            writer.writerow(["trace2", "2024-01-01 10:00:00"])
        
        parser = LangfuseCSVParser(str(csv_file))
        records = parser.parse()
        
        assert len(records) == 2
        # Both should have timestamp parsed (or at least present)
        assert "timestamp" in records[0] or records[0].get("timestamp") is not None


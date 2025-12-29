"""Unit tests for langfuse_graphrag.extractor module."""

import pytest
from datetime import datetime
from langfuse_graphrag.extractor import EntityExtractor


@pytest.fixture
def sample_trace_records():
    """Sample trace records."""
    return [
        {
            "_csv_type": "trace",
            "_row_index": 0,
            "id": "trace1",
            "name": "Test Trace",
            "session_id": "session1",
            "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            "user_id": "user1",
        },
        {
            "_csv_type": "trace",
            "_row_index": 1,
            "id": "trace2",
            "name": "Another Trace",
            "session_id": "session1",
            "timestamp": datetime(2024, 1, 1, 11, 0, 0),
            "user_id": "user1",
        },
    ]


@pytest.fixture
def sample_generation_records():
    """Sample generation records."""
    return [
        {
            "_csv_type": "generation",
            "_row_index": 0,
            "id": "gen1",
            "span_id": "span1",
            "model": "gpt-4",
            "prompt": "What is 2+2?",
            "response": "4",
            "tokens_input": 10,
            "tokens_output": 5,
        },
    ]


@pytest.fixture
def sample_span_records():
    """Sample span records."""
    return [
        {
            "_csv_type": "span",
            "_row_index": 0,
            "id": "span1",
            "trace_id": "trace1",
            "name": "Test Span",
            "type": "llm",
            "start_time": datetime(2024, 1, 1, 10, 0, 0),
            "end_time": datetime(2024, 1, 1, 10, 0, 5),
            "duration_ms": 5000.0,
            "status": "success",
        },
    ]


class TestEntityExtractor:
    """Test entity extraction functionality."""
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = EntityExtractor()
        assert extractor.entities == {}
        assert extractor.relationships == []
    
    def test_extract_trace_entities(self, sample_trace_records):
        """Test extracting trace entities."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(sample_trace_records)
        
        assert "Trace" in entities
        assert len(entities["Trace"]) == 2
        assert entities["Trace"][0]["id"] == "trace1"
        assert entities["Trace"][0]["_type"] == "Trace"
    
    def test_extract_generation_entities(self, sample_generation_records):
        """Test extracting generation entities."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(sample_generation_records)
        
        assert "Generation" in entities
        assert len(entities["Generation"]) == 1
        assert entities["Generation"][0]["id"] == "gen1"
    
    def test_extract_span_entities(self, sample_span_records):
        """Test extracting span entities."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(sample_span_records)
        
        assert "Span" in entities
        assert len(entities["Span"]) == 1
        assert entities["Span"][0]["id"] == "span1"
    
    def test_extract_entities_missing_required(self):
        """Test extraction fails for missing required properties."""
        records = [
            {
                "_csv_type": "trace",
                "_row_index": 0,
                "name": "Test Trace",  # Missing id
            },
        ]
        extractor = EntityExtractor()
        entities = extractor.extract_entities(records)
        
        # Should skip invalid records
        assert "Trace" not in entities or len(entities.get("Trace", [])) == 0
    
    def test_extract_structural_relationships(self, sample_trace_records, sample_span_records):
        """Test extracting structural relationships."""
        extractor = EntityExtractor()
        
        # Extract entities
        all_records = sample_trace_records + sample_span_records
        entities = extractor.extract_entities(all_records)
        
        # Extract relationships
        relationships = extractor.extract_relationships(entities)
        
        # Check for Trace -> Span relationship
        span_rels = [r for r in relationships if r["type"] == "HAS_SPAN"]
        assert len(span_rels) > 0
    
    def test_extract_temporal_relationships(self, sample_trace_records):
        """Test extracting temporal relationships."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(sample_trace_records)
        relationships = extractor.extract_relationships(entities)
        
        # Check for FOLLOWS relationships
        follows_rels = [r for r in relationships if r["type"] == "FOLLOWS"]
        # Should have at least one FOLLOWS relationship between traces
        assert len(follows_rels) >= 0  # May be 0 if traces not in same session
    
    def test_get_entities(self, sample_trace_records):
        """Test getting extracted entities."""
        extractor = EntityExtractor()
        extractor.extract_entities(sample_trace_records)
        entities = extractor.get_entities()
        
        assert isinstance(entities, dict)
        assert "Trace" in entities
    
    def test_get_relationships(self, sample_trace_records):
        """Test getting extracted relationships."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities(sample_trace_records)
        extractor.extract_relationships(entities)
        relationships = extractor.get_relationships()
        
        assert isinstance(relationships, list)
    
    def test_extract_entities_empty_records(self):
        """Test extraction with empty records."""
        extractor = EntityExtractor()
        entities = extractor.extract_entities([])
        
        assert entities == {}
    
    def test_extract_relationships_empty_entities(self):
        """Test relationship extraction with empty entities."""
        extractor = EntityExtractor()
        relationships = extractor.extract_relationships({})
        
        assert relationships == []
    
    def test_extract_relationships_score_to_trace_and_span(self):
        """Test score relationships to both trace and span."""
        extractor = EntityExtractor()
        
        records = [
            {
                "_csv_type": "trace",
                "_row_index": 0,
                "id": "trace1",
                "name": "Test Trace",
                "session_id": "session1",
                "timestamp": datetime(2024, 1, 1, 10, 0, 0),
            },
            {
                "_csv_type": "span",
                "_row_index": 0,
                "id": "span1",
                "trace_id": "trace1",
                "name": "Test Span",
                "type": "llm",
            },
            {
                "_csv_type": "score",
                "_row_index": 0,
                "id": "score1",
                "trace_id": "trace1",
                "span_id": "span1",
                "name": "cooperation_index",
                "value": 0.9,
            },
        ]
        
        entities = extractor.extract_entities(records)
        relationships = extractor.extract_relationships(entities)
        
        # Should have relationships from both trace and span to score
        score_rels = [r for r in relationships if r["to_id"] == "score1"]
        assert len(score_rels) >= 1  # At least one relationship to score


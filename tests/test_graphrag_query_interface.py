"""Unit tests for langfuse_graphrag.query_interface module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langfuse_graphrag.query_interface import QueryInterface


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "default_query_limit": 10,
        "vector_similarity_threshold": 0.7,
        "openai_api_key": "test-key",
        "embedding_model": "text-embedding-3-small",
    }


@pytest.fixture
def mock_neo4j_manager():
    """Mock Neo4j manager."""
    manager = MagicMock()
    manager.execute_query.return_value = []
    return manager


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    generator = MagicMock()
    generator.generate_embedding.return_value = [0.1] * 1536
    return generator


class TestQueryInterface:
    """Test query interface functionality."""
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_interface_initialization(self, mock_emb, mock_neo4j, mock_config):
        """Test query interface initialization."""
        interface = QueryInterface(mock_config)
        assert interface.config == mock_config
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_semantic_search(self, mock_emb, mock_neo4j, mock_config, mock_embedding_generator, mock_neo4j_manager):
        """Test semantic search."""
        mock_emb.return_value = mock_embedding_generator
        mock_neo4j.return_value = mock_neo4j_manager
        
        # Mock query results
        mock_neo4j_manager.execute_query.return_value = [
            {"n": {"id": "gen1", "prompt": "test"}, "similarity": 0.85},
        ]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager, mock_embedding_generator)
        results = interface.semantic_search("test query")
        
        assert isinstance(results, list)
        mock_embedding_generator.generate_embedding.assert_called_once()
        mock_neo4j_manager.execute_query.assert_called_once()
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_pattern_analysis(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test pattern analysis."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_neo4j_manager.execute_query.return_value = [
            {"session_id": "session1", "trace_count": 5},
        ]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.pattern_analysis("session_traces")
        
        assert isinstance(results, list)
        mock_neo4j_manager.execute_query.assert_called_once()
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_error_analysis(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test error analysis."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_neo4j_manager.execute_query.return_value = [
            {"error_id": "err1", "error_type": "ValueError", "error_message": "test"},
        ]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.error_analysis()
        
        assert isinstance(results, list)
        mock_neo4j_manager.execute_query.assert_called_once()
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_performance_analysis(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test performance analysis."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_neo4j_manager.execute_query.return_value = [
            {"model": "gpt-4", "total_cost": 10.5},
        ]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.performance_analysis("cost", group_by="model")
        
        assert isinstance(results, list)
        mock_neo4j_manager.execute_query.assert_called_once()
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_execute_custom_query(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test executing custom query."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_neo4j_manager.execute_query.return_value = [{"result": "value"}]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.execute_custom_query("MATCH (n) RETURN n", {})
        
        assert isinstance(results, list)
        mock_neo4j_manager.execute_query.assert_called_once()
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_semantic_search_empty_embedding(self, mock_emb, mock_neo4j, mock_config, mock_embedding_generator, mock_neo4j_manager):
        """Test semantic search when embedding generation fails."""
        mock_emb.return_value = mock_embedding_generator
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_embedding_generator.generate_embedding.return_value = []
        
        interface = QueryInterface(mock_config, mock_neo4j_manager, mock_embedding_generator)
        results = interface.semantic_search("test query")
        
        assert results == []
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_pattern_analysis_unknown_type(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test pattern analysis with unknown pattern type."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.pattern_analysis("unknown_pattern_type")
        
        assert results == []
    
    @patch('langfuse_graphrag.query_interface.Neo4jManager')
    @patch('langfuse_graphrag.query_interface.EmbeddingGenerator')
    def test_error_analysis_with_filters(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager):
        """Test error analysis with filters."""
        mock_emb.return_value = MagicMock()
        mock_neo4j.return_value = mock_neo4j_manager
        
        mock_neo4j_manager.execute_query.return_value = [
            {"error_id": "err1", "error_type": "ValueError", "trace_id": "trace1"}
        ]
        
        interface = QueryInterface(mock_config, mock_neo4j_manager)
        results = interface.error_analysis(error_type="ValueError", trace_id="trace1")
        
        assert isinstance(results, list)
        mock_neo4j_manager.execute_query.assert_called_once()


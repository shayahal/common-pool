"""Unit tests for langfuse_graphrag.neo4j_manager module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from neo4j.exceptions import ServiceUnavailable
from langfuse_graphrag.neo4j_manager import Neo4jManager


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "neo4j_database": "neo4j",
        "neo4j_batch_size": 1000,
        "embedding_dimension": 1536,
    }


@pytest.fixture
def mock_driver():
    """Mock Neo4j driver."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None
    session.run.return_value.consume.return_value = None
    session.run.return_value = MagicMock()
    return driver


class TestNeo4jManager:
    """Test Neo4j manager functionality."""
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_manager_initialization(self, mock_graph_db, mock_config, mock_driver):
        """Test manager initialization."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        assert manager.config == mock_config
        assert manager.driver is not None
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_connection_failure(self, mock_graph_db, mock_config):
        """Test connection failure handling."""
        mock_graph_db.driver.side_effect = ServiceUnavailable("Connection failed")
        
        with pytest.raises(ServiceUnavailable):
            Neo4jManager(mock_config)
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_create_schema(self, mock_graph_db, mock_config, mock_driver):
        """Test schema creation."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        manager.create_schema()
        
        # Verify session.run was called (for constraints and indexes)
        assert mock_driver.session.return_value.__enter__.return_value.run.called
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_create_nodes(self, mock_graph_db, mock_config, mock_driver):
        """Test node creation."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        entities = {
            "Trace": [
                {"id": "trace1", "name": "Test Trace", "_type": "Trace"},
            ],
        }
        
        manager.create_nodes(entities)
        
        # Verify session.run was called
        assert mock_driver.session.return_value.__enter__.return_value.run.called
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_create_relationships(self, mock_graph_db, mock_config, mock_driver):
        """Test relationship creation."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        relationships = [
            {
                "type": "CONTAINS",
                "from_type": "Session",
                "from_id": "session1",
                "to_type": "Trace",
                "to_id": "trace1",
            },
        ]
        
        manager.create_relationships(relationships)
        
        # Verify session.run was called
        assert mock_driver.session.return_value.__enter__.return_value.run.called
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_update_node_embedding(self, mock_graph_db, mock_config, mock_driver):
        """Test updating node embedding."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        manager.update_node_embedding("Generation", "gen1", "prompt_embedding", embedding)
        
        # Verify session.run was called
        assert mock_driver.session.return_value.__enter__.return_value.run.called
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_execute_query(self, mock_graph_db, mock_config, mock_driver):
        """Test query execution."""
        mock_graph_db.driver.return_value = mock_driver
        
        # Mock query result
        mock_record = MagicMock()
        mock_record.__iter__ = Mock(return_value=iter([{"key": "value"}]))
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_driver.session.return_value.__enter__.return_value.run.return_value = mock_result
        
        manager = Neo4jManager(mock_config)
        results = manager.execute_query("MATCH (n) RETURN n", {})
        
        assert isinstance(results, list)
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_get_stats(self, mock_graph_db, mock_config, mock_driver):
        """Test getting database statistics."""
        mock_graph_db.driver.return_value = mock_driver
        
        # Mock query results - use dict directly instead of MagicMock
        mock_node_record = {"label": "Trace", "count": 10}
        mock_rel_record = {"type": "CONTAINS", "count": 5}
        
        mock_node_result = MagicMock()
        mock_node_result.__iter__ = Mock(return_value=iter([mock_node_record]))
        mock_rel_result = MagicMock()
        mock_rel_result.__iter__ = Mock(return_value=iter([mock_rel_record]))
        
        mock_session = mock_driver.session.return_value.__enter__.return_value
        # Use a list for side_effect so it can be called multiple times
        def run_side_effect(query, **kwargs):
            if "MATCH (n)" in query and "labels(n)" in query:
                return mock_node_result
            elif "MATCH ()-[r]->()" in query:
                return mock_rel_result
            return MagicMock()
        
        mock_session.run.side_effect = run_side_effect
        
        manager = Neo4jManager(mock_config)
        stats = manager.get_stats()
        
        assert "nodes" in stats
        assert "relationships" in stats
        assert "total_nodes" in stats
        assert "total_relationships" in stats
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_close(self, mock_graph_db, mock_config, mock_driver):
        """Test closing connection."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        manager.close()
        
        assert mock_driver.close.called
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_create_nodes_empty(self, mock_graph_db, mock_config, mock_driver):
        """Test creating nodes with empty entities."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        manager.create_nodes({})
        
        # Should not crash with empty entities
        assert True
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_create_relationships_empty(self, mock_graph_db, mock_config, mock_driver):
        """Test creating relationships with empty list."""
        mock_graph_db.driver.return_value = mock_driver
        
        manager = Neo4jManager(mock_config)
        manager.create_relationships([])
        
        # Should not crash with empty relationships
        assert True
    
    @patch('langfuse_graphrag.neo4j_manager.GraphDatabase')
    def test_execute_query_with_parameters(self, mock_graph_db, mock_config, mock_driver):
        """Test executing query with parameters."""
        mock_graph_db.driver.return_value = mock_driver
        
        mock_record = MagicMock()
        mock_record.__iter__ = Mock(return_value=iter([{"key": "value"}]))
        mock_result = MagicMock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_driver.session.return_value.__enter__.return_value.run.return_value = mock_result
        
        manager = Neo4jManager(mock_config)
        results = manager.execute_query("MATCH (n {id: $id}) RETURN n", {"id": "test"})
        
        assert isinstance(results, list)
        # Verify parameters were passed
        call_args = mock_driver.session.return_value.__enter__.return_value.run.call_args
        assert "id" in call_args[1] or "test" in str(call_args)


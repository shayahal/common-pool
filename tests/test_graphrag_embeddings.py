"""Unit tests for langfuse_graphrag.embeddings module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from langfuse_graphrag.embeddings import EmbeddingGenerator


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "openai_api_key": "test-key",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
        "embedding_batch_size": 100,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = mock_embedding
    return client


@pytest.fixture
def mock_neo4j_manager():
    """Mock Neo4j manager."""
    return MagicMock()


class TestEmbeddingGenerator:
    """Test embedding generator functionality."""
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generator_initialization(self, mock_neo4j, mock_openai, mock_config):
        """Test generator initialization."""
        mock_openai.return_value = MagicMock()
        
        generator = EmbeddingGenerator(mock_config)
        assert generator.config == mock_config
        assert generator.model == "text-embedding-3-small"
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generator_missing_api_key(self, mock_neo4j, mock_openai):
        """Test generator fails without API key."""
        config = {
            "openai_api_key": "",
            "embedding_model": "text-embedding-3-small",
        }
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            EmbeddingGenerator(config)
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generate_embedding(self, mock_neo4j, mock_openai, mock_config, mock_openai_client):
        """Test generating single embedding."""
        mock_openai.return_value = mock_openai_client
        
        generator = EmbeddingGenerator(mock_config)
        embedding = generator.generate_embedding("test text")
        
        assert isinstance(embedding, list)
        assert len(embedding) == 1536
        mock_openai_client.embeddings.create.assert_called_once()
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generate_embedding_empty_text(self, mock_neo4j, mock_openai, mock_config):
        """Test generating embedding for empty text."""
        mock_openai.return_value = MagicMock()
        
        generator = EmbeddingGenerator(mock_config)
        embedding = generator.generate_embedding("")
        
        assert embedding == []
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generate_embeddings_batch(self, mock_neo4j, mock_openai, mock_config, mock_openai_client):
        """Test generating batch embeddings."""
        # Mock batch response
        mock_batch_embedding = MagicMock()
        mock_batch_embedding.data = [
            MagicMock(embedding=[0.1] * 1536),
            MagicMock(embedding=[0.2] * 1536),
        ]
        mock_openai_client.embeddings.create.return_value = mock_batch_embedding
        mock_openai.return_value = mock_openai_client
        
        generator = EmbeddingGenerator(mock_config)
        texts = ["text1", "text2"]
        embeddings = generator.generate_embeddings_batch(texts)
        
        assert len(embeddings) == 2
        assert all(len(e) == 1536 for e in embeddings if e)
    
    @patch('langfuse_graphrag.embeddings.OpenAI')
    @patch('langfuse_graphrag.embeddings.Neo4jManager')
    def test_generate_and_store_generation_embeddings(
        self, mock_neo4j, mock_openai, mock_config, mock_openai_client, mock_neo4j_manager
    ):
        """Test generating and storing generation embeddings."""
        mock_openai.return_value = mock_openai_client
        mock_neo4j.return_value = mock_neo4j_manager
        
        generator = EmbeddingGenerator(mock_config, mock_neo4j_manager)
        entities = {
            "Generation": [
                {
                    "id": "gen1",
                    "prompt": "test prompt",
                    "response": "test response",
                    "reasoning": "test reasoning",
                },
            ],
        }
        
        generator.generate_and_store_generation_embeddings(entities)
        
        # Verify embeddings were generated and stored
        assert mock_openai_client.embeddings.create.called
        assert mock_neo4j_manager.update_node_embedding.called


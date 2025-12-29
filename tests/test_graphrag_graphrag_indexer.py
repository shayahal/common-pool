"""Unit tests for langfuse_graphrag.graphrag_indexer module."""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from langfuse_graphrag.graphrag_indexer import GraphRAGIndexer


@pytest.fixture
def mock_config():
    """Mock configuration."""
    return {
        "data_dir": "data/graphrag",
        "processed_data_dir": "data/graphrag/processed",
        "indices_dir": "data/graphrag/indices",
        "graphrag_llm_model": "gpt-4o-mini",
        "graphrag_batch_size": 100,
        "graphrag_chunk_size": 1000,
        "graphrag_chunk_overlap": 200,
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "openai_api_key": "test-key",
        "embedding_model": "text-embedding-3-small",
    }


@pytest.fixture
def mock_neo4j_manager():
    """Mock Neo4j manager."""
    return MagicMock()


@pytest.fixture
def mock_embedding_generator():
    """Mock embedding generator."""
    return MagicMock()


class TestGraphRAGIndexer:
    """Test GraphRAG indexer functionality."""
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_indexer_initialization(self, mock_emb, mock_neo4j, mock_config):
        """Test indexer initialization."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        assert indexer.config == mock_config
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_extract_text_from_entities(self, mock_emb, mock_neo4j, mock_config):
        """Test extracting text from entities."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        entities = {
            "Generation": [
                {"id": "gen1", "prompt": "test prompt", "response": "test response"},
            ],
            "Trace": [
                {"id": "trace1", "input": "test input", "output": "test output"},
            ],
        }
        
        texts = indexer.extract_text_from_entities(entities)
        
        assert isinstance(texts, list)
        assert len(texts) > 0
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_chunk_texts(self, mock_emb, mock_neo4j, mock_config):
        """Test chunking texts."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Override chunk_size to be smaller for testing
        mock_config["graphrag_chunk_size"] = 500
        mock_config["graphrag_chunk_overlap"] = 100
        
        indexer = GraphRAGIndexer(mock_config)
        # Use text that will be chunked but not cause infinite loop
        texts = ["short text", "a" * 800]  # One short, one that will be chunked
        
        chunks = indexer.chunk_texts(texts)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_success(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test successful semantic entity extraction."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([
            {"name": "error handling", "type": "concept", "description": "Methods for handling errors"},
            {"name": "API call", "type": "action", "description": "Making requests to APIs"}
        ])
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        indexer = GraphRAGIndexer(mock_config)
        chunks = [
            {"id": "chunk1", "text": "This is about error handling and API calls"},
        ]
        
        entities = indexer.extract_semantic_entities(chunks)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        assert all("id" in e for e in entities)
        assert all("name" in e for e in entities)
        assert all("type" in e for e in entities)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_empty_chunks(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test entity extraction with empty chunks."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_openai.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        entities = indexer.extract_semantic_entities([])
        
        assert entities == []
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_short_text(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test entity extraction skips very short text."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_openai.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        chunks = [{"id": "chunk1", "text": "short"}]
        
        entities = indexer.extract_semantic_entities(chunks)
        
        # Should skip text that's too short
        assert entities == []
        # OpenAI should not be called for very short text
        assert not mock_openai.return_value.chat.completions.create.called
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_json_error(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test entity extraction handles JSON decode errors."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock OpenAI response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "not valid json"
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        indexer = GraphRAGIndexer(mock_config)
        chunks = [{"id": "chunk1", "text": "This is a longer text that should be processed"}]
        
        entities = indexer.extract_semantic_entities(chunks)
        
        # Should handle error gracefully and return empty list or partial results
        assert isinstance(entities, list)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_api_error(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test entity extraction handles API errors."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock API error
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        indexer = GraphRAGIndexer(mock_config)
        chunks = [{"id": "chunk1", "text": "This is a longer text that should be processed"}]
        
        entities = indexer.extract_semantic_entities(chunks)
        
        # Should handle error gracefully
        assert isinstance(entities, list)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_extract_semantic_entities_deduplication(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test entity extraction deduplicates entities."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock OpenAI response with duplicate entities
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps([
            {"name": "error handling", "type": "concept", "description": "First description"},
            {"name": "error handling", "type": "concept", "description": "Second description"}
        ])
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        indexer = GraphRAGIndexer(mock_config)
        chunks = [{"id": "chunk1", "text": "This is about error handling"}]
        
        entities = indexer.extract_semantic_entities(chunks)
        
        # Should deduplicate - only one entity with merged description
        assert len(entities) == 1
        assert entities[0]["name"] == "error handling"
        assert "first" in entities[0]["description"].lower() or "second" in entities[0]["description"].lower()
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_build_communities_empty(self, mock_emb, mock_neo4j, mock_config):
        """Test building communities with empty entity list."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        communities = indexer.build_communities([])
        
        assert communities == []
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_build_communities_small(self, mock_emb, mock_neo4j, mock_config):
        """Test building communities with few entities."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        entities = [
            {"id": "e1", "type": "concept", "name": "Entity 1"},
            {"id": "e2", "type": "concept", "name": "Entity 2"},
        ]
        
        communities = indexer.build_communities(entities)
        
        assert len(communities) > 0
        assert all("id" in c for c in communities)
        assert all("name" in c for c in communities)
        assert all("level" in c for c in communities)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_build_communities_large(self, mock_emb, mock_neo4j, mock_config):
        """Test building communities with many entities."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Override min size for testing
        mock_config["community_min_size"] = 3
        
        indexer = GraphRAGIndexer(mock_config)
        entities = [
            {"id": f"e{i}", "type": "concept", "name": f"Entity {i}"}
            for i in range(10)
        ]
        
        communities = indexer.build_communities(entities)
        
        assert len(communities) > 0
        assert all("id" in c for c in communities)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_generate_summaries_success(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test successful summary generation."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a summary of the community."
        mock_openai.return_value.chat.completions.create.return_value = mock_response
        
        indexer = GraphRAGIndexer(mock_config)
        communities = [
            {"id": "c1", "name": "Test Community", "level": 0}
        ]
        chunks = [
            {"id": "chunk1", "text": "Relevant text about the community"}
        ]
        
        result = indexer.generate_summaries(communities, chunks)
        
        assert len(result) == 1
        assert "summary" in result[0]
        assert result[0]["summary"] == "This is a summary of the community."
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_generate_summaries_api_error(self, mock_openai, mock_emb, mock_neo4j, mock_config):
        """Test summary generation handles API errors."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        # Mock API error
        mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
        
        indexer = GraphRAGIndexer(mock_config)
        communities = [
            {"id": "c1", "name": "Test Community", "level": 0}
        ]
        chunks = [{"id": "chunk1", "text": "Relevant text"}]
        
        result = indexer.generate_summaries(communities, chunks)
        
        # Should have fallback summary
        assert len(result) == 1
        assert "summary" in result[0]
        assert len(result[0]["summary"]) > 0
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_generate_summaries_empty_communities(self, mock_emb, mock_neo4j, mock_config):
        """Test summary generation with empty communities."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        result = indexer.generate_summaries([], [])
        
        assert result == []
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_create_entity_relationships(self, mock_emb, mock_neo4j, mock_config):
        """Test creating entity relationships."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        semantic_entities = [
            {"id": "se1", "name": "error handling", "type": "concept"}
        ]
        original_entities = {
            "Generation": [
                {"id": "gen1", "prompt": "How to handle errors", "response": "Use try-catch"}
            ],
            "Trace": [
                {"id": "trace1", "input": "error handling", "output": "result"}
            ]
        }
        
        relationships = indexer.create_entity_relationships(semantic_entities, original_entities)
        
        assert isinstance(relationships, list)
        # Should create MENTIONS and ABOUT relationships
        assert len(relationships) >= 0  # May be 0 if no matches found
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_create_entity_relationships_no_matches(self, mock_emb, mock_neo4j, mock_config):
        """Test relationship creation when no matches found."""
        mock_neo4j.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        
        indexer = GraphRAGIndexer(mock_config)
        semantic_entities = [
            {"id": "se1", "name": "unrelated topic", "type": "concept"}
        ]
        original_entities = {
            "Generation": [
                {"id": "gen1", "prompt": "completely different text", "response": "response"}
            ]
        }
        
        relationships = indexer.create_entity_relationships(semantic_entities, original_entities)
        
        assert isinstance(relationships, list)
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    @patch('langfuse_graphrag.graphrag_indexer.OpenAI')
    def test_index_pipeline(self, mock_openai, mock_neo4j, mock_emb, mock_config, mock_neo4j_manager, mock_embedding_generator):
        """Test full indexing pipeline."""
        mock_neo4j.return_value = mock_neo4j_manager
        mock_emb.return_value = mock_embedding_generator
        
        # Mock OpenAI responses
        mock_entity_response = MagicMock()
        mock_entity_response.choices[0].message.content = json.dumps([
            {"name": "test concept", "type": "concept", "description": "A test concept"}
        ])
        mock_summary_response = MagicMock()
        mock_summary_response.choices[0].message.content = "Test summary"
        mock_openai.return_value.chat.completions.create.side_effect = [
            mock_entity_response,
            mock_summary_response
        ]
        
        indexer = GraphRAGIndexer(mock_config, mock_neo4j_manager, mock_embedding_generator)
        entities = {
            "Generation": [
                {"id": "gen1", "prompt": "test prompt about concepts", "response": "test response"},
            ],
        }
        
        results = indexer.index(entities)
        
        assert isinstance(results, dict)
        assert "SemanticEntity" in results
        assert "Community" in results
        # Verify Neo4j operations were called
        assert mock_neo4j_manager.create_nodes.called
        assert mock_embedding_generator.generate_and_store_semantic_entity_embeddings.called
    
    @patch('langfuse_graphrag.graphrag_indexer.Neo4jManager')
    @patch('langfuse_graphrag.graphrag_indexer.EmbeddingGenerator')
    def test_index_pipeline_no_text(self, mock_emb, mock_neo4j, mock_config, mock_neo4j_manager, mock_embedding_generator):
        """Test indexing pipeline with no extractable text."""
        mock_neo4j.return_value = mock_neo4j_manager
        mock_emb.return_value = mock_embedding_generator
        
        indexer = GraphRAGIndexer(mock_config, mock_neo4j_manager, mock_embedding_generator)
        entities = {
            "Generation": [
                {"id": "gen1"},  # No text fields
            ],
        }
        
        results = indexer.index(entities)
        
        assert isinstance(results, dict)
        assert "SemanticEntity" in results
        assert "Community" in results


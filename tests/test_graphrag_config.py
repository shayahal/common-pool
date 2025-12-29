"""Unit tests for langfuse_graphrag.config module."""

import pytest
import os
from unittest.mock import patch
from langfuse_graphrag.config import (
    get_config,
    validate_config,
    CONFIG,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
)


class TestConfig:
    """Test configuration loading and validation."""
    
    def test_get_config_returns_dict(self):
        """Test that get_config returns a dictionary."""
        config = get_config()
        assert isinstance(config, dict)
        assert "neo4j_uri" in config
        assert "neo4j_user" in config
        assert "neo4j_password" in config
    
    def test_config_has_required_keys(self):
        """Test that config has all required keys."""
        config = get_config()
        required_keys = [
            "neo4j_uri",
            "neo4j_user",
            "neo4j_password",
            "embedding_model",
            "embedding_dimension",
            "openai_api_key",
        ]
        for key in required_keys:
            assert key in config
    
    def test_validate_config_success(self):
        """Test successful config validation."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "openai_api_key": "test-key",
            "embedding_dimension": 1536,
            "neo4j_batch_size": 1000,
            "embedding_batch_size": 100,
        }
        assert validate_config(config) is True
    
    def test_validate_config_missing_uri(self):
        """Test validation fails with missing URI."""
        config = {
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "openai_api_key": "test-key",
        }
        with pytest.raises(ValueError, match="NEO4J_URI"):
            validate_config(config)
    
    def test_validate_config_missing_user(self):
        """Test validation fails with missing user."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_password": "password",
            "openai_api_key": "test-key",
        }
        with pytest.raises(ValueError, match="NEO4J_USER"):
            validate_config(config)
    
    def test_validate_config_missing_password(self):
        """Test validation fails with missing password."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "openai_api_key": "test-key",
        }
        with pytest.raises(ValueError, match="NEO4J_PASSWORD"):
            validate_config(config)
    
    def test_validate_config_missing_openai_key(self):
        """Test validation fails with missing OpenAI key."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
        }
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            validate_config(config)
    
    def test_validate_config_invalid_embedding_dimension(self):
        """Test validation fails with invalid embedding dimension."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "openai_api_key": "test-key",
            "embedding_dimension": 0,
        }
        with pytest.raises(ValueError, match="embedding_dimension"):
            validate_config(config)
    
    def test_validate_config_invalid_batch_size(self):
        """Test validation fails with invalid batch size."""
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "openai_api_key": "test-key",
            "embedding_dimension": 1536,
            "neo4j_batch_size": 0,
        }
        with pytest.raises(ValueError, match="neo4j_batch_size"):
            validate_config(config)


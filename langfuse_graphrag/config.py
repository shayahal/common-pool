"""Configuration for Langfuse GraphRAG System.

Settings for Neo4j connection, GraphRAG processing, and embedding generation.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# Neo4j Configuration
# ============================================================================

NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")

# ============================================================================
# GraphRAG Configuration
# ============================================================================

# LLM Model for GraphRAG entity extraction and summarization
GRAPHRAG_LLM_MODEL: str = os.getenv("GRAPHRAG_LLM_MODEL", "gpt-4o-mini")
GRAPHRAG_LLM_TEMPERATURE: float = float(os.getenv("GRAPHRAG_LLM_TEMPERATURE", "0.0"))
GRAPHRAG_LLM_MAX_TOKENS: int = int(os.getenv("GRAPHRAG_LLM_MAX_TOKENS", "4000"))

# Embedding Model
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))  # text-embedding-3-small default

# OpenAI API Key (for embeddings and GraphRAG LLM)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# GraphRAG Processing Settings
GRAPHRAG_BATCH_SIZE: int = int(os.getenv("GRAPHRAG_BATCH_SIZE", "100"))
GRAPHRAG_CHUNK_SIZE: int = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "1000"))
GRAPHRAG_CHUNK_OVERLAP: int = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "200"))

# Community Detection Settings
COMMUNITY_MIN_SIZE: int = int(os.getenv("COMMUNITY_MIN_SIZE", "3"))
COMMUNITY_MAX_LEVELS: int = int(os.getenv("COMMUNITY_MAX_LEVELS", "3"))

# ============================================================================
# Processing Configuration
# ============================================================================

# Batch sizes for Neo4j operations
NEO4J_BATCH_SIZE: int = int(os.getenv("NEO4J_BATCH_SIZE", "1000"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

# Data directories
DATA_DIR: str = os.getenv("GRAPHRAG_DATA_DIR", "data/graphrag")
RAW_DATA_DIR: str = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR: str = os.path.join(DATA_DIR, "processed")
INDICES_DIR: str = os.path.join(DATA_DIR, "indices")

# ============================================================================
# Vector Index Configuration
# ============================================================================

# Vector index settings for Neo4j
VECTOR_INDEX_NAME: str = "generation_embeddings"
VECTOR_INDEX_DIMENSION: int = EMBEDDING_DIMENSION
VECTOR_SIMILARITY_THRESHOLD: float = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))

# ============================================================================
# Query Configuration
# ============================================================================

# Default query limits
DEFAULT_QUERY_LIMIT: int = int(os.getenv("DEFAULT_QUERY_LIMIT", "10"))
MAX_QUERY_LIMIT: int = int(os.getenv("MAX_QUERY_LIMIT", "100"))

# ============================================================================
# Configuration Dictionary
# ============================================================================

CONFIG: Dict = {
    # Neo4j
    "neo4j_uri": NEO4J_URI,
    "neo4j_user": NEO4J_USER,
    "neo4j_password": NEO4J_PASSWORD,
    "neo4j_database": NEO4J_DATABASE,
    
    # GraphRAG
    "graphrag_llm_model": GRAPHRAG_LLM_MODEL,
    "graphrag_llm_temperature": GRAPHRAG_LLM_TEMPERATURE,
    "graphrag_llm_max_tokens": GRAPHRAG_LLM_MAX_TOKENS,
    
    # Embeddings
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimension": EMBEDDING_DIMENSION,
    "openai_api_key": OPENAI_API_KEY,
    
    # Processing
    "graphrag_batch_size": GRAPHRAG_BATCH_SIZE,
    "graphrag_chunk_size": GRAPHRAG_CHUNK_SIZE,
    "graphrag_chunk_overlap": GRAPHRAG_CHUNK_OVERLAP,
    "community_min_size": COMMUNITY_MIN_SIZE,
    "community_max_levels": COMMUNITY_MAX_LEVELS,
    "neo4j_batch_size": NEO4J_BATCH_SIZE,
    "embedding_batch_size": EMBEDDING_BATCH_SIZE,
    
    # Directories
    "data_dir": DATA_DIR,
    "raw_data_dir": RAW_DATA_DIR,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "indices_dir": INDICES_DIR,
    
    # Vector Index
    "vector_index_name": VECTOR_INDEX_NAME,
    "vector_index_dimension": VECTOR_INDEX_DIMENSION,
    "vector_similarity_threshold": VECTOR_SIMILARITY_THRESHOLD,
    
    # Query
    "default_query_limit": DEFAULT_QUERY_LIMIT,
    "max_query_limit": MAX_QUERY_LIMIT,
}


def get_config() -> Dict:
    """Return the complete configuration dictionary.
    
    Returns:
        Dict: Configuration parameters for GraphRAG system
    """
    return CONFIG.copy()


def validate_config(config: Optional[Dict] = None) -> bool:
    """Validate configuration parameters.
    
    Args:
        config: Optional configuration dictionary. If None, uses CONFIG.
    
    Returns:
        bool: True if config is valid
    
    Raises:
        ValueError: If configuration has invalid values
    """
    if config is None:
        config = CONFIG
    
    # Validate Neo4j settings
    if not config.get("neo4j_uri"):
        raise ValueError("NEO4J_URI is required")
    
    if not config.get("neo4j_user"):
        raise ValueError("NEO4J_USER is required")
    
    if not config.get("neo4j_password"):
        raise ValueError("NEO4J_PASSWORD is required")
    
    # Validate OpenAI API key if using OpenAI for embeddings
    if not config.get("openai_api_key"):
        raise ValueError("OPENAI_API_KEY is required for embeddings and GraphRAG")
    
    # Validate embedding dimension
    if config.get("embedding_dimension", 0) <= 0:
        raise ValueError("embedding_dimension must be positive")
    
    # Validate batch sizes
    if config.get("neo4j_batch_size", 0) <= 0:
        raise ValueError("neo4j_batch_size must be positive")
    
    if config.get("embedding_batch_size", 0) <= 0:
        raise ValueError("embedding_batch_size must be positive")
    
    return True


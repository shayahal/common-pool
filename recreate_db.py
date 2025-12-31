#!/usr/bin/env python3
"""Recreate the database by ingesting the CSV file."""

import sys
import os
from pathlib import Path
from langfuse_graphrag.csv_parser import LangfuseCSVParser
from langfuse_graphrag.vibe_csv_parser import VibeBenchCSVParser
from langfuse_graphrag.extractor import EntityExtractor
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.embeddings import EmbeddingGenerator
from langfuse_graphrag.graphrag_indexer import GraphRAGIndexer
from langfuse_graphrag.config import get_config, validate_config
import logging

# Setup logging with file handlers (same as CLI)
def setup_logging() -> logging.Logger:
    """Configure logging with file handlers and console output."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    
    # Create file handlers for each log level
    log_levels = [
        ('debug.log', logging.DEBUG),
        ('info.log', logging.INFO),
        ('warning.log', logging.WARNING),
        ('error.log', logging.ERROR),
    ]
    
    for filename, level in log_levels:
        handler = logging.FileHandler(logs_dir / filename, encoding='utf-8')
        handler.setLevel(level)
        handler.setFormatter(file_formatter)
        root_logger.addHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = os.environ.get('STDOUT_LOG_LEVEL', 'INFO').upper()
    console_handler.setLevel(getattr(logging, console_level, logging.INFO))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose third-party library logging
    noisy_loggers = [
        'httpx',
        'httpcore',
        'urllib3',
        'requests',
        'openai',
        'neo4j',
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

def main():
    # Use Vibe Bench CSV instead of games CSV
    csv_path = "data/vibe-bench-1.csv"
    
    logger.info(f"Ingesting CSV file: {csv_path}")
    
    try:
        config = get_config()
        validate_config(config)
        
        # Parse CSV using Vibe Bench parser
        parser = VibeBenchCSVParser(csv_path)
        summary = parser.get_summary()
        logger.info(f"CSV summary: {summary}")
        
        records = parser.parse()
        logger.info(f"Parsed {len(records)} records")
        
        # Extract entities and relationships
        extractor = EntityExtractor()
        entities = extractor.extract_entities(records)
        relationships = extractor.extract_relationships(entities)
        
        logger.info(f"Extracted {sum(len(v) for v in entities.values())} entities")
        logger.info(f"Extracted {len(relationships)} relationships")
        
        # Initialize Neo4j
        neo4j_manager = Neo4jManager(config)
        
        # Clear database
        logger.warning("Clearing existing database")
        neo4j_manager.clear_database()
        
        # Create schema
        neo4j_manager.create_schema()
        
        # Create nodes
        neo4j_manager.create_nodes(entities)
        
        # Create relationships
        neo4j_manager.create_relationships(relationships)
        
        # Generate embeddings
        embedding_generator = EmbeddingGenerator(config, neo4j_manager)
        embedding_generator.generate_and_store_generation_embeddings(entities)
        embedding_generator.generate_and_store_error_embeddings(entities)
        
        # Run GraphRAG indexing
        graphrag_indexer = GraphRAGIndexer(config, neo4j_manager, embedding_generator)
        graphrag_results = graphrag_indexer.index(entities)
        
        logger.info(f"GraphRAG extracted {len(graphrag_results.get('SemanticEntity', []))} semantic entities")
        logger.info(f"GraphRAG created {len(graphrag_results.get('Community', []))} communities")
        
        # Get final stats
        node_counts = neo4j_manager.execute_query(
            'MATCH (n) RETURN labels(n)[0] as type, count(*) as cnt ORDER BY cnt DESC'
        )
        logger.info("Final node counts:")
        for row in node_counts:
            logger.info(f"  {row['type']}: {row['cnt']}")
        
        rel_counts = neo4j_manager.execute_query(
            'MATCH ()-[r]->() RETURN type(r) as type, count(*) as cnt ORDER BY cnt DESC'
        )
        logger.info("Final relationship counts:")
        for row in rel_counts:
            logger.info(f"  {row['type']}: {row['cnt']}")
        
        neo4j_manager.close()
        logger.info("Ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        logger.error("FATAL: Ingestion failed. Exiting immediately.")
        sys.exit(1)

if __name__ == "__main__":
    main()

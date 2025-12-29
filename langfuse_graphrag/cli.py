"""CLI interface for Langfuse GraphRAG system.

Provides commands for ingestion, querying, searching, and analysis.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional

from langfuse_graphrag.config import get_config, validate_config
from langfuse_graphrag.csv_parser import LangfuseCSVParser
from langfuse_graphrag.extractor import EntityExtractor
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.embeddings import EmbeddingGenerator
from langfuse_graphrag.graphrag_indexer import GraphRAGIndexer
from langfuse_graphrag.query_interface import QueryInterface


def setup_logging() -> logging.Logger:
    """Configure logging with file handlers and console output.
    
    Creates log files in logs/ directory with different levels:
    - logs/debug.log - DEBUG and above
    - logs/info.log - INFO and above  
    - logs/warning.log - WARNING and above
    - logs/error.log - ERROR and above
    
    Returns:
        Logger instance for the CLI module
    """
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
    # These should be at WARNING level to avoid spamming INFO logs
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


# Setup logging at module load
logger = setup_logging()


def ingest_command(args):
    """Ingest CSV file and build knowledge graph."""
    csv_path = args.csv_file
    clear_db = args.clear
    
    logger.info(f"Ingesting CSV file: {csv_path}")
    
    try:
        config = get_config()
        validate_config(config)
        
        # Parse CSV
        parser = LangfuseCSVParser(csv_path)
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
        
        if clear_db:
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
        stats = neo4j_manager.get_stats()
        logger.info(f"Database stats: {stats}")
        
        logger.info("Ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        logger.error("FATAL: Ingestion failed. Exiting immediately.")
        sys.exit(1)
    finally:
        if 'neo4j_manager' in locals():
            neo4j_manager.close()


def query_command(args):
    """Execute a query."""
    query_type = args.type
    pattern_type = args.pattern_type
    limit = args.limit
    
    logger.info(f"Executing query: {query_type}")
    
    try:
        config = get_config()
        query_interface = QueryInterface(config)
        
        if query_type == "pattern":
            if not pattern_type:
                logger.error("pattern_type is required for pattern queries")
                sys.exit(1)
            results = query_interface.pattern_analysis(pattern_type, limit=limit)
        elif query_type == "error":
            results = query_interface.error_analysis(limit=limit)
        elif query_type == "performance":
            metric = args.metric or "cost"
            group_by = args.group_by
            results = query_interface.performance_analysis(metric, group_by, limit=limit)
        else:
            logger.error(f"Unknown query type: {query_type}")
            sys.exit(1)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results written to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        sys.exit(1)


def explore_command(args):
    """Run exploratory queries to understand the graph structure."""
    query_type = args.query_type or "overview"
    
    logger.info(f"Running exploratory query: {query_type}")
    
    try:
        config = get_config()
        neo4j_manager = Neo4jManager(config)
        
        queries = {
            "overview": """
                MATCH (n)
                RETURN labels(n)[0] AS node_type, count(n) AS count
                ORDER BY count DESC
            """,
            "relationships": """
                MATCH ()-[r]->()
                RETURN type(r) AS relationship_type, count(r) AS count
                ORDER BY count DESC
            """,
            "structure": """
                MATCH (n)
                WITH labels(n)[0] AS label, keys(n) AS props
                RETURN DISTINCT label, collect(DISTINCT props)[0..5] AS sample_properties
                ORDER BY label
            """,
            "sessions": """
                MATCH (s:Session)
                RETURN s.id, s.name, s.user_id, s.created_at
                LIMIT 20
            """,
            "isolated": """
                MATCH (n)
                WHERE NOT (n)--()
                RETURN labels(n)[0] AS node_type, count(*) AS isolated_count
            """,
            "stats": """
                MATCH (n)
                RETURN 
                  count(DISTINCT labels(n)) AS unique_node_types,
                  count(n) AS total_nodes,
                  count{(n)-[]->()} AS total_outgoing_relationships,
                  count{()-[r]->()} AS total_relationships
            """,
        }
        
        if query_type not in queries:
            logger.error(f"Unknown query type: {query_type}. Available: {', '.join(queries.keys())}")
            sys.exit(1)
        
        results = neo4j_manager.execute_query(queries[query_type])
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results written to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))
        
        neo4j_manager.close()
        
    except Exception as e:
        logger.error(f"Error running exploratory query: {e}", exc_info=True)
        sys.exit(1)


def search_command(args):
    """Perform semantic search."""
    query_text = args.query
    entity_type = args.entity_type or "Generation"
    property_name = args.property or "prompt_embedding"
    limit = args.limit
    threshold = args.threshold
    
    logger.info(f"Semantic search: '{query_text}'")
    
    try:
        config = get_config()
        query_interface = QueryInterface(config)
        
        results = query_interface.semantic_search(
            query_text,
            entity_type=entity_type,
            property_name=property_name,
            limit=limit,
            threshold=threshold
        )
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results written to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}", exc_info=True)
        sys.exit(1)


def analyze_command(args):
    """Run analysis queries."""
    analysis_type = args.type
    
    logger.info(f"Running analysis: {analysis_type}")
    
    try:
        config = get_config()
        query_interface = QueryInterface(config)
        
        if analysis_type == "errors":
            results = query_interface.error_analysis(limit=args.limit)
        elif analysis_type == "performance":
            results = query_interface.performance_analysis(
                metric=args.metric or "cost",
                group_by=args.group_by,
                limit=args.limit
            )
        elif analysis_type == "patterns":
            results = query_interface.pattern_analysis(
                pattern_type=args.pattern_type or "session_traces",
                limit=args.limit
            )
        else:
            logger.error(f"Unknown analysis type: {analysis_type}")
            sys.exit(1)
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results written to {output_path}")
        else:
            print(json.dumps(results, indent=2, default=str))
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}", exc_info=True)
        sys.exit(1)


def stats_command(args):
    """Get database statistics."""
    logger.info("Getting database statistics")
    
    try:
        config = get_config()
        neo4j_manager = Neo4jManager(config)
        
        stats = neo4j_manager.get_stats()
        
        # Output stats
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Stats written to {output_path}")
        else:
            print(json.dumps(stats, indent=2, default=str))
        
        neo4j_manager.close()
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Langfuse GraphRAG System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CSV file and build knowledge graph")
    ingest_parser.add_argument("csv_file", help="Path to Langfuse CSV export file")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear database before ingestion")
    ingest_parser.set_defaults(func=ingest_command)
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Execute a query")
    query_parser.add_argument("type", choices=["pattern", "error", "performance"], help="Query type")
    query_parser.add_argument("--pattern-type", help="Pattern type (for pattern queries)")
    query_parser.add_argument("--metric", help="Metric name (for performance queries)")
    query_parser.add_argument("--group-by", help="Grouping field (for performance queries)")
    query_parser.add_argument("--limit", type=int, help="Maximum number of results")
    query_parser.add_argument("--output", help="Output file path (JSON)")
    query_parser.set_defaults(func=query_command)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Perform semantic search")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--entity-type", default="Generation", help="Entity type to search")
    search_parser.add_argument("--property", default="prompt_embedding", help="Embedding property name")
    search_parser.add_argument("--limit", type=int, help="Maximum number of results")
    search_parser.add_argument("--threshold", type=float, help="Similarity threshold")
    search_parser.add_argument("--output", help="Output file path (JSON)")
    search_parser.set_defaults(func=search_command)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis queries")
    analyze_parser.add_argument("type", choices=["errors", "performance", "patterns"], help="Analysis type")
    analyze_parser.add_argument("--metric", help="Metric name (for performance analysis)")
    analyze_parser.add_argument("--group-by", help="Grouping field")
    analyze_parser.add_argument("--pattern-type", help="Pattern type (for pattern analysis)")
    analyze_parser.add_argument("--limit", type=int, help="Maximum number of results")
    analyze_parser.add_argument("--output", help="Output file path (JSON)")
    analyze_parser.set_defaults(func=analyze_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get database statistics")
    stats_parser.add_argument("--output", help="Output file path (JSON)")
    stats_parser.set_defaults(func=stats_command)
    
    # Explore command
    explore_parser = subparsers.add_parser("explore", help="Run exploratory queries to understand graph structure")
    explore_parser.add_argument("--query-type", choices=["overview", "relationships", "structure", "sessions", "isolated", "stats"], 
                                default="overview", help="Type of exploratory query")
    explore_parser.add_argument("--output", help="Output file path (JSON)")
    explore_parser.set_defaults(func=explore_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.error("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"FATAL: Unhandled exception in CLI: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


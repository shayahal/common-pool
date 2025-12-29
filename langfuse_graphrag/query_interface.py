"""Query interface for GraphRAG system.

Provides query methods for pattern analysis, semantic search, error analysis,
and performance analysis.
"""

import logging
from typing import Dict, List, Optional, Any

from langfuse_graphrag.config import get_config
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class QueryInterface:
    """Interface for querying the GraphRAG knowledge graph."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """Initialize query interface.
        
        Args:
            config: Optional configuration dictionary. If None, uses default config.
            neo4j_manager: Optional Neo4jManager instance. If None, creates new one.
            embedding_generator: Optional EmbeddingGenerator instance. If None, creates new one.
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.neo4j_manager = neo4j_manager or Neo4jManager(config)
        self.embedding_generator = embedding_generator or EmbeddingGenerator(config, self.neo4j_manager)
        self.default_limit = config.get("default_query_limit", 10)
    
    def semantic_search(
        self,
        query_text: str,
        entity_type: str = "Generation",
        property_name: str = "prompt_embedding",
        limit: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity.
        
        Args:
            query_text: Query text to search for
            entity_type: Type of entity to search (Generation, SemanticEntity, etc.)
            property_name: Name of embedding property to search
            limit: Maximum number of results
            threshold: Minimum similarity threshold
        
        Returns:
            List of matching entities with similarity scores
        """
        if limit is None:
            limit = self.default_limit
        
        if threshold is None:
            threshold = self.config.get("vector_similarity_threshold", 0.7)
        
        logger.info(f"Semantic search: '{query_text}' (type={entity_type}, property={property_name})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            raise RuntimeError("Failed to generate query embedding for semantic search")
        
        # Vector similarity search query
        # Note: Neo4j vector search syntax may vary by version
        query = f"""
        MATCH (n:{entity_type})
        WHERE n.{property_name} IS NOT NULL
        WITH n, vector.similarity.cosine(n.{property_name}, $query_embedding) AS similarity
        WHERE similarity >= $threshold
        RETURN n, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        try:
            results = self.neo4j_manager.execute_query(
                query,
                {
                    "query_embedding": query_embedding,
                    "threshold": threshold,
                    "limit": limit,
                }
            )
            
            # Format results
            formatted_results = []
            for record in results:
                node = dict(record["n"])
                similarity = record["similarity"]
                formatted_results.append({
                    "entity": node,
                    "similarity": float(similarity),
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}", exc_info=True)
            raise RuntimeError(f"Semantic search failed: {e}") from e
    
    def pattern_analysis(
        self,
        pattern_type: str = "session_traces",
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyze patterns across traces and sessions.
        
        Args:
            pattern_type: Type of pattern to analyze
                - "session_traces": Traces grouped by session
                - "trace_spans": Spans grouped by trace
                - "span_generations": Generations grouped by span
                - "temporal_flow": Temporal flow of traces/spans
            filters: Optional filters (e.g., {"session_id": "xxx", "user_id": "yyy"})
            limit: Maximum number of results
        
        Returns:
            List of pattern results
        """
        if limit is None:
            limit = self.default_limit
        
        if filters is None:
            filters = {}
        
        logger.info(f"Pattern analysis: {pattern_type}")
        
        queries = {
            "session_traces": """
                MATCH (s:Session)-[:CONTAINS]->(t:Trace)
                RETURN s.id AS session_id, s.name AS session_name,
                       collect(t.id) AS trace_ids, count(t) AS trace_count
                ORDER BY trace_count DESC
                LIMIT $limit
            """,
            "trace_spans": """
                MATCH (t:Trace)-[:HAS_SPAN]->(sp:Span)
                RETURN t.id AS trace_id, t.name AS trace_name,
                       collect(sp.id) AS span_ids, count(sp) AS span_count
                ORDER BY span_count DESC
                LIMIT $limit
            """,
            "span_generations": """
                MATCH (sp:Span)-[:GENERATES]->(g:Generation)
                RETURN sp.id AS span_id, sp.name AS span_name,
                       collect(g.id) AS generation_ids, count(g) AS generation_count
                ORDER BY generation_count DESC
                LIMIT $limit
            """,
            "temporal_flow": """
                MATCH path = (t1:Trace)-[:FOLLOWS*]->(t2:Trace)
                RETURN t1.id AS from_trace, t2.id AS to_trace,
                       length(path) AS path_length
                ORDER BY path_length DESC
                LIMIT $limit
            """,
        }
        
        query = queries.get(pattern_type)
        if not query:
            logger.error(f"Unknown pattern type: {pattern_type}")
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        try:
            results = self.neo4j_manager.execute_query(query, {"limit": limit})
            logger.info(f"Found {len(results)} pattern results")
            return results
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to perform pattern analysis: {e}") from e
    
    def error_analysis(
        self,
        error_type: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyze errors in traces and spans.
        
        Args:
            error_type: Optional error type filter
            trace_id: Optional trace ID filter
            limit: Maximum number of results
        
        Returns:
            List of error analysis results
        """
        if limit is None:
            limit = self.default_limit
        
        logger.info(f"Error analysis (type={error_type}, trace_id={trace_id})")
        
        # Build query with optional filters
        query = """
        MATCH (e:Error)
        """
        
        conditions = []
        params = {}
        
        if error_type:
            conditions.append("e.type = $error_type")
            params["error_type"] = error_type
        
        if trace_id:
            query += """
            MATCH (t:Trace {id: $trace_id})-[:HAS_ERROR]->(e)
            """
            params["trace_id"] = trace_id
        else:
            query += """
            OPTIONAL MATCH (t:Trace)-[:HAS_ERROR]->(e)
            OPTIONAL MATCH (sp:Span)-[:HAS_ERROR]->(e)
            """
        
        if conditions:
            query += "WHERE " + " AND ".join(conditions)
        
        query += """
        RETURN e.id AS error_id, e.type AS error_type, e.message AS error_message,
               e.timestamp AS error_timestamp,
               t.id AS trace_id, t.name AS trace_name,
               sp.id AS span_id, sp.name AS span_name
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        params["limit"] = limit
        
        try:
            results = self.neo4j_manager.execute_query(query, params)
            logger.info(f"Found {len(results)} error results")
            return results
        except Exception as e:
            logger.error(f"Error in error analysis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to perform error analysis: {e}") from e
    
    def performance_analysis(
        self,
        metric: str = "cost",
        group_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyze performance metrics (cost, latency, tokens).
        
        Args:
            metric: Metric to analyze ("cost", "latency_ms", "tokens_input", "tokens_output")
            group_by: Optional grouping ("model", "trace", "span")
            limit: Maximum number of results
        
        Returns:
            List of performance analysis results
        """
        if limit is None:
            limit = self.default_limit
        
        logger.info(f"Performance analysis: {metric} (group_by={group_by})")
        
        if group_by == "model":
            query = f"""
            MATCH (g:Generation)
            WHERE g.{metric} IS NOT NULL
            RETURN g.model AS model,
                   count(g) AS count,
                   avg(g.{metric}) AS avg_{metric},
                   sum(g.{metric}) AS total_{metric},
                   min(g.{metric}) AS min_{metric},
                   max(g.{metric}) AS max_{metric}
            ORDER BY total_{metric} DESC
            LIMIT $limit
            """
        elif group_by == "trace":
            query = f"""
            MATCH (t:Trace)-[:HAS_SPAN]->(sp:Span)-[:GENERATES]->(g:Generation)
            WHERE g.{metric} IS NOT NULL
            WITH t, g
            RETURN t.id AS trace_id, t.name AS trace_name,
                   count(g) AS generation_count,
                   avg(g.{metric}) AS avg_{metric},
                   sum(g.{metric}) AS total_{metric}
            ORDER BY total_{metric} DESC
            LIMIT $limit
            """
        else:
            # Overall statistics
            query = f"""
            MATCH (g:Generation)
            WHERE g.{metric} IS NOT NULL
            RETURN count(g) AS count,
                   avg(g.{metric}) AS avg_{metric},
                   sum(g.{metric}) AS total_{metric},
                   min(g.{metric}) AS min_{metric},
                   max(g.{metric}) AS max_{metric}
            LIMIT 1
            """
        
        try:
            results = self.neo4j_manager.execute_query(query, {"limit": limit})
            logger.info(f"Found {len(results)} performance results")
            return results
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}", exc_info=True)
            raise RuntimeError(f"Failed to perform performance analysis: {e}") from e
    
    def find_similar_traces(
        self,
        trace_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find traces similar to a given trace using semantic similarity.
        
        Args:
            trace_id: ID of trace to find similar ones for
            limit: Maximum number of results
        
        Returns:
            List of similar traces with similarity scores
        """
        if limit is None:
            limit = self.default_limit
        
        logger.info(f"Finding traces similar to: {trace_id}")
        
        # Get trace's generation embeddings and compute average
        query = """
        MATCH (t:Trace {id: $trace_id})-[:HAS_SPAN]->(sp:Span)-[:GENERATES]->(g:Generation)
        WHERE g.prompt_embedding IS NOT NULL
        WITH t, collect(g.prompt_embedding) AS embeddings
        RETURN t, embeddings
        LIMIT 1
        """
        
        try:
            results = self.neo4j_manager.execute_query(query, {"trace_id": trace_id})
            if not results:
                logger.warning(f"Trace not found: {trace_id}")
                return []
            
            # For now, use first generation's embedding as trace embedding
            # In a more sophisticated implementation, you could average embeddings
            trace_embedding_query = """
            MATCH (t:Trace {id: $trace_id})-[:HAS_SPAN]->(sp:Span)-[:GENERATES]->(g:Generation)
            WHERE g.prompt_embedding IS NOT NULL
            RETURN g.prompt_embedding AS embedding
            LIMIT 1
            """
            
            embedding_results = self.neo4j_manager.execute_query(
                trace_embedding_query,
                {"trace_id": trace_id}
            )
            
            if not embedding_results or not embedding_results[0].get("embedding"):
                logger.error(f"No embeddings found for trace: {trace_id}")
                raise RuntimeError(f"No embeddings found for trace: {trace_id}")
            
            trace_embedding = embedding_results[0]["embedding"]
            
            # Find similar traces
            similarity_query = """
            MATCH (t:Trace)-[:HAS_SPAN]->(sp:Span)-[:GENERATES]->(g:Generation)
            WHERE g.prompt_embedding IS NOT NULL AND t.id <> $trace_id
            WITH t, g.prompt_embedding AS embedding
            WITH t, collect(embedding)[0] AS trace_embedding
            WHERE trace_embedding IS NOT NULL
            WITH t, vector.similarity.cosine(trace_embedding, $query_embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN t, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            threshold = self.config.get("vector_similarity_threshold", 0.7)
            results = self.neo4j_manager.execute_query(
                similarity_query,
                {
                    "trace_id": trace_id,
                    "query_embedding": trace_embedding,
                    "threshold": threshold,
                    "limit": limit,
                }
            )
            
            formatted_results = []
            for record in results:
                node = dict(record["t"])
                similarity = record["similarity"]
                formatted_results.append({
                    "trace": node,
                    "similarity": float(similarity),
                })
            
            logger.info(f"Found {len(formatted_results)} similar traces")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error finding similar traces: {e}", exc_info=True)
            raise RuntimeError(f"Failed to find similar traces: {e}") from e
    
    def execute_custom_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
        
        Returns:
            Query results
        """
        logger.info("Executing custom query")
        
        try:
            results = self.neo4j_manager.execute_query(query, parameters or {})
            logger.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error executing custom query: {e}", exc_info=True)
            raise


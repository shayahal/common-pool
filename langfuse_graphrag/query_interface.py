"""Query interface for GraphRAG system.

Provides query methods for pattern analysis, semantic search, error analysis,
performance analysis, and LLM-based question answering.
"""

import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from opentelemetry import trace

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

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
        
        # Initialize LLM for question answering
        self._init_llm()
    
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
        
        # Ensure limit and threshold are not None for type checking
        limit_val = int(limit) if limit is not None else self.default_limit
        threshold_val = float(threshold) if threshold is not None else 0.7
        
        logger.info(f"Semantic search: '{query_text}' (type={entity_type}, property={property_name})")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            raise RuntimeError("Failed to generate query embedding for semantic search")
        
        # Use vector index query - REQUIRES Neo4j 5.11+ Enterprise with vector indexes
        index_name = f"{entity_type.lower()}_{property_name}_vector_idx"
        
        query = f"""
        CALL db.index.vector.queryNodes(
            '{index_name}',
            $top_k,
            $query_embedding
        )
        YIELD node, score
        WHERE node:{entity_type} AND node.{property_name} IS NOT NULL
        WITH node, score AS similarity
        WHERE similarity >= $threshold
        RETURN node AS n, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        results = self.neo4j_manager.execute_query(
            query,
            {
                "query_embedding": query_embedding,
                "top_k": limit_val * 2,  # Get more candidates to filter by threshold
                "threshold": threshold_val,
                "limit": limit_val,
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
    
    def _init_llm(self) -> None:
        """Initialize LLM for question answering."""
        llm_model = self.config.get("graphrag_llm_model", "gpt-3.5-turbo")
        llm_temperature = self.config.get("graphrag_llm_temperature", 0.0)
        llm_max_tokens = self.config.get("graphrag_llm_max_tokens", 4000)
        openai_api_key = self.config.get("openai_api_key")
        
        if not openai_api_key:
            logger.error("OpenAI API key not found in config")
            raise ValueError("OpenAI API key is required for LLM question answering. Set OPENAI_API_KEY environment variable.")
        
        # Initialize ChatOpenAI - max_tokens may not be available in all versions
        try:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,  # type: ignore
                api_key=openai_api_key,
            )
        except TypeError:
            # Fallback if max_tokens is not supported
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=llm_temperature,
                api_key=openai_api_key,
            )
        logger.debug(f"Initialized LLM for question answering: {llm_model}")
    
    def answer_question(
        self,
        question: str,
        max_context_items: int = 10,
        include_graph_context: bool = True,
        max_graph_depth: int = 2
    ) -> Dict[str, Any]:
        """Answer a question using LLM with context from the GraphRAG knowledge graph.
        
        This method:
        1. Performs semantic search to find relevant entities
        2. Retrieves graph context (neighbors, relationships)
        3. Formats context for LLM
        4. Generates answer using LLM
        
        Args:
            question: Natural language question to answer
            max_context_items: Maximum number of context items to retrieve
            include_graph_context: Whether to include graph neighbors and relationships
            max_graph_depth: Maximum depth for graph traversal (1 = direct neighbors, 2 = neighbors of neighbors)
        
        Returns:
            Dictionary with:
                - answer: The generated answer
                - sources: List of source entities used
                - context_summary: Summary of context used
                - metadata: Additional metadata (token usage, etc.)
        
        Raises:
            RuntimeError: If question answering fails
        """
        logger.info(f"Answering question: '{question}'")
        
        try:
            # Step 1: Semantic search to find relevant entities
            logger.debug("Step 1: Performing semantic search")
            relevant_generations = self.semantic_search(
                query_text=question,
                entity_type="Generation",
                property_name="prompt_embedding",
                limit=max_context_items // 2,
                threshold=0.6  # Lower threshold for broader context
            )
            
            relevant_entities = self.semantic_search(
                query_text=question,
                entity_type="SemanticEntity",
                property_name="embedding",
                limit=max_context_items // 2,
                threshold=0.6
            )
            
            # Step 2: Retrieve graph context if requested
            context_items = []
            source_ids = set()
            
            # Add Generation context
            for gen_result in relevant_generations:
                gen = gen_result["entity"]
                gen_id = gen.get("id")
                if gen_id:
                    source_ids.add(f"Generation:{gen_id}")
                    
                    context_item = {
                        "type": "Generation",
                        "id": gen_id,
                        "prompt": gen.get("prompt", ""),
                        "response": gen.get("response", ""),
                        "reasoning": gen.get("reasoning", ""),
                        "similarity": gen_result.get("similarity", 0.0),
                    }
                    context_items.append(context_item)
                    
                    # Get graph context if requested
                    if include_graph_context:
                        graph_context = self._get_graph_context(
                            entity_type="Generation",
                            entity_id=gen_id,
                            max_depth=max_graph_depth
                        )
                        context_item["graph_context"] = graph_context
            
            # Add SemanticEntity context
            for entity_result in relevant_entities:
                entity = entity_result["entity"]
                entity_id = entity.get("id")
                if entity_id:
                    source_ids.add(f"SemanticEntity:{entity_id}")
                    
                    context_item = {
                        "type": "SemanticEntity",
                        "id": entity_id,
                        "name": entity.get("name", ""),
                        "type": entity.get("type", ""),
                        "description": entity.get("description", ""),
                        "similarity": entity_result.get("similarity", 0.0),
                    }
                    context_items.append(context_item)
                    
                    # Get graph context if requested
                    if include_graph_context:
                        graph_context = self._get_graph_context(
                            entity_type="SemanticEntity",
                            entity_id=entity_id,
                            max_depth=max_graph_depth
                        )
                        context_item["graph_context"] = graph_context
            
            if not context_items:
                logger.warning("No relevant context found for question")
                return {
                    "answer": "I couldn't find any relevant information in the knowledge graph to answer this question.",
                    "sources": [],
                    "context_summary": "No context retrieved",
                    "metadata": {}
                }
            
            # Step 3: Format context for LLM
            logger.debug(f"Step 3: Formatting {len(context_items)} context items for LLM")
            formatted_context = self._format_context_for_llm(context_items)
            
            # Step 4: Generate answer using LLM
            logger.debug("Step 4: Generating answer with LLM")
            answer = self._generate_answer_with_llm(question, formatted_context)
            
            # Step 5: Prepare response
            response = {
                "answer": answer,
                "sources": list(source_ids),
                "context_summary": f"Retrieved {len(context_items)} context items ({len(relevant_generations)} generations, {len(relevant_entities)} semantic entities)",
                "metadata": {
                    "context_items_count": len(context_items),
                    "generations_count": len(relevant_generations),
                    "entities_count": len(relevant_entities),
                }
            }
            
            logger.info(f"Successfully generated answer (sources: {len(source_ids)})")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            raise RuntimeError(f"Failed to answer question: {e}") from e
    
    def _get_graph_context(
        self,
        entity_type: str,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get graph context (neighbors and relationships) for an entity.
        
        Args:
            entity_type: Type of entity (Generation, SemanticEntity, etc.)
            entity_id: ID of the entity
            max_depth: Maximum depth for graph traversal
        
        Returns:
            Dictionary with graph context information
        """
        try:
            # Get direct neighbors (depth 1)
            query = f"""
            MATCH (n:{entity_type} {{id: $entity_id}})
            OPTIONAL MATCH (n)-[r1]->(neighbor1)
            OPTIONAL MATCH (neighbor1)-[r2]->(neighbor2)
            RETURN 
                collect(DISTINCT {{type: type(r1), to: labels(neighbor1)[0], to_id: neighbor1.id, to_name: neighbor1.name}}) AS depth1,
                collect(DISTINCT {{type: type(r2), to: labels(neighbor2)[0], to_id: neighbor2.id, to_name: neighbor2.name}}) AS depth2
            LIMIT 1
            """
            
            results = self.neo4j_manager.execute_query(
                query,
                {"entity_id": entity_id}
            )
            
            if not results:
                return {"depth1": [], "depth2": []}
            
            result = results[0]
            depth1 = result.get("depth1", [])
            depth2 = result.get("depth2", []) if max_depth >= 2 else []
            
            # Filter out None values
            depth1 = [item for item in depth1 if item.get("to_id")]
            depth2 = [item for item in depth2 if item.get("to_id")]
            
            return {
                "depth1": depth1[:10],  # Limit to 10 neighbors per depth
                "depth2": depth2[:10] if max_depth >= 2 else []
            }
            
        except Exception as e:
            logger.error(f"Error getting graph context for {entity_type}:{entity_id}: {e}", exc_info=True)
            # Return empty context rather than failing
            return {"depth1": [], "depth2": []}
    
    def _format_context_for_llm(self, context_items: List[Dict[str, Any]]) -> str:
        """Format context items into a string for LLM consumption.
        
        Args:
            context_items: List of context item dictionaries
        
        Returns:
            Formatted context string
        """
        formatted_parts = []
        
        for i, item in enumerate(context_items, 1):
            part = ""
            if item["type"] == "Generation":
                part = f"""Context Item {i} (Generation: {item.get('id', 'unknown')}):
- Prompt: {item.get('prompt', 'N/A')[:500]}
- Response: {item.get('response', 'N/A')[:500]}
- Reasoning: {item.get('reasoning', 'N/A')[:300]}
- Similarity Score: {item.get('similarity', 0.0):.3f}
"""
                if item.get("graph_context"):
                    graph = item["graph_context"]
                    if graph.get("depth1"):
                        part += f"- Related Entities: {', '.join([n.get('to_name', n.get('to_id', 'unknown')) for n in graph['depth1'][:5]])}\n"
            elif item["type"] == "SemanticEntity":
                part = f"""Context Item {i} (SemanticEntity: {item.get('name', 'unknown')}):
- Type: {item.get('type', 'N/A')}
- Description: {item.get('description', 'N/A')[:500]}
- Similarity Score: {item.get('similarity', 0.0):.3f}
"""
                if item.get("graph_context"):
                    graph = item["graph_context"]
                    if graph.get("depth1"):
                        part += f"- Related Entities: {', '.join([n.get('to_name', n.get('to_id', 'unknown')) for n in graph['depth1'][:5]])}\n"
            else:
                # Fallback for unknown types
                part = f"""Context Item {i} ({item.get('type', 'Unknown')}: {item.get('id', 'unknown')}):
- Data: {str(item)[:500]}
"""
            
            if part:
                formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def _generate_answer_with_llm(
        self,
        question: str,
        context: str
    ) -> str:
        """Generate answer using LLM with provided context.
        
        Args:
            question: The question to answer
            context: Formatted context string
        
        Returns:
            Generated answer string
        
        Raises:
            RuntimeError: If LLM generation fails
        """
        system_prompt = """You are a helpful assistant that answers questions based on context from a knowledge graph.
The context provided comes from a GraphRAG (Graph Retrieval-Augmented Generation) system that contains:
- Generations: LLM prompts, responses, and reasoning from traces
- SemanticEntities: Extracted concepts, topics, and entities
- Relationships: Connections between entities in the graph

Your task is to:
1. Analyze the provided context carefully
2. Synthesize information from multiple sources when relevant
3. Provide a clear, accurate answer to the question
4. If the context doesn't contain enough information, say so clearly
5. Cite specific context items when making claims

Answer in a clear, concise manner. If you reference specific information, mention which context items it came from."""
        
        user_prompt = f"""Question: {question}

Context from Knowledge Graph:
{context}

Please answer the question based on the context provided above. If the context doesn't contain enough information to fully answer the question, please say so."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle different response types
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    # Handle list of content blocks (e.g., from some models)
                    answer = " ".join(str(item) for item in content)
                else:
                    answer = str(content)
            else:
                answer = str(response)
            
            logger.debug(f"Generated answer (length: {len(answer)} chars)")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}", exc_info=True)
            raise RuntimeError(f"LLM generation failed: {e}") from e


class InteractiveChat:
    """Interactive chat interface with GraphRAG context support.
    
    Maintains conversation history and automatically retrieves relevant
    context from the GraphRAG knowledge graph when needed.
    All operations are traced via OpenTelemetry to Langfuse.
    """
    
    def __init__(
        self,
        query_interface: Optional[QueryInterface] = None,
        config: Optional[Dict[str, Any]] = None,
        use_graphrag_context: bool = True,
        auto_retrieve_context: bool = True,
        max_context_items: int = 10,
        session_id: Optional[str] = None
    ):
        """Initialize interactive chat.
        
        Args:
            query_interface: Optional QueryInterface instance. If None, creates new one.
            config: Optional configuration dictionary.
            use_graphrag_context: Whether to use GraphRAG context for answers.
            auto_retrieve_context: Whether to automatically retrieve context for each message.
            max_context_items: Maximum number of context items to retrieve.
            session_id: Optional session ID for grouping traces. If None, generates one.
        """
        self.query_interface = query_interface or QueryInterface(config)
        self.use_graphrag_context = use_graphrag_context
        self.auto_retrieve_context = auto_retrieve_context
        self.max_context_items = max_context_items
        
        # Generate session ID for grouping traces
        if session_id is None:
            from datetime import datetime
            session_id = f"graphrag_chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.session_id = session_id
        
        # Initialize OpenTelemetry tracing
        try:
            from cpr_game.otel_manager import OTelManager
            self.otel_manager = OTelManager(config)
            self.tracer = self.otel_manager.get_tracer()
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize OpenTelemetry for chat tracing: {e}") from e
        
        # Conversation history
        self.conversation_history: List[BaseMessage] = []
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant with access to a GraphRAG (Graph Retrieval-Augmented Generation) knowledge graph.

The knowledge graph contains:
- Generations: LLM prompts, responses, and reasoning from traces
- SemanticEntities: Extracted concepts, topics, and entities
- Relationships: Connections between entities in the graph

You can answer questions about the data in the knowledge graph. When context is provided, use it to give accurate, detailed answers. If context is not provided or insufficient, you can still answer general questions, but let the user know when you're using general knowledge vs. specific data from the graph.

Be conversational, helpful, and clear. Maintain context from previous messages in the conversation."""
        
        # Initialize with system message
        self.conversation_history.append(SystemMessage(content=self.system_prompt))
        
        # Message counter for trace naming
        self.message_count = 0
        
        logger.info(f"Initialized interactive chat interface (session_id: {self.session_id})")
    
    def chat(
        self,
        user_message: str,
        use_context: Optional[bool] = None
    ) -> str:
        """Send a message and get a response.
        
        All operations are traced via OpenTelemetry to Langfuse.
        
        Args:
            user_message: The user's message
            use_context: Whether to retrieve GraphRAG context (overrides auto_retrieve_context if set)
        
        Returns:
            The assistant's response
        
        Raises:
            RuntimeError: If chat fails
        """
        logger.debug(f"Chat message received: {user_message[:100]}...")
        
        self.message_count += 1
        trace_name = f"graphrag_chat_message_{self.message_count}"
        trace_start_time = datetime.utcnow()
        
        # Create trace for this chat message
        if self.tracer is None:
            logger.error("Tracer is None - OpenTelemetry not initialized")
            raise RuntimeError("OpenTelemetry tracer not available - cannot trace chat operations")
        
        # Create trace attributes
        trace_attributes = {
            "trace_id": f"{self.session_id}_msg_{self.message_count}",
            "name": trace_name,
            "session_id": self.session_id,
            "user_message": user_message[:500],  # Truncate for attribute size limits
            "timestamp": trace_start_time.isoformat(),
            "langfuse.trace.name": trace_name,
            "langfuse.session.id": self.session_id,
        }
        
        # Determine if we should retrieve context
        should_use_context = use_context if use_context is not None else self.auto_retrieve_context
        trace_attributes["use_graphrag_context"] = str(should_use_context and self.use_graphrag_context)
        
        # Create root span for this chat message
        with self.tracer.start_as_current_span(
            trace_name,
            attributes=trace_attributes
        ) as root_span:
            context_text = ""
            relevant_generations = []
            relevant_entities = []
            
            if should_use_context and self.use_graphrag_context:
                logger.debug("Retrieving GraphRAG context")
                
                # Span for semantic search operations
                with self.tracer.start_as_current_span(
                    "graphrag_semantic_search",
                    attributes={
                        "query": user_message[:500],
                        "entity_types": "Generation,SemanticEntity",
                    }
                ) as search_span:
                    # Retrieve relevant context from GraphRAG
                    relevant_generations = self.query_interface.semantic_search(
                        query_text=user_message,
                        entity_type="Generation",
                        property_name="prompt_embedding",
                        limit=self.max_context_items // 2,
                        threshold=0.6
                    )
                    
                    relevant_entities = self.query_interface.semantic_search(
                        query_text=user_message,
                        entity_type="SemanticEntity",
                        property_name="embedding",
                        limit=self.max_context_items // 2,
                        threshold=0.6
                    )
                    
                    # Set span attributes with results
                    search_span.set_attribute("generations_found", str(len(relevant_generations)))
                    search_span.set_attribute("entities_found", str(len(relevant_entities)))
                    
                    # Set output as JSON for Langfuse visibility
                    import json
                    search_output = {
                        "generations": len(relevant_generations),
                        "entities": len(relevant_entities),
                        "generation_ids": [g["entity"].get("id", "unknown") for g in relevant_generations[:5]],
                        "entity_names": [e["entity"].get("name", "unknown") for e in relevant_entities[:5]],
                    }
                    search_span.set_attribute("output", json.dumps(search_output))
                    
                    # Also set as event for better visibility
                    search_span.add_event("semantic_search_complete", {
                        "generations_count": len(relevant_generations),
                        "entities_count": len(relevant_entities),
                    })
            
            # Format context
            if relevant_generations or relevant_entities:
                context_parts = []
                if relevant_generations:
                    context_parts.append("Relevant Generations from Knowledge Graph:")
                    for i, gen_result in enumerate(relevant_generations[:5], 1):  # Limit to 5
                        gen = gen_result["entity"]
                        context_parts.append(
                            f"{i}. Prompt: {gen.get('prompt', '')[:200]}...\n"
                            f"   Response: {gen.get('response', '')[:200]}..."
                        )
                
                if relevant_entities:
                    context_parts.append("\nRelevant Semantic Entities:")
                    for i, entity_result in enumerate(relevant_entities[:5], 1):  # Limit to 5
                        entity = entity_result["entity"]
                        context_parts.append(
                            f"{i}. {entity.get('name', 'Unknown')} ({entity.get('type', 'unknown')}): "
                            f"{entity.get('description', '')[:200]}..."
                        )
                
                context_text = "\n".join(context_parts)
                logger.debug(f"Retrieved context ({len(relevant_generations)} generations, {len(relevant_entities)} entities)")
            
            # Build user message with optional context
            if context_text:
                full_user_message = f"{user_message}\n\n[Context from Knowledge Graph]\n{context_text}"
            else:
                full_user_message = user_message
            
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=full_user_message))
            
            # Create explicit span for LLM generation with input/output
            # Use Langfuse-specific attributes so generations show up properly
            llm_model = self.query_interface.config.get("graphrag_llm_model", "unknown")
            llm_temperature = self.query_interface.config.get("graphrag_llm_temperature", 0.0)
            
            with self.tracer.start_as_current_span(
                "llm_generation",
                attributes={
                    "type": "llm",
                    "model": llm_model,
                    "temperature": str(llm_temperature),
                    # Langfuse-specific attributes for proper generation display
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.name": "graphrag_chat_llm_generation",
                    "langfuse.observation.model.name": llm_model,
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": llm_model,
                    # Input attributes (multiple formats for compatibility)
                    "input": full_user_message,
                    "langfuse.observation.input": full_user_message,
                    "gen_ai.prompt": full_user_message,
                    "prompt": full_user_message,
                }
            ) as llm_span:
                llm_span_start = datetime.utcnow()
                
                # Get response from LLM (auto-instrumented by LangChain)
                # The LLM call will automatically create spans via LangChain instrumentation
                response = self.query_interface.llm.invoke(self.conversation_history)
                
                # Extract response content
                if hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, str):
                        assistant_response = content
                    elif isinstance(content, list):
                        assistant_response = " ".join(str(item) for item in content)
                    else:
                        assistant_response = str(content)
                else:
                    assistant_response = str(response)
                
                llm_span_end = datetime.utcnow()
                llm_duration_ms = (llm_span_end - llm_span_start).total_seconds() * 1000
                
                # Set output attributes (multiple formats for compatibility)
                # Ensure assistant_response is a string
                output_str = str(assistant_response) if not isinstance(assistant_response, str) else assistant_response
                llm_span.set_attribute("output", output_str)
                llm_span.set_attribute("langfuse.observation.output", output_str)
                llm_span.set_attribute("gen_ai.completion", output_str)
                llm_span.set_attribute("response", output_str)
                
                # Set system prompt if available
                if self.conversation_history and isinstance(self.conversation_history[0], SystemMessage):
                    system_prompt_content = self.conversation_history[0].content
                    system_prompt_str = str(system_prompt_content) if not isinstance(system_prompt_content, str) else system_prompt_content
                    llm_span.set_attribute("system_prompt", system_prompt_str)
                    llm_span.set_attribute("gen_ai.system_prompt", system_prompt_str)
                
                # Set duration and status
                llm_span.set_attribute("duration_ms", str(llm_duration_ms))
                llm_span.set_attribute("end_time", llm_span_end.isoformat())
                llm_span.set_attribute("status", "success")
                
                # Try to get token usage from response if available
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if metadata and 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        if 'prompt_tokens' in token_usage:
                            tokens_input = int(token_usage['prompt_tokens'])
                            llm_span.set_attribute("tokens_input", str(tokens_input))
                            llm_span.set_attribute("llm.prompt_tokens", str(tokens_input))
                            llm_span.set_attribute("langfuse.observation.usage.input", str(tokens_input))
                        if 'completion_tokens' in token_usage:
                            tokens_output = int(token_usage['completion_tokens'])
                            llm_span.set_attribute("tokens_output", str(tokens_output))
                            llm_span.set_attribute("llm.completion_tokens", str(tokens_output))
                            llm_span.set_attribute("langfuse.observation.usage.output", str(tokens_output))
                        if 'total_tokens' in token_usage:
                            llm_span.set_attribute("llm.total_tokens", str(token_usage['total_tokens']))
                        if 'total_cost' in token_usage:
                            llm_span.set_attribute("cost", str(token_usage['total_cost']))
                            llm_span.set_attribute("langfuse.observation.usage.cost", str(token_usage['total_cost']))
            
            # Add assistant response to history
            self.conversation_history.append(AIMessage(content=assistant_response))
            
            # Set trace end attributes
            trace_end_time = datetime.utcnow()
            duration_ms = (trace_end_time - trace_start_time).total_seconds() * 1000
            root_span.set_attribute("end_time", trace_end_time.isoformat())
            root_span.set_attribute("duration_ms", str(duration_ms))
            root_span.set_attribute("response_length", str(len(assistant_response)))
            root_span.set_attribute("context_used", str(bool(context_text)))
            
            # Set input/output for Langfuse visibility
            root_span.set_attribute("input", user_message[:1000])  # Truncate for size
            root_span.set_attribute("output", assistant_response[:1000])  # Truncate for size
            root_span.set_attribute("langfuse.input", user_message[:1000])
            root_span.set_attribute("langfuse.output", assistant_response[:1000])
            
            logger.debug(f"Generated response (length: {len(assistant_response)} chars)")
            return assistant_response
    
    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = [SystemMessage(content=self.system_prompt)]
        logger.info("Conversation history reset")
    
    def get_history(self) -> List[BaseMessage]:
        """Get conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history (alias for reset)."""
        self.reset()


class VectorRAGChat:
    """Interactive chat interface using only vector RAG (semantic search).
    
    This chat uses semantic search to retrieve relevant context from embeddings,
    but does not use the full graph structure, relationships, or communities.
    All operations are traced via OpenTelemetry to Langfuse.
    """
    
    def __init__(
        self,
        query_interface: Optional[QueryInterface] = None,
        config: Optional[Dict[str, Any]] = None,
        max_context_items: int = 10,
        session_id: Optional[str] = None
    ):
        """Initialize vector RAG chat.
        
        Args:
            query_interface: Optional QueryInterface instance. If None, creates new one.
            config: Optional configuration dictionary.
            max_context_items: Maximum number of context items to retrieve.
            session_id: Optional session ID for grouping traces. If None, generates one.
        """
        self.query_interface = query_interface or QueryInterface(config)
        self.max_context_items = max_context_items
        
        # Generate session ID for grouping traces
        if session_id is None:
            session_id = f"vector_rag_chat_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.session_id = session_id
        
        # Initialize OpenTelemetry tracing
        try:
            from cpr_game.otel_manager import OTelManager
            self.otel_manager = OTelManager(config)
            self.tracer = self.otel_manager.get_tracer()
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize OpenTelemetry for chat tracing: {e}") from e
        
        # Conversation history
        self.conversation_history: List[BaseMessage] = []
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant with access to a knowledge base via vector search.

The knowledge base contains:
- Generations: LLM prompts, responses, and reasoning from traces
- SemanticEntities: Extracted concepts, topics, and entities

You can answer questions about the data in the knowledge base. When context is provided from vector search, use it to give accurate, detailed answers. If context is not provided or insufficient, you can still answer general questions, but let the user know when you're using general knowledge vs. specific data from the knowledge base.

Be conversational, helpful, and clear. Maintain context from previous messages in the conversation."""
        
        # Initialize with system message
        self.conversation_history.append(SystemMessage(content=self.system_prompt))
        
        # Message counter for trace naming
        self.message_count = 0
        
        logger.info(f"Initialized vector RAG chat interface (session_id: {self.session_id})")
    
    def chat(
        self,
        user_message: str
    ) -> str:
        """Send a message and get a response using vector RAG.
        
        All operations are traced via OpenTelemetry to Langfuse.
        
        Args:
            user_message: The user's message
        
        Returns:
            The assistant's response
        
        Raises:
            RuntimeError: If chat fails
        """
        logger.debug(f"Vector RAG chat message received: {user_message[:100]}...")
        
        self.message_count += 1
        trace_name = f"vector_rag_chat_message_{self.message_count}"
        trace_start_time = datetime.utcnow()
        
        # Create trace for this chat message
        if self.tracer is None:
            logger.error("Tracer is None - OpenTelemetry not initialized")
            raise RuntimeError("OpenTelemetry tracer not available - cannot trace chat operations")
        
        # Create trace attributes
        trace_attributes = {
            "trace_id": f"{self.session_id}_msg_{self.message_count}",
            "name": trace_name,
            "session_id": self.session_id,
            "user_message": user_message[:500],
            "timestamp": trace_start_time.isoformat(),
            "langfuse.trace.name": trace_name,
            "langfuse.session.id": self.session_id,
            "rag_type": "vector_only",
        }
        
        # Create root span for this chat message
        with self.tracer.start_as_current_span(
            trace_name,
            attributes=trace_attributes
        ) as root_span:
            context_text = ""
            relevant_generations = []
            relevant_entities = []
            
            # Vector search for context
            logger.debug("Performing vector search for context")
            
            with self.tracer.start_as_current_span(
                "vector_semantic_search",
                attributes={
                    "query": user_message[:500],
                    "entity_types": "Generation,SemanticEntity",
                }
            ) as search_span:
                # Retrieve relevant context using vector search only
                relevant_generations = self.query_interface.semantic_search(
                    query_text=user_message,
                    entity_type="Generation",
                    property_name="prompt_embedding",
                    limit=self.max_context_items // 2,
                    threshold=0.6
                )
                
                relevant_entities = self.query_interface.semantic_search(
                    query_text=user_message,
                    entity_type="SemanticEntity",
                    property_name="embedding",
                    limit=self.max_context_items // 2,
                    threshold=0.6
                )
                
                # Set span attributes with results
                search_span.set_attribute("generations_found", str(len(relevant_generations)))
                search_span.set_attribute("entities_found", str(len(relevant_entities)))
                
                # Set output as JSON for Langfuse visibility
                import json
                search_output = {
                    "generations": len(relevant_generations),
                    "entities": len(relevant_entities),
                    "generation_ids": [g["entity"].get("id", "unknown") for g in relevant_generations[:5]],
                    "entity_names": [e["entity"].get("name", "unknown") for e in relevant_entities[:5]],
                }
                search_span.set_attribute("output", json.dumps(search_output))
            
            # Format context from vector search results
            if relevant_generations or relevant_entities:
                context_parts = []
                if relevant_generations:
                    context_parts.append("Relevant Generations from Knowledge Base:")
                    for i, gen_result in enumerate(relevant_generations[:5], 1):
                        gen = gen_result["entity"]
                        context_parts.append(
                            f"{i}. Prompt: {gen.get('prompt', '')[:200]}...\n"
                            f"   Response: {gen.get('response', '')[:200]}..."
                        )
                
                if relevant_entities:
                    context_parts.append("\nRelevant Semantic Entities:")
                    for i, entity_result in enumerate(relevant_entities[:5], 1):
                        entity = entity_result["entity"]
                        context_parts.append(
                            f"{i}. {entity.get('name', 'Unknown')} ({entity.get('type', 'unknown')}): "
                            f"{entity.get('description', '')[:200]}..."
                        )
                
                context_text = "\n".join(context_parts)
                logger.debug(f"Retrieved context ({len(relevant_generations)} generations, {len(relevant_entities)} entities)")
            
            # Build user message with context
            if context_text:
                full_user_message = f"{user_message}\n\n[Context from Vector Search]\n{context_text}"
            else:
                full_user_message = user_message
            
            # Add user message to history
            self.conversation_history.append(HumanMessage(content=full_user_message))
            
            # Create explicit span for LLM generation with input/output
            llm_model = self.query_interface.config.get("graphrag_llm_model", "unknown")
            llm_temperature = self.query_interface.config.get("graphrag_llm_temperature", 0.0)
            
            with self.tracer.start_as_current_span(
                "llm_generation",
                attributes={
                    "type": "llm",
                    "model": llm_model,
                    "temperature": str(llm_temperature),
                    # Langfuse-specific attributes for proper generation display
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.name": "vector_rag_llm_generation",
                    "langfuse.observation.model.name": llm_model,
                    "gen_ai.system": "openai",
                    "gen_ai.request.model": llm_model,
                    # Input attributes (multiple formats for compatibility)
                    "input": full_user_message,
                    "langfuse.observation.input": full_user_message,
                    "gen_ai.prompt": full_user_message,
                    "prompt": full_user_message,
                }
            ) as llm_span:
                llm_span_start = datetime.utcnow()
                
                # Get response from LLM
                response = self.query_interface.llm.invoke(self.conversation_history)
                
                # Extract response content
                if hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, str):
                        assistant_response = content
                    elif isinstance(content, list):
                        assistant_response = " ".join(str(item) for item in content)
                    else:
                        assistant_response = str(content)
                else:
                    assistant_response = str(response)
                
                llm_span_end = datetime.utcnow()
                llm_duration_ms = (llm_span_end - llm_span_start).total_seconds() * 1000
                
                # Set output attributes (multiple formats for compatibility)
                output_str = str(assistant_response) if not isinstance(assistant_response, str) else assistant_response
                llm_span.set_attribute("output", output_str)
                llm_span.set_attribute("langfuse.observation.output", output_str)
                llm_span.set_attribute("gen_ai.completion", output_str)
                llm_span.set_attribute("response", output_str)
                
                # Set system prompt if available
                if self.conversation_history and isinstance(self.conversation_history[0], SystemMessage):
                    system_prompt_content = self.conversation_history[0].content
                    system_prompt_str = str(system_prompt_content) if not isinstance(system_prompt_content, str) else system_prompt_content
                    llm_span.set_attribute("system_prompt", system_prompt_str)
                    llm_span.set_attribute("gen_ai.system_prompt", system_prompt_str)
                
                # Set duration and status
                llm_span.set_attribute("duration_ms", str(llm_duration_ms))
                llm_span.set_attribute("end_time", llm_span_end.isoformat())
                llm_span.set_attribute("status", "success")
                
                # Try to get token usage from response if available
                if hasattr(response, 'response_metadata'):
                    metadata = response.response_metadata
                    if metadata and 'token_usage' in metadata:
                        token_usage = metadata['token_usage']
                        if 'prompt_tokens' in token_usage:
                            tokens_input = int(token_usage['prompt_tokens'])
                            llm_span.set_attribute("tokens_input", str(tokens_input))
                            llm_span.set_attribute("llm.prompt_tokens", str(tokens_input))
                            llm_span.set_attribute("langfuse.observation.usage.input", str(tokens_input))
                        if 'completion_tokens' in token_usage:
                            tokens_output = int(token_usage['completion_tokens'])
                            llm_span.set_attribute("tokens_output", str(tokens_output))
                            llm_span.set_attribute("llm.completion_tokens", str(tokens_output))
                            llm_span.set_attribute("langfuse.observation.usage.output", str(tokens_output))
                        if 'total_tokens' in token_usage:
                            llm_span.set_attribute("llm.total_tokens", str(token_usage['total_tokens']))
                        if 'total_cost' in token_usage:
                            llm_span.set_attribute("cost", str(token_usage['total_cost']))
                            llm_span.set_attribute("langfuse.observation.usage.cost", str(token_usage['total_cost']))
            
            # Add assistant response to history
            self.conversation_history.append(AIMessage(content=assistant_response))
            
            # Set trace end attributes
            trace_end_time = datetime.utcnow()
            duration_ms = (trace_end_time - trace_start_time).total_seconds() * 1000
            root_span.set_attribute("end_time", trace_end_time.isoformat())
            root_span.set_attribute("duration_ms", str(duration_ms))
            root_span.set_attribute("response_length", str(len(assistant_response)))
            root_span.set_attribute("context_used", str(bool(context_text)))
            
            # Set input/output for Langfuse visibility
            root_span.set_attribute("input", user_message[:1000])
            root_span.set_attribute("output", assistant_response[:1000])
            root_span.set_attribute("langfuse.input", user_message[:1000])
            root_span.set_attribute("langfuse.output", assistant_response[:1000])
            
            logger.debug(f"Generated response (length: {len(assistant_response)} chars)")
            return assistant_response
    
    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = [SystemMessage(content=self.system_prompt)]
        logger.info("Conversation history reset")
    
    def get_history(self) -> List[BaseMessage]:
        """Get conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self.conversation_history.copy()
    
    def clear_history(self) -> None:
        """Clear conversation history (alias for reset)."""
        self.reset()


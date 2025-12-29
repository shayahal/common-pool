"""Microsoft GraphRAG integration for semantic extraction.

Uses GraphRAG to extract semantic entities, build communities, and generate summaries
from trace data.
"""

import logging
import json
import uuid
import hashlib
import time
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError

from langfuse_graphrag.config import get_config
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    AgglomerativeClustering = None
    cosine_similarity = None
    logger.warning("scikit-learn not available. Hierarchical clustering will use fallback method.")


class GraphRAGIndexer:
    """Indexer using Microsoft GraphRAG for semantic extraction."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        neo4j_manager: Optional[Neo4jManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """Initialize GraphRAG indexer.
        
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
        
        # GraphRAG settings
        self.llm_model = config.get("graphrag_llm_model", "gpt-3.5-turbo")
        self.batch_size = config.get("graphrag_batch_size", 100)
        self.chunk_size = config.get("graphrag_chunk_size", 1000)
        self.chunk_overlap = config.get("graphrag_chunk_overlap", 200)
        
        # Data directories
        self.data_dir = Path(config.get("data_dir", "data/graphrag"))
        self.processed_dir = Path(config.get("processed_data_dir", self.data_dir / "processed"))
        self.indices_dir = Path(config.get("indices_dir", self.data_dir / "indices"))
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client for LLM operations
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GraphRAG")
        self.openai_client = OpenAI(api_key=api_key)
        
        # Community detection settings
        self.community_min_size = config.get("community_min_size", 3)
        self.community_max_levels = config.get("community_max_levels", 3)
        
        logger.info("Initialized GraphRAGIndexer")
    
    def _call_openai_with_retry(
        self,
        api_call_func,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        operation_name: str = "OpenAI API call"
    ):
        """Call OpenAI API with exponential backoff retry for rate limits.
        
        Args:
            api_call_func: Function that makes the OpenAI API call
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
            max_delay: Maximum delay in seconds
            operation_name: Name of the operation for logging
        
        Returns:
            Result of the API call
        
        Raises:
            RateLimitError: If rate limit persists after max retries
            APIError: For other API errors
        """
        for attempt in range(max_retries):
            try:
                return api_call_func()
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    # Rate limits often need more time - double the delay
                    delay = min(delay * 2, max_delay)
                    jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"{operation_name}: Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    # Max retries reached, re-raise the error
                    logger.error(
                        f"{operation_name}: Rate limit error after {max_retries} attempts"
                    )
                    raise
            except (APIConnectionError, APITimeoutError) as e:
                # Network errors - retry with shorter delay
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"{operation_name}: Connection/timeout error (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    logger.error(f"{operation_name}: Connection/timeout error after {max_retries} attempts")
                    raise
            except APIError as e:
                # Check if error message indicates rate limit
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay = min(delay * 2, max_delay)  # Double for rate limits
                        jitter = random.uniform(0, delay * 0.1)
                        total_delay = delay + jitter
                        
                        logger.warning(
                            f"{operation_name}: Rate limit detected in error message (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {total_delay:.2f}s..."
                        )
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"{operation_name}: Rate limit error after {max_retries} attempts")
                        raise
                else:
                    # Not a rate limit error, re-raise immediately
                    logger.error(f"{operation_name}: API error: {e}")
                    raise
            except Exception as e:
                # Unexpected error - log and re-raise
                logger.error(f"{operation_name}: Unexpected error: {e}", exc_info=True)
                raise
    
    def extract_text_from_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract text content from entities for GraphRAG processing with deduplication.
        
        Args:
            entities: Dictionary of entity types to entity lists
        
        Returns:
            List of text dictionaries with metadata (deduplicated by raw text content)
        """
        texts = []
        seen_texts = {}  # Map raw text -> text dict for deduplication
        
        # Extract from Generation entities (prompts, responses, reasoning)
        generations = entities.get("Generation", [])
        for gen in generations:
            prompt = gen.get("prompt", "") or ""
            response = gen.get("response", "") or ""
            reasoning = gen.get("reasoning", "") or ""
            
            if prompt and prompt not in seen_texts:
                text_dict = {
                    "text": prompt,
                    "type": "prompt",
                    "prefix": "Prompt: ",
                    "hash": hashlib.md5(prompt.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[prompt] = text_dict
            if response and response not in seen_texts:
                text_dict = {
                    "text": response,
                    "type": "response",
                    "prefix": "Response: ",
                    "hash": hashlib.md5(response.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[response] = text_dict
            if reasoning and reasoning not in seen_texts:
                text_dict = {
                    "text": reasoning,
                    "type": "reasoning",
                    "prefix": "Reasoning: ",
                    "hash": hashlib.md5(reasoning.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[reasoning] = text_dict
        
        # Extract from Trace entities (input, output)
        traces = entities.get("Trace", [])
        for trace in traces:
            input_text = trace.get("input", "") or ""
            output_text = trace.get("output", "") or ""
            
            if input_text and input_text not in seen_texts:
                text_dict = {
                    "text": input_text,
                    "type": "trace_input",
                    "prefix": "Trace Input: ",
                    "hash": hashlib.md5(input_text.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[input_text] = text_dict
            if output_text and output_text not in seen_texts:
                text_dict = {
                    "text": output_text,
                    "type": "trace_output",
                    "prefix": "Trace Output: ",
                    "hash": hashlib.md5(output_text.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[output_text] = text_dict
        
        # Extract from Error entities (messages)
        errors = entities.get("Error", [])
        for error in errors:
            message = error.get("message", "") or ""
            if message and message not in seen_texts:
                text_dict = {
                    "text": message,
                    "type": "error",
                    "prefix": "Error: ",
                    "hash": hashlib.md5(message.encode()).hexdigest(),
                }
                texts.append(text_dict)
                seen_texts[message] = text_dict
        
        logger.info(f"Extracted {len(texts)} unique text chunks from entities (deduplicated by raw content)")
        return texts
    
    def _chunk_at_boundaries(self, text: str) -> List[str]:
        """Chunk text at semantic boundaries (sentence/paragraph) to maximize deduplication.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks at semantic boundaries
        """
        import re
        
        # Try to split at paragraph boundaries first (double newline)
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If paragraph fits, add it
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too long, split at sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def chunk_texts(self, texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk texts for processing with optimized deduplication.
        
        Chunks are created at semantic boundaries (paragraphs/sentences) to maximize
        deduplication opportunities. Full texts are deduplicated first, then chunks
        are deduplicated by content hash.
        
        Args:
            texts: List of text dictionaries with 'text', 'prefix', 'type', 'hash' keys
        
        Returns:
            List of unique chunk dictionaries with text and metadata
        """
        chunks = []
        chunk_id = 0
        seen_chunks = {}  # Map chunk hash -> chunk dict for deduplication
        
        for text_dict in texts:
            raw_text = text_dict.get("text", "")
            prefix = text_dict.get("prefix", "")
            text_type = text_dict.get("type", "unknown")
            original_hash = text_dict.get("hash")
            
            if not raw_text:
                continue
            
            # Add prefix for context
            full_text = prefix + raw_text if prefix else raw_text
            text_length = len(full_text)
            
            if text_length <= self.chunk_size:
                # Text fits in one chunk - use full text with prefix
                chunk_text = full_text
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                
                if chunk_hash not in seen_chunks:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "start": 0,
                        "end": text_length,
                        "hash": chunk_hash,
                        "type": text_type,
                        "original_hash": original_hash,
                    })
                    seen_chunks[chunk_hash] = chunks[-1]
                    chunk_id += 1
            else:
                # Text is too long - chunk at semantic boundaries
                # Chunk the raw text first (without prefix) to maximize deduplication
                raw_chunks = self._chunk_at_boundaries(raw_text)
                
                for raw_chunk in raw_chunks:
                    # Add prefix to each chunk
                    chunk_text = prefix + raw_chunk if prefix else raw_chunk
                    chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                    
                    if chunk_hash not in seen_chunks:
                        chunks.append({
                            "id": f"chunk_{chunk_id}",
                            "text": chunk_text,
                            "start": 0,  # Start position within original text
                            "end": len(chunk_text),
                            "hash": chunk_hash,
                            "type": text_type,
                            "original_hash": original_hash,
                        })
                        seen_chunks[chunk_hash] = chunks[-1]
                        chunk_id += 1
        
        logger.info(f"Created {len(chunks)} unique text chunks (deduplicated by content hash)")
        return chunks
    
    def _extract_entities_from_chunk(self, chunk: Dict[str, Any], cache: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract entities from a single chunk (used for parallel processing).
        
        Args:
            chunk: Chunk dictionary
            cache: Cache dictionary mapping text hash to entities
        
        Returns:
            List of entity dictionaries
        """
        text = chunk.get("text", "")
        if not text or len(text.strip()) < 10:
            return []
        
        # Check cache first
        text_hash = chunk.get("hash")
        if text_hash and text_hash in cache:
            return cache[text_hash]
        
        # Use LLM to extract entities from text
        prompt = f"""Extract semantic entities from the following text. 
For each entity, identify:
1. Entity name
2. Entity type (concept, topic, intent, action, person, organization, etc.)
3. Brief description

Text: {text[:3000]}

Return a JSON array of entities, each with "name", "type", and "description" fields.
Example format:
[
  {{"name": "error handling", "type": "concept", "description": "Methods for handling errors in code"}},
  {{"name": "API call", "type": "action", "description": "Making requests to external APIs"}}
]

If no entities are found, return an empty array []."""

        # Make API call with rate limit retry
        try:
            response = self._call_openai_with_retry(
                lambda: self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting semantic entities from text. Always return valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config.get("graphrag_llm_temperature", 0.0),
                    max_tokens=self.config.get("graphrag_llm_max_tokens", 4000),
                ),
                operation_name=f"Entity extraction for chunk {chunk.get('id', 'unknown')}"
            )
        except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
            logger.error(f"Failed to extract entities from chunk {chunk.get('id')} after retries: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract entities from chunk {chunk.get('id')}: {e}") from e
        
        if not response or not response.choices or len(response.choices) == 0:
            raise RuntimeError(f"Empty response from OpenAI API for chunk {chunk.get('id')}")
        
        content = response.choices[0].message.content
        if content:
            content = content.strip()
        else:
            content = ""
        
        # Parse JSON response
        try:
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            chunk_entities = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse entity extraction JSON for chunk {chunk.get('id')}: {e}", exc_info=True)
            logger.error(f"Response content: {content[:500]}")
            raise RuntimeError(f"Failed to parse entity extraction JSON for chunk {chunk.get('id')}: {e}") from e
        
        # Cache the result
        if text_hash:
            cache[text_hash] = chunk_entities
        
        return chunk_entities
    
    def extract_semantic_entities(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract semantic entities from text chunks using LLM with parallel processing and caching.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            List of semantic entity dictionaries
        """
        logger.info(f"Extracting semantic entities from {len(chunks)} chunks (parallel processing enabled)")
        
        entities = []
        entity_map = {}  # Track entities by name to avoid duplicates
        cache = {}  # Cache for identical text chunks
        
        # Get max workers from config or use default
        max_workers = self.config.get("graphrag_max_workers", 10)
        
        # Process chunks in parallel batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            logger.debug(f"Processing entity extraction batch {batch_num} ({len(batch)} chunks) with {max_workers} workers")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._extract_entities_from_chunk, chunk, cache): chunk
                    for chunk in batch
                }
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_entities = future.result()
                        
                        # Create entity objects
                        for entity_data in chunk_entities:
                            if not isinstance(entity_data, dict):
                                continue
                            
                            name = entity_data.get("name", "").strip()
                            if not name:
                                continue
                            
                            # Use name as key to avoid duplicates
                            entity_key = f"{entity_data.get('type', 'concept')}:{name.lower()}"
                            
                            if entity_key not in entity_map:
                                entity_id = str(uuid.uuid4())
                                entity = {
                                    "id": entity_id,
                                    "type": entity_data.get("type", "concept"),
                                    "name": name,
                                    "description": entity_data.get("description", ""),
                                }
                                entity_map[entity_key] = entity
                                entities.append(entity)
                            else:
                                # Merge descriptions if entity already exists
                                existing = entity_map[entity_key]
                                existing_desc = existing.get("description", "")
                                new_desc = entity_data.get("description", "")
                                if new_desc and new_desc not in existing_desc:
                                    existing["description"] = f"{existing_desc}. {new_desc}".strip()
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.get('id')}: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to process chunk {chunk.get('id')}: {e}") from e
        
        logger.info(f"Extracted {len(entities)} unique semantic entities (cache hits: {sum(1 for h in cache.keys() if chunks and any(c.get('hash') == h for c in chunks))})")
        return entities
    
    def build_communities(
        self,
        entities: List[Dict[str, Any]],
        entity_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build hierarchical community structure from entities using embedding-based clustering.
        
        Implements the GraphRAG approach: uses entity embeddings to create a hierarchy
        of communities from top-level (broad) to bottom-level (specific).
        
        Args:
            entities: List of semantic entity dictionaries
            entity_embeddings: Optional dictionary mapping entity_id -> embedding vector.
                             If None, will try to retrieve from Neo4j.
        
        Returns:
            Tuple of (communities list, parent-child relationships list)
        """
        logger.info(f"Building hierarchical communities from {len(entities)} entities")
        
        if not entities:
            return [], []
        
        # If embeddings not provided, try to get from Neo4j
        if entity_embeddings is None:
            entity_ids = [e.get("id") for e in entities]
            
            # Get embeddings from Neo4j
            embeddings_query = """
            MATCH (se:SemanticEntity)
            WHERE se.id IN $entity_ids AND se.embedding IS NOT NULL
            RETURN se.id as id, se.embedding as embedding
            """
            
            try:
                embedding_results = self.neo4j_manager.execute_query(
                    embeddings_query,
                    {"entity_ids": entity_ids}
                )
                
                # Create mapping of entity_id -> embedding
                entity_embeddings = {}
                for result in embedding_results:
                    embedding = result.get("embedding")
                    if embedding:
                        entity_embeddings[result["id"]] = embedding
                
                logger.debug(f"Retrieved {len(entity_embeddings)} entity embeddings from Neo4j")
                
            except Exception as e:
                logger.error(f"Could not retrieve embeddings from Neo4j: {e}", exc_info=True)
                raise RuntimeError(f"Failed to retrieve entity embeddings from Neo4j: {e}") from e
        
        # If we have embeddings, use hierarchical clustering
        if entity_embeddings and SKLEARN_AVAILABLE and len(entity_embeddings) >= self.community_min_size:
            return self._build_communities_hierarchical(entities, entity_embeddings)
        else:
            # Fallback to simple grouping if no embeddings or sklearn not available
            logger.info("Using fallback community building (no embeddings or sklearn unavailable)")
            return self._build_communities_fallback(entities)
    
    def _build_communities_hierarchical(
        self,
        entities: List[Dict[str, Any]],
        entity_embeddings: Dict[str, List[float]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build hierarchical communities using AgglomerativeClustering on embeddings.
        
        Creates multiple levels of communities based on similarity thresholds.
        """
        # Filter entities that have embeddings
        entities_with_embeddings = [
            e for e in entities if e.get("id") in entity_embeddings
        ]
        
        if len(entities_with_embeddings) < self.community_min_size:
            return self._build_communities_fallback(entities)
        
        # Prepare embedding matrix
        entity_ids = [e.get("id") for e in entities_with_embeddings]
        embedding_matrix = np.array([
            entity_embeddings[eid] 
            for eid in entity_ids 
            if eid in entity_embeddings and entity_embeddings[eid] is not None
        ])
        
        # Filter entity_ids to match embedding_matrix
        valid_entity_ids = [
            eid for eid in entity_ids 
            if eid in entity_embeddings and entity_embeddings[eid] is not None
        ]
        
        if len(valid_entity_ids) < self.community_min_size:
            return self._build_communities_fallback(entities)
        
        # Update entities_with_embeddings to match valid IDs
        entities_with_embeddings = [
            e for e in entities_with_embeddings 
            if e.get("id") in valid_entity_ids
        ]
        entity_ids = valid_entity_ids
        
        communities = []
        parent_child_rels = []
        max_levels = self.community_max_levels
        
        # Create communities at multiple levels using different distance thresholds
        # Level 0 = most general (fewer, larger communities)
        # Higher levels = more specific (more, smaller communities)
        
        # Use distance thresholds that create progressively finer clusters
        # Start with larger distance (more permissive) for level 0, then decrease
        distance_thresholds = [0.7, 0.5, 0.3]  # Cosine distance thresholds
        
        level_communities = {}  # level -> list of (community, entity_indices)
        
        for level in range(max_levels):
            if level >= len(distance_thresholds):
                break
            
            distance_threshold = distance_thresholds[level]
            
            # Perform hierarchical clustering
            if not SKLEARN_AVAILABLE or AgglomerativeClustering is None:
                logger.warning("scikit-learn not available, skipping hierarchical clustering")
                break
            
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage='average',
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(embedding_matrix)
            
            # Create communities for each cluster
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(idx)
            
            level_communities[level] = []
            
            for cluster_id, entity_indices in clusters.items():
                if len(entity_indices) < self.community_min_size:
                    continue
                
                cluster_entities = [entities_with_embeddings[i] for i in entity_indices]
                
                # Determine community name from entity types and names
                entity_types = [e.get("type", "concept") for e in cluster_entities]
                most_common_type = max(set(entity_types), key=entity_types.count)
                
                # Get representative entity names
                entity_names = [e.get("name", "") for e in cluster_entities[:5]]
                name_prefix = entity_names[0].split()[0] if entity_names else "entities"
                
                community = {
                    "id": str(uuid.uuid4()),
                    "name": f"{most_common_type.title()}: {name_prefix.title()}",
                    "level": level,
                    "summary": f"Community of {len(cluster_entities)} related {most_common_type} entities",
                    "_entity_indices": entity_indices,  # Temporary, for parent-child linking
                    "_entity_ids": [entity_ids[i] for i in entity_indices],
                }
                
                communities.append(community)
                level_communities[level].append((community, entity_indices))
            
            logger.info(f"Created {len(level_communities[level])} communities at level {level}")
        
        # Create parent-child relationships between levels
        # A parent community contains all entities that its child communities contain
        for level in range(1, max_levels):
            if level not in level_communities or (level - 1) not in level_communities:
                continue
            
            parent_communities = level_communities[level - 1]
            child_communities = level_communities[level]
            
            for child_comm, child_indices in child_communities:
                # Find parent community that contains most of the child's entities
                best_parent = None
                best_overlap = 0
                
                for parent_comm, parent_indices in parent_communities:
                    # Check overlap: how many child entities are in parent?
                    overlap = len(set(child_indices) & set(parent_indices))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent_comm
                
                # Create CONTAINS relationship if significant overlap
                if best_parent and best_overlap >= len(child_indices) * 0.5:  # At least 50% overlap
                    parent_child_rels.append({
                        "type": "CONTAINS",
                        "from_type": "Community",
                        "from_id": best_parent.get("id"),
                        "to_type": "Community",
                        "to_id": child_comm.get("id"),
                        "properties": {},
                    })
        
        # Remove temporary fields
        for comm in communities:
            comm.pop("_entity_indices", None)
            comm.pop("_entity_ids", None)
        
        logger.info(f"Created {len(communities)} communities across {max_levels} levels with {len(parent_child_rels)} parent-child relationships")
        return communities, parent_child_rels
    
    def _build_communities_fallback(
        self,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fallback community building using simple grouping (original method)."""
        communities = []
        
        # Group entities by type first
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type", "concept")
            entities_by_type[entity_type].append(entity)
        
        # Create communities for each entity type
        for entity_type, type_entities in entities_by_type.items():
            if len(type_entities) < self.community_min_size:
                # Too few entities, create a single community
                if len(type_entities) > 0:
                    community = {
                        "id": str(uuid.uuid4()),
                        "name": f"{entity_type.title()} Community",
                        "level": 0,
                        "summary": f"Collection of {len(type_entities)} {entity_type} entities",
                    }
                    communities.append(community)
            else:
                # Create multiple communities by clustering similar entities
                # Simple approach: group by name similarity (first word/prefix)
                name_groups = defaultdict(list)
                for entity in type_entities:
                    name = entity.get("name", "")
                    # Use first significant word as grouping key
                    first_word = name.split()[0].lower() if name else "other"
                    name_groups[first_word].append(entity)
                
                # Create communities for each group
                for group_key, group_entities in name_groups.items():
                    if len(group_entities) >= self.community_min_size:
                        community = {
                            "id": str(uuid.uuid4()),
                            "name": f"{entity_type.title()}: {group_key.title()}",
                            "level": 0,
                            "summary": f"Community of {len(group_entities)} related {entity_type} entities",
                        }
                        communities.append(community)
                    else:
                        # Merge small groups into a general community
                        if "general" not in [c.get("name", "").lower() for c in communities if c.get("level") == 0]:
                            community = {
                                "id": str(uuid.uuid4()),
                                "name": f"{entity_type.title()}: General",
                                "level": 0,
                                "summary": f"General {entity_type} entities",
                            }
                            communities.append(community)
        
        logger.info(f"Created {len(communities)} communities (fallback method)")
        return communities, []
    
    def _assign_entities_to_communities(
        self,
        entities: List[Dict[str, Any]],
        communities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Assign entities to communities using embedding similarity.
        
        Uses cosine similarity between entity embeddings and community embeddings
        to determine which community each entity belongs to. Entities are assigned
        to the most similar community at the lowest level (most specific).
        
        Args:
            entities: List of semantic entity dictionaries
            communities: List of community dictionaries
        
        Returns:
            List of BELONGS_TO relationship dictionaries
        """
        if not entities or not communities:
            return []
        
        # Get entity embeddings from Neo4j
        entity_ids = [e.get("id") for e in entities]
        entity_emb_query = """
        MATCH (se:SemanticEntity)
        WHERE se.id IN $entity_ids AND se.embedding IS NOT NULL
        RETURN se.id as id, se.embedding as embedding
        """
        
        try:
            entity_emb_results = self.neo4j_manager.execute_query(
                entity_emb_query,
                {"entity_ids": entity_ids}
            )
            entity_embeddings = {r["id"]: r["embedding"] for r in entity_emb_results if r.get("embedding")}
        except Exception as e:
            logger.error(f"Could not retrieve entity embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve entity embeddings: {e}") from e
        
        # Get community embeddings from Neo4j
        community_ids = [c.get("id") for c in communities]
        comm_emb_query = """
        MATCH (c:Community)
        WHERE c.id IN $community_ids AND c.embedding IS NOT NULL
        RETURN c.id as id, c.embedding as embedding, c.level as level
        """
        
        try:
            comm_emb_results = self.neo4j_manager.execute_query(
                comm_emb_query,
                {"community_ids": community_ids}
            )
            community_embeddings = {
                r["id"]: {"embedding": r["embedding"], "level": r.get("level", 0)}
                for r in comm_emb_results if r.get("embedding")
            }
        except Exception as e:
            logger.error(f"Could not retrieve community embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve community embeddings: {e}") from e
        
        relationships = []
        
        # If we have embeddings, use similarity-based assignment
        if entity_embeddings and community_embeddings and SKLEARN_AVAILABLE and cosine_similarity is not None:
            for entity in entities:
                entity_id = entity.get("id")
                entity_emb = entity_embeddings.get(entity_id)
                
                if not entity_emb:
                    continue
                
                # Find most similar community, preferring lower levels (more specific)
                best_community = None
                best_similarity = -1
                
                for community in communities:
                    comm_id = community.get("id")
                    comm_data = community_embeddings.get(comm_id)
                    
                    if not comm_data:
                        continue
                    
                    comm_emb = comm_data["embedding"]
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [entity_emb],
                        [comm_emb]
                    )[0][0]
                    
                    # Prefer communities at lower levels (more specific) if similarity is close
                    if similarity > best_similarity or (
                        similarity > best_similarity * 0.9 and 
                        comm_data["level"] > (community_embeddings.get(best_community.get("id"), {}).get("level", 0) if best_community else 0)
                    ):
                        best_similarity = similarity
                        best_community = community
                
                # Assign to best community if similarity is above threshold
                if best_community and best_similarity > 0.3:  # Similarity threshold
                    relationships.append({
                        "type": "BELONGS_TO",
                        "from_type": "SemanticEntity",
                        "from_id": entity_id,
                        "to_type": "Community",
                        "to_id": best_community.get("id"),
                        "properties": {"similarity": float(best_similarity)},
                    })
        else:
            # Fallback: use name/type matching
            for community in communities:
                community_id = community.get("id")
                community_name = community.get("name", "").lower()
                
                for entity in entities:
                    entity_type = entity.get("type", "").lower()
                    entity_name = entity.get("name", "").lower()
                    
                    if community_name and entity_type:
                        if entity_type in community_name or any(word in community_name for word in entity_name.split()[:2]):
                            relationships.append({
                                "type": "BELONGS_TO",
                                "from_type": "SemanticEntity",
                                "from_id": entity.get("id"),
                                "to_type": "Community",
                                "to_id": community_id,
                                "properties": {},
                            })
        
        return relationships
    
    def generate_summaries(
        self,
        communities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate summaries for communities using LLM (bottom-up approach).
        
        Implements GraphRAG bottom-up summarization: generates summaries starting
        from the lowest level (most specific) and moving up to higher levels.
        Higher-level summaries incorporate information from lower-level summaries.

        Args:
            communities: List of community dictionaries
            chunks: List of text chunks

        Returns:
            List of communities with summaries
        """
        logger.info(f"Generating summaries for {len(communities)} communities (bottom-up)")

        # Sort communities by level (lowest first for bottom-up summarization)
        communities_by_level = defaultdict(list)
        for community in communities:
            level = community.get("level", 0)
            communities_by_level[level].append(community)
        
        max_level = max(communities_by_level.keys()) if communities_by_level else 0
        
        # Process from bottom (highest level) to top (level 0)
        # This allows higher-level summaries to reference lower-level ones
        for level in range(max_level, -1, -1):
            if level not in communities_by_level:
                continue
            
            level_communities = communities_by_level[level]
            logger.debug(f"Generating summaries for {len(level_communities)} communities at level {level}")
            
            # Get relevant chunks for each community based on community name
            for community in level_communities:
                if "summary" in community and community.get("summary"):
                    # Already has a summary, skip
                    continue
                
                community_name = community.get("name", "")
                community_type = community_name.split(":")[0].lower() if ":" in community_name else ""
                
                # Find relevant chunks that mention this community type
                relevant_chunks = []
                for chunk in chunks:
                    text = chunk.get("text", "")
                    if text:
                        text = text.lower()
                        if community_type and community_type in text:
                            relevant_chunks.append(chunk)
                        elif not community_type and len(relevant_chunks) < 5:
                            # If no specific type, take first few chunks
                            relevant_chunks.append(chunk)
                
                # Generate summary using LLM
                if relevant_chunks:
                    # Combine relevant chunk texts
                    combined_text = "\n\n".join([
                        chunk.get("text", "")[:500] 
                        for chunk in relevant_chunks[:5]  # Limit to 5 chunks
                    ])
                    
                    prompt = f"""Generate a concise summary (2-3 sentences) for the following community of related concepts/topics.

Community Name: {community_name}

Relevant Text Excerpts:
{combined_text[:2000]}

Provide a summary that describes what this community represents and what concepts/topics it contains."""

                    try:
                        response = self._call_openai_with_retry(
                            lambda: self.openai_client.chat.completions.create(
                                model=self.llm_model,
                                messages=[
                                    {"role": "system", "content": "You are an expert at creating concise summaries of related concepts."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=self.config.get("graphrag_llm_temperature", 0.0),
                                max_tokens=200,
                            ),
                            operation_name=f"Summary generation for community '{community_name}'"
                        )
                    except (RateLimitError, APIError, APIConnectionError, APITimeoutError) as e:
                        logger.error(f"Failed to generate summary for community '{community_name}' after retries: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to generate summary for community '{community_name}': {e}") from e
                    
                    if not response or not response.choices or len(response.choices) == 0:
                        raise RuntimeError(f"Empty response from OpenAI API for community '{community_name}'")
                    
                    summary_text = response.choices[0].message.content
                    if summary_text:
                        community["summary"] = summary_text.strip()
                    else:
                        community["summary"] = f"Community: {community_name}"
                else:
                    # No relevant chunks, use default summary
                    community["summary"] = f"Community: {community_name}"
        
        logger.info(f"Generated summaries for {len(communities)} communities")
        return communities
    
    def index(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run full GraphRAG indexing pipeline.
        
        Args:
            entities: Dictionary of entity types to entity lists
        
        Returns:
            Dictionary with extracted semantic entities and communities
        """
        logger.info("Starting GraphRAG indexing pipeline")
        
        # Step 1: Extract text from entities
        texts = self.extract_text_from_entities(entities)
        
        if not texts:
            raise ValueError(
                "No text content found in entities for GraphRAG indexing. "
                "Ensure entities contain text fields like 'prompt', 'response', 'reasoning', 'input', 'output', or 'message'."
            )
        
        # Step 2: Chunk texts
        chunks = self.chunk_texts(texts)
        
        # Step 3: Extract semantic entities
        semantic_entities = self.extract_semantic_entities(chunks)
        
        # Step 4: Generate embeddings for entities IN MEMORY (needed for clustering)
        # We'll store them in Neo4j after communities are built
        entity_embeddings_dict = {}
        if semantic_entities:
            logger.info(f"Generating embeddings for {len(semantic_entities)} semantic entities (for clustering)")
            # Generate embeddings in batches
            entity_texts = []
            entity_ids = []
            for entity in semantic_entities:
                # Create text representation for embedding
                entity_text = f"{entity.get('name', '')} {entity.get('description', '')} {entity.get('type', '')}"
                entity_texts.append(entity_text)
                entity_ids.append(entity.get("id"))
            
            # Generate embeddings in batches
            for i in range(0, len(entity_texts), self.embedding_generator.batch_size):
                batch_texts = entity_texts[i:i + self.embedding_generator.batch_size]
                batch_ids = entity_ids[i:i + self.embedding_generator.batch_size]
                
                try:
                    batch_embeddings = self.embedding_generator.generate_embeddings_batch(batch_texts)
                    for entity_id, embedding in zip(batch_ids, batch_embeddings):
                        if embedding:  # Only store non-empty embeddings
                            entity_embeddings_dict[entity_id] = embedding
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to generate embeddings for entity batch: {e}") from e
        
        # Step 5: Store entities in Neo4j
        if semantic_entities:
            logger.info(f"Storing {len(semantic_entities)} semantic entities in Neo4j")
            self.neo4j_manager.create_nodes({"SemanticEntity": semantic_entities})
            
            # Store embeddings in Neo4j
            if entity_embeddings_dict:
                logger.info(f"Storing embeddings for {len(entity_embeddings_dict)} entities")
                for entity_id, embedding in entity_embeddings_dict.items():
                    self.neo4j_manager.update_node_embedding("SemanticEntity", entity_id, "embedding", embedding)
        
        # Step 6: Build hierarchical communities (using in-memory embeddings)
        communities, community_relationships = self.build_communities(semantic_entities, entity_embeddings_dict)
        
        # Step 7: Generate summaries (bottom-up: lowest level first)
        communities = self.generate_summaries(communities, chunks)
        
        if communities:
            logger.info(f"Storing {len(communities)} communities in Neo4j")
            self.neo4j_manager.create_nodes({"Community": communities})
            
            # Generate embeddings for communities
            self.embedding_generator.generate_and_store_community_embeddings(communities)
            
            # Create parent-child CONTAINS relationships between communities
            if community_relationships:
                logger.info(f"Creating {len(community_relationships)} CONTAINS relationships between communities")
                self.neo4j_manager.create_relationships(community_relationships)
            
            # Create BELONGS_TO relationships from entities to communities
            # Use embedding similarity to assign entities to communities
            entity_community_rels = self._assign_entities_to_communities(semantic_entities, communities)
            
            if entity_community_rels:
                logger.info(f"Creating {len(entity_community_rels)} BELONGS_TO relationships")
                self.neo4j_manager.create_relationships(entity_community_rels)
        
        # Step 7: Create relationships between semantic entities and original entities
        entity_rels = self.create_entity_relationships(semantic_entities, entities)
        if entity_rels:
            logger.info(f"Creating {len(entity_rels)} entity relationships")
            self.neo4j_manager.create_relationships(entity_rels)
        
        logger.info("GraphRAG indexing pipeline completed")
        
        return {
            "SemanticEntity": semantic_entities,
            "Community": communities,
        }
    
    def create_entity_relationships(
        self,
        semantic_entities: List[Dict[str, Any]],
        original_entities: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Create relationships between semantic entities and original entities.
        
        Args:
            semantic_entities: List of semantic entity dictionaries
            original_entities: Original entity dictionary
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Create MENTIONS relationships from Generation to SemanticEntity
        generations = original_entities.get("Generation", [])
        
        # Build a lookup map for semantic entities by name (case-insensitive)
        semantic_entity_map = {}
        for entity in semantic_entities:
            name_lower = entity.get("name", "").lower()
            if name_lower:
                if name_lower not in semantic_entity_map:
                    semantic_entity_map[name_lower] = []
                semantic_entity_map[name_lower].append(entity)
        
        # Check each generation for mentions of semantic entities
        for generation in generations:
            gen_id = generation.get("id")
            if not gen_id:
                continue
            
            # Check prompt, response, and reasoning for entity mentions
            texts_to_check = [
                generation.get("prompt", ""),
                generation.get("response", ""),
                generation.get("reasoning", ""),
            ]
            
            combined_text = " ".join([t for t in texts_to_check if t]).lower()
            
            # Find mentioned entities
            mentioned_entity_ids = set()
            for entity_name, entities in semantic_entity_map.items():
                # Check if entity name appears in text (simple substring match)
                if entity_name in combined_text:
                    mentioned_entity_ids.update(e["id"] for e in entities)
            
            # Create MENTIONS relationships
            for entity in semantic_entities:
                if entity["id"] in mentioned_entity_ids:
                    relationships.append({
                        "type": "MENTIONS",
                        "from_type": "Generation",
                        "from_id": gen_id,
                        "to_type": "SemanticEntity",
                        "to_id": entity.get("id"),
                        "properties": {},
                    })
        
        # Create ABOUT relationships from Trace to SemanticEntity
        traces = original_entities.get("Trace", [])
        for trace in traces:
            trace_id = trace.get("id")
            if not trace_id:
                continue
            
            # Check trace input/output for entity mentions
            trace_text = " ".join([
                trace.get("input") or "",
                trace.get("output") or "",
            ]).lower()
            
            mentioned_entity_ids = set()
            for entity_name, entities in semantic_entity_map.items():
                if entity_name in trace_text:
                    mentioned_entity_ids.update(e["id"] for e in entities)
            
            # Create ABOUT relationships (limit to top 3 most relevant)
            mentioned_entities_list = [e for e in semantic_entities if e["id"] in mentioned_entity_ids]
            for entity in mentioned_entities_list[:3]:
                relationships.append({
                    "type": "ABOUT",
                    "from_type": "Trace",
                    "from_id": trace_id,
                    "to_type": "SemanticEntity",
                    "to_id": entity.get("id"),
                    "properties": {},
                })
        
        # Create BELONGS_TO relationships from SemanticEntity to Community
        # This will be done after communities are created and stored
        
        logger.info(f"Created {len(relationships)} entity relationships")
        return relationships


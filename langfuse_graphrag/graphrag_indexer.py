"""Microsoft GraphRAG integration for semantic extraction.

Uses GraphRAG to extract semantic entities, build communities, and generate summaries
from trace data.
"""

import logging
import json
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

from langfuse_graphrag.config import get_config
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


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
        self.llm_model = config.get("graphrag_llm_model", "gpt-4o-mini")
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
    
    def extract_text_from_entities(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Extract text content from entities for GraphRAG processing.
        
        Args:
            entities: Dictionary of entity types to entity lists
        
        Returns:
            List of text chunks
        """
        texts = []
        
        # Extract from Generation entities (prompts, responses, reasoning)
        generations = entities.get("Generation", [])
        for gen in generations:
            prompt = gen.get("prompt", "") or ""
            response = gen.get("response", "") or ""
            reasoning = gen.get("reasoning", "") or ""
            
            if prompt:
                texts.append(f"Prompt: {prompt}")
            if response:
                texts.append(f"Response: {response}")
            if reasoning:
                texts.append(f"Reasoning: {reasoning}")
        
        # Extract from Trace entities (input, output)
        traces = entities.get("Trace", [])
        for trace in traces:
            input_text = trace.get("input", "") or ""
            output_text = trace.get("output", "") or ""
            
            if input_text:
                texts.append(f"Trace Input: {input_text}")
            if output_text:
                texts.append(f"Trace Output: {output_text}")
        
        # Extract from Error entities (messages)
        errors = entities.get("Error", [])
        for error in errors:
            message = error.get("message", "") or ""
            if message:
                texts.append(f"Error: {message}")
        
        logger.info(f"Extracted {len(texts)} text chunks from entities")
        return texts
    
    def chunk_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Chunk texts for processing.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        chunk_id = 0
        
        for text in texts:
            # Simple chunking by character count with overlap
            text_length = len(text)
            if text_length <= self.chunk_size:
                chunks.append({
                    "id": f"chunk_{chunk_id}",
                    "text": text,
                    "start": 0,
                    "end": text_length,
                })
                chunk_id += 1
            else:
                # Split into overlapping chunks
                start = 0
                while start < text_length:
                    end = min(start + self.chunk_size, text_length)
                    chunk_text = text[start:end]
                    
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "start": start,
                        "end": end,
                    })
                    chunk_id += 1
                    
                    # Move start with overlap, but ensure progress
                    next_start = end - self.chunk_overlap
                    if next_start <= start:
                        # Ensure we always make progress
                        next_start = start + 1
                    start = next_start
                    
                    if start >= text_length:
                        break
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def extract_semantic_entities(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract semantic entities from text chunks using LLM.
        
        Args:
            chunks: List of text chunks
        
        Returns:
            List of semantic entity dictionaries
        """
        logger.info(f"Extracting semantic entities from {len(chunks)} chunks")
        
        entities = []
        entity_map = {}  # Track entities by name to avoid duplicates
        
        # Process chunks in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            logger.debug(f"Processing entity extraction batch {i // self.batch_size + 1} ({len(batch)} chunks)")
            
            for chunk in batch:
                text = chunk.get("text", "")
                if not text or len(text.strip()) < 10:
                    continue
                
                try:
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

                    response = self.openai_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at extracting semantic entities from text. Always return valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.get("graphrag_llm_temperature", 0.0),
                        max_tokens=self.config.get("graphrag_llm_max_tokens", 4000),
                    )
                    
                    content = response.choices[0].message.content.strip()
                    
                    # Parse JSON response
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    chunk_entities = json.loads(content)
                    
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
                
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse entity extraction JSON: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error extracting entities from chunk: {e}")
                    continue
        
        logger.info(f"Extracted {len(entities)} unique semantic entities")
        return entities
    
    def build_communities(
        self,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build community hierarchy from entities using clustering.
        
        Args:
            entities: List of semantic entity dictionaries
        
        Returns:
            List of community dictionaries
        """
        logger.info(f"Building communities from {len(entities)} entities")
        
        if not entities:
            return []
        
        communities = []
        
        # Group entities by type first
        entities_by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type", "concept")
            entities_by_type[entity_type].append(entity)
        
        # Create communities for each entity type
        community_id = 0
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
        
        logger.info(f"Created {len(communities)} communities")
        return communities
    
    def generate_summaries(
        self,
        communities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate summaries for communities using LLM.
        
        Args:
            communities: List of community dictionaries
            chunks: List of text chunks
        
        Returns:
            List of communities with summaries
        """
        logger.info(f"Generating summaries for {len(communities)} communities")
        
        # Get relevant chunks for each community based on community name
        for community in communities:
            if "summary" in community and community["summary"]:
                # Already has a summary, skip
                continue
            
            community_name = community.get("name", "")
            community_type = community_name.split(":")[0].lower() if ":" in community_name else ""
            
            # Find relevant chunks that mention this community type
            relevant_chunks = []
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                if community_type and community_type in text:
                    relevant_chunks.append(chunk)
                elif not community_type and len(relevant_chunks) < 5:
                    # If no specific type, take first few chunks
                    relevant_chunks.append(chunk)
            
            # Generate summary using LLM
            if relevant_chunks:
                try:
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

                    response = self.openai_client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "You are an expert at creating concise summaries of related concepts."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.get("graphrag_llm_temperature", 0.0),
                        max_tokens=200,
                    )
                    
                    summary = response.choices[0].message.content.strip()
                    community["summary"] = summary
                    
                except Exception as e:
                    logger.warning(f"Error generating summary for community {community_name}: {e}")
                    # Fallback to default summary
                    community["summary"] = f"Community containing related {community_type or 'entities'}"
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
            logger.warning("No text found in entities for indexing")
            return {"SemanticEntity": [], "Community": []}
        
        # Step 2: Chunk texts
        chunks = self.chunk_texts(texts)
        
        # Step 3: Extract semantic entities
        semantic_entities = self.extract_semantic_entities(chunks)
        
        # Step 4: Build communities
        communities = self.build_communities(semantic_entities)
        
        # Step 5: Generate summaries
        communities = self.generate_summaries(communities, chunks)
        
        # Step 6: Store in Neo4j
        if semantic_entities:
            logger.info(f"Storing {len(semantic_entities)} semantic entities in Neo4j")
            self.neo4j_manager.create_nodes({"SemanticEntity": semantic_entities})
            
            # Generate embeddings for semantic entities
            self.embedding_generator.generate_and_store_semantic_entity_embeddings(semantic_entities)
        
        if communities:
            logger.info(f"Storing {len(communities)} communities in Neo4j")
            self.neo4j_manager.create_nodes({"Community": communities})
            
            # Generate embeddings for communities
            self.embedding_generator.generate_and_store_community_embeddings(communities)
            
            # Create BELONGS_TO relationships from entities to communities
            entity_community_rels = []
            for community in communities:
                community_id = community.get("id")
                community_name = community.get("name", "").lower()
                
                # Match entities to communities based on type/name similarity
                for entity in semantic_entities:
                    entity_type = entity.get("type", "").lower()
                    entity_name = entity.get("name", "").lower()
                    
                    # Check if entity belongs to this community
                    if community_name and entity_type:
                        # Simple matching: if community name contains entity type or vice versa
                        if entity_type in community_name or any(word in community_name for word in entity_name.split()[:2]):
                            entity_community_rels.append({
                                "type": "BELONGS_TO",
                                "from_type": "SemanticEntity",
                                "from_id": entity.get("id"),
                                "to_type": "Community",
                                "to_id": community_id,
                                "properties": {},
                            })
            
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


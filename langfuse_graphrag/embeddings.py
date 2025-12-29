"""Embedding generation and storage for GraphRAG system.

Generates embeddings using OpenAI and stores them in Neo4j as vector types.
"""

import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI

from langfuse_graphrag.config import get_config
from langfuse_graphrag.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates and stores embeddings in Neo4j."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, neo4j_manager: Optional[Neo4jManager] = None):
        """Initialize embedding generator.
        
        Args:
            config: Optional configuration dictionary. If None, uses default config.
            neo4j_manager: Optional Neo4jManager instance. If None, creates new one.
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.neo4j_manager = neo4j_manager or Neo4jManager(config)
        
        # Initialize OpenAI client
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings")
        
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("embedding_model", "text-embedding-3-small")
        self.batch_size = config.get("embedding_batch_size", 100)
        
        logger.info(f"Initialized EmbeddingGenerator with model: {self.model}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text[:8000]  # Limit text length
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t[:8000] if t else "" for t in texts]
        valid_indices = [i for i, t in enumerate(valid_texts) if t.strip()]
        valid_texts_filtered = [valid_texts[i] for i in valid_indices]
        
        if not valid_texts_filtered:
            logger.warning("No valid texts for embedding")
            return [[] for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts_filtered
            )
            
            # Map embeddings back to original indices
            embeddings = [[] for _ in texts]
            for idx, valid_idx in enumerate(valid_indices):
                embeddings[valid_idx] = response.data[idx].embedding
            
            logger.debug(f"Generated {len(valid_texts_filtered)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
            raise
    
    def generate_and_store_generation_embeddings(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Generate and store embeddings for Generation entities.
        
        Args:
            entities: Dictionary of entity types to entity lists
        """
        generations = entities.get("Generation", [])
        if not generations:
            logger.info("No Generation entities to embed")
            return
        
        logger.info(f"Generating embeddings for {len(generations)} Generation entities")
        
        # Collect texts to embed
        prompt_texts = []
        response_texts = []
        reasoning_texts = []
        generation_ids = []
        
        for gen in generations:
            gen_id = gen.get("id")
            if not gen_id:
                continue
            
            generation_ids.append(gen_id)
            prompt_texts.append(gen.get("prompt", "") or "")
            response_texts.append(gen.get("response", "") or "")
            reasoning_texts.append(gen.get("reasoning", "") or "")
        
        # Generate embeddings in batches
        total_processed = 0
        
        for i in range(0, len(generation_ids), self.batch_size):
            batch_ids = generation_ids[i:i + self.batch_size]
            batch_prompts = prompt_texts[i:i + self.batch_size]
            batch_responses = response_texts[i:i + self.batch_size]
            batch_reasonings = reasoning_texts[i:i + self.batch_size]
            
            logger.debug(f"Processing embedding batch {i // self.batch_size + 1} ({len(batch_ids)} items)")
            
            # Generate embeddings
            prompt_embeddings = self.generate_embeddings_batch(batch_prompts)
            response_embeddings = self.generate_embeddings_batch(batch_responses)
            reasoning_embeddings = self.generate_embeddings_batch(batch_reasonings)
            
            # Store embeddings
            for j, gen_id in enumerate(batch_ids):
                if prompt_embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "Generation",
                        gen_id,
                        "prompt_embedding",
                        prompt_embeddings[j]
                    )
                
                if response_embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "Generation",
                        gen_id,
                        "response_embedding",
                        response_embeddings[j]
                    )
                
                if reasoning_embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "Generation",
                        gen_id,
                        "reasoning_embedding",
                        reasoning_embeddings[j]
                    )
            
            total_processed += len(batch_ids)
            logger.debug(f"Processed {total_processed}/{len(generation_ids)} generations")
        
        logger.info(f"Completed embedding generation for {total_processed} Generation entities")
    
    def generate_and_store_semantic_entity_embeddings(
        self,
        entities: List[Dict[str, Any]]
    ) -> None:
        """Generate and store embeddings for SemanticEntity nodes.
        
        Args:
            entities: List of SemanticEntity dictionaries
        """
        if not entities:
            logger.info("No SemanticEntity entities to embed")
            return
        
        logger.info(f"Generating embeddings for {len(entities)} SemanticEntity entities")
        
        # Collect texts to embed (combine name and description)
        texts = []
        entity_ids = []
        
        for entity in entities:
            entity_id = entity.get("id")
            if not entity_id:
                continue
            
            entity_ids.append(entity_id)
            name = entity.get("name", "") or ""
            description = entity.get("description", "") or ""
            text = f"{name}. {description}".strip()
            texts.append(text)
        
        # Generate embeddings in batches
        total_processed = 0
        
        for i in range(0, len(entity_ids), self.batch_size):
            batch_ids = entity_ids[i:i + self.batch_size]
            batch_texts = texts[i:i + self.batch_size]
            
            logger.debug(f"Processing SemanticEntity embedding batch {i // self.batch_size + 1} ({len(batch_ids)} items)")
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(batch_texts)
            
            # Store embeddings
            for j, entity_id in enumerate(batch_ids):
                if embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "SemanticEntity",
                        entity_id,
                        "embedding",
                        embeddings[j]
                    )
            
            total_processed += len(batch_ids)
            logger.debug(f"Processed {total_processed}/{len(entity_ids)} semantic entities")
        
        logger.info(f"Completed embedding generation for {total_processed} SemanticEntity entities")
    
    def generate_and_store_error_embeddings(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Generate and store embeddings for Error entities.
        
        Args:
            entities: Dictionary of entity types to entity lists
        """
        errors = entities.get("Error", [])
        if not errors:
            logger.info("No Error entities to embed")
            return
        
        logger.info(f"Generating embeddings for {len(errors)} Error entities")
        
        # Collect error messages
        texts = []
        error_ids = []
        
        for error in errors:
            error_id = error.get("id")
            if not error_id:
                continue
            
            error_ids.append(error_id)
            message = error.get("message", "") or ""
            texts.append(message)
        
        # Generate embeddings in batches
        total_processed = 0
        
        for i in range(0, len(error_ids), self.batch_size):
            batch_ids = error_ids[i:i + self.batch_size]
            batch_texts = texts[i:i + self.batch_size]
            
            logger.debug(f"Processing Error embedding batch {i // self.batch_size + 1} ({len(batch_ids)} items)")
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(batch_texts)
            
            # Store embeddings
            for j, error_id in enumerate(batch_ids):
                if embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "Error",
                        error_id,
                        "message_embedding",
                        embeddings[j]
                    )
            
            total_processed += len(batch_ids)
            logger.debug(f"Processed {total_processed}/{len(error_ids)} errors")
        
        logger.info(f"Completed embedding generation for {total_processed} Error entities")
    
    def generate_and_store_community_embeddings(
        self,
        entities: List[Dict[str, Any]]
    ) -> None:
        """Generate and store embeddings for Community nodes.
        
        Args:
            entities: List of Community dictionaries
        """
        if not entities:
            logger.info("No Community entities to embed")
            return
        
        logger.info(f"Generating embeddings for {len(entities)} Community entities")
        
        # Collect summary texts
        texts = []
        community_ids = []
        
        for community in entities:
            community_id = community.get("id")
            if not community_id:
                continue
            
            community_ids.append(community_id)
            summary = community.get("summary", "") or ""
            texts.append(summary)
        
        # Generate embeddings in batches
        total_processed = 0
        
        for i in range(0, len(community_ids), self.batch_size):
            batch_ids = community_ids[i:i + self.batch_size]
            batch_texts = texts[i:i + self.batch_size]
            
            logger.debug(f"Processing Community embedding batch {i // self.batch_size + 1} ({len(batch_ids)} items)")
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(batch_texts)
            
            # Store embeddings
            for j, community_id in enumerate(batch_ids):
                if embeddings[j]:
                    self.neo4j_manager.update_node_embedding(
                        "Community",
                        community_id,
                        "embedding",
                        embeddings[j]
                    )
            
            total_processed += len(batch_ids)
            logger.debug(f"Processed {total_processed}/{len(community_ids)} communities")
        
        logger.info(f"Completed embedding generation for {total_processed} Community entities")


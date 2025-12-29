"""Rebuild hierarchical communities from existing semantic entities.

This script rebuilds communities using hierarchical clustering without
clearing the database. It:
1. Retrieves all semantic entities and their embeddings from Neo4j
2. Rebuilds communities using hierarchical clustering (now that sklearn is installed)
3. Updates communities in Neo4j, preserving existing data
"""

import logging
import sys
from langfuse_graphrag.config import get_config, validate_config
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.graphrag_indexer import GraphRAGIndexer
from langfuse_graphrag.embeddings import EmbeddingGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Rebuild hierarchical communities."""
    logger.info("Starting community rebuild process")
    
    try:
        config = get_config()
        validate_config(config)
        
        # Initialize managers
        neo4j_manager = Neo4jManager(config)
        embedding_generator = EmbeddingGenerator(config, neo4j_manager)
        graphrag_indexer = GraphRAGIndexer(config, neo4j_manager, embedding_generator)
        
        # Get all semantic entities from Neo4j
        logger.info("Retrieving semantic entities from Neo4j")
        query = """
        MATCH (se:SemanticEntity)
        WHERE se.embedding IS NOT NULL
        RETURN se.id as id, se.type as type, se.name as name, 
               se.description as description, se.embedding as embedding
        """
        
        results = neo4j_manager.execute_query(query)
        semantic_entities = []
        entity_embeddings = {}
        
        for result in results:
            entity = {
                "id": result["id"],
                "type": result.get("type", "concept"),
                "name": result.get("name", ""),
                "description": result.get("description", ""),
            }
            semantic_entities.append(entity)
            if result.get("embedding"):
                entity_embeddings[result["id"]] = result["embedding"]
        
        logger.info(f"Retrieved {len(semantic_entities)} semantic entities with {len(entity_embeddings)} embeddings")
        
        if len(semantic_entities) < graphrag_indexer.community_min_size:
            logger.warning(f"Not enough entities ({len(semantic_entities)}) for hierarchical clustering. Minimum: {graphrag_indexer.community_min_size}")
            return
        
        # Delete old communities and their relationships
        logger.info("Deleting old communities and relationships")
        delete_query = """
        MATCH (c:Community)
        DETACH DELETE c
        """
        neo4j_manager.execute_query(delete_query)
        logger.info("Deleted old communities")
        
        # Rebuild communities using hierarchical clustering
        logger.info("Rebuilding communities with hierarchical clustering")
        communities, community_relationships = graphrag_indexer.build_communities(
            semantic_entities, 
            entity_embeddings
        )
        
        logger.info(f"Created {len(communities)} communities across multiple levels")
        logger.info(f"Created {len(community_relationships)} parent-child relationships")
        
        # Generate summaries for communities
        logger.info("Generating summaries for communities")
        # Get chunks from existing traces/generations for context
        chunks_query = """
        MATCH (g:Generation)
        WHERE g.prompt IS NOT NULL OR g.response IS NOT NULL
        RETURN g.prompt as prompt, g.response as response
        LIMIT 100
        """
        chunk_results = neo4j_manager.execute_query(chunks_query)
        chunks = []
        for result in chunk_results:
            text = f"{result.get('prompt', '')} {result.get('response', '')}".strip()
            if text:
                chunks.append({"text": text})
        
        communities = graphrag_indexer.generate_summaries(communities, chunks)
        
        # Store new communities
        if communities:
            logger.info(f"Storing {len(communities)} communities in Neo4j")
            neo4j_manager.create_nodes({"Community": communities})
            
            # Generate embeddings for communities
            embedding_generator.generate_and_store_community_embeddings(communities)
            
            # Create parent-child CONTAINS relationships
            if community_relationships:
                logger.info(f"Creating {len(community_relationships)} CONTAINS relationships")
                neo4j_manager.create_relationships(community_relationships)
            
            # Reassign entities to communities using embedding similarity
            logger.info("Reassigning entities to communities")
            entity_community_rels = graphrag_indexer._assign_entities_to_communities(
                semantic_entities, 
                communities
            )
            
            if entity_community_rels:
                # Delete old BELONGS_TO relationships
                delete_belongs_query = """
                MATCH ()-[r:BELONGS_TO]->()
                DELETE r
                """
                neo4j_manager.execute_query(delete_belongs_query)
                
                logger.info(f"Creating {len(entity_community_rels)} BELONGS_TO relationships")
                neo4j_manager.create_relationships(entity_community_rels)
        
        # Get final stats
        stats = neo4j_manager.get_stats()
        logger.info(f"Final database stats: {stats}")
        
        # Show community levels
        levels_query = """
        MATCH (c:Community)
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
        WITH c, count(se) as entity_count
        RETURN c.level as level, count(c) as num_communities,
               sum(entity_count) as total_entities,
               round(avg(entity_count), 2) as avg_entities
        ORDER BY level ASC
        """
        levels_result = neo4j_manager.execute_query(levels_query)
        logger.info("\nCommunity levels:")
        for row in levels_result:
            logger.info(f"  Level {row['level']}: {row['num_communities']} communities, "
                       f"{row['total_entities']} entities, avg {row['avg_entities']} per community")
        
        logger.info("Community rebuild completed successfully")
        
    except Exception as e:
        logger.error(f"Error during community rebuild: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'neo4j_manager' in locals():
            neo4j_manager.close()

if __name__ == "__main__":
    main()


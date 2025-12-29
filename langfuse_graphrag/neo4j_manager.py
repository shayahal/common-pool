"""Neo4j database manager for GraphRAG system.

Handles connection, schema creation, and operations for Neo4j graph database.
"""

import logging
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError, DatabaseUnavailable

from langfuse_graphrag.config import get_config
from langfuse_graphrag.ontology import (
    get_entity_schema,
    get_all_entity_types,
    get_relationship_schemas,
)

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manager for Neo4j database operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Neo4j manager.
        
        Args:
            config: Optional configuration dictionary. If None, uses default config.
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.driver: Optional[GraphDatabase.driver] = None
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to Neo4j."""
        uri = self.config["neo4j_uri"]
        user = self.config["neo4j_user"]
        password = self.config["neo4j_password"]
        database = self.config.get("neo4j_database", "neo4j")
        
        logger.info(f"Connecting to Neo4j at {uri} (database: {database})")
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection - try system database first to check availability
            import time
            max_retries = 10
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # First try system database to verify connection
                    with self.driver.session(database="system") as session:
                        result = session.run("SHOW DATABASES")
                        databases = [record["name"] for record in result]
                        logger.debug(f"Available databases: {databases}")
                        
                        # Check database status
                        status_result = session.run("SHOW DATABASES YIELD name, currentStatus WHERE name = $db RETURN name, currentStatus", db=database)
                        status_record = status_result.single()
                        if status_record:
                            status = status_record["currentStatus"]
                            logger.debug(f"Database '{database}' status: {status}")
                        
                        # Check if target database exists
                        if database not in databases:
                            # Try to create it if it doesn't exist
                            logger.info(f"Database '{database}' not found. Attempting to create it...")
                            try:
                                session.run(f"CREATE DATABASE {database} IF NOT EXISTS")
                                logger.info(f"Created database '{database}'")
                                # Wait a bit for database to be created
                                time.sleep(3)
                            except Exception as create_error:
                                logger.error(f"Could not create database '{database}': {create_error}", exc_info=True)
                                # Try using default database as fallback
                                if "neo4j" in databases:
                                    database = "neo4j"
                                    logger.info(f"Using default database 'neo4j' as fallback")
                                else:
                                    # Use first available database as fallback
                                    if databases:
                                        database = databases[0]
                                        logger.info(f"Using available database '{database}' as fallback")
                                    else:
                                        raise RuntimeError(f"Failed to create database '{database}' and no fallback databases available: {create_error}") from create_error
                    
                    # Now test the target database with retry
                    db_ready = False
                    for db_attempt in range(5):
                        try:
                            with self.driver.session(database=database) as session:
                                session.run("RETURN 1")
                            db_ready = True
                            break
                        except DatabaseUnavailable:
                            if db_attempt < 4:
                                logger.debug(f"Database '{database}' not ready yet, waiting {retry_delay}s...")
                                time.sleep(retry_delay)
                            else:
                                raise
                    
                    if db_ready:
                        logger.info(f"Successfully connected to Neo4j database '{database}'")
                        self.config["neo4j_database"] = database  # Update config with actual database
                        break
                except DatabaseUnavailable as db_error:
                    if attempt < max_retries - 1:
                        logger.debug(f"Database '{database}' not ready yet, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(retry_delay)
                    else:
                        raise RuntimeError(
                            f"Database '{database}' is unavailable after {max_retries} attempts. "
                            f"This usually means Neo4j is still starting up. "
                            f"Please wait a few minutes and try again, or check Neo4j logs for errors."
                        ) from db_error
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}", exc_info=True)
            raise
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j connection")
    
    def create_schema(self) -> None:
        """Create database schema (constraints, indexes, vector indexes)."""
        logger.info("Creating Neo4j schema")
        database = self.config.get("neo4j_database", "neo4j")
        
        with self.driver.session(database=database) as session:
            # Create constraints and indexes for each entity type
            for entity_type in get_all_entity_types():
                schema = get_entity_schema(entity_type)
                if not schema:
                    continue
                
                # Create unique constraint on id
                if "id" in schema.required_properties:
                    constraint_query = f"""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type})
                    REQUIRE n.id IS UNIQUE
                    """
                    try:
                        session.run(constraint_query)
                        logger.debug(f"Created unique constraint on {entity_type}.id")
                    except Exception as e:
                        logger.error(f"Failed to create constraint on {entity_type}.id: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to create unique constraint on {entity_type}.id") from e
                
                # Create indexes on indexed properties
                for prop in schema.indexes:
                    if prop == "id":  # Already has unique constraint
                        continue
                    
                    index_query = f"""
                    CREATE INDEX IF NOT EXISTS FOR (n:{entity_type})
                    ON (n.{prop})
                    """
                    try:
                        session.run(index_query)
                        logger.debug(f"Created index on {entity_type}.{prop}")
                    except Exception as e:
                        logger.error(f"Failed to create index on {entity_type}.{prop}: {e}", exc_info=True)
                        raise RuntimeError(f"Failed to create index on {entity_type}.{prop}") from e
            
            # Create vector indexes for embedding properties
            # Vector indexes require Neo4j 5.11+ Enterprise Edition or AuraDB with vector search capabilities
            embedding_dimension = self.config.get("embedding_dimension", 1536)
            
            # Check Neo4j version and capabilities first
            self._check_vector_index_support(session)
            
            # Generation embeddings
            vector_indexes = [
                ("Generation", "prompt_embedding", embedding_dimension),
                ("Generation", "response_embedding", embedding_dimension),
                ("Generation", "reasoning_embedding", embedding_dimension),
                ("SemanticEntity", "embedding", embedding_dimension),
                ("Community", "embedding", embedding_dimension),
                ("Error", "message_embedding", embedding_dimension),
            ]
            
            for entity_type, prop, dim in vector_indexes:
                index_name = f"{entity_type.lower()}_{prop}_vector_idx"
                self._create_vector_index(session, index_name, entity_type, prop, dim)
        
        logger.info("Schema creation completed")
    
    def _check_vector_index_support(self, session) -> None:
        """Check if Neo4j supports vector indexes.
        
        Args:
            session: Neo4j session
            
        Raises:
            RuntimeError: If vector indexes are not supported
        """
        try:
            # Check Neo4j version
            version_result = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions[0] as version, edition")
            version_record = version_result.single()
            
            if version_record:
                edition = version_record.get("edition", "").upper()
                version = version_record.get("version", "")
                
                # Vector indexes require Neo4j Enterprise or AuraDB
                if "COMMUNITY" in edition:
                    raise RuntimeError(
                        f"Vector indexes are not supported in Neo4j Community Edition. "
                        f"Detected: {edition} {version}. "
                        f"Please use Neo4j Enterprise Edition 5.11+ or Neo4j AuraDB with vector search capabilities. "
                        f"Alternatively, you can use a different vector store (e.g., Weaviate, Pinecone) and disable vector indexes."
                    )
                
                # Check if version is 5.11+
                try:
                    major, minor = map(int, version.split(".")[:2])
                    if major < 5 or (major == 5 and minor < 11):
                        raise RuntimeError(
                            f"Vector indexes require Neo4j 5.11+. "
                            f"Detected version: {version}. "
                            f"Please upgrade to Neo4j 5.11+ Enterprise Edition or AuraDB."
                        )
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse Neo4j version: {version}")
            
            # Try to check if vector index procedure exists
            try:
                test_result = session.run("CALL dbms.procedures() YIELD name WHERE name = 'db.index.vector.createNodeIndex' RETURN count(*) as count")
                count = test_result.single()["count"]
                if count == 0:
                    raise RuntimeError(
                        "Vector index procedure 'db.index.vector.createNodeIndex' is not available. "
                        "This indicates that vector search is not enabled in your Neo4j instance. "
                        "Please use Neo4j Enterprise Edition 5.11+ or AuraDB with vector search capabilities."
                    )
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                # If procedure check fails, we'll try to create the index and fail with a better error
                logger.debug(f"Could not check vector index procedure availability: {e}")
                
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Could not check Neo4j vector index support: {e}", exc_info=True)
            raise RuntimeError(f"Failed to check Neo4j vector index support: {e}") from e
    
    def _create_vector_index(self, session, index_name: str, entity_type: str, prop: str, dim: int) -> None:
        """Create a vector index, trying multiple syntax options.
        
        Args:
            session: Neo4j session
            index_name: Name of the vector index
            entity_type: Entity type label
            prop: Property name for the embedding
            dim: Embedding dimension
        
        Raises:
            RuntimeError: If vector index creation fails with all syntax options
        """
        # Try the newer syntax first (Neo4j 5.11+)
        e1 = None
        try:
            vector_index_query = f"""
            CALL db.index.vector.createNodeIndex(
                '{index_name}',
                '{entity_type}',
                '{prop}',
                {dim},
                'cosine'
            )
            """
            session.run(vector_index_query)
            logger.debug(f"Created vector index {index_name}")
            return
        except Exception as e:
            e1 = e
            # Check if index already exists - that's fine, treat as success
            error_str = str(e)
            if "already exists" in error_str.lower() or "EquivalentSchemaRuleAlreadyExistsException" in error_str:
                logger.debug(f"Vector index {index_name} already exists, skipping creation")
                return
            logger.debug(f"db.index.vector.createNodeIndex failed for {index_name}: {e1}")
        
        # Try the older CREATE VECTOR INDEX syntax
        try:
            vector_index_query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{entity_type})
            ON n.{prop}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
            session.run(vector_index_query)
            logger.debug(f"Created vector index {index_name} using CREATE syntax")
            return
        except Exception as e2:
            logger.debug(f"CREATE VECTOR INDEX failed for {index_name}: {e2}")
            # Both methods failed - raise error
            first_error = f"Error from first attempt: {e1}" if e1 else "First attempt not executed"
            error_msg = (
                f"Failed to create vector index {index_name}. "
                f"Vector indexes require Neo4j 5.11+ Enterprise Edition or AuraDB with vector search capabilities. "
                f"{first_error}. Error from second attempt: {e2}. "
                f"\n\nTo fix this issue:\n"
                f"1. Use Neo4j Enterprise Edition 5.11+ or Neo4j AuraDB\n"
                f"2. Ensure vector search is enabled in your Neo4j instance\n"
                f"3. Check that your Neo4j license includes vector search capabilities"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e2
    
    def create_nodes(self, entities: Dict[str, List[Dict[str, Any]]], batch_size: Optional[int] = None) -> None:
        """Create nodes in Neo4j from entities.
        
        Args:
            entities: Dictionary mapping entity types to entity lists
            batch_size: Optional batch size for transactions
        """
        if batch_size is None:
            batch_size = self.config.get("neo4j_batch_size", 1000)
        
        database = self.config.get("neo4j_database", "neo4j")
        total_created = 0
        
        logger.info(f"Creating nodes in Neo4j (batch_size={batch_size})")
        
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
            
            logger.info(f"Creating {len(entity_list)} {entity_type} nodes")
            
            # Process in batches
            for i in range(0, len(entity_list), batch_size):
                batch = entity_list[i:i + batch_size]
                
                query = f"""
                UNWIND $batch AS entity
                MERGE (n:{entity_type} {{id: entity.id}})
                SET n += entity
                """
                
                # Prepare batch data (remove _type, convert datetime to string)
                batch_data = []
                for entity in batch:
                    entity_data = {k: v for k, v in entity.items() if k != "_type"}
                    # Convert datetime to ISO format string
                    for key, value in entity_data.items():
                        if hasattr(value, 'isoformat'):  # datetime object
                            entity_data[key] = value.isoformat()
                    batch_data.append(entity_data)
                
                try:
                    with self.driver.session(database=database) as session:
                        result = session.run(query, batch=batch_data)
                        result.consume()
                        total_created += len(batch)
                        logger.debug(f"Created batch of {len(batch)} {entity_type} nodes")
                except Exception as e:
                    logger.error(f"Error creating {entity_type} nodes: {e}", exc_info=True)
                    raise
        
        logger.info(f"Created {total_created} nodes total")
    
    def create_relationships(self, relationships: List[Dict[str, Any]], batch_size: Optional[int] = None) -> None:
        """Create relationships in Neo4j.
        
        Args:
            relationships: List of relationship dictionaries
            batch_size: Optional batch size for transactions
        """
        if batch_size is None:
            batch_size = self.config.get("neo4j_batch_size", 1000)
        
        database = self.config.get("neo4j_database", "neo4j")
        
        logger.info(f"Creating {len(relationships)} relationships (batch_size={batch_size})")
        
        # Group relationships by type for efficiency
        relationships_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for rel in relationships:
            rel_type = rel["type"]
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            relationships_by_type[rel_type].append(rel)
        
        total_created = 0
        
        for rel_type, rel_list in relationships_by_type.items():
            logger.debug(f"Creating {len(rel_list)} {rel_type} relationships")
            
            # Process in batches
            for i in range(0, len(rel_list), batch_size):
                batch = rel_list[i:i + batch_size]
                
                # Build query based on relationship structure
                query = f"""
                UNWIND $batch AS rel
                MATCH (from:{batch[0]['from_type']} {{id: rel.from_id}})
                MATCH (to:{batch[0]['to_type']} {{id: rel.to_id}})
                MERGE (from)-[r:{rel_type}]->(to)
                """
                
                # Add properties if present
                if batch[0].get("properties"):
                    query += "SET r += rel.properties"
                
                # Prepare batch data
                batch_data = []
                for rel in batch:
                    rel_data = {
                        "from_id": rel["from_id"],
                        "to_id": rel["to_id"],
                    }
                    if rel.get("properties"):
                        rel_data["properties"] = rel["properties"]
                    batch_data.append(rel_data)
                
                try:
                    with self.driver.session(database=database) as session:
                        result = session.run(query, batch=batch_data)
                        result.consume()
                        total_created += len(batch)
                        logger.debug(f"Created batch of {len(batch)} {rel_type} relationships")
                except Exception as e:
                    logger.error(f"Error creating {rel_type} relationships: {e}", exc_info=True)
                    raise
        
        logger.info(f"Created {total_created} relationships total")
    
    def update_node_embedding(
        self,
        entity_type: str,
        entity_id: str,
        property_name: str,
        embedding: List[float]
    ) -> None:
        """Update a node's embedding property.
        
        Args:
            entity_type: Entity type label
            entity_id: Entity ID
            property_name: Name of embedding property
            embedding: Embedding vector
        """
        database = self.config.get("neo4j_database", "neo4j")
        
        query = f"""
        MATCH (n:{entity_type} {{id: $entity_id}})
        SET n.{property_name} = $embedding
        """
        
        try:
            with self.driver.session(database=database) as session:
                session.run(query, entity_id=entity_id, embedding=embedding)
                logger.debug(f"Updated {entity_type}.{property_name} for {entity_id}")
        except Exception as e:
            logger.error(f"Error updating embedding: {e}", exc_info=True)
            raise
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Optional query parameters
        
        Returns:
            List of result dictionaries
        """
        database = self.config.get("neo4j_database", "neo4j")
        
        if parameters is None:
            parameters = {}
        
        try:
            with self.driver.session(database=database) as session:
                result = session.run(query, parameters)
                records = [dict(record) for record in result]
                return records
        except Exception as e:
            logger.error(f"Error executing query: {e}", exc_info=True)
            raise
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        logger.warning("Clearing all data from Neo4j database")
        database = self.config.get("neo4j_database", "neo4j")
        
        query = "MATCH (n) DETACH DELETE n"
        
        try:
            with self.driver.session(database=database) as session:
                result = session.run(query)
                result.consume()
                logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}", exc_info=True)
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with node counts, relationship counts, etc.
        """
        database = self.config.get("neo4j_database", "neo4j")
        
        stats = {}
        
        # Count nodes by type
        node_counts_query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        """
        
        # Count relationships by type
        rel_counts_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        """
        
        try:
            with self.driver.session(database=database) as session:
                node_result = session.run(node_counts_query)
                stats["nodes"] = {record["label"]: record["count"] for record in node_result}
                
                rel_result = session.run(rel_counts_query)
                stats["relationships"] = {record["type"]: record["count"] for record in rel_result}
                
                # Total counts
                stats["total_nodes"] = sum(stats["nodes"].values())
                stats["total_relationships"] = sum(stats["relationships"].values())
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            raise
        
        return stats


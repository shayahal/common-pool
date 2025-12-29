"""Unit tests for langfuse_graphrag.ontology module."""

import pytest
from langfuse_graphrag.ontology import (
    EntityType,
    RelationshipType,
    get_entity_schema,
    get_relationship_schemas,
    get_all_entity_types,
    get_all_relationship_types,
    ENTITY_SCHEMAS,
    RELATIONSHIP_SCHEMAS,
)


class TestEntityType:
    """Test EntityType enum."""
    
    def test_entity_type_values(self):
        """Test entity type enum values."""
        assert EntityType.SESSION.value == "Session"
        assert EntityType.TRACE.value == "Trace"
        assert EntityType.SPAN.value == "Span"
        assert EntityType.GENERATION.value == "Generation"


class TestRelationshipType:
    """Test RelationshipType enum."""
    
    def test_relationship_type_values(self):
        """Test relationship type enum values."""
        assert RelationshipType.CONTAINS.value == "CONTAINS"
        assert RelationshipType.FOLLOWS.value == "FOLLOWS"
        assert RelationshipType.MENTIONS.value == "MENTIONS"


class TestEntitySchemas:
    """Test entity schema functions."""
    
    def test_get_entity_schema_exists(self):
        """Test getting existing entity schema."""
        schema = get_entity_schema("Session")
        assert schema is not None
        assert schema.label == "Session"
        assert "id" in schema.properties
    
    def test_get_entity_schema_not_found(self):
        """Test getting non-existent entity schema."""
        schema = get_entity_schema("NonExistent")
        assert schema is None
    
    def test_all_entity_types(self):
        """Test getting all entity types."""
        types = get_all_entity_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert "Session" in types
        assert "Trace" in types
    
    def test_session_schema_properties(self):
        """Test Session schema has required properties."""
        schema = get_entity_schema("Session")
        assert "id" in schema.required_properties
        assert "id" in schema.properties
        assert "name" in schema.properties
        assert "user_id" in schema.properties
    
    def test_trace_schema_properties(self):
        """Test Trace schema has required properties."""
        schema = get_entity_schema("Trace")
        assert "id" in schema.required_properties
        assert "session_id" in schema.properties
        assert "timestamp" in schema.properties
    
    def test_generation_schema_embeddings(self):
        """Test Generation schema has embedding properties."""
        schema = get_entity_schema("Generation")
        assert "prompt_embedding" in schema.properties
        assert "response_embedding" in schema.properties
        assert "reasoning_embedding" in schema.properties


class TestRelationshipSchemas:
    """Test relationship schema functions."""
    
    def test_get_relationship_schemas_all(self):
        """Test getting all relationship schemas."""
        schemas = get_relationship_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0
    
    def test_get_relationship_schemas_filtered(self):
        """Test getting filtered relationship schemas."""
        schemas = get_relationship_schemas("CONTAINS")
        assert isinstance(schemas, list)
        assert all(s.type == "CONTAINS" for s in schemas)
    
    def test_all_relationship_types(self):
        """Test getting all relationship types."""
        types = get_all_relationship_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert "CONTAINS" in types
        assert "FOLLOWS" in types
    
    def test_contains_relationship(self):
        """Test CONTAINS relationship schema."""
        schemas = get_relationship_schemas("CONTAINS")
        assert len(schemas) > 0
        contains_rel = schemas[0]
        assert contains_rel.type == "CONTAINS"
        assert contains_rel.from_entity in ["Session", "Community"]
        assert contains_rel.to_entity in ["Trace", "Community"]


class TestSchemaStructure:
    """Test schema structure and consistency."""
    
    def test_all_entity_schemas_have_id(self):
        """Test all entity schemas have id property."""
        for entity_type, schema in ENTITY_SCHEMAS.items():
            assert "id" in schema.properties, f"{entity_type} missing id property"
            assert "id" in schema.required_properties, f"{entity_type} id not required"
    
    def test_relationship_schemas_have_types(self):
        """Test all relationship schemas have valid types."""
        for schema in RELATIONSHIP_SCHEMAS:
            assert schema.type is not None
            assert schema.from_entity is not None
            assert schema.to_entity is not None


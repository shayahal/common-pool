"""Ontology definitions for Langfuse GraphRAG system.

Defines entities, relationships, and their properties for the knowledge graph.
This ontology is generic to all traces and sessions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    """Entity types in the knowledge graph."""
    SESSION = "Session"
    TRACE = "Trace"
    SPAN = "Span"
    GENERATION = "Generation"
    SCORE = "Score"
    ERROR = "Error"
    SEMANTIC_ENTITY = "SemanticEntity"
    COMMUNITY = "Community"
    MODEL = "Model"


class RelationshipType(str, Enum):
    """Relationship types in the knowledge graph."""
    # Structural
    CONTAINS = "CONTAINS"
    HAS_SPAN = "HAS_SPAN"
    GENERATES = "GENERATES"
    HAS_SCORE = "HAS_SCORE"
    HAS_ERROR = "HAS_ERROR"
    PARENT_OF = "PARENT_OF"  # Parent-child span hierarchy
    
    # Temporal
    FOLLOWS = "FOLLOWS"
    NEXT_ROUND = "NEXT_ROUND"  # Game-specific: links rounds
    
    # Game-specific
    SAME_ROUND = "SAME_ROUND"  # Links traces in same round
    SAME_GAME = "SAME_GAME"  # Links traces in same game
    
    # Semantic
    MENTIONS = "MENTIONS"
    ABOUT = "ABOUT"
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    
    # Community
    BELONGS_TO = "BELONGS_TO"
    
    # Performance
    USES_MODEL = "USES_MODEL"
    HAS_COST = "HAS_COST"
    HAS_LATENCY = "HAS_LATENCY"


@dataclass
class EntitySchema:
    """Schema definition for a graph entity."""
    label: str
    properties: Dict[str, str]  # property_name -> type
    required_properties: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)  # Properties to index
    unique_constraints: List[str] = field(default_factory=list)  # Properties with unique constraint


@dataclass
class RelationshipSchema:
    """Schema definition for a graph relationship."""
    type: str
    from_entity: str
    to_entity: str
    properties: Dict[str, str] = field(default_factory=dict)  # property_name -> type


# ============================================================================
# Entity Schemas
# ============================================================================

SESSION_SCHEMA = EntitySchema(
    label="Session",
    properties={
        "id": "string",
        "name": "string",
        "user_id": "string",
        "created_at": "datetime",
        "updated_at": "datetime",
        "metadata": "string",  # JSON string
    },
    required_properties=["id"],
    indexes=["id", "user_id", "created_at"],
    unique_constraints=["id"],
)

TRACE_SCHEMA = EntitySchema(
    label="Trace",
    properties={
        "id": "string",
        "name": "string",
        "session_id": "string",
        "timestamp": "datetime",
        "duration_ms": "float",
        "user_id": "string",
        "metadata": "string",  # JSON string
        "input": "string",  # Text content
        "output": "string",  # Text content
        # CSV standard fields
        "release": "string",
        "version": "string",
        "environment": "string",
        "tags": "string",  # JSON string or comma-separated
        "bookmarked": "boolean",
        "public": "boolean",
        "comments": "string",
        # Game metrics (from game_summary output)
        "total_rounds": "integer",
        "final_resource": "float",
        "tragedy_occurred": "boolean",
        "avg_cooperation_index": "float",
        "gini_coefficient": "float",
        "payoff_fairness": "float",
        # Round metrics (from round_X_metrics rows, extracted from CSV columns)
        "round": "integer",
        "cooperation_index": "float",
        "resource_level": "float",
        "total_extraction": "float",
        # Metadata-extracted fields
        "game_id": "string",
        "player_id": "string",
        "action_extraction": "integer",  # from metadata.attributes.action.extraction
        "reasoning": "string",  # from metadata.attributes.reasoning
        "llm_model": "string",  # from metadata.attributes.llm.model
        "llm_temperature": "float",  # from metadata.attributes.llm.temperature
        "end_time": "datetime",  # from metadata.attributes.end_time
    },
    required_properties=["id"],
    indexes=["id", "session_id", "timestamp", "user_id", "game_id", "player_id", "round"],
    unique_constraints=["id"],
)

SPAN_SCHEMA = EntitySchema(
    label="Span",
    properties={
        "id": "string",
        "trace_id": "string",
        "name": "string",
        "type": "string",  # e.g., "llm", "tool", "function"
        "start_time": "datetime",
        "end_time": "datetime",
        "duration_ms": "float",
        "status": "string",  # "success" or "error"
        "input": "string",  # Text content
        "output": "string",  # Text content
        "reasoning": "string",  # Text content (from metadata.attributes.reasoning)
        "metadata": "string",  # JSON string
    },
    required_properties=["id", "trace_id"],
    indexes=["id", "trace_id", "type", "status", "start_time"],
    unique_constraints=["id"],
)

GENERATION_SCHEMA = EntitySchema(
    label="Generation",
    properties={
        "id": "string",
        "span_id": "string",
        "model": "string",
        "prompt": "string",  # Text content
        "response": "string",  # Text content
        "system_prompt": "string",  # Text content
        "reasoning": "string",  # Text content
        "tokens_input": "integer",
        "tokens_output": "integer",
        "cost": "float",
        "latency_ms": "float",
        "temperature": "float",
        "metadata": "string",  # JSON string
        # Vector embeddings (stored as Neo4j vector type)
        "prompt_embedding": "vector",
        "response_embedding": "vector",
        "reasoning_embedding": "vector",
    },
    required_properties=["id"],
    indexes=["id", "span_id", "model", "tokens_input", "tokens_output", "cost"],
    unique_constraints=["id"],
)

SCORE_SCHEMA = EntitySchema(
    label="Score",
    properties={
        "id": "string",
        "trace_id": "string",
        "span_id": "string",
        "name": "string",
        "value": "float",
        "comment": "string",
        "timestamp": "datetime",
    },
    required_properties=["id", "name", "value"],
    indexes=["id", "trace_id", "span_id", "name", "value", "timestamp"],
    unique_constraints=["id"],
)

ERROR_SCHEMA = EntitySchema(
    label="Error",
    properties={
        "id": "string",
        "trace_id": "string",
        "span_id": "string",
        "type": "string",
        "message": "string",
        "stack_trace": "string",
        "timestamp": "datetime",
        "metadata": "string",  # JSON string
        "message_embedding": "vector",  # For error pattern matching
    },
    required_properties=["id", "type", "message"],
    indexes=["id", "trace_id", "span_id", "type", "timestamp"],
    unique_constraints=["id"],
)

SEMANTIC_ENTITY_SCHEMA = EntitySchema(
    label="SemanticEntity",
    properties={
        "id": "string",
        "type": "string",  # e.g., "concept", "topic", "intent", "action"
        "name": "string",
        "description": "string",
        "embedding": "vector",
        "metadata": "string",  # JSON string
    },
    required_properties=["id", "type", "name"],
    indexes=["id", "type", "name"],
    unique_constraints=["id"],
)

COMMUNITY_SCHEMA = EntitySchema(
    label="Community",
    properties={
        "id": "string",
        "name": "string",
        "level": "integer",
        "summary": "string",
        "embedding": "vector",
    },
    required_properties=["id", "level"],
    indexes=["id", "level"],
    unique_constraints=["id"],
)

MODEL_SCHEMA = EntitySchema(
    label="Model",
    properties={
        "id": "string",
        "name": "string",
        "provider": "string",
        "version": "string",
    },
    required_properties=["id", "name"],
    indexes=["id", "name", "provider"],
    unique_constraints=["id"],
)

# ============================================================================
# Relationship Schemas
# ============================================================================

RELATIONSHIP_SCHEMAS: List[RelationshipSchema] = [
    # Structural relationships
    RelationshipSchema(
        type=RelationshipType.CONTAINS,
        from_entity="Session",
        to_entity="Trace",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.HAS_SPAN,
        from_entity="Trace",
        to_entity="Span",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.GENERATES,
        from_entity="Span",
        to_entity="Generation",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.HAS_SCORE,
        from_entity="Trace",
        to_entity="Score",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.HAS_SCORE,
        from_entity="Span",
        to_entity="Score",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.HAS_ERROR,
        from_entity="Trace",
        to_entity="Error",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.HAS_ERROR,
        from_entity="Span",
        to_entity="Error",
        properties={},
    ),
    
    # Temporal relationships
    RelationshipSchema(
        type=RelationshipType.FOLLOWS,
        from_entity="Trace",
        to_entity="Trace",
        properties={"order": "integer"},
    ),
    RelationshipSchema(
        type=RelationshipType.FOLLOWS,
        from_entity="Span",
        to_entity="Span",
        properties={"order": "integer"},
    ),
    RelationshipSchema(
        type=RelationshipType.FOLLOWS,
        from_entity="Generation",
        to_entity="Generation",
        properties={"order": "integer"},
    ),
    
    # Semantic relationships
    RelationshipSchema(
        type=RelationshipType.MENTIONS,
        from_entity="Generation",
        to_entity="SemanticEntity",
        properties={"context": "string"},
    ),
    RelationshipSchema(
        type=RelationshipType.ABOUT,
        from_entity="Trace",
        to_entity="SemanticEntity",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.RELATED_TO,
        from_entity="Error",
        to_entity="SemanticEntity",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.SIMILAR_TO,
        from_entity="SemanticEntity",
        to_entity="SemanticEntity",
        properties={"similarity": "float"},
    ),
    
    # Community relationships
    RelationshipSchema(
        type=RelationshipType.BELONGS_TO,
        from_entity="SemanticEntity",
        to_entity="Community",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.BELONGS_TO,
        from_entity="Trace",
        to_entity="Community",
        properties={},
    ),
    RelationshipSchema(
        type=RelationshipType.CONTAINS,
        from_entity="Community",
        to_entity="Community",
        properties={},
    ),
    
    # Performance relationships
    RelationshipSchema(
        type=RelationshipType.USES_MODEL,
        from_entity="Generation",
        to_entity="Model",
        properties={},
    ),
]

# ============================================================================
# Entity Schema Registry
# ============================================================================

ENTITY_SCHEMAS: Dict[str, EntitySchema] = {
    "Session": SESSION_SCHEMA,
    "Trace": TRACE_SCHEMA,
    "Span": SPAN_SCHEMA,
    "Generation": GENERATION_SCHEMA,
    "Score": SCORE_SCHEMA,
    "Error": ERROR_SCHEMA,
    "SemanticEntity": SEMANTIC_ENTITY_SCHEMA,
    "Community": COMMUNITY_SCHEMA,
    "Model": MODEL_SCHEMA,
}


def get_entity_schema(entity_type: str) -> Optional[EntitySchema]:
    """Get schema for an entity type.
    
    Args:
        entity_type: Entity type name
    
    Returns:
        EntitySchema or None if not found
    """
    return ENTITY_SCHEMAS.get(entity_type)


def get_relationship_schemas(
    relationship_type: Optional[str] = None
) -> List[RelationshipSchema]:
    """Get relationship schemas, optionally filtered by type.
    
    Args:
        relationship_type: Optional relationship type to filter by
    
    Returns:
        List of RelationshipSchema objects
    """
    if relationship_type is None:
        return RELATIONSHIP_SCHEMAS
    
    return [
        schema for schema in RELATIONSHIP_SCHEMAS
        if schema.type == relationship_type
    ]


def get_all_entity_types() -> List[str]:
    """Get all entity type names.
    
    Returns:
        List of entity type names
    """
    return list(ENTITY_SCHEMAS.keys())


def get_all_relationship_types() -> List[str]:
    """Get all relationship type names.
    
    Returns:
        List of relationship type names
    """
    return list(set(schema.type for schema in RELATIONSHIP_SCHEMAS))


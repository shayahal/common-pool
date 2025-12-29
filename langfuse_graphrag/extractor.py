"""Entity and relationship extractor from parsed CSV data.

Extracts structural entities (Session, Trace, Span, Generation, Score, Error)
and builds relationships from CSV data.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from langfuse_graphrag.ontology import (
    EntityType,
    RelationshipType,
    get_entity_schema,
    get_relationship_schemas,
)

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities and relationships from parsed CSV records."""
    
    def __init__(self):
        """Initialize the extractor."""
        self.entities: Dict[str, List[Dict[str, Any]]] = {}
        self.relationships: List[Dict[str, Any]] = []
        
    def extract_entities(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from CSV records.
        
        Args:
            records: List of parsed CSV records
        
        Returns:
            Dictionary mapping entity types to lists of entity dictionaries
        """
        logger.info(f"Extracting entities from {len(records)} records")
        
        # Group records by CSV type
        records_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            csv_type = record.get("_csv_type", "trace")
            if csv_type not in records_by_type:
                records_by_type[csv_type] = []
            records_by_type[csv_type].append(record)
        
        # Extract entities for each type
        for csv_type, type_records in records_by_type.items():
            logger.debug(f"Processing {len(type_records)} {csv_type} records")
            
            if csv_type == "session":
                self._extract_sessions(type_records)
            elif csv_type == "trace":
                self._extract_traces(type_records)
            elif csv_type == "span":
                self._extract_spans(type_records)
            elif csv_type == "generation":
                self._extract_generations(type_records)
            elif csv_type == "score":
                self._extract_scores(type_records)
            else:
                logger.warning(f"Unknown CSV type: {csv_type}, skipping entity extraction")
        
        # Log summary
        for entity_type, entities in self.entities.items():
            logger.info(f"Extracted {len(entities)} {entity_type} entities")
        
        return self.entities
    
    def _extract_sessions(self, records: List[Dict[str, Any]]) -> None:
        """Extract Session and Trace entities from Langfuse CSV exports.
        
        Langfuse exports can be either:
        1. Session-level exports (aggregated data per session)
        2. Trace-level exports (each row is a trace with sessionId)
        
        We detect which type it is and handle accordingly.
        """
        entity_type = EntityType.SESSION.value
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        schema = get_entity_schema(entity_type)
        if not schema:
            logger.error(f"No schema found for {entity_type}")
            return
        
        # Track unique sessions by sessionId (for grouping traces into sessions)
        session_map: Dict[str, Dict[str, Any]] = {}
        trace_records: List[Dict[str, Any]] = []
        
        for record in records:
            # Check if this row has both id and sessionId
            row_id = record.get("id")
            session_id = record.get("sessionId") or record.get("session_id")
            trace_name = record.get("name", "")
            
            # Determine if this is a trace or session based on available data
            is_trace = False
            
            # If sessionId exists and is different from id, this is a trace
            if session_id and row_id and session_id != row_id:
                is_trace = True
            # If name looks like a trace (player action, round, etc.), treat as trace
            elif trace_name and any(x in trace_name for x in ["player_", "round_", "action", "generation", "game_"]):
                is_trace = True
                if not session_id:
                    # Use id as session_id if not provided
                    session_id = row_id
            # If there's input/output data, it's likely a trace not a session
            elif record.get("input") or record.get("output"):
                is_trace = True
                if not session_id:
                    session_id = row_id
            
            if is_trace:
                # Track this record for trace creation
                trace_records.append(record)
                
                # Create/update the session
                if session_id and session_id not in session_map:
                    session_entity = {
                        "id": session_id,
                        "session_id": session_id,
                        "name": session_id,
                        "created_at": record.get("timestamp") or record.get("createdAt"),
                        "user_id": record.get("userId") or record.get("user_id"),
                    }
                    session_map[session_id] = session_entity
            else:
                # This is a session-level record
                entity = self._build_entity(record, schema, entity_type)
                if entity:
                    self.entities[entity_type].append(entity)
        
        # Add unique sessions to entities
        for session_entity in session_map.values():
            self.entities[entity_type].append(session_entity)
        
        # Create Trace entities from trace records
        for record in trace_records:
            session_id = record.get("sessionId") or record.get("session_id") or record.get("id")
            self._create_trace_from_session_record(record, session_id)
    
    def _create_trace_from_session_record(self, record: Dict[str, Any], session_id: str) -> None:
        """Create a Trace entity from a session CSV record.
        
        Langfuse sometimes exports traces as session rows. This method extracts
        trace information and creates proper Trace entities.
        """
        trace_type = EntityType.TRACE.value
        if trace_type not in self.entities:
            self.entities[trace_type] = []
        
        trace_entity = {
            "id": record.get("id"),
            "trace_id": record.get("id"),
            "session_id": session_id,
            "name": record.get("name"),
            "timestamp": record.get("timestamp") or record.get("createdAt"),
            "input": record.get("input"),
            "output": record.get("output"),
            "metadata": record.get("metadata"),
            "user_id": record.get("userId") or record.get("user_id"),
        }
        
        # Extract additional metrics from metadata
        metadata = record.get("metadata")
        if metadata:
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            if isinstance(metadata, dict):
                trace_entity["game_id"] = metadata.get("game_id")
                trace_entity["player_id"] = metadata.get("player_id")
                trace_entity["round"] = metadata.get("round")
        
        self.entities[trace_type].append(trace_entity)
    
    def _extract_traces(self, records: List[Dict[str, Any]]) -> None:
        """Extract Trace entities."""
        entity_type = EntityType.TRACE.value
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        schema = get_entity_schema(entity_type)
        if not schema:
            logger.error(f"No schema found for {entity_type}")
            return
        
        for record in records:
            entity = self._build_entity(record, schema, entity_type)
            if entity:
                self.entities[entity_type].append(entity)
    
    def _extract_spans(self, records: List[Dict[str, Any]]) -> None:
        """Extract Span entities."""
        entity_type = EntityType.SPAN.value
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        schema = get_entity_schema(entity_type)
        if not schema:
            logger.error(f"No schema found for {entity_type}")
            return
        
        for record in records:
            entity = self._build_entity(record, schema, entity_type)
            if entity:
                self.entities[entity_type].append(entity)
    
    def _extract_generations(self, records: List[Dict[str, Any]]) -> None:
        """Extract Generation entities."""
        entity_type = EntityType.GENERATION.value
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        schema = get_entity_schema(entity_type)
        if not schema:
            logger.error(f"No schema found for {entity_type}")
            return
        
        for record in records:
            entity = self._build_entity(record, schema, entity_type)
            if entity:
                self.entities[entity_type].append(entity)
    
    def _extract_scores(self, records: List[Dict[str, Any]]) -> None:
        """Extract Score entities."""
        entity_type = EntityType.SCORE.value
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        schema = get_entity_schema(entity_type)
        if not schema:
            logger.error(f"No schema found for {entity_type}")
            return
        
        for record in records:
            entity = self._build_entity(record, schema, entity_type)
            if entity:
                self.entities[entity_type].append(entity)
    
    def _build_entity(
        self,
        record: Dict[str, Any],
        schema: Any,
        entity_type: str
    ) -> Optional[Dict[str, Any]]:
        """Build entity dictionary from record according to schema.
        
        Args:
            record: CSV record
            schema: Entity schema
            entity_type: Entity type name
        
        Returns:
            Entity dictionary or None if required properties missing
        """
        entity = {"_type": entity_type}
        
        # Check required properties
        for prop in schema.required_properties:
            if prop not in record or record[prop] is None:
                logger.warning(
                    f"Missing required property '{prop}' for {entity_type} "
                    f"(row {record.get('_row_index', 'unknown')})"
                )
                return None
        
        # Add all schema properties
        for prop, prop_type in schema.properties.items():
            if prop in record:
                value = record[prop]
                # Convert types if needed
                if prop_type == "integer" and value is not None:
                    try:
                        entity[prop] = int(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {prop} to integer: {value}")
                        entity[prop] = value
                elif prop_type == "float" and value is not None:
                    try:
                        entity[prop] = float(value)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert {prop} to float: {value}")
                        entity[prop] = value
                elif prop_type == "datetime" and value is not None:
                    if isinstance(value, datetime):
                        entity[prop] = value
                    else:
                        entity[prop] = value  # Keep as-is, let Neo4j handle conversion
                else:
                    entity[prop] = value
        
        return entity
    
    def extract_relationships(self, entities: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Extract relationships from entities.
        
        Args:
            entities: Dictionary of entity types to entity lists
        
        Returns:
            List of relationship dictionaries
        """
        logger.info("Extracting relationships from entities")
        
        self.relationships = []
        
        # Extract structural relationships
        self._extract_structural_relationships(entities)
        
        # Extract temporal relationships
        self._extract_temporal_relationships(entities)
        
        logger.info(f"Extracted {len(self.relationships)} relationships")
        return self.relationships
    
    def _extract_structural_relationships(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Extract structural relationships (CONTAINS, HAS_SPAN, etc.)."""
        # Session -> Trace
        sessions = entities.get(EntityType.SESSION.value, [])
        traces = entities.get(EntityType.TRACE.value, [])
        
        # Build session lookup map
        session_ids = {s.get("id") or s.get("session_id") for s in sessions}
        
        for trace in traces:
            session_id = trace.get("session_id")
            if session_id:
                # Create relationship if session exists or if we should create it
                if session_id in session_ids:
                    self.relationships.append({
                        "type": RelationshipType.CONTAINS.value,
                        "from_type": EntityType.SESSION.value,
                        "from_id": session_id,
                        "to_type": EntityType.TRACE.value,
                        "to_id": trace.get("id"),
                    })
                else:
                    # Auto-create session if it doesn't exist
                    session_ids.add(session_id)
                    if EntityType.SESSION.value not in entities:
                        entities[EntityType.SESSION.value] = []
                    entities[EntityType.SESSION.value].append({
                        "id": session_id,
                        "session_id": session_id,
                        "name": session_id,
                    })
                    self.relationships.append({
                        "type": RelationshipType.CONTAINS.value,
                        "from_type": EntityType.SESSION.value,
                        "from_id": session_id,
                        "to_type": EntityType.TRACE.value,
                        "to_id": trace.get("id"),
                    })
        
        # Trace -> Span
        spans = entities.get(EntityType.SPAN.value, [])
        for span in spans:
            trace_id = span.get("trace_id")
            if trace_id:
                self.relationships.append({
                    "type": RelationshipType.HAS_SPAN.value,
                    "from_type": EntityType.TRACE.value,
                    "from_id": trace_id,
                    "to_type": EntityType.SPAN.value,
                    "to_id": span.get("id"),
                })
        
        # Span -> Generation
        generations = entities.get(EntityType.GENERATION.value, [])
        for generation in generations:
            span_id = generation.get("span_id")
            trace_id = generation.get("trace_id")
            
            if span_id:
                self.relationships.append({
                    "type": RelationshipType.GENERATES.value,
                    "from_type": EntityType.SPAN.value,
                    "from_id": span_id,
                    "to_type": EntityType.GENERATION.value,
                    "to_id": generation.get("id"),
                })
            elif trace_id:
                # If no span_id but has trace_id, link directly to trace
                self.relationships.append({
                    "type": RelationshipType.GENERATES.value,
                    "from_type": EntityType.TRACE.value,
                    "from_id": trace_id,
                    "to_type": EntityType.GENERATION.value,
                    "to_id": generation.get("id"),
                })
        
        # Trace/Span -> Score
        scores = entities.get(EntityType.SCORE.value, [])
        for score in scores:
            trace_id = score.get("trace_id")
            span_id = score.get("span_id")
            
            if trace_id:
                self.relationships.append({
                    "type": RelationshipType.HAS_SCORE.value,
                    "from_type": EntityType.TRACE.value,
                    "from_id": trace_id,
                    "to_type": EntityType.SCORE.value,
                    "to_id": score.get("id"),
                })
            
            if span_id:
                self.relationships.append({
                    "type": RelationshipType.HAS_SCORE.value,
                    "from_type": EntityType.SPAN.value,
                    "from_id": span_id,
                    "to_type": EntityType.SCORE.value,
                    "to_id": score.get("id"),
                })
        
        # Extract game-specific relationships from metadata
        self._extract_game_relationships(entities)
    
    def _extract_game_relationships(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Extract game-specific relationships from trace metadata.
        
        Creates relationships like:
        - Player traces within the same round (SAME_ROUND)
        - Sequential rounds (NEXT_ROUND)
        - Game summary to session
        """
        traces = entities.get(EntityType.TRACE.value, [])
        
        # Group traces by game_id and round
        traces_by_game: Dict[str, List[Dict[str, Any]]] = {}
        traces_by_game_round: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
        
        for trace in traces:
            game_id = trace.get("game_id") or trace.get("session_id")
            if not game_id:
                # Try to extract from name or metadata
                name = trace.get("name", "")
                if "game_" in name:
                    game_id = name.split("_")[0] + "_" + name.split("_")[1] if len(name.split("_")) > 1 else name
            
            if game_id:
                if game_id not in traces_by_game:
                    traces_by_game[game_id] = []
                    traces_by_game_round[game_id] = {}
                traces_by_game[game_id].append(trace)
                
                # Extract round number from name or metadata
                round_num = trace.get("round")
                if round_num is None:
                    name = trace.get("name", "")
                    if "round_" in name:
                        try:
                            parts = name.split("round_")
                            if len(parts) > 1:
                                round_str = parts[1].split("_")[0]
                                round_num = int(round_str)
                        except (ValueError, IndexError):
                            pass
                
                if round_num is not None:
                    if round_num not in traces_by_game_round[game_id]:
                        traces_by_game_round[game_id][round_num] = []
                    traces_by_game_round[game_id][round_num].append(trace)
        
        # Create SAME_ROUND relationships (player traces in same round)
        for game_id, rounds in traces_by_game_round.items():
            for round_num, round_traces in rounds.items():
                # Link all traces in the same round
                for i, trace1 in enumerate(round_traces):
                    for trace2 in round_traces[i+1:]:
                        self.relationships.append({
                            "type": "SAME_ROUND",
                            "from_type": EntityType.TRACE.value,
                            "from_id": trace1.get("id"),
                            "to_type": EntityType.TRACE.value,
                            "to_id": trace2.get("id"),
                            "properties": {"round": round_num, "game_id": game_id},
                        })
            
            # Create NEXT_ROUND relationships
            sorted_rounds = sorted(rounds.keys())
            for i in range(len(sorted_rounds) - 1):
                current_round = sorted_rounds[i]
                next_round = sorted_rounds[i + 1]
                
                # Link last trace of current round to first trace of next round
                if rounds[current_round] and rounds[next_round]:
                    # Sort by timestamp if available
                    current_traces = sorted(
                        rounds[current_round],
                        key=lambda t: t.get("timestamp") or ""
                    )
                    next_traces = sorted(
                        rounds[next_round],
                        key=lambda t: t.get("timestamp") or ""
                    )
                    
                    self.relationships.append({
                        "type": "NEXT_ROUND",
                        "from_type": EntityType.TRACE.value,
                        "from_id": current_traces[-1].get("id"),
                        "to_type": EntityType.TRACE.value,
                        "to_id": next_traces[0].get("id"),
                        "properties": {
                            "from_round": current_round,
                            "to_round": next_round,
                            "game_id": game_id
                        },
                    })

    def _extract_temporal_relationships(
        self,
        entities: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Extract temporal relationships (FOLLOWS)."""
        # Trace -> Trace (by timestamp)
        traces = entities.get(EntityType.TRACE.value, [])
        traces_with_time = [
            (t, t.get("timestamp"))
            for t in traces
            if t.get("timestamp") is not None
        ]
        traces_with_time.sort(key=lambda x: x[1] if x[1] else datetime.min)
        
        for i in range(len(traces_with_time) - 1):
            current_trace, _ = traces_with_time[i]
            next_trace, _ = traces_with_time[i + 1]
            
            # Only create FOLLOWS if they're in the same session
            if current_trace.get("session_id") == next_trace.get("session_id"):
                self.relationships.append({
                    "type": RelationshipType.FOLLOWS.value,
                    "from_type": EntityType.TRACE.value,
                    "from_id": current_trace.get("id"),
                    "to_type": EntityType.TRACE.value,
                    "to_id": next_trace.get("id"),
                    "properties": {"order": i + 1},
                })
        
        # Span -> Span (within same trace, by start_time)
        spans = entities.get(EntityType.SPAN.value, [])
        spans_by_trace: Dict[str, List[Dict[str, Any]]] = {}
        
        for span in spans:
            trace_id = span.get("trace_id")
            if trace_id:
                if trace_id not in spans_by_trace:
                    spans_by_trace[trace_id] = []
                spans_by_trace[trace_id].append(span)
        
        for trace_id, trace_spans in spans_by_trace.items():
            spans_with_time = [
                (s, s.get("start_time"))
                for s in trace_spans
                if s.get("start_time") is not None
            ]
            spans_with_time.sort(key=lambda x: x[1] if x[1] else datetime.min)
            
            for i in range(len(spans_with_time) - 1):
                current_span, _ = spans_with_time[i]
                next_span, _ = spans_with_time[i + 1]
                
                self.relationships.append({
                    "type": RelationshipType.FOLLOWS.value,
                    "from_type": EntityType.SPAN.value,
                    "from_id": current_span.get("id"),
                    "to_type": EntityType.SPAN.value,
                    "to_id": next_span.get("id"),
                    "properties": {"order": i + 1},
                })
        
        # Generation -> Generation (within same span, by order in CSV)
        generations = entities.get(EntityType.GENERATION.value, [])
        generations_by_span: Dict[str, List[Dict[str, Any]]] = {}
        
        for generation in generations:
            span_id = generation.get("span_id")
            if span_id:
                if span_id not in generations_by_span:
                    generations_by_span[span_id] = []
                generations_by_span[span_id].append(generation)
        
        for span_id, span_generations in generations_by_span.items():
            for i in range(len(span_generations) - 1):
                current_gen = span_generations[i]
                next_gen = span_generations[i + 1]
                
                self.relationships.append({
                    "type": RelationshipType.FOLLOWS.value,
                    "from_type": EntityType.GENERATION.value,
                    "from_id": current_gen.get("id"),
                    "to_type": EntityType.GENERATION.value,
                    "to_id": next_gen.get("id"),
                    "properties": {"order": i + 1},
                })
    
    def get_entities(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get extracted entities.
        
        Returns:
            Dictionary mapping entity types to entity lists
        """
        return self.entities
    
    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get extracted relationships.
        
        Returns:
            List of relationship dictionaries
        """
        return self.relationships


# Entity Hierarchy: Session → Trace → Span → Generation

## Proper Hierarchy Structure

```
Session (Top Level)
  │
  ├─[CONTAINS]─→ Trace (Game/Workflow Level)
  │                │
  │                ├─[HAS_SPAN]─→ Span (Operation Level)
  │                │                │
  │                │                ├─[GENERATES]─→ Generation (LLM Call Level)
  │                │                │
  │                │                └─[HAS_ERROR]─→ Error (if errors occur)
  │                │
  │                ├─[HAS_SCORE]─→ Score (evaluations)
  │                │
  │                └─[HAS_ERROR]─→ Error (trace-level errors)
  │
  └─[CONTAINS]─→ Trace (another trace in same session)
```

## Entity Definitions

### 1. **Session** (Top Level)
- **Purpose**: Represents a user session or experiment run
- **Example**: `exp_bb6fc969_game_0009` (a game session)
- **Properties**: `id`, `name`, `user_id`, `created_at`, `updated_at`, `metadata`
- **Relationships**: 
  - `[CONTAINS]` → Trace (one-to-many)

### 2. **Trace** (Workflow/Game Level)
- **Purpose**: Represents a complete workflow execution or game
- **Example**: `game_summary`, `round_X_metrics`, or the game session itself
- **Properties**: `id`, `name`, `session_id`, `timestamp`, `duration_ms`, `user_id`, `metadata`, `input`, `output`
- **Relationships**:
  - `[CONTAINS]` ← Session (many-to-one)
  - `[HAS_SPAN]` → Span (one-to-many)
  - `[HAS_SCORE]` → Score (one-to-many)
  - `[HAS_ERROR]` → Error (one-to-many)
  - `[FOLLOWS]` → Trace (temporal ordering)
  - `[ABOUT]` → SemanticEntity (semantic relationships)

### 3. **Span** (Operation Level)
- **Purpose**: Represents a single operation or step within a trace
- **Example**: `player_0_action_round_17` (a player's action in a round)
- **Properties**: `id`, `trace_id`, `name`, `type` (e.g., "llm", "tool", "function"), `start_time`, `end_time`, `duration_ms`, `status`, `metadata`, `input`, `output`
- **Relationships**:
  - `[HAS_SPAN]` ← Trace (many-to-one)
  - `[GENERATES]` → Generation (one-to-many)
  - `[HAS_SCORE]` → Score (one-to-many)
  - `[HAS_ERROR]` → Error (one-to-many)
  - `[FOLLOWS]` → Span (temporal ordering within same trace)
  - `[PARENT_OF]` → Span (nested span hierarchy, if supported)

### 4. **Generation** (LLM Call Level)
- **Purpose**: Represents a single LLM API call with prompt/response
- **Example**: The actual LLM call that generated a player's action
- **Properties**: `id`, `span_id`, `trace_id`, `model`, `prompt`, `response`, `system_prompt`, `reasoning`, `tokens_input`, `tokens_output`, `cost`, `latency_ms`, `temperature`, `metadata`, `prompt_embedding`, `response_embedding`, `reasoning_embedding`
- **Relationships**:
  - `[GENERATES]` ← Span (many-to-one)
  - `[GENERATES]` ← Trace (fallback if no span_id, many-to-one)
  - `[MENTIONS]` → SemanticEntity (semantic relationships)
  - `[USES_MODEL]` → Model (performance tracking)
  - `[FOLLOWS]` → Generation (temporal ordering within same span)

## Relationship Types

### Structural Relationships
- **CONTAINS**: Session → Trace (session contains traces)
- **HAS_SPAN**: Trace → Span (trace contains spans)
- **GENERATES**: Span → Generation (span generates LLM calls)
- **HAS_SCORE**: Trace/Span → Score (evaluations)
- **HAS_ERROR**: Trace/Span → Error (error tracking)

### Temporal Relationships
- **FOLLOWS**: Trace → Trace, Span → Span, Generation → Generation (sequential ordering)

### Semantic Relationships
- **MENTIONS**: Generation → SemanticEntity (content mentions concepts)
- **ABOUT**: Trace → SemanticEntity (trace is about concepts)
- **RELATED_TO**: Error → SemanticEntity (errors related to concepts)

## Current Problem in Your Data

**What's happening now:**
- All 543 `player_X_action` rows are being created as **Trace entities**
- They should be **Span entities** instead!
- The metadata shows `langfuse.observation.type: span` indicating these are spans

**What should happen:**
```
Session: exp_bb6fc969_game_0009
  └─[CONTAINS]─→ Trace: exp_bb6fc969_game_0009 (the game itself)
       └─[HAS_SPAN]─→ Span: player_0_action_round_17
            └─[GENERATES]─→ Generation: (LLM call for player action)
```

**Current incorrect structure:**
```
Session: exp_bb6fc969_game_0009
  └─[CONTAINS]─→ Trace: player_0_action_round_17 (WRONG! Should be Span)
       └─[CONTAINS]─→ Trace: player_1_action_round_17 (WRONG! Should be Span)
```

## Key Properties for Each Level

### Session
- Groups related traces together
- Represents a user session or experiment

### Trace  
- Represents a complete workflow/game
- Can have multiple spans (operations)
- Has overall input/output

### Span
- Represents a single operation within a trace
- Has start_time, end_time, duration_ms
- Can have type: "llm", "tool", "function", etc.
- Contains the actual operation details

### Generation
- Represents a single LLM API call
- Contains prompt, response, reasoning
- Has cost, latency, token counts
- Has embeddings for semantic search

## For Your Game Data

**Proper structure should be:**
```
Session: exp_bb6fc969_game_0009
  └─[CONTAINS]─→ Trace: exp_bb6fc969_game_0009 (game session)
       ├─[HAS_SPAN]─→ Span: player_0_action_round_0
       │    └─[GENERATES]─→ Generation: (LLM call)
       ├─[HAS_SPAN]─→ Span: player_1_action_round_0
       │    └─[GENERATES]─→ Generation: (LLM call)
       ├─[HAS_SPAN]─→ Span: player_0_action_round_1
       │    └─[GENERATES]─→ Generation: (LLM call)
       └─[HAS_SPAN]─→ Span: game_summary
            └─[GENERATES]─→ Generation: (summary generation)
```

This hierarchy allows you to:
- Query all spans in a game (trace)
- Find all LLM calls (generations) in a span
- Track temporal ordering (FOLLOWS relationships)
- Link semantic entities to the right level (MENTIONS from Generation, ABOUT from Trace)


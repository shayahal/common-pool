# GraphRAG Query Guide

This guide explains what you can query from the GraphRAG system and how to use it.

## Quick Start

All commands use the CLI: `python -m langfuse_graphrag.cli <command>`

## Available Commands

### 1. **Semantic Search** 🔍
Find similar content using vector similarity search.

```bash
# Search for similar prompts/responses
python -m langfuse_graphrag.cli search "cooperation strategies" --entity-type Generation --property prompt_embedding

# Search semantic entities
python -m langfuse_graphrag.cli search "resource management" --entity-type SemanticEntity --property embedding

# Search communities
python -m langfuse_graphrag.cli search "game strategies" --entity-type Community --property embedding

# Options:
#   --entity-type: Generation, SemanticEntity, Community, Error (default: Generation)
#   --property: embedding property name (default: prompt_embedding)
#   --limit: max results (default: 10)
#   --threshold: similarity threshold 0-1 (default: 0.7)
#   --output: save to JSON file
```

**What you can find:**
- Similar prompts, responses, or reasoning from LLM generations
- Related semantic entities (concepts, actions, topics)
- Related communities of concepts
- Similar error messages

---

### 2. **Pattern Analysis** 📊
Analyze structural patterns in your traces and sessions.

```bash
# Analyze traces grouped by session
python -m langfuse_graphrag.cli analyze patterns --pattern-type session_traces

# Analyze spans within traces
python -m langfuse_graphrag.cli analyze patterns --pattern-type trace_spans

# Analyze generations within spans
python -m langfuse_graphrag.cli analyze patterns --pattern-type span_generations

# Analyze temporal flow (how traces follow each other)
python -m langfuse_graphrag.cli analyze patterns --pattern-type temporal_flow

# Options:
#   --pattern-type: session_traces, trace_spans, span_generations, temporal_flow
#   --limit: max results (default: 10)
#   --output: save to JSON file
```

**What you can discover:**
- Which sessions have the most traces
- Which traces have the most spans
- How traces flow temporally
- Generation patterns across spans

---

### 3. **Error Analysis** ⚠️
Find and analyze errors in your traces.

```bash
# Find all errors
python -m langfuse_graphrag.cli analyze errors

# Find errors by type
python -m langfuse_graphrag.cli analyze errors --error-type "APIError"

# Options:
#   --error-type: filter by error type
#   --trace-id: filter by specific trace
#   --limit: max results (default: 10)
#   --output: save to JSON file
```

**What you can find:**
- All errors across all traces
- Errors by type
- Errors in specific traces
- Error frequency and patterns

---

### 4. **Performance Analysis** ⚡
Analyze performance metrics (cost, latency, tokens).

```bash
# Analyze by cost
python -m langfuse_graphrag.cli analyze performance --metric cost

# Analyze by latency
python -m langfuse_graphrag.cli analyze performance --metric latency

# Analyze by tokens
python -m langfuse_graphrag.cli analyze performance --metric tokens

# Group by model
python -m langfuse_graphrag.cli analyze performance --metric cost --group-by model

# Options:
#   --metric: cost, latency, tokens, tokens_input, tokens_output
#   --group-by: model, trace_id, span_id, etc.
#   --limit: max results (default: 10)
#   --output: save to JSON file
```

**What you can discover:**
- Most expensive operations
- Slowest traces/spans
- Token usage patterns
- Performance by model or other dimensions

---

### 5. **Database Statistics** 📈
Get overview statistics about your graph.

```bash
# Get all stats
python -m langfuse_graphrag.cli stats

# Save to file
python -m langfuse_graphrag.cli stats --output stats.json
```

**What you get:**
- Node counts by type (Trace, Span, Generation, SemanticEntity, Community, etc.)
- Relationship counts by type
- Total nodes and relationships

---

### 6. **Exploratory Queries** 🔎
Explore the graph structure to understand your data.

```bash
# Overview of node types and counts
python -m langfuse_graphrag.cli explore --query-type overview

# Relationship types and counts
python -m langfuse_graphrag.cli explore --query-type relationships

# Node structure (properties)
python -m langfuse_graphrag.cli explore --query-type structure

# Sample sessions
python -m langfuse_graphrag.cli explore --query-type sessions

# Isolated nodes (no relationships)
python -m langfuse_graphrag.cli explore --query-type isolated

# Overall statistics
python -m langfuse_graphrag.cli explore --query-type stats
```

---

### 7. **Custom Cypher Queries** 💻
Execute custom Cypher queries directly.

```bash
# Query semantic entities
python -m langfuse_graphrag.cli query "MATCH (se:SemanticEntity) RETURN se.name, se.type LIMIT 10"

# Query communities
python -m langfuse_graphrag.cli query "MATCH (c:Community) WHERE c.level = 0 RETURN c.name, c.summary LIMIT 10"

# Query relationships
python -m langfuse_graphrag.cli query "MATCH (t:Trace)-[r:ABOUT]->(se:SemanticEntity) RETURN t.name, se.name LIMIT 10"
```

**Note:** The `query` command expects a query type (pattern, error, performance), not raw Cypher. For custom Cypher, use Neo4j Browser or the Python API directly.

---

## Using the Python API

You can also use the `QueryInterface` directly in Python:

```python
from langfuse_graphrag.query_interface import QueryInterface
from langfuse_graphrag.config import get_config

config = get_config()
query_interface = QueryInterface(config)

# Semantic search
results = query_interface.semantic_search(
    "cooperation strategies",
    entity_type="SemanticEntity",
    limit=5
)

# Pattern analysis
patterns = query_interface.pattern_analysis(
    pattern_type="session_traces",
    limit=10
)

# Error analysis
errors = query_interface.error_analysis(
    error_type="APIError",
    limit=20
)

# Performance analysis
performance = query_interface.performance_analysis(
    metric="cost",
    group_by="model",
    limit=10
)
```

---

## Example Use Cases

### Find all traces about "resource extraction"
```bash
python -m langfuse_graphrag.cli search "resource extraction" --entity-type Trace --property input_embedding
```

### Find communities related to "cooperation"
```bash
python -m langfuse_graphrag.cli search "cooperation" --entity-type Community --property embedding
```

### Analyze which sessions have the most traces
```bash
python -m langfuse_graphrag.cli analyze patterns --pattern-type session_traces --limit 20
```

### Find the most expensive operations
```bash
python -m langfuse_graphrag.cli analyze performance --metric cost --limit 20
```

### Find all errors
```bash
python -m langfuse_graphrag.cli analyze errors --limit 50
```

---

## Neo4j Browser Queries

You can also use Neo4j Browser directly for visual exploration. See the `.cypher` files in `langfuse_graphrag/`:
- `graph_queries.cypher` - General graph queries
- `community_hierarchy_queries.cypher` - Community exploration
- `content_exploration_queries.cypher` - Semantic entity exploration
- `graphical_queries.cypher` - Visualization queries

---

## Tips

1. **Start with `stats`** to understand your data volume
2. **Use `explore`** to understand the structure
3. **Use `search`** for semantic similarity queries
4. **Use `analyze`** for pattern and performance insights
5. **Save results** with `--output` for further analysis
6. **Adjust `--limit`** and `--threshold`** based on your needs


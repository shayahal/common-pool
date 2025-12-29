// ============================================
// Neo4j GraphRAG Graphical/Visualization Queries
// These queries return nodes and relationships for graph visualization
// ============================================

// 1. Visualize all nodes and relationships (limit for performance)
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100;

// 2. Visualize all Session nodes (even without relationships)
MATCH (s:Session)
RETURN s
LIMIT 50;

// 3. Visualize relationships by type - CONTAINS
MATCH (session:Session)-[r:CONTAINS]->(trace:Trace)
RETURN session, r, trace
LIMIT 50;

// 4. Visualize relationships by type - HAS_SPAN
MATCH (trace:Trace)-[r:HAS_SPAN]->(span:Span)
RETURN trace, r, span
LIMIT 50;

// 5. Visualize relationships by type - GENERATES
MATCH (span:Span)-[r:GENERATES]->(gen:Generation)
RETURN span, r, gen
LIMIT 50;

// 6. Visualize full trace hierarchy (Session → Trace → Span → Generation)
MATCH path = (session:Session)-[:CONTAINS]->(trace:Trace)-[:HAS_SPAN]->(span:Span)-[:GENERATES]->(gen:Generation)
RETURN path
LIMIT 20;

// 7. Visualize relationships with scores
MATCH (parent)-[r:HAS_SCORE]->(score:Score)
WHERE 'Trace' IN labels(parent) OR 'Span' IN labels(parent)
RETURN parent, r, score
LIMIT 30;

// 8. Visualize relationships with errors
MATCH (parent)-[r:HAS_ERROR]->(error:Error)
WHERE 'Trace' IN labels(parent) OR 'Span' IN labels(parent)
RETURN parent, r, error
LIMIT 30;

// 9. Visualize semantic entity relationships - MENTIONS
MATCH (gen:Generation)-[r:MENTIONS]->(entity:SemanticEntity)
RETURN gen, r, entity
LIMIT 50;

// 10. Visualize semantic entity relationships - ABOUT
MATCH (trace:Trace)-[r:ABOUT]->(entity:SemanticEntity)
RETURN trace, r, entity
LIMIT 50;

// 11. Visualize community structure
MATCH (entity:SemanticEntity)-[r:BELONGS_TO]->(community:Community)
RETURN entity, r, community
LIMIT 50;

// 12. Visualize temporal relationships - FOLLOWS
MATCH (a)-[r:FOLLOWS]->(b)
RETURN a, r, b
LIMIT 50;

// 13. Visualize model usage
MATCH (gen:Generation)-[r:USES_MODEL]->(model:Model)
RETURN gen, r, model
LIMIT 30;

// 14. Visualize similarity relationships
MATCH (a)-[r:SIMILAR_TO]->(b)
WHERE 'SemanticEntity' IN labels(a) OR 'Community' IN labels(a)
RETURN a, r, b
LIMIT 30;

// 15. Visualize error relationships to semantic entities
MATCH (error:Error)-[r:RELATED_TO]->(entity:SemanticEntity)
RETURN error, r, entity
LIMIT 30;

// 16. Visualize a specific session and all its connected nodes
MATCH path = (s:Session {id: $session_id})-[*1..3]-(connected)
RETURN path
LIMIT 100;
// Usage: Set $session_id parameter, e.g., "cmiolylif00t3ad08q3us4ui9"

// 17. Visualize nodes with most connections (hubs)
MATCH (n)-[r]-(connected)
WITH n, count(r) AS connection_count
ORDER BY connection_count DESC
LIMIT 20
MATCH (n)-[r]-(connected)
RETURN n, r, connected
LIMIT 100;

// 18. Visualize all relationships in a 2-hop neighborhood
MATCH path = (start:Session)-[*1..2]-(connected)
RETURN path
LIMIT 50;

// 19. Visualize error propagation (errors connected to traces/spans)
MATCH path = (error:Error)-[:HAS_ERROR*0..1]-(parent)-[*0..2]-(related)
RETURN path
LIMIT 30;

// 20. Visualize semantic entity clusters (entities in same community)
MATCH (e1:SemanticEntity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(e2:SemanticEntity)
WHERE e1 <> e2
RETURN e1, c, e2
LIMIT 50;

// 21. Visualize generation chains (generations that follow each other)
MATCH path = (g1:Generation)-[:FOLLOWS*1..3]->(g2:Generation)
RETURN path
LIMIT 20;

// 22. Visualize trace execution flow
MATCH path = (t1:Trace)-[:FOLLOWS*1..5]->(t2:Trace)
RETURN path
LIMIT 20;

// 23. Visualize all nodes connected to a specific trace
MATCH (t:Trace {id: $trace_id})-[*0..2]-(connected)
RETURN t, connected
LIMIT 100;
// Usage: Set $trace_id parameter

// 24. Visualize score relationships in detail
MATCH (parent)-[r:HAS_SCORE]->(score:Score)
RETURN parent, r, score
LIMIT 50;

// 25. Visualize the complete knowledge graph (use with caution on large datasets)
MATCH (n)-[r]->(m)
RETURN n, r, m;


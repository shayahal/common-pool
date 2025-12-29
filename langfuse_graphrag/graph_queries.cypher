// ============================================================
// Neo4j Graph Queries - View in Neo4j Browser (localhost:7474)
// Copy-paste these queries into the Neo4j Browser query box
// ============================================================

// ---------------------------------------------
// 1. GRAPH OVERVIEW - See everything
// ---------------------------------------------

// Show all nodes and relationships (limit for performance)
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100

// Count by type
MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC

// Count relationships
MATCH ()-[r]->() RETURN type(r) as type, count(*) as count ORDER BY count DESC

// Show full graph structure
MATCH path = (n)-[*1..3]-(m)
RETURN path
LIMIT 50

// ---------------------------------------------
// 2. SESSION → TRACE → SPAN HIERARCHY
// ---------------------------------------------

// Sessions with their traces
MATCH (s:Session)-[r:HAS_TRACE]->(t:Trace)
RETURN s, r, t
LIMIT 50

// Full hierarchy: Session → Trace → Span
MATCH path = (s:Session)-[:HAS_TRACE]->(t:Trace)-[:HAS_SPAN]->(sp:Span)
RETURN path
LIMIT 50

// Session → Trace → Generation (LLM calls)
MATCH path = (s:Session)-[:HAS_TRACE]->(t:Trace)-[:HAS_SPAN]->(sp:Span)-[:HAS_GENERATION]->(g:Generation)
RETURN path
LIMIT 50

// ---------------------------------------------
// 3. TEMPORAL RELATIONSHIPS
// ---------------------------------------------

// Traces ordered by time (NEXT relationships)
MATCH path = (t1:Trace)-[:NEXT*1..5]->(t2:Trace)
RETURN path
LIMIT 25

// Session timeline - all traces in a session ordered
MATCH (s:Session)-[:HAS_TRACE]->(t:Trace)
WHERE s.name STARTS WITH "game_exp"
RETURN s.name as session, collect(t.name) as traces
ORDER BY s.name

// ---------------------------------------------
// 4. GENERATION ANALYSIS
// ---------------------------------------------

// All generations with their parent spans
MATCH (sp:Span)-[:HAS_GENERATION]->(g:Generation)
RETURN sp.name as span, g.model as model, g.prompt as prompt, g.response as response
LIMIT 20

// Generations by model
MATCH (g:Generation)
RETURN g.model as model, count(*) as count
ORDER BY count DESC

// Generation cost analysis
MATCH (g:Generation)
WHERE g.cost IS NOT NULL
RETURN g.model as model, sum(g.cost) as total_cost, avg(g.cost) as avg_cost, count(*) as count
ORDER BY total_cost DESC

// ---------------------------------------------
// 5. ERROR TRACKING
// ---------------------------------------------

// Errors linked to their traces
MATCH (t:Trace)-[:HAS_ERROR]->(e:Error)
RETURN t.name as trace, e.error_type as type, e.message as message
LIMIT 20

// Errors with full context
MATCH path = (s:Session)-[:HAS_TRACE]->(t:Trace)-[:HAS_ERROR]->(e:Error)
RETURN path
LIMIT 25

// Error frequency by type
MATCH (e:Error)
RETURN e.error_type as type, count(*) as count
ORDER BY count DESC

// ---------------------------------------------
// 6. SCORE/METRICS ANALYSIS
// ---------------------------------------------

// Scores linked to traces
MATCH (t:Trace)-[:HAS_SCORE]->(sc:Score)
RETURN t.name as trace, sc.name as metric, sc.value as value
LIMIT 50

// Average scores by metric name
MATCH (sc:Score)
RETURN sc.name as metric, avg(sc.value) as avg_value, count(*) as count
ORDER BY count DESC

// ---------------------------------------------
// 7. SEMANTIC ENTITIES (GraphRAG)
// ---------------------------------------------

// All semantic entities
MATCH (se:SemanticEntity)
RETURN se.name as name, se.type as type, se.description as description
ORDER BY se.type, se.name
LIMIT 50

// Semantic entities by type
MATCH (se:SemanticEntity)
RETURN se.type as entity_type, count(*) as count
ORDER BY count DESC

// Entities with their community memberships
MATCH (se:SemanticEntity)-[:BELONGS_TO]->(c:Community)
RETURN se.name as entity, se.type as type, c.summary as community
LIMIT 30

// Entity co-occurrence network (entities in same community)
MATCH (e1:SemanticEntity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(e2:SemanticEntity)
WHERE id(e1) < id(e2)
RETURN e1.name as entity1, e2.name as entity2, count(c) as shared_communities
ORDER BY shared_communities DESC
LIMIT 50

// Community structure with members
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, collect(se.name) as members, count(se) as member_count
RETURN c.summary as community, member_count, members[0..5] as sample_members
ORDER BY member_count DESC
LIMIT 10

// Trace → Entity connections (via ABOUT)
MATCH (t:Trace)-[:ABOUT]->(se:SemanticEntity)
RETURN t.name as trace, collect(se.name) as entities
LIMIT 20

// ---------------------------------------------
// 8. PLAYER/GAME ANALYSIS (CPR-specific)
// ---------------------------------------------

// All player actions in order
MATCH (s:Session)
WHERE s.name CONTAINS "player_" AND s.name CONTAINS "_action_"
RETURN s.name as action, s.created_at as time
ORDER BY s.created_at

// Game summary traces
MATCH (s:Session)
WHERE s.name = "game_summary"
RETURN s

// Round metrics
MATCH (s:Session)
WHERE s.name CONTAINS "round_" AND s.name CONTAINS "_metrics"
RETURN s.name as round, s.created_at as time
ORDER BY s.created_at

// ---------------------------------------------
// 9. PATH QUERIES
// ---------------------------------------------

// Find all paths between two nodes
MATCH path = shortestPath((a)-[*]-(b))
WHERE a.name = "session_start" AND b.name = "game_summary"
RETURN path

// Find related entities within 2 hops
MATCH path = (n)-[*1..2]-(m)
WHERE n.name CONTAINS "player_0"
RETURN path
LIMIT 50

// ---------------------------------------------
// 10. AGGREGATION QUERIES
// ---------------------------------------------

// Count all node types
MATCH (n)
RETURN labels(n)[0] as type, count(*) as count
ORDER BY count DESC

// Count all relationship types
MATCH ()-[r]->()
RETURN type(r) as relationship, count(*) as count
ORDER BY count DESC

// Graph density metrics
MATCH (n)
WITH count(n) as nodes
MATCH ()-[r]->()
WITH nodes, count(r) as rels
RETURN nodes, rels, toFloat(rels) / nodes as avg_rels_per_node


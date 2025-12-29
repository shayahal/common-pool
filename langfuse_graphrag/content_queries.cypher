// ============================================
// Neo4j GraphRAG Content Exploration Queries
// Explore entities, claims, and semantic content
// ============================================

// 1. View all SemanticEntity nodes
MATCH (e:SemanticEntity)
RETURN e
LIMIT 50;

// 2. View all SemanticEntity nodes with details
MATCH (e:SemanticEntity)
RETURN e.id, e.name, e.type, e.description, e
LIMIT 30;

// 3. Visualize SemanticEntity relationships - MENTIONS
MATCH (gen:Generation)-[r:MENTIONS]->(entity:SemanticEntity)
RETURN gen, r, entity
LIMIT 50;

// 4. Visualize SemanticEntity relationships - ABOUT
MATCH (trace:Trace)-[r:ABOUT]->(entity:SemanticEntity)
RETURN trace, r, entity
LIMIT 50;

// 5. Visualize SemanticEntity relationships - RELATED_TO
MATCH (error:Error)-[r:RELATED_TO]->(entity:SemanticEntity)
RETURN error, r, entity
LIMIT 30;

// 6. View Communities and their semantic entities
MATCH (e:SemanticEntity)-[r:BELONGS_TO]->(c:Community)
RETURN e, r, c
LIMIT 50;

// 7. View all Communities
MATCH (c:Community)
RETURN c
LIMIT 30;

// 8. Communities with their summaries
MATCH (c:Community)
RETURN c.id, c.name, c.level, c.summary, c
LIMIT 20;

// 9. Semantic entities by type
MATCH (e:SemanticEntity)
WHERE e.type = $entity_type
RETURN e
LIMIT 50;
// Usage: Set parameter $entity_type, e.g., "concept", "topic", "action"

// 10. Visualize similarity relationships between entities
MATCH (e1:SemanticEntity)-[r:SIMILAR_TO]->(e2:SemanticEntity)
RETURN e1, r, e2
LIMIT 30;

// 11. Find entities mentioned in specific content
MATCH (gen:Generation {id: $generation_id})-[r:MENTIONS]->(entity:SemanticEntity)
RETURN gen, r, entity;
// Usage: Set $generation_id parameter

// 12. View Generation content with their mentioned entities
MATCH (gen:Generation)-[r:MENTIONS]->(entity:SemanticEntity)
RETURN gen.id, gen.prompt, gen.response, entity.name, entity.type
LIMIT 20;

// 13. View Trace content with their about entities
MATCH (trace:Trace)-[r:ABOUT]->(entity:SemanticEntity)
RETURN trace.id, trace.input, trace.output, entity.name, entity.type
LIMIT 20;

// 14. Find entities in a specific community
MATCH (e:SemanticEntity)-[:BELONGS_TO]->(c:Community {id: $community_id})
RETURN e, c;
// Usage: Set $community_id parameter

// 15. Visualize community hierarchy (if multi-level)
MATCH (c1:Community)-[r:SIMILAR_TO]->(c2:Community)
RETURN c1, r, c2
LIMIT 30;

// 16. Find most mentioned entities
MATCH (gen:Generation)-[:MENTIONS]->(entity:SemanticEntity)
WITH entity, count(*) AS mention_count
ORDER BY mention_count DESC
LIMIT 10
MATCH (gen:Generation)-[r:MENTIONS]->(entity)
RETURN gen, r, entity
LIMIT 50;

// 17. View Session input/output content (if available)
MATCH (s:Session)
WHERE s.input IS NOT NULL OR s.output IS NOT NULL
RETURN s.id, s.input, s.output, s
LIMIT 20;

// 18. View Trace input/output content
MATCH (t:Trace)
WHERE t.input IS NOT NULL OR t.output IS NOT NULL
RETURN t.id, t.input, t.output, t
LIMIT 20;

// 19. View Generation prompts and responses
MATCH (g:Generation)
WHERE g.prompt IS NOT NULL OR g.response IS NOT NULL
RETURN g.id, g.prompt, g.response, g.reasoning, g
LIMIT 20;

// 20. Complete content graph: Generations → Entities → Communities
MATCH path = (gen:Generation)-[:MENTIONS]->(entity:SemanticEntity)-[:BELONGS_TO]->(comm:Community)
RETURN path
LIMIT 30;

// 21. Find entities related to errors
MATCH (error:Error)-[:RELATED_TO]->(entity:SemanticEntity)
RETURN error, entity
LIMIT 30;

// 22. View all content nodes (Sessions, Traces, Generations) with text
MATCH (n)
WHERE (n:Session OR n:Trace OR n:Generation) 
  AND (n.input IS NOT NULL OR n.output IS NOT NULL OR n.prompt IS NOT NULL OR n.response IS NOT NULL)
RETURN n
LIMIT 50;

// 23. Semantic entity clusters (entities in same community)
MATCH (e1:SemanticEntity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(e2:SemanticEntity)
WHERE e1 <> e2
RETURN e1, c, e2
LIMIT 50;

// 24. Find entities by keyword in name or description
MATCH (e:SemanticEntity)
WHERE toLower(e.name) CONTAINS toLower($keyword) 
   OR toLower(e.description) CONTAINS toLower($keyword)
RETURN e;
// Usage: Set $keyword parameter, e.g., "error", "cooperation", "resource"

// 25. Complete semantic knowledge graph
MATCH (n)-[r]->(m)
WHERE 'SemanticEntity' IN labels(n) OR 'SemanticEntity' IN labels(m)
   OR 'Community' IN labels(n) OR 'Community' IN labels(m)
RETURN n, r, m
LIMIT 100;


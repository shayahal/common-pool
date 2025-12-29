// ============================================
// Neo4j GraphRAG Content & Entity Exploration Queries
// Explore semantic entities, claims, and content
// ============================================

// 1. Check if SemanticEntity nodes exist
MATCH (e:SemanticEntity)
RETURN count(e) AS semantic_entity_count;

// 2. View all SemanticEntity nodes (graphical)
MATCH (e:SemanticEntity)
RETURN e
LIMIT 50;

// 3. View SemanticEntity details
MATCH (e:SemanticEntity)
RETURN e.id, e.name, e.type, e.description
ORDER BY e.type, e.name
LIMIT 30;

// 4. View all Community nodes (graphical)
MATCH (c:Community)
RETURN c
LIMIT 30;

// 5. View Community details with summaries
MATCH (c:Community)
RETURN c.id, c.name, c.level, c.summary
ORDER BY c.level, c.name
LIMIT 20;

// 6. Visualize SemanticEntity → Community relationships
MATCH (e:SemanticEntity)-[r:BELONGS_TO]->(c:Community)
RETURN e, r, c
LIMIT 50;

// 7. Visualize Generation → SemanticEntity (MENTIONS)
MATCH (g:Generation)-[r:MENTIONS]->(e:SemanticEntity)
RETURN g, r, e
LIMIT 50;

// 8. Visualize Trace → SemanticEntity (ABOUT)
MATCH (t:Trace)-[r:ABOUT]->(e:SemanticEntity)
RETURN t, r, e
LIMIT 50;

// 9. Visualize Error → SemanticEntity (RELATED_TO)
MATCH (err:Error)-[r:RELATED_TO]->(e:SemanticEntity)
RETURN err, r, e
LIMIT 30;

// 10. Complete semantic knowledge graph
MATCH (n)-[r]->(m)
WHERE 'SemanticEntity' IN labels(n) OR 'SemanticEntity' IN labels(m)
   OR 'Community' IN labels(n) OR 'Community' IN labels(m)
RETURN n, r, m
LIMIT 100;

// 11. Find most mentioned semantic entities
MATCH (g:Generation)-[:MENTIONS]->(e:SemanticEntity)
WITH e, count(*) AS mention_count
ORDER BY mention_count DESC
LIMIT 10
MATCH (g:Generation)-[r:MENTIONS]->(e)
RETURN g, r, e
LIMIT 50;

// 12. Semantic entities by type
MATCH (e:SemanticEntity)
WHERE e.type = $entity_type
RETURN e
LIMIT 50;
// Usage: Set parameter $entity_type (e.g., "concept", "topic", "action", "person")

// 13. Find entities by keyword search
MATCH (e:SemanticEntity)
WHERE toLower(e.name) CONTAINS toLower($keyword) 
   OR toLower(e.description) CONTAINS toLower($keyword)
RETURN e;
// Usage: Set $keyword parameter

// 14. View Generation content with mentioned entities
MATCH (g:Generation)-[r:MENTIONS]->(e:SemanticEntity)
RETURN g.id, g.prompt, g.response, e.name AS entity_name, e.type AS entity_type
LIMIT 20;

// 15. View Trace content with about entities
MATCH (t:Trace)-[r:ABOUT]->(e:SemanticEntity)
RETURN t.id, t.input, t.output, e.name AS entity_name, e.type AS entity_type
LIMIT 20;

// 16. Entities in a specific community
MATCH (e:SemanticEntity)-[:BELONGS_TO]->(c:Community {id: $community_id})
RETURN e, c;
// Usage: Set $community_id parameter

// 17. Community clusters (entities in same community)
MATCH (e1:SemanticEntity)-[:BELONGS_TO]->(c:Community)<-[:BELONGS_TO]-(e2:SemanticEntity)
WHERE e1 <> e2
RETURN e1, c, e2
LIMIT 50;

// 18. Similar entities (SIMILAR_TO relationships)
MATCH (e1:SemanticEntity)-[r:SIMILAR_TO]->(e2:SemanticEntity)
RETURN e1, r, e2
LIMIT 30;

// 19. Complete content flow: Generation → Entity → Community
MATCH path = (g:Generation)-[:MENTIONS]->(e:SemanticEntity)-[:BELONGS_TO]->(c:Community)
RETURN path
LIMIT 30;

// 20. View Session metadata content (if it contains text/claims)
MATCH (s:Session)
WHERE s.metadata IS NOT NULL
RETURN s.id, s.name, s.metadata
LIMIT 10;

// 21. View Generation prompts and responses (content)
MATCH (g:Generation)
WHERE g.prompt IS NOT NULL OR g.response IS NOT NULL
RETURN g.id, g.prompt, g.response, g.reasoning, g
LIMIT 20;

// 22. View Trace input/output (content)
MATCH (t:Trace)
WHERE t.input IS NOT NULL OR t.output IS NOT NULL
RETURN t.id, t.input, t.output, t
LIMIT 20;

// 23. All content nodes with text fields
MATCH (n)
WHERE (n:Session OR n:Trace OR n:Generation) 
  AND (n.input IS NOT NULL OR n.output IS NOT NULL 
       OR n.prompt IS NOT NULL OR n.response IS NOT NULL
       OR n.metadata IS NOT NULL)
RETURN n
LIMIT 50;

// 24. Error content with related entities
MATCH (err:Error)-[:RELATED_TO]->(e:SemanticEntity)
RETURN err.id, err.message, err.type, e.name AS related_entity
LIMIT 20;

// 25. Full semantic graph with all relationships
MATCH (n)-[r]->(m)
WHERE 'SemanticEntity' IN labels(n) OR 'SemanticEntity' IN labels(m)
   OR 'Community' IN labels(n) OR 'Community' IN labels(m)
   OR 'Generation' IN labels(n) OR 'Trace' IN labels(n)
RETURN n, r, m
LIMIT 100;


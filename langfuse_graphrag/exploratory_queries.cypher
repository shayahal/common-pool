// ============================================
// Neo4j GraphRAG Exploratory Queries
// ============================================

// 1. Overview: Count all node types
MATCH (n)
RETURN labels(n)[0] AS node_type, count(n) AS count
ORDER BY count DESC;

// 2. Overview: Count all relationship types
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC;

// 3. Graph structure: See all node types and their properties
MATCH (n)
WITH labels(n)[0] AS label, keys(n) AS props
RETURN DISTINCT label, collect(DISTINCT props)[0..5] AS sample_properties
ORDER BY label;

// 4. Sample Session nodes with their properties
MATCH (s:Session)
RETURN s.id, s.name, s.user_id, s.created_at, s.metadata
LIMIT 10;

// 5. Check if there are any relationships
MATCH (a)-[r]->(b)
RETURN type(r) AS rel_type, labels(a)[0] AS from_type, labels(b)[0] AS to_type, count(*) AS count
ORDER BY count DESC
LIMIT 20;

// 6. Find nodes with the most relationships (if any exist)
MATCH (n)-[r]-()
RETURN labels(n)[0] AS node_type, n.id AS node_id, count(r) AS relationship_count
ORDER BY relationship_count DESC
LIMIT 20;

// 7. Explore Session metadata structure
MATCH (s:Session)
WHERE s.metadata IS NOT NULL
RETURN s.id, s.metadata
LIMIT 5;

// 8. Check for any Generation, Trace, or Span nodes
MATCH (n)
WHERE 'Generation' IN labels(n) OR 'Trace' IN labels(n) OR 'Span' IN labels(n)
RETURN labels(n)[0] AS node_type, count(*) AS count;

// 9. Find Sessions with specific properties (e.g., high scores)
MATCH (s:Session)
WHERE s.metadata IS NOT NULL
RETURN s.id, s.name, 
       s.metadata.avg_cooperation_index AS avg_coop,
       s.metadata.final_resource_level AS final_level,
       s.metadata.tragedy_occurred AS tragedy
ORDER BY s.metadata.avg_cooperation_index DESC NULLS LAST
LIMIT 10;

// 10. Visualize the full graph structure (be careful with large datasets)
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100;

// 11. Find isolated nodes (nodes with no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] AS node_type, count(*) AS isolated_count;

// 12. Check for duplicate IDs
MATCH (n)
WITH labels(n)[0] AS label, n.id AS id, count(*) AS count
WHERE count > 1
RETURN label, id, count
ORDER BY count DESC;

// 13. Sample all unique property keys across all nodes
MATCH (n)
UNWIND keys(n) AS key
RETURN DISTINCT key, count(*) AS frequency
ORDER BY frequency DESC;

// 14. Find Sessions with metadata containing specific fields
MATCH (s:Session)
WHERE s.metadata IS NOT NULL AND s.metadata.avg_cooperation_index IS NOT NULL
RETURN s.id, s.name, 
       s.metadata.avg_cooperation_index AS cooperation,
       s.metadata.total_rounds AS rounds,
       s.metadata.sustainability_score AS sustainability
ORDER BY s.metadata.avg_cooperation_index DESC
LIMIT 10;

// 15. Check database statistics
MATCH (n)
RETURN 
  count(DISTINCT labels(n)) AS unique_node_types,
  count(n) AS total_nodes,
  count{(n)-[]->()} AS total_outgoing_relationships,
  count{()-[r]->()} AS total_relationships;


// ============================================================
// Community Hierarchy Queries - Top to Bottom
// View communities from top-level (level 0) to bottom-level
// ============================================================

// ---------------------------------------------
// 1. COMMUNITIES BY LEVEL (Top to Bottom)
// ---------------------------------------------

// All communities ordered by level (top-level first)
MATCH (c:Community)
RETURN c.level as level, c.name as name, c.summary as summary, c.id as id
ORDER BY c.level ASC, c.name ASC

// Count communities by level
MATCH (c:Community)
RETURN c.level as level, count(*) as community_count
ORDER BY level ASC

// Top-level communities only (level 0)
MATCH (c:Community)
WHERE c.level = 0
RETURN c.name as name, c.summary as summary, c.id as id
ORDER BY c.name ASC

// Bottom-level communities (highest level number)
MATCH (c:Community)
WITH max(c.level) as max_level
MATCH (c:Community)
WHERE c.level = max_level
RETURN c.name as name, c.summary as summary, c.level as level, c.id as id
ORDER BY c.name ASC

// ---------------------------------------------
// 2. COMMUNITY HIERARCHY (Parent-Child)
// ---------------------------------------------

// Community hierarchy: Parent → Child (via CONTAINS)
MATCH path = (parent:Community)-[:CONTAINS]->(child:Community)
RETURN parent.name as parent, parent.level as parent_level, 
       child.name as child, child.level as child_level
ORDER BY parent.level ASC, parent.name ASC, child.level ASC

// Full community hierarchy tree (all levels)
MATCH path = (top:Community)-[:CONTAINS*]->(bottom:Community)
WHERE top.level = 0
RETURN path
ORDER BY length(path) ASC
LIMIT 50

// Communities with their parent communities
MATCH (child:Community)<-[:CONTAINS]-(parent:Community)
RETURN parent.name as parent, parent.level as parent_level,
       child.name as child, child.level as child_level
ORDER BY parent.level ASC, child.level ASC

// Top-level communities with all their descendants
MATCH (top:Community)-[:CONTAINS*0..]->(descendant:Community)
WHERE top.level = 0
WITH top, collect(DISTINCT descendant) as descendants
RETURN top.name as top_community, top.level as top_level,
       size(descendants) as total_communities_in_tree,
       [d IN descendants WHERE d.level = top.level + 1 | d.name] as direct_children
ORDER BY top.name ASC
LIMIT 20

// ---------------------------------------------
// 3. COMMUNITIES WITH MEMBER ENTITIES (Top to Bottom)
// ---------------------------------------------

// Communities with their semantic entities, ordered by level
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, collect(se.name) as entities, count(se) as entity_count
RETURN c.level as level, c.name as community, c.summary as summary,
       entity_count, entities[0..10] as sample_entities
ORDER BY c.level ASC, entity_count DESC

// Top-level communities with all their entities
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WHERE c.level = 0
WITH c, collect(se.name) as entities, count(se) as entity_count
RETURN c.name as community, c.summary as summary, entity_count,
       entities[0..15] as sample_entities
ORDER BY entity_count DESC

// Communities by level with entity counts
MATCH (c:Community)
OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, count(se) as entity_count
RETURN c.level as level, count(c) as communities_at_level,
       sum(entity_count) as total_entities,
       avg(entity_count) as avg_entities_per_community
ORDER BY level ASC

// ---------------------------------------------
// 4. DETAILED COMMUNITY STRUCTURE (Top to Bottom)
// ---------------------------------------------

// Complete community structure: Level → Name → Summary → Entity Count
MATCH (c:Community)
OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, count(se) as entity_count
RETURN c.level as level, c.name as name, c.summary as summary,
       entity_count, c.id as id
ORDER BY c.level ASC, entity_count DESC, c.name ASC

// Communities with their entity types breakdown
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, se.type as entity_type, count(se) as type_count
WITH c, collect({type: entity_type, count: type_count}) as type_breakdown
RETURN c.level as level, c.name as community, c.summary as summary,
       type_breakdown
ORDER BY c.level ASC, c.name ASC

// ---------------------------------------------
// 5. VISUALIZATION QUERIES
// ---------------------------------------------

// Visualize community hierarchy (for Neo4j Browser)
MATCH path = (top:Community)-[:CONTAINS*0..2]->(child:Community)
WHERE top.level = 0
RETURN path
LIMIT 30

// Visualize communities with their entities (level 0 only for clarity)
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WHERE c.level = 0
RETURN c, se
LIMIT 100

// Visualize full community-entity network (all levels)
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
RETURN c, se
LIMIT 200

// ---------------------------------------------
// 6. STATISTICS BY LEVEL
// ---------------------------------------------

// Summary statistics by community level
MATCH (c:Community)
OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c.level as level, c, count(se) as entity_count
WITH level, 
     count(c) as num_communities,
     sum(entity_count) as total_entities,
     avg(entity_count) as avg_entities,
     min(entity_count) as min_entities,
     max(entity_count) as max_entities
RETURN level, num_communities, total_entities, 
       round(avg_entities, 2) as avg_entities,
       min_entities, max_entities
ORDER BY level ASC

// Most populated communities at each level
MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, count(se) as entity_count
WITH c.level as level, c, entity_count
ORDER BY level ASC, entity_count DESC
WITH level, collect({name: c.name, count: entity_count, summary: c.summary})[0..5] as top_communities
RETURN level, top_communities
ORDER BY level ASC

// ---------------------------------------------
// 7. NAVIGATION QUERIES
// ---------------------------------------------

// Find a specific community and see its level and members
MATCH (c:Community {name: $community_name})<-[:BELONGS_TO]-(se:SemanticEntity)
RETURN c.level as level, c.name as community, c.summary as summary,
       collect(se.name) as entities, count(se) as entity_count
// Usage: Set $community_name parameter (e.g., "Concept: Game")

// Find communities at a specific level
MATCH (c:Community)
WHERE c.level = $level
OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
WITH c, count(se) as entity_count
RETURN c.name as name, c.summary as summary, entity_count, c.id as id
ORDER BY entity_count DESC
// Usage: Set $level parameter (e.g., 0 for top-level)

// Find path from top-level to a specific community
MATCH path = (top:Community)-[:CONTAINS*]->(target:Community {id: $community_id})
WHERE top.level = 0
RETURN path, 
       [n IN nodes(path) | n.name] as path_names,
       [n IN nodes(path) | n.level] as path_levels
// Usage: Set $community_id parameter

// ---------------------------------------------
// 8. COMPREHENSIVE TOP-TO-BOTTOM VIEW
// ---------------------------------------------

// Complete hierarchical view: Level → Communities → Entities
MATCH (c:Community)
OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
OPTIONAL MATCH (c)-[:CONTAINS]->(child:Community)
WITH c, 
     count(DISTINCT se) as entity_count,
     count(DISTINCT child) as child_count,
     collect(DISTINCT se.name)[0..5] as sample_entities
RETURN c.level as level,
       c.name as community_name,
       c.summary as summary,
       entity_count as entities,
       child_count as child_communities,
       sample_entities
ORDER BY c.level ASC, entity_count DESC, c.name ASC


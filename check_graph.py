#!/usr/bin/env python3
"""Check the Neo4j graph contents."""

from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.config import get_config

config = get_config()
manager = Neo4jManager(config)

print("=" * 60)
print("GRAPH OVERVIEW")
print("=" * 60)

# Node counts
result = manager.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(*) as count ORDER BY count DESC')
print("\nNode Types:")
for row in result:
    print(f"  {row['type']}: {row['count']}")

# Relationship counts
result = manager.execute_query('MATCH ()-[r]->() RETURN type(r) as rel, count(*) as count ORDER BY count DESC')
print("\nRelationship Types:")
if result:
    for row in result:
        print(f"  {row['rel']}: {row['count']}")
else:
    print("  (no relationships yet)")

# Sample relationships
result = manager.execute_query('MATCH (n)-[r]->(m) RETURN n.name as src, type(r) as rel, m.name as dst LIMIT 10')
print("\nSample Relationships:")
if result:
    for row in result:
        print(f"  ({row['src']}) --[{row['rel']}]--> ({row['dst']})")
else:
    print("  (none found - need to ingest Trace/Generation exports)")

print("\n" + "=" * 60)
print("SAMPLE SESSIONS")
print("=" * 60)
result = manager.execute_query('''
    MATCH (s:Session) 
    RETURN s.id as id, s.name as name, s.created_at as created_at
    ORDER BY s.created_at DESC
    LIMIT 5
''')
for row in result:
    node_id = row['id'][:16] + "..." if row['id'] else 'N/A'
    print(f"  {node_id} | {row['name']} | {row['created_at']}")

# Game sessions
print("\n" + "=" * 60)
print("GAME SESSIONS (game_exp_*)")
print("=" * 60)
result = manager.execute_query('''
    MATCH (s:Session) 
    WHERE s.name STARTS WITH "game_exp"
    RETURN s.name as name
    LIMIT 5
''')
if result:
    for row in result:
        print(f"  {row['name']}")
else:
    print("  (no game_exp sessions found)")

# Player actions
print("\n" + "=" * 60)
print("PLAYER ACTIONS")
print("=" * 60)
result = manager.execute_query('''
    MATCH (s:Session) 
    WHERE s.name CONTAINS "player_" AND s.name CONTAINS "_action_"
    RETURN s.name as name, s.created_at as time
    ORDER BY s.created_at
    LIMIT 10
''')
if result:
    for row in result:
        print(f"  {row['name']} @ {row['time']}")
else:
    print("  (no player actions found)")

# Semantic entities
print("\n" + "=" * 60)
print("SEMANTIC ENTITIES (top 15)")
print("=" * 60)
result = manager.execute_query('''
    MATCH (se:SemanticEntity)
    RETURN se.name as name, se.type as type, se.description as desc
    LIMIT 15
''')
if result:
    for row in result:
        desc = row['desc'][:60] + "..." if row['desc'] and len(row['desc']) > 60 else row['desc']
        print(f"  [{row['type']}] {row['name']}: {desc}")
else:
    print("  (no semantic entities found)")

# Communities
print("\n" + "=" * 60)
print("COMMUNITIES (top 5)")
print("=" * 60)
result = manager.execute_query('''
    MATCH (c:Community)-[:BELONGS_TO]-(se:SemanticEntity)
    WITH c, count(se) as member_count
    RETURN c.summary as summary, member_count
    ORDER BY member_count DESC
    LIMIT 5
''')
if result:
    for row in result:
        summary = row['summary'][:80] + "..." if row['summary'] and len(row['summary']) > 80 else row['summary']
        print(f"  ({row['member_count']} members) {summary}")
else:
    print("  (no communities found)")

# Entity relationships (MENTIONS)
print("\n" + "=" * 60)
print("CONTENT RELATIONSHIPS")
print("=" * 60)
result = manager.execute_query('''
    MATCH (t:Trace)-[r:MENTIONS]->(se:SemanticEntity)
    RETURN t.name as trace, type(r) as rel, se.name as entity
    LIMIT 10
''')
if result:
    for row in result:
        print(f"  {row['trace']} --[{row['rel']}]--> {row['entity']}")
else:
    # Try other relationship types
    result = manager.execute_query('''
        MATCH (n)-[r]->(m)
        WHERE NOT type(r) IN ['CONTAINS', 'BELONGS_TO']
        RETURN labels(n)[0] as from_type, type(r) as rel, labels(m)[0] as to_type, count(*) as cnt
        ORDER BY cnt DESC
        LIMIT 5
    ''')
    if result:
        for row in result:
            print(f"  {row['from_type']} --[{row['rel']}]--> {row['to_type']}: {row['cnt']} relationships")
    else:
        print("  (no content relationships yet - run GraphRAG to extract)")

manager.close()
print("\nDone!")


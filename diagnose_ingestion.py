"""Diagnostic script to investigate data ingestion issues."""

import csv
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.config import get_config

# Read CSV structure
print("=== CSV STRUCTURE ===")
with open('data/1-exp-10-games-traces.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f"Total rows: {len(rows)}")
    print(f"Columns: {len(reader.fieldnames)}")
    
    # Sample first few rows
    print("\nFirst 3 rows sample:")
    for i, row in enumerate(rows[:3]):
        print(f"\nRow {i+1}:")
        print(f"  id: {row.get('id')}")
        print(f"  sessionId: {row.get('sessionId')}")
        print(f"  name: {row.get('name')}")
        print(f"  userId: {row.get('userId')}")
        if row.get('metadata'):
            print(f"  metadata (first 200 chars): {row.get('metadata')[:200]}")

# Query Neo4j
print("\n\n=== NEO4J DATABASE ===")
manager = Neo4jManager(get_config())

# Node counts
result = manager.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY type')
print("\nNode counts:")
for record in result:
    print(f"  {record['type']}: {record['count']}")

# Relationship counts
result = manager.execute_query('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY rel_type')
print("\nRelationship counts:")
for record in result:
    print(f"  {record['rel_type']}: {record['count']}")

# Sessions and their trace counts
result = manager.execute_query('''
    MATCH (s:Session)-[r:CONTAINS]->(t:Trace)
    RETURN s.id as session_id, count(t) as trace_count
    ORDER BY trace_count DESC
    LIMIT 10
''')
print("\nSessions and their trace counts:")
for record in result:
    print(f"  Session {record['session_id']}: {record['trace_count']} traces")

# Check for self-relationships
result = manager.execute_query('''
    MATCH (t1:Trace)-[r:SAME_ROUND|NEXT_ROUND]->(t2:Trace)
    WHERE t1.id = t2.id
    RETURN t1.id as trace_id, type(r) as rel_type, count(r) as count
    LIMIT 10
''')
print("\nSelf-relationships (SAME_ROUND/NEXT_ROUND):")
self_rels = list(result)
if self_rels:
    for record in self_rels:
        print(f"  Trace {record['trace_id']}: {record['count']} {record['rel_type']} to itself")
else:
    print("  None found")

# Sample trace IDs and their names
result = manager.execute_query('''
    MATCH (t:Trace)
    RETURN t.id as trace_id, t.name as trace_name, t.session_id as session_id
    LIMIT 10
''')
print("\nSample Trace IDs and names:")
for record in result:
    print(f"  ID: {record['trace_id']}, Name: {record['trace_name']}, Session: {record['session_id']}")

# Sample session IDs
result = manager.execute_query('''
    MATCH (s:Session)
    RETURN s.id as session_id
    LIMIT 10
''')
print("\nSample Session IDs:")
for record in result:
    print(f"  {record['session_id']}")

# Check SAME_ROUND relationships (all, not just self)
result = manager.execute_query('''
    MATCH (t1:Trace)-[r:SAME_ROUND]->(t2:Trace)
    RETURN t1.id as t1_id, t1.name as t1_name, t2.id as t2_id, t2.name as t2_name
    LIMIT 10
''')
print("\nSample SAME_ROUND relationships:")
for record in result:
    print(f"  {record['t1_id']} ({record['t1_name']}) -> {record['t2_id']} ({record['t2_name']})")

# Check NEXT_ROUND relationships
result = manager.execute_query('''
    MATCH (t1:Trace)-[r:NEXT_ROUND]->(t2:Trace)
    RETURN t1.id as t1_id, t1.name as t1_name, t2.id as t2_id, t2.name as t2_name
    LIMIT 10
''')
print("\nSample NEXT_ROUND relationships:")
for record in result:
    print(f"  {record['t1_id']} ({record['t1_name']}) -> {record['t2_id']} ({record['t2_name']})")

manager.close()


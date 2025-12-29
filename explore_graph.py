#!/usr/bin/env python3
"""Explore the Neo4j graph with semantic entities and relationships."""

from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.config import get_config

manager = Neo4jManager(get_config())

print("=" * 70)
print("GRAPH STRUCTURE")
print("=" * 70)

# Node types
result = manager.execute_query('MATCH (n) RETURN labels(n)[0] as type, count(*) as cnt ORDER BY cnt DESC')
print("\nNode Types:")
for r in result:
    print(f"  {r['type']}: {r['cnt']}")

# Relationship types
result = manager.execute_query('MATCH ()-[r]->() RETURN type(r) as type, count(*) as cnt ORDER BY cnt DESC')
print("\nRelationship Types:")
for r in result:
    print(f"  {r['type']}: {r['cnt']}")

print("\n" + "=" * 70)
print("STRUCTURAL RELATIONSHIPS")
print("=" * 70)

# Session -> Trace
result = manager.execute_query('''
    MATCH (s:Session)-[r:CONTAINS]->(t:Trace)
    RETURN s.name as session, t.name as trace
    LIMIT 10
''')
print("\nSession -> Trace (CONTAINS):")
if result:
    for r in result:
        print(f"  {r['session'][:30]}... -> {r['trace']}")
else:
    print("  (none)")

print("\n" + "=" * 70)
print("SEMANTIC ENTITIES")
print("=" * 70)

# Entity types
result = manager.execute_query('MATCH (se:SemanticEntity) RETURN se.type as type, count(*) as cnt ORDER BY cnt DESC')
print("\nEntity Types:")
for r in result:
    print(f"  {r['type']}: {r['cnt']}")

# Sample entities
result = manager.execute_query('''
    MATCH (se:SemanticEntity)
    RETURN se.name as name, se.type as type, left(se.description, 80) as desc
    LIMIT 10
''')
print("\nSample Entities:")
for r in result:
    print(f"  [{r['type']}] {r['name']}")
    if r['desc']:
        print(f"    -> {r['desc']}...")

print("\n" + "=" * 70)
print("COMMUNITIES")
print("=" * 70)

# Communities with member counts
result = manager.execute_query('''
    MATCH (c:Community)<-[:BELONGS_TO]-(se:SemanticEntity)
    WITH c, count(se) as cnt, collect(se.name)[0..3] as sample
    RETURN c.summary as summary, cnt, sample
    ORDER BY cnt DESC
    LIMIT 5
''')
print("\nTop Communities:")
for r in result:
    summary = r['summary'][:60] + "..." if r['summary'] and len(r['summary']) > 60 else r['summary']
    print(f"  ({r['cnt']} members) {summary}")
    print(f"    Sample: {', '.join(r['sample'][:3])}")

print("\n" + "=" * 70)
print("CONTENT CONNECTIONS")
print("=" * 70)

# Trace -> Entity (ABOUT)
result = manager.execute_query('''
    MATCH (t:Trace)-[:ABOUT]->(se:SemanticEntity)
    RETURN t.name as trace, collect(se.name) as entities
    LIMIT 5
''')
print("\nTrace -> Entity (ABOUT):")
if result:
    for r in result:
        print(f"  {r['trace']}: {', '.join(r['entities'][:3])}")
else:
    print("  (none - ABOUT relationships not yet created)")

manager.close()
print("\n" + "=" * 70)
print("View in Neo4j Browser: http://localhost:7474")
print("=" * 70)


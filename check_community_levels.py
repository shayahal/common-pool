"""Quick script to check community levels in Neo4j."""
from langfuse_graphrag.neo4j_manager import Neo4jManager
from langfuse_graphrag.config import get_config

mgr = Neo4jManager(get_config())
result = mgr.execute_query("""
    MATCH (c:Community)
    RETURN c.level as level, count(*) as count
    ORDER BY level ASC
""")

print("Community levels:")
for r in result:
    print(f"  Level {r['level']}: {r['count']} communities")

# Show sample communities at each level
print("\nSample communities by level:")
for r in result:
    level = r['level']
    sample = mgr.execute_query(f"""
        MATCH (c:Community)
        WHERE c.level = {level}
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(se:SemanticEntity)
        WITH c, count(se) as entity_count
        RETURN c.name as name, c.summary as summary, entity_count
        ORDER BY entity_count DESC
        LIMIT 3
    """)
    print(f"\n  Level {level}:")
    for s in sample:
        print(f"    - {s['name']}: {s['entity_count']} entities")
        print(f"      Summary: {s['summary'][:80]}...")

mgr.close()


#!/usr/bin/env python3
"""Interactive Graphiti querying playground for exploring game traces.

This script provides an interactive REPL for querying the Graphiti knowledge graph
with natural language questions about player behavior, game outcomes, and more.

Examples:
    - "what does player 1 usually do?"
    - "show me all games where the resource was depleted"
    - "what are the most common actions taken by players?"
    - "find traces related to player 2 in round 5"
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Any, Union

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from cpr_game.config import CONFIG
from cpr_game.logger_setup import get_logger, setup_logging

# Setup logging (but keep it minimal for interactive use)
setup_logging()
logger = get_logger(__name__)

try:
    from graphiti_core import Graphiti
    from graphiti_core.driver.falkordb_driver import FalkorDriver
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    print("ERROR: Graphiti not available. Install with: pip install graphiti-core[falkordb]")
    sys.exit(1)


class GraphitiPlayground:
    """Interactive playground for querying Graphiti knowledge graph."""
    
    def __init__(self):
        """Initialize the playground with Graphiti connection."""
        self.graphiti: Optional[Graphiti] = None
        self.group_id = CONFIG.get("falkordb_group_id", "cpr-game-traces")
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to FalkorDB via Graphiti.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Get port as int (CONFIG already has it as int)
            port = CONFIG.get("falkordb_port", 6379)
            if isinstance(port, str):
                port = int(port)
            
            # Try to use group_id as database name, fallback to default_db
            # In FalkorDB, the graph name might match the group_id
            database_name = CONFIG.get("falkordb_group_id", "cpr-game-traces")
            
            falkor_driver = FalkorDriver(
                host=CONFIG.get("falkordb_host", "localhost"),
                port=port,
                username=CONFIG.get("falkordb_username"),
                password=CONFIG.get("falkordb_password"),
                database=database_name,  # Use group_id as database name
            )
            self.graphiti = Graphiti(graph_driver=falkor_driver)
            
            # Try to build indices (might already exist, that's ok)
            try:
                await self.graphiti.build_indices_and_constraints()
            except Exception as e:
                # Indices might already exist, continue anyway
                logger.debug(f"Index initialization note: {e}")
            
            self._connected = True
            logger.info("Connected to Graphiti/FalkorDB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Graphiti/FalkorDB: {e}", exc_info=True)
            self._connected = False
            return False
    
    async def search(self, query: str, limit: int = 10) -> List[Any]:
        """Search the knowledge graph with a natural language query.
        
        Args:
            query: Natural language query string
            limit: Maximum number of results to return
            
        Returns:
            List of search results
        """
        if not self._connected or not self.graphiti:
            raise RuntimeError("Not connected to Graphiti. Call connect() first.")
        
        try:
            results = await self.graphiti.search(query)
            # Convert to list and limit
            result_list = list(results)
            return result_list[:limit]
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise
    
    async def query_database_directly(self) -> str:
        """Query the database directly using Cypher to see what's stored.
        
        Returns:
            Formatted results showing node and episode counts
        """
        if not self._connected or not self.graphiti:
            return "Not connected to Graphiti."
        
        try:
            driver = self.graphiti.driver
            results_lines = ["\n=== Direct Database Query ===\n"]
            
            # Count all nodes
            try:
                node_result = await driver.execute_query("MATCH (n) RETURN count(n) as node_count")
                if node_result:
                    records, _, _ = node_result
                    if records and len(records) > 0:
                        node_count = records[0].get('node_count', 0)
                        results_lines.append(f"Total nodes: {node_count}")
            except Exception as e:
                results_lines.append(f"Error counting nodes: {e}")
            
            # Count episodic nodes (episodes)
            try:
                episode_result = await driver.execute_query(
                    "MATCH (n:EpisodicNode) RETURN count(n) as episode_count"
                )
                if episode_result:
                    records, _, _ = episode_result
                    if records and len(records) > 0:
                        episode_count = records[0].get('episode_count', 0)
                        results_lines.append(f"Episodic nodes (episodes): {episode_count}")
            except Exception as e:
                results_lines.append(f"Error counting episodes: {e}")
            
            # Count entity nodes
            try:
                entity_result = await driver.execute_query(
                    "MATCH (n:EntityNode) RETURN count(n) as entity_count"
                )
                if entity_result:
                    records, _, _ = entity_result
                    if records and len(records) > 0:
                        entity_count = records[0].get('entity_count', 0)
                        results_lines.append(f"Entity nodes: {entity_count}")
            except Exception as e:
                results_lines.append(f"Error counting entities: {e}")
            
            # Get sample episode names
            try:
                sample_result = await driver.execute_query(
                    "MATCH (n:EpisodicNode) RETURN n.name as name, n.description as description LIMIT 5"
                )
                if sample_result:
                    records, _, _ = sample_result
                    if records and len(records) > 0:
                        results_lines.append("\nSample episodes:")
                        for record in records[:5]:
                            name = record.get('name', 'N/A')
                            desc = record.get('description', 'N/A') or 'N/A'
                            results_lines.append(f"  - {name}: {desc[:80]}...")
            except Exception as e:
                results_lines.append(f"Error getting sample episodes: {e}")
            
            results_lines.append("\n" + "=" * 60)
            return "\n".join(results_lines)
        
        except Exception as e:
            return f"Error querying database directly: {e}"
    
    async def explore_graph(self) -> str:
        """Explore what's actually in the graph by trying various queries.
        
        Returns:
            Formatted exploration results
        """
        if not self._connected or not self.graphiti:
            return "Not connected to Graphiti."
        
        exploration_lines = ["\n=== Exploring Graphiti Knowledge Graph ===\n"]
        
        # First, check database directly
        direct_query_result = await self.query_database_directly()
        exploration_lines.append(direct_query_result)
        exploration_lines.append("\n=== Trying Graphiti Search ===\n")
        
        # Try a very broad search first
        broad_queries = [
            "span",
            "trace",
            "episode",
            "game",
            "player",
            "action",
            "round",
            "extraction",
            "resource",
        ]
        
        found_anything = False
        
        for query in broad_queries:
            try:
                results = await self.search(query, limit=5)
                
                if results:
                    found_anything = True
                    exploration_lines.append(f"\nQuery: '{query}'")
                    exploration_lines.append(f"  Found {len(results)} result(s)")
                    
                    # Show first result details
                    first_result = results[0]
                    if hasattr(first_result, 'fact'):
                        exploration_lines.append(f"  Sample fact: {first_result.fact[:100]}...")
                    elif hasattr(first_result, 'content') and hasattr(type(first_result), '__getattr__'):
                        try:
                            content = str(getattr(first_result, 'content', ''))[:100]
                            exploration_lines.append(f"  Sample content: {content}...")
                        except (AttributeError, TypeError):
                            pass
            except Exception as e:
                exploration_lines.append(f"\nQuery: '{query}' - Error: {e}")
        
        if not found_anything:
            exploration_lines.append("\n⚠️  No results found with Graphiti search.")
            if "Total nodes: 0" in direct_query_result:
                exploration_lines.append("\n📝 The database is empty. You need to run a game first!")
                exploration_lines.append("\nTo generate traces, run one of these:")
                exploration_lines.append("  1. Test game (quick): python test_falkordb_integration.py")
                exploration_lines.append("  2. Full game: python main.py")
                exploration_lines.append("  3. Experiment: python experiment_worker.py")
                exploration_lines.append("\nAfter running a game, wait a few minutes for traces to be")
                exploration_lines.append("processed and indexed, then run 'explore' again.")
            else:
                exploration_lines.append("However, if nodes exist above, this might indicate:")
                exploration_lines.append("  1. Search indexing is still in progress")
                exploration_lines.append("  2. Episodes need time to be processed by Graphiti")
                exploration_lines.append("  3. Search requires different query format")
                exploration_lines.append("\nWait a few minutes and try 'explore' again.")
        
        exploration_lines.append("\n" + "=" * 60)
        
        return "\n".join(exploration_lines)
    
    def format_result(self, result: Any, index: Optional[int] = None) -> str:
        """Format a search result for display.
        
        Args:
            result: Search result object from Graphiti
            index: Optional index number for the result
            
        Returns:
            Formatted string representation
        """
        lines = []
        
        if index is not None:
            lines.append(f"\n--- Result {index + 1} ---")
        else:
            lines.append("\n--- Result ---")
        
        # Extract fact/content
        if hasattr(result, 'fact'):
            lines.append(f"Fact: {result.fact}")
        
        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, str):
                try:
                    # Try to parse as JSON for pretty printing
                    content_obj = json.loads(content)
                    lines.append(f"Content (JSON):")
                    lines.append(json.dumps(content_obj, indent=2))
                except json.JSONDecodeError:
                    lines.append(f"Content: {content}")
            else:
                lines.append(f"Content: {content}")
        
        # Extract metadata
        metadata = {}
        if hasattr(result, 'source_node_uuid'):
            metadata['node_uuid'] = result.source_node_uuid
        if hasattr(result, 'description'):
            metadata['description'] = result.description
        if hasattr(result, 'name'):
            metadata['name'] = result.name
        
        if metadata:
            lines.append("Metadata:")
            for key, value in metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    async def query(self, question: str) -> str:
        """Execute a query and return formatted results.
        
        Args:
            question: Natural language question to ask
            
        Returns:
            Formatted response string
        """
        try:
            results = await self.search(question, limit=10)
            
            if not results:
                return f"\nNo results found for: '{question}'\n"
            
            response_lines = [f"\nFound {len(results)} result(s) for: '{question}'\n"]
            
            for i, result in enumerate(results):
                response_lines.append(self.format_result(result, index=i))
            
            return "\n".join(response_lines)
        
        except Exception as e:
            return f"\nError executing query: {e}\n"
    
    async def get_stats(self) -> str:
        """Get basic statistics about the knowledge graph.
        
        Returns:
            Formatted statistics string
        """
        if not self._connected or not self.graphiti:
            return "Not connected to Graphiti."
        
        try:
            # Try to get some basic info by searching for common terms
            total_results = 0
            sample_queries = [
                "player",
                "game",
                "round",
                "extraction",
                "resource"
            ]
            
            stats_lines = ["\n=== Graphiti Knowledge Graph Statistics ===\n"]
            
            for query_term in sample_queries:
                try:
                    results = await self.search(query_term, limit=1)
                    count = len(results)
                    if count > 0:
                        stats_lines.append(f"  '{query_term}': {count}+ results found")
                except Exception:
                    pass
            
            stats_lines.append("\nNote: These are sample queries. Use search queries to explore the full graph.")
            
            return "\n".join(stats_lines)
        
        except Exception as e:
            return f"Error getting statistics: {e}"


async def run_interactive_loop(playground: GraphitiPlayground):
    """Run the interactive REPL loop.
    
    Args:
        playground: Initialized GraphitiPlayground instance
    """
    print("\n" + "=" * 80)
    print("Graphiti Query Playground")
    print("=" * 80)
    print("\nAsk questions about your game traces!")
    print("\nExample queries:")
    print("  - 'what does player 1 usually do?'")
    print("  - 'show me games where resource was depleted'")
    print("  - 'what are the most common player actions?'")
    print("  - 'find traces for player 2 in round 5'")
    print("\nCommands:")
    print("  'stats' - Show graph statistics")
    print("  'explore' or 'list' - Explore what's in the graph")
    print("  'help' - Show this help message")
    print("  'quit' or 'exit' - Exit the playground")
    print("=" * 80 + "\n")
    
    while True:
        try:
            # Get user input
            question = input("Query> ").strip()
            
            if not question:
                continue
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if question.lower() in ['help', 'h']:
                print("\nExample queries:")
                print("  - 'what does player 1 usually do?'")
                print("  - 'show me games where resource was depleted'")
                print("  - 'what are the most common player actions?'")
                print("  - 'find traces for player 2 in round 5'")
                print("\nCommands:")
                print("  'stats' - Show graph statistics")
                print("  'explore' or 'list' - Explore what's in the graph")
                print("  'help' - Show this help message")
                print("  'quit' or 'exit' - Exit the playground")
                print()
                continue
            
            if question.lower() == 'stats':
                stats = await playground.get_stats()
                print(stats)
                continue
            
            if question.lower() in ['explore', 'list', 'show']:
                exploration = await playground.explore_graph()
                print(exploration)
                continue
            
            # Execute query
            print("\nSearching...")
            try:
                response = await playground.query(question)
                print(response)
                
                # If no results, suggest exploring
                if "No results found" in response:
                    print("\n💡 Tip: Try 'explore' to see what's actually in the graph")
            except Exception as e:
                logger.error(f"Query error: {e}", exc_info=True)
                print(f"\nError: {e}")
                print("\n💡 Tip: Try 'explore' to see what's actually in the graph")
            print()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Use 'quit' to exit or continue querying.")
            continue
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in interactive loop: {e}", exc_info=True)
            print(f"\nError: {e}\n")


async def main():
    """Main entry point."""
    playground = GraphitiPlayground()
    
    # Connect to Graphiti
    print("Connecting to Graphiti/FalkorDB...")
    connected = await playground.connect()
    
    if not connected:
        print("\nERROR: Failed to connect to Graphiti/FalkorDB.")
        print("\nTroubleshooting:")
        print("  1. Make sure FalkorDB is running:")
        print("     docker ps | grep falkordb")
        print("  2. If not running, start it:")
        print("     docker run -d --name falkordb -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest")
        print("  3. Check connection settings in your .env file or config")
        sys.exit(1)
    
    # Run interactive loop
    await run_interactive_loop(playground)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        sys.exit(0)


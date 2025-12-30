"""Example script demonstrating LLM question answering on GraphRAG data.

This script shows how to use the QueryInterface to answer questions
using context from the GraphRAG knowledge graph.
"""

import logging
from langfuse_graphrag.config import get_config
from langfuse_graphrag.query_interface import QueryInterface

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Example: Answer questions using GraphRAG."""
    # Initialize query interface
    config = get_config()
    query_interface = QueryInterface(config)
    
    # Example questions
    questions = [
        "What are the main patterns in the traces?",
        "What errors occurred most frequently?",
        "What models were used and what were their costs?",
        "What semantic entities are most commonly mentioned?",
    ]
    
    print("="*80)
    print("GraphRAG Question Answering Example")
    print("="*80)
    print()
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print("="*80)
        
        try:
            result = query_interface.answer_question(
                question=question,
                max_context_items=10,
                include_graph_context=True,
                max_graph_depth=2
            )
            
            print(f"\nAnswer:\n{result['answer']}\n")
            print(f"Sources ({len(result['sources'])}):")
            for source in result['sources'][:5]:  # Show first 5
                print(f"  - {source}")
            if len(result['sources']) > 5:
                print(f"  ... and {len(result['sources']) - 5} more")
            print(f"\nContext: {result['context_summary']}")
            
        except Exception as e:
            logger.error(f"Error answering question: {e}", exc_info=True)
            print(f"Error: {e}")
        
        print()


if __name__ == "__main__":
    main()


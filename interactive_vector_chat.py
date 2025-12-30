"""Interactive vector RAG chat interface.

Provides a conversational interface using only vector semantic search
(no graph structure, relationships, or communities).
All operations are traced via OpenTelemetry to Langfuse.
"""

import sys
import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langfuse_graphrag.config import get_config
from langfuse_graphrag.query_interface import VectorRAGChat

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for interactive use
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_welcome():
    """Print welcome message."""
    print("\n" + "="*80)
    print("Vector RAG Interactive Chat")
    print("="*80)
    print("\nAsk questions using vector semantic search (no graph structure).")
    print("The chat will automatically retrieve relevant context via vector search.")
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /reset    - Reset conversation history")
    print("  /exit     - Exit the chat")
    print("\n" + "="*80 + "\n")


def print_help():
    """Print help message."""
    print("""
Commands:
  /help     - Show this help message
  /reset    - Reset conversation history
  /exit     - Exit the chat

Just type your question and press Enter to get an answer!
""")


def main():
    """Run interactive vector RAG chat."""
    print_welcome()
    
    try:
        # Initialize chat
        config = get_config()
        chat = VectorRAGChat(
            config=config,
            max_context_items=10
        )
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/exit" or command == "/quit":
                        print("\nGoodbye!\n")
                        break
                    
                    elif command == "/help":
                        print_help()
                        continue
                    
                    elif command == "/reset":
                        chat.reset()
                        print("Conversation history reset.\n")
                        continue
                    
                    else:
                        print(f"Unknown command: {user_input}. Type /help for available commands.\n")
                        continue
                
                # Regular message - get response
                print("\nAssistant: ", end="", flush=True)
                
                try:
                    response = chat.chat(user_input)
                    print(response)
                    print()  # Blank line for readability
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type /exit to quit or continue chatting.\n")
                    continue
                except Exception as e:
                    logger.error(f"Error getting response: {e}", exc_info=True)
                    print(f"\nError: {e}\n")
                    continue
                    
            except EOFError:
                # Handle Ctrl+D
                print("\n\nGoodbye!\n")
                break
            except KeyboardInterrupt:
                # Handle Ctrl+C
                print("\n\nType /exit to quit or continue chatting.\n")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


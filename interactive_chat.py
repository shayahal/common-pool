"""Interactive chat interface for GraphRAG question answering.

Provides a conversational interface where you can ask questions about your
GraphRAG knowledge graph data. The chat maintains conversation history
and automatically retrieves relevant context from the graph.
"""

import sys
import logging
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langfuse_graphrag.config import get_config
from langfuse_graphrag.query_interface import QueryInterface, InteractiveChat

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for interactive use
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_welcome():
    """Print welcome message."""
    print("\n" + "="*80)
    print("GraphRAG Interactive Chat")
    print("="*80)
    print("\nAsk questions about your knowledge graph data.")
    print("The chat will automatically retrieve relevant context from the graph.")
    print("\nCommands:")
    print("  /help     - Show this help message")
    print("  /reset    - Reset conversation history")
    print("  /context  - Toggle automatic context retrieval")
    print("  /history  - Show conversation history")
    print("  /clear    - Clear screen")
    print("  /exit     - Exit the chat")
    print("\n" + "="*80 + "\n")


def print_help():
    """Print help message."""
    print("""
Commands:
  /help     - Show this help message
  /reset    - Reset conversation history
  /context  - Toggle automatic context retrieval (on/off)
  /history  - Show conversation history
  /clear    - Clear screen
  /exit     - Exit the chat

Just type your question and press Enter to get an answer!
""")


def main():
    """Run interactive chat."""
    print_welcome()
    
    try:
        # Initialize chat
        config = get_config()
        chat = InteractiveChat(
            config=config,
            use_graphrag_context=True,
            auto_retrieve_context=True,
            max_context_items=10
        )
        
        context_enabled = True
        
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
                    
                    elif command == "/context":
                        context_enabled = not context_enabled
                        chat.auto_retrieve_context = context_enabled
                        status = "enabled" if context_enabled else "disabled"
                        print(f"Automatic context retrieval {status}.\n")
                        continue
                    
                    elif command == "/history":
                        history = chat.get_history()
                        print("\n" + "="*80)
                        print("Conversation History:")
                        print("="*80)
                        for i, msg in enumerate(history, 1):
                            if isinstance(msg, SystemMessage):
                                print(f"{i}. [SYSTEM] {msg.content[:100]}...")
                            elif isinstance(msg, HumanMessage):
                                print(f"{i}. [USER] {msg.content[:200]}...")
                            elif isinstance(msg, AIMessage):
                                print(f"{i}. [ASSISTANT] {msg.content[:200]}...")
                        print("="*80 + "\n")
                        continue
                    
                    elif command == "/clear":
                        import os
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print_welcome()
                        continue
                    
                    else:
                        print(f"Unknown command: {user_input}. Type /help for available commands.\n")
                        continue
                
                # Regular message - get response
                print("\nAssistant: ", end="", flush=True)
                
                try:
                    response = chat.chat(user_input, use_context=context_enabled)
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


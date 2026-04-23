"""
PHN Technology WhatsApp Agent — CLI Test Mode

Interactive command-line interface for testing the agent
without needing WhatsApp integration. Same LangGraph agent,
just using terminal input/output.

Usage:
    # From host (with native Ollama):
    python -m scripts.test_agent

    # Or inside Docker container:
    docker exec -it phn-whatsapp-agent python -m scripts.test_agent
"""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for local testing (if not already set)
if not os.environ.get("OLLAMA_BASE_URL"):
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
if not os.environ.get("CHROMA_PERSIST_DIR"):
    os.environ["CHROMA_PERSIST_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "chroma_data_local"
    )
if not os.environ.get("SQLITE_DB_PATH"):
    os.environ["SQLITE_DB_PATH"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "test_agent.db"
    )
if not os.environ.get("KNOWLEDGE_BASE_DIR"):
    os.environ["KNOWLEDGE_BASE_DIR"] = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_base"
    )

from langchain_core.messages import HumanMessage
from app.agent.graph import get_agent
from app.db.models import init_database
from app.rag.indexer import index_knowledge_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 60)
    print("🤖 PHN Technology WhatsApp Agent — CLI Test Mode")
    print("=" * 60)
    print("  Test the agent without WhatsApp integration.")
    print("  Type your messages and see the agent's responses.")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'reset' to clear conversation history.")
    print("  Type 'status' to see current conversation state.")
    print("=" * 60 + "\n")


def main():
    """Run interactive CLI test."""
    print_banner()

    # Initialize
    print("🔄 Initializing database...")
    init_database()

    print("📚 Indexing knowledge base...")
    try:
        index_knowledge_base()
    except Exception as e:
        print(f"⚠️  Knowledge base indexing failed: {e}")
        print("   Make sure Ollama is running with the required models!")
        print(f"   Ollama URL: {os.environ.get('OLLAMA_BASE_URL', 'not set')}")
        return

    print("✅ Agent ready! Start chatting.\n")

    # Create agent
    agent = get_agent()
    test_phone = "test_user_cli"
    config = {"configurable": {"thread_id": test_phone}}

    while True:
        try:
            user_input = input("🧑 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 Bye! Have a great day!")
            break

        if user_input.lower() == "reset":
            # Reset conversation by creating new agent
            agent = get_agent()
            config = {"configurable": {"thread_id": f"test_user_cli_{id(agent)}"}}
            print("🔄 Conversation reset!\n")
            continue

        if user_input.lower() == "status":
            # Show current state info
            print("\n📊 Current Session:")
            print(f"   Thread ID: {config['configurable']['thread_id']}")
            print(f"   Ollama URL: {os.environ.get('OLLAMA_BASE_URL')}")
            print(f"   Model: {os.environ.get('OLLAMA_CHAT_MODEL', 'qwen2.5:7b-instruct')}")
            print()
            continue

        # Invoke agent
        print("🤔 Thinking...", end="", flush=True)
        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_phone": test_phone,
                    "user_name": "Test User",
                },
                config=config,
            )

            response = result.get("response_text", "No response generated.")
            intent = result.get("intent", "unknown")
            lead_score = result.get("lead_score", "cold")
            language = result.get("language", "en")
            
            # Extracted profile fields
            extracted_name = result.get("extracted_name", "")
            city = result.get("city", "")
            occupation = result.get("occupation", "")
            interest_field = result.get("interest_field", "")
            is_interested = result.get("is_interested", "")
            interested_courses = result.get("interested_courses", "")

            # Mirror the webhook DB saving logic so test_agent.db stays updated
            from app.db.models import save_lead, update_extracted_info, update_lead_score, log_conversation
            save_lead(test_phone, "Test User", language)
            update_extracted_info(
                phone=test_phone,
                extracted_name=extracted_name,
                city=city,
                occupation=occupation,
                interest_field=interest_field,
                is_interested=is_interested
            )
            update_lead_score(test_phone, lead_score, interested_courses)
            log_conversation(test_phone, "incoming", user_input, intent, response, lead_score)

            # Clear "Thinking..." and show response
            print(f"\r🤖 Agent: {response}")
            print(f"   [Intent: {intent} | Score: {lead_score} | Lang: {language}]")
            print(f"   [Profile -> Name: '{extracted_name}', City: '{city}', Occupation: '{occupation}', Courses: '{interested_courses}']\n")

        except Exception as e:
            print(f"\r❌ Error: {e}\n")
            logger.error(f"Agent error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

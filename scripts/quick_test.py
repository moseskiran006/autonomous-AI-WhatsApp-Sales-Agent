import sys
sys.path.insert(0, "/app")

from langchain_core.messages import HumanMessage
from app.agent.graph import get_agent

agent = get_agent()

def test(msg, name, tid):
    config = {"configurable": {"thread_id": tid}}
    r = agent.invoke({
        "messages": [HumanMessage(content=msg)],
        "user_phone": "test", "user_name": "Student",
    }, config=config)
    print("=" * 60)
    print("TEST: " + name)
    print("Q: " + msg)
    print("-" * 60)
    print(r.get("response_text", "No response"))
    print()

test("Hi, I want to learn Data Science", "Test FOMO and Name Prompt", "fomo1")

print("ALL TESTS DONE!")

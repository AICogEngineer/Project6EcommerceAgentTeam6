from langchain_core.messages import HumanMessage
from langgraph.errors import Interrupt

from agent.graph import build_graph
from agent.state import AgentState

def main():
    graph = build_graph()

    state = AgentState(
        messages=[HumanMessage(content="I want a refund for my order")],
        user_id="user_123",
        order_id="order_456",
        identity_verified=True,
    )

    try:
        result = graph.invoke(state)
        print("=== FINAL DECISION ===")
        print(result["decision_summary"])
        print("Human Review Required:", result["human_review_required"])

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

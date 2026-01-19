from langchain_core.messages import HumanMessage
from agent.agent import build_graph

graph = build_graph()

initial_state = {
    "messages": [HumanMessage(content="I want a refund for my order")],
    "user_id": "user_123",
    "order_id": "order_456",
    "identity_verified": False,
}

try: 
    result = graph.invoke(initial_state)
    print(result)
except Exception as e:
    print("INTERRUPT:", e)
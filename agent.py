import os
import traceback
from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.errors import Interrupt
from langchain.chat_models import init_chat_model
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


from adapters.gold_mock import GOLD_ORDER_LOOKUP, GOLD_RISK_LOOKUP
# from adapters.snowflake_adapter import fetch_gold_order

# Load environment variables
load_dotenv()

# Ensure envionment variables are set
required_env_vars = (
    "PINECONE_API_KEY",
    "OPENAI_API_KEY"
)
debug_env_vars = (
    "LANGSMITH_PROJECT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_TRACING"
)
for r in required_env_vars:
    if r not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {r}")
for d in debug_env_vars:
    if d not in os.environ:
        print(f"Warning: Missing debug environment variable: {d}")

# Initialize model
model = init_chat_model("openai:gpt-4o-mini")

# ========================== Pinecone Setup ==========================

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Must match dimensions for OpenAIEmbeddings
index = pc.Index("project-6-ecommerce-agent")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text"
)

# ========================== Tools ==========================

@tool
def fetch_from_snowflake(user_id: str, order_id: str) -> dict:
        """
        Fetches Gold-layer order and risk data for a given user and order.
        This tool **must be called first**.
        """
        # TODO: replace with snowflake connector
        gold_order = GOLD_ORDER_LOOKUP.get(order_id, {})
        gold_risk = GOLD_RISK_LOOKUP.get(user_id, {})

        return {
            **gold_order,
            **gold_risk
        }        

@tool
def fetch_from_pinecone(item_category: str) -> dict:
    """
    Retrieves the refund policy clause for the given item_category.
    Only call this **after** fetching Snowflake data to get item_category.
    """
    results = vectorstore.similarity_search(
        query=item_category,
        k=1
    )
    if not results:
        return {"policy_clause": "No refund policy found for this category."}

    return {"policy_clause": results[0].page_content}

tools = [fetch_from_snowflake, fetch_from_pinecone]

# ====================== Create Agent =======================

class RefundPolicy(BaseModel):
    policy_clause: str

agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(RefundPolicy)
)

# ========================== State ==========================

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

    # routing
    intent: Optional[str] = None
    human_review_required: bool = False

    # identity
    user_id: str
    order_id: str
    identity_verified: bool = False

    # order context (Gold)
    item_category: Optional[str] = None
    order_date: Optional[str] = None
    item_price_usd: Optional[float] = None
 
    # fraud signals (Gold)
    refund_count_window: Optional[int] = None
    returnless_refund_count_window: Optional[int] = None
    chargeback_flag: Optional[bool] = None
    address_distance_miles: Optional[float] = None

    # Policy
    policy_clause: Optional[str] = None

    # Explanation
    decision_summary: Optional[dict] = None

# ========================== Nodes ==========================

# Intent detection
def intent_node(state: AgentState) -> dict:
    last_message = state.messages[-1].content.lower()
    if "refund" in last_message.lower():
        return {"intent": "refund"}
    else:
        return {"intent": "other"}

# Handle cases where the user needs to verify who they are, such as PII
# or financial information
def identity_gate_node(state: AgentState) -> dict:
    if not state.identity_verified:
        raise Interrupt("Identity verification required")
    return {}

# Fetch the order date, item category, and transactional data (even user login)
# via Pydantic-validated models
def snowflake_node(state: AgentState) -> dict:   
    data = fetch_from_snowflake(state.user_id, state.order_id)   
    return data

# Retrieve policy clause via Pinecone
def pinecone_node(state: AgentState) -> dict:
    if not state.item_category:
        return {"policy_clause": "No policy available"}
    return fetch_from_pinecone(state.item_category) 

# Retrieves the order based on the user, and order id
# Crucially, it calls the Snowflake and Pinecone tools sequentially
# using LLM commands. This may be probablistic and not recommended.
def retrieval_agent_node(state: AgentState) -> dict:
    # Prepare a prompt context combining needed state
    user_input = (
        f"User said: {state.messages[-1].content}\n"
        f"User ID: {state.user_id}, Order ID: {state.order_id}\n"
        # f"Item Category: {state.item_category}"
    )

    result = agent.invoke({
        "input": user_input,
        "messages": state.messages,
    })
    # Extract structured policy if present
    structured = result.get("structured_response")
    if structured and "policy_clause" in structured:
        return {"policy_clause": structured["policy_clause"]}

    # Fallback: parse from last model message
    last_msg = result["messages"][-1].content
    return {"policy_clause": last_msg}

# Execute fraud and abuse signals.
def fraud_detection_node(state: AgentState) -> dict:
    if state.chargeback_flag:
        return {"human_review_required": True}
    if state.refund_count_window and state.refund_count_window > 3:
        return {"human_review_required": True}
    if state.address_distance_miles and state.address_distance_miles > 500:
        return {"human_review_required": True}
    
    return {}

# Generate an explanation for the decision path
def decision_summary_node(state: AgentState) -> dict:
    summary = {
        "reason": "Refund request within policy window",
        "policy": state.policy_clause,
        "refund_count_window": state.refund_count_window,
        "chargeback_flag": state.chargeback_flag,
        "geo_distance_miles": state.address_distance_miles,
        "human_review_required": state.human_review_required
    }
    return {"decision_summary": summary}

# Final mandatory HITL review
def human_review_node(state: AgentState) -> dict:
    return {"human_review_required": True}

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("identity", identity_gate_node)
    builder.add_node("snowflake", snowflake_node)
    builder.add_node("pinecone", pinecone_node)
    builder.add_node("retrieval", retrieval_agent_node)
    builder.add_node("fraud", fraud_detection_node)
    builder.add_node("summary", decision_summary_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "identity")
    # builder.add_edge("identity", "snowflake")
    # builder.add_edge("snowflake", "pinecone")
    # builder.add_edge("pinecone", "fraud")
    builder.add_edge("identity", "retrieval")
    builder.add_edge("retrieval", "fraud")
    builder.add_edge("fraud", "summary")

    builder.add_conditional_edges(
        "summary", 
        lambda s: "human_review" if s.human_review_required else END,
        {
            "human_review": "human_review",
            END: END,
        },
    )
   
    builder.add_edge("human_review", END)

    return builder.compile()

def run_test_case(
    user_message: str,
    user_id: str,
    order_id: str,
    identity_verified: bool = True,
):
    graph = build_graph()

    initial_state = AgentState(
        messages=[HumanMessage(content=user_message)],
        user_id=user_id,
        order_id=order_id,
        identity_verified=identity_verified,
    )

    try:
        final_state = graph.invoke(initial_state)
        return final_state
    except Exception as e:
        print("Execution halted:")
        traceback.print_exc()
        return None

def main():
    print("=== Running Agent Test ===")

    # Sample refund-related query (returns dict)
    result = run_test_case(
        user_message="I want a refund for my order",
        user_id="user_123",
        order_id="order_456",
        identity_verified=True,  # bypass identity gate for testing
    )

    if result:
        print("\n=== FINAL STATE ===")
        print("Decision Summary:")
        print(result["decision_summary"])

        print("\nHuman review required:", result["human_review_required"])

if __name__ == "__main__":
    main()
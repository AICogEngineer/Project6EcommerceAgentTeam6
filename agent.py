from dotenv import load_dotenv
from typing import Annotated, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.errors import Interrupt
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")

# ========================== Tools ==========================

# Fetches order date, item category, and transactional data, even user log-in
# via Pydantic-validated models
@tool
def fetch_from_snowflake(user_id: str, order_id: str) -> dict:
    return {
        "item_category": "electronics",
        "order_date": "2023-10-01",
        "item_price_usd": 199.99,
        "refund_count_window": 2,
        "returnless_refund_count_window": 0,
        "chargeback_flag": False,
        "address_distance_miles": 42.0
    }

# Fetches specific return/refund clauses relevant to the item category
@tool
def fetch_from_pinecone(item_category: str) -> dict:
    return{
        "policy_clause": "Electronics are eligible for return within 30 days."
    }

# ========================== State ==========================

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

    # routing
    intent: str
    human_review_required: bool = False

    # identity
    user_id: str
    order_id: str
    identity_verified: bool = False

    # order context (Gold)
    item_category: Optional[str] = None
    order_date: Optional[str] = None
    item_price_usd: Optional[str] = None

    # fraud signals (Gold)
    refund_count_window: Optional[int] = None
    returnless_refund_count_window: Optional[int] = None
    chargeback_flag: Optional[bool] = None
    address_distance_miles: Optional[float] = None

# ========================== Nodes ==========================

# TODO this is a simple string check in the message - should we use something
# more sophisticated instead?
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
    data = fetch_from_snowflake(state.user_id, state.order_id)   # dict type
    return data

def pinecone_node(state: AgentState) -> dict:
    fetch_from_pinecone(state.item_category) 
    return {}

def fraud_detection_node(state: AgentState) -> dict:
    if state.chargeback_flag:
        return {"human_review_required": True}
    if state.refund_count_window and state.refund_count_window > 3:
        return {"human_review_required": True}
    if state.address_distance_miles and state.address_distance_miles > 500:
        return {"human_review_required": True}

    return{}

def human_review_node(state: AgentState) -> dict:
    return {"human_review_required": True}


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("identity", identity_gate_node)
    builder.add_node("snowflake", snowflake_node)
    builder.add_node("pinecone", pinecone_node)
    builder.add_node("fraud", fraud_detection_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "identity")
    builder.add_edge("identity", "snowflake")
    builder.add_edge("snowflake", "pinecone")
    builder.add_edge("pinecone", "fraud")

    builder.add_conditional_edges(
        "fraud", 
        lambda s: "human_review" if s.human_review_required else END,
        {
            "human_review": "human_review",
            END: END
        }
    )
   
    builder.add_edge("human_review", END)

    return builder.compile()

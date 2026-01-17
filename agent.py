from dotenv import load_dotenv
from typing import Annotated, BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o-mini")

# ========================== Tools ==========================

# Fetches order date, item category, and transactional data, even user log-in
# via Pydantic-validated models
@tool
def fetch_from_snowflake() -> dict:
    pass

# Fetches specific return/refund clauses relevant to the item category
@tool
def fetch_from_pinecone(item_category: str) -> dict:
    pass

# ========================== State ==========================

class AgentState(BaseModel):
    intent: str
    identity_verified: bool = False
    user_id: int
    messages: Annotated[list[AnyMessage], add_messages]
    item_category: str
    order_date: str # TODO date?
    user_address: str
    shipping_address: str
    transaction_history: list[dict]  # Should especially indicate refunds
    user_chargebacks: list[dict]
    human_review_required: bool = False


# ========================== Nodes ==========================

# TODO this is a simple string check in the message - should we use something
# more sophisticated instead?
def intent_node(state: AgentState) -> dict:
    last_message = state["messages"][-1] if state["messages"] else ""
    # is_refund_classification = model.invoke(
    #     f"Classify the following message as 'refund' or 'other': {last_message}"
    # )
    if "refund" in last_message.lower():
        return {"intent": "refund"}
    else:
        return {"intent": "other"}

# Handle cases where the user needs to verify who they are, such as PII
# or financial information
def identity_gate_node(state: AgentState) -> dict:
    if not state["identity_verified"]:
        # LangGraph interrupt
        # TODO IMPLEMENT
        pass
    return {}

# Fetch the order date, item category, and transactional data (even user login)
# via Pydantic-validated models
def snowflake_node(state: AgentState) -> dict:
    # TODO implement
    fetch_from_snowflake()   # dict type
    return {
        "item_category": "electronics",
        "order_date": "2023-10-01"
        # TODO etc
    }

def pinecone_node(state: AgentState) -> dict:
    # TODO implement
    item_category = state["item_category"]
    fetch_from_pinecone(item_category)   # dict type
    return {}

def fraud_detection_node(state: AgentState) -> dict:
    # Distance disrepancy
    user_address = state["user_address"]
    shipping_address = state["shipping_address"]
    # TODO - compare distance between addresses - drift triggers auto flag

    # Refund velocity
    transaction_history = state["transaction_history"]
    refund_count = 0
    for t in transaction_history:
        if t["refunded"]:
            refund_count += 1
        if t > 3:
            # TODO flag
            pass
    
    # Chargeback risk
    chargebacks = state["user_chargebacks"]
    # TODO implement
    return {}

def human_review_node(state: AgentState) -> dict:
    # TODO implement
    pass


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("intent", intent_node)
    builder.add_node("identity", identity_gate_node)
    builder.add_node("snowflake", snowflake_node)
    builder.add_node("pinecone", pinecone_node)
    builder.add_node("fraud_detection", fraud_detection_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START,"intent")
    builder.add_edge("intent", "identity")
    builder.add_edge("identity", "snowflake")
    builder.add_edge("snowflake", "pinecone")
    builder.add_edge("pinecone", "fraud_detection")
    builder.add_edge("fraud_detection", END)
    # TODO is this right? Might not be
    builder.add_conditional_edge("fraud_detection", "human_review",
                                 lambda state: state["human_review_required"])
    builder.add_edge("human_review", END)

def main():
    # TODO - test cases?
    graph = build_graph()

if __name__ == "main":
    main()
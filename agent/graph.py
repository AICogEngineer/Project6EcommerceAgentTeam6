from langgraph.graph import StateGraph, START, END
from langgraph.errors import Interrupt
from langchain_core.messages import HumanMessage

from agent.state import AgentState
from adapters.snowflake_adapter import fetch_from_snowflake
from adapters.pinecone_adapter import fetch_from_pinecone
from langchain.chat_models import init_chat_model

# Initialize model (used ONLY for explanation, not decisions)
model = init_chat_model("openai:gpt-4o-mini")


# =========================
# Nodes
# =========================

def intent_node(state: AgentState) -> dict:
    """Detect high-level user intent."""
    last_message = state.messages[-1].content.lower()
    refund_keywords = ["refund", "return", "money back", "chargeback"]

    if any(k in last_message for k in refund_keywords):
        return {"intent": "refund"}
    return {"intent": "other"}


def identity_gate_node(state: AgentState) -> dict:
    """Block workflow until identity is verified."""
    if not state.identity_verified:
        raise Interrupt("Identity verification required before proceeding.")
    return {}


def snowflake_node(state: AgentState) -> dict:
    """Fetch Gold-layer order + risk data (mock or real via adapter)."""
    return fetch_from_snowflake(state.user_id, state.order_id)


def pinecone_node(state: AgentState) -> dict:
    """Retrieve policy clause using item category."""
    if not state.item_category:
        return {"policy_clause": "No policy available for this item."}
    return fetch_from_pinecone(state.item_category)


def fraud_detection_node(state: AgentState) -> dict:
    """Deterministic fraud / abuse checks."""
    if state.chargeback_flag:
        return {"human_review_required": True}

    if state.refund_count_window and state.refund_count_window > 3:
        return {"human_review_required": True}

    if state.address_distance_miles and state.address_distance_miles > 500:
        return {"human_review_required": True}

    return {}


def decision_summary_node(state: AgentState) -> dict:
    """Create a structured, deterministic decision summary."""
    summary = {
        "reason": "Refund evaluated against policy and risk signals",
        "policy": state.policy_clause,
        "refund_count_window": state.refund_count_window,
        "chargeback_flag": state.chargeback_flag,
        "geo_distance_miles": state.address_distance_miles,
        "human_review_required": state.human_review_required,
    }
    return {"decision_summary": summary}


def llm_explanation_node(state: AgentState) -> dict:
    """
    LLM explanation layer.
    Produces customer-facing language only.
    """

    prompt = f"""
You are an e-commerce refund assistant.

Order details:
- Item category: {state.item_category}
- Item price: {state.item_price_usd}
- Recent refunds: {state.refund_count_window}
- Chargeback flag: {state.chargeback_flag}
- Address distance (miles): {state.address_distance_miles}

Policy:
{state.policy_clause}

Human review required: {state.human_review_required}

Write a clear, professional explanation for the customer.
Do NOT mention internal fraud thresholds or risk scoring.
"""

    response = model.invoke([HumanMessage(content=prompt)])
    return {"final_explanation": response.content}


def human_review_node(state: AgentState) -> dict:
    """Final mandatory HITL step."""
    return {"human_review_required": True}


# =========================
# Graph Builder
# =========================

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("identity", identity_gate_node)
    builder.add_node("snowflake", snowflake_node)
    builder.add_node("pinecone", pinecone_node)
    builder.add_node("fraud", fraud_detection_node)
    builder.add_node("summary", decision_summary_node)
    builder.add_node("llm_explain", llm_explanation_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "identity")
    builder.add_edge("identity", "snowflake")
    builder.add_edge("snowflake", "pinecone")
    builder.add_edge("pinecone", "fraud")
    builder.add_edge("fraud", "summary")
    builder.add_edge("summary", "llm_explain")

    builder.add_conditional_edges(
        "llm_explain",
        lambda s: "human_review" if s.human_review_required else END,
        {
            "human_review": "human_review",
            END: END,
        },
    )

    builder.add_edge("human_review", END)

    return builder.compile()

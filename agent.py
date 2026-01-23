"""
Agentic E-Commerce Refund Orchestrator
"""

import os
from typing import Annotated, Optional, Literal
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------
load_dotenv()

# ------------------------------------------------------------------
# STATE
# ------------------------------------------------------------------

class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]

    # routing
    request_type: Optional[Literal["policy", "sensitive"]] = None

    # identity
    username: Optional[str] = None
    email: Optional[str] = None
    verified: bool = False
    attempts: int = 0

    # snowflake (gold layer)
    user_info: Optional[dict] = None

    # risk
    is_risky: Optional[bool] = None

    # refund decision
    refund_route: Optional[Literal["Impossible", "Risky", "Trusted"]] = None
    refund_decision_reasoning: Optional[str] = None


# ------------------------------------------------------------------
# TOOLS (SAFE MOCKS)
# ------------------------------------------------------------------

@tool
def fetch_policy_tool(query: str) -> str:
    """
    Fetch refund policy text using Pinecone (RAG).
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
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

    results = vectorstore.similarity_search(query, k=1)
    if not results:
        return "No relevant refund policy found."

    return results[0].page_content


@tool
def verify_user_tool(username: str, email: str) -> bool:
    """
    Mock identity verification.
    """
    return username != "" and email != ""


@tool
def fetch_snowflake_tool(username: str, email: str) -> dict:
    """
    Mock Gold-layer Snowflake data.
    """
    return {
        "days_since_purchase": 12,
        "chargeback_amount": 0,
        "refund_amount": 49.99,
        "products": [("Wireless Headphones", "electronics")]
    }


@tool
def calculate_risk_tool(chargeback_amount: float, refund_amount: float) -> bool:
    """
    Simple risk heuristic.
    """
    return chargeback_amount > 0 or refund_amount > 200


# ------------------------------------------------------------------
# NODES
# ------------------------------------------------------------------

def classify_request_node(state: AgentState):
    """
    Policy vs Sensitive router.
    """
    text = state.messages[-1].content.lower()

    if any(word in text for word in ["refund", "return", "order", "charged", "my account"]):
        return {"request_type": "sensitive"}

    return {"request_type": "policy"}


def respond_policy_node(state: AgentState):
    """
    Answer policy-only questions using Pinecone.
    """
    model = init_chat_model("openai:gpt-4o-mini", temperature=0.7)

    agent = create_agent(
        model=model,
        tools=[fetch_policy_tool],
        system_prompt="Answer the user's question using policy context only.",
        name="policy_agent"
    )

    response = agent.invoke({"messages": state.messages})
    return {"messages": [response["messages"][-1]]}


def collect_credentials_node(state: AgentState):
    """
    Interrupt to collect identity info.
    """
    if state.attempts >= 3:
        return {
            "messages": [AIMessage(content="Too many failed attempts. Contact support.")],
            "verified": False
        }

    payload = {
        "type": "identity_verification",
        "attempt": state.attempts + 1,
        "fields": ["username", "email"]
    }

    user_data = interrupt(payload)

    return {
        "username": user_data.get("username"),
        "email": user_data.get("email")
    }


def verify_user_node(state: AgentState):
    """
    Verify identity.
    """
    attempts = state.attempts + 1

    verified = verify_user_tool.invoke({
        "username": state.username or "",
        "email": state.email or ""
    })

    if verified:
        return {"verified": True, "attempts": 0}

    if attempts >= 3:
        return {
            "verified": False,
            "attempts": 0,
            "messages": [AIMessage(content="Verification failed. Contact support.")]
        }

    return {"verified": False, "attempts": attempts}


def fetch_snowflake_node(state: AgentState):
    """
    Fetch Gold-layer data.
    """
    user_info = fetch_snowflake_tool.invoke({
        "username": state.username,
        "email": state.email
    })

    return {"user_info": user_info}


def risk_scoring_node(state: AgentState):
    """
    Risk classification.
    """
    info = state.user_info or {}
    is_risky = calculate_risk_tool.invoke({
        "chargeback_amount": info.get("chargeback_amount", 0),
        "refund_amount": info.get("refund_amount", 0)
    })

    return {"is_risky": is_risky}


class RefundRoute(BaseModel):
    decision: Literal["Impossible", "Risky", "Trusted"]
    reason: str = Field(max_length=200)


def refund_routing_node(state: AgentState):
    """
    Final refund routing decision.
    """
    if state.is_risky:
        return {
            "refund_route": "Risky",
            "refund_decision_reasoning": "Risk signals detected.",
            "messages": [AIMessage(content="Your request is under human review.")]
        }

    return {
        "refund_route": "Trusted",
        "refund_decision_reasoning": "Request meets refund policy.",
        "messages": [AIMessage(content="Your refund has been approved.")]
    }


def human_review_node(state: AgentState):
    """
    HITL interrupt.
    """
    payload = {
        "reason": state.refund_decision_reasoning,
        "fields": [{"decision": "approve or reject"}]
    }

    decision = interrupt(payload).get("decision", "reject")

    if decision == "approve":
        return {"messages": [AIMessage(content="Refund approved by agent.")]}
    return {"messages": [AIMessage(content="Refund denied by agent.")]}


# ------------------------------------------------------------------
# GRAPH
# ------------------------------------------------------------------

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify", classify_request_node)
    builder.add_node("policy", respond_policy_node)
    builder.add_node("collect_identity", collect_credentials_node)
    builder.add_node("verify", verify_user_node)
    builder.add_node("snowflake", fetch_snowflake_node)
    builder.add_node("risk", risk_scoring_node)
    builder.add_node("route", refund_routing_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        lambda s: s.request_type,
        {
            "policy": "policy",
            "sensitive": "collect_identity"
        }
    )

    builder.add_edge("policy", END)

    builder.add_edge("collect_identity", "verify")

    builder.add_conditional_edges(
        "verify",
        lambda s: "verified" if s.verified else "retry",
        {
            "verified": "snowflake",
            "retry": "collect_identity"
        }
    )

    builder.add_edge("snowflake", "risk")
    builder.add_edge("risk", "route")

    builder.add_conditional_edges(
        "route",
        lambda s: "human" if s.refund_route == "Risky" else END,
        {
            "human": "human_review",
            END: END
        }
    )

    builder.add_edge("human_review", END)

    return builder.compile(checkpointer=InMemorySaver())


# ------------------------------------------------------------------
# ENTRY POINT (for CLI testing)
# ------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_graph()
    config = {"configurable": {"thread_id": "demo"}}

    msg = input("User: ")
    result = graph.invoke(
        AgentState(messages=[HumanMessage(content=msg)]),
        config=config
    )

    print(result)

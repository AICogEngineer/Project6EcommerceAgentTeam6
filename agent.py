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
from langgraph.types import Command
from langgraph.types import interrupt
from typing import Literal
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

from adapters.snowflake_adapter import snowflake_verify_user_email, snowflake_fetch_user_orders, snowflake_fetch_item_category, snowflake_fetch_order, snowflake_update_order_refund, snowflake_count_refunded

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
def fetch_user_orders(user_id: str) -> dict:
    """
    Queries the user's orders from the Snowflake gold database
    by their user_id, returning a list.
    """
    print("fetch_user_orders called!")
    return snowflake_fetch_user_orders(user_id)

@tool
def fetch_item_category(transaction_id: str) -> str | None:
    """
    Queries the Snowflake gold database to get the item_category.
    After this, call the tool fetch_from_pinecone - unless you
    received a None type here - in which case, end and return.
    """
    print("fetch_item_category called!")
    return snowflake_fetch_item_category(transaction_id);

@tool
def fetch_from_pinecone(item_category: str) -> dict:
    """
    Retrieves the refund policy clause for the given item_category.
    Only call this **after** fetching Snowflake data to get item_category.
    """
    print("fetch_from_pinecone called!")
    results = vectorstore.similarity_search(
        query=item_category,
        k=1
    )
    if not results:
        return {"policy_clause": "No refund policy found for this category."}

    return {"policy_clause": results[0].page_content}

tools = [fetch_user_orders, fetch_item_category, fetch_from_pinecone]

# ====================== Create Agent =======================

class RefundPolicy(BaseModel):
    policy_clause: str

agent = create_agent(
    name="ecommerce_agent",
    model="openai:gpt-4o-mini",
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
    user_id: Optional[str] = None   # Must consider starting from scratch
    order_id: Optional[str] = None
    identity_verified: bool = False

    # order context (Gold)
    orders: Optional[list[dict]] = None
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
    messages = state.messages
    if not messages:
        return {}
    
    last_msg = messages[-1]
    content = last_msg.content[0]['text']

    # Detect intent
    print(content)
    if "refund" in content.lower():
        return {"intent": "refund"}
    else:
        return {"intent": "other"}

# Handle cases where the user needs to verify who they are, such as PII
# or financial information
def identity_gate_node(state: AgentState) -> dict:
    if state.identity_verified:
        return {}
    email = interrupt({"message": "Verify identity (email)"})
    return snowflake_verify_user_email(email)

# Fetch the order date, item category, and transactional data (even user login)
# via Pydantic-validated models
def snowflake_node(state: AgentState) -> dict:
    user_orders = snowflake_fetch_user_orders(state.user_id)
    print(user_orders)
    return {"orders": user_orders}

# Retrieve policy clause via Pinecone
def pinecone_node(state: AgentState) -> dict:
    if not state.item_category:
        return {"policy_clause": "No policy available"}
    return fetch_from_pinecone(state.item_category) 

# Retrieves the order based on the user, and order id
# Crucially, it calls the Snowflake and Pinecone tools sequentially
# using LLM commands. This may be probablistic and not recommended.
def retrieval_agent_node(state: AgentState) -> dict:
    order_list = snowflake_fetch_user_orders(state.user_id)
    # Interrupt to see which item the user wants to return
    return_transaction_id = interrupt({
        f"Which of these transactions would you like to return? {order_list}\nEnter the transaction id"
    })
    order = snowflake_fetch_order(return_transaction_id)
    item_category = order.get("category")
    results = vectorstore.similarity_search(
        query=item_category,
        k=1
    )
    policy_clause = "No refund policy found for this category."
    if results:
        policy_clause = results[0].page_content
    # Mark order as refunded
    snowflake_update_order_refund(return_transaction_id)

    return {
        "policy_clause": policy_clause,
        "refund_count_window": snowflake_count_refunded(state.user_id),
        "order_id": return_transaction_id
    }

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
        "human_review_required": True
    }
    return {"decision_summary": summary}

# Final mandatory HITL review
def human_review_node(state: AgentState) -> dict:
    decision = interrupt({f"Human review for transaction {state.order_id} required: yes or no?"})
    if decision and decision.lower() == "yes":
        return {"human_review_required": False}
    return {"human_review_required": True}

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("intent", intent_node)
    builder.add_node("identity", identity_gate_node)
    builder.add_node("snowflake", snowflake_node)
    builder.add_node("retrieval", retrieval_agent_node)
    builder.add_node("fraud", fraud_detection_node)
    builder.add_node("summary", decision_summary_node)
    builder.add_node("human_review", human_review_node)

    builder.add_edge(START, "intent")
    builder.add_edge("intent", "identity")
    builder.add_edge("identity", "snowflake")
    builder.add_edge("snowflake", "retrieval")
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

    # InMemoryServer needed to maintain state during interrupts
    return builder.compile(checkpointer=InMemorySaver())

def interactive_stateful_cli():
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-1"}}  # Needed for InMemorySaver()
    state = AgentState(messages=[])

    # Get and add user's first message
    user_input = input(
        "\nHello, I am your refund agent! How can I help you today? "
    )
    state.messages.append(HumanMessage(content=user_input))
    # Call graph from user's initial message
    result = graph.invoke(state, config=config)

    # If an interrupt was detected (i.e. need identity verification, handle it)
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]

        payload = interrupt_obj.value
        print("\n[AGENT]:", payload["message"])

        # Verify user credentials for their identity (TODO sample creds for now)
        email = input("Email: ")
        verify_results = snowflake_verify_user_email(email)
        if verify_results["identity_verified"]:
            # Resume
            result = graph.invoke(
                # Contains email, identity_verified, user_id (for querying orders)
                Command(resume=verify_results),
                config=config
            )
        else:
            print("Identity verification failed. Exiting.")
            return
            
    # On completion of agent cycle, print the agent's decision
    if "decision_summary" in result:
        print("\nAgent decision summary:")
        print(result["decision_summary"])

def main():
    print("=== Running Agent Test ===")
    interactive_stateful_cli()

if __name__ == "__main__":
    main()
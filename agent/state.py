from typing import Annotated, Optional
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

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
    item_price_usd: Optional[str] = None
 
    # fraud signals (Gold)
    refund_count_window: Optional[int] = None
    returnless_refund_count_window: Optional[int] = None
    chargeback_flag: Optional[bool] = None
    address_distance_miles: Optional[float] = None

    # Policy
    policy_clause: Optional[str] = None

    # Explanation
    decision_summary: Optional[dict] = None
    final_explanation: Optional[str] = None
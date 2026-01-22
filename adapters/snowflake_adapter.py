import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()
def get_snowflake_connection():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"],
        role=os.environ["SNOWFLAKE_ROLE"]
    )

def snowflake_verify_user_email(email: str) -> dict: 
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT USER_ID FROM DIM_CUSTOMERS WHERE EMAIL = %s",
                    (email,)
                )
                row = cur.fetchone()
                return {
                    "email": email,
                    "identity_verified": True if row else False,
                    "user_id": row[0] if row else None
                }
    except Exception as e:
        print("ERROR verifying email in Snowflake:", e)
        return {
            "email": email,
            "identity_verified": False,
            "user_id": None
        }

def fetch_from_snowflake(user_id: str, order_id: str) -> dict:
    """
    Read-only Snowflake adapter
    Maps Snowflake data into agent-compatible fields.
    """
    print("[SNOWFLAKE] Querying MENU table...")

    try:
        cs.execute("""
            SELECT
                p.CATEGORY,
                s.UNIT_PRICE,
                s.QUANTITY,
                s.REVENUE
            FROM FACT_SALES s
            JOIN DIM_PRODUCTS p
              ON s.PRODUCT_ID = p.PRODUCT_ID
            WHERE s.USER_ID = %s
            LIMIT 1
        """, (user_id,))

        row = cs.fetchone()

        if not row:
            print("[SNOWFLAKE] No rows returned (expected in training env)")
            return {}
        
        category, unit_price, quantity, revenue = row

        return {
            "item_category": category,
            "item_price_usd": float(unit_price),
            "refund_count_window": 0,
            "chargeback_flag": False,
            "address_distance_miles": None
        }
    
    finally:
        cs.close()
        ctx.close()
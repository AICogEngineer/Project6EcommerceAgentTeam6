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
    
def snowflake_fetch_user_orders(user_id: str) -> list[dict]:
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    select f.*, p.product_name, p.category
                    from fact_sales f
                    join dim_products p
                    on f.product_id = p.product_id
                    where f.user_id = %s
                    limit 5 
                    """
                cur.execute(query,(user_id,))
                # Just get everything
                # Get column names from cursor description
                columns = [col[0].lower() for col in cur.description]

                # Build list of dictionaries dynamically
                orders = [dict(zip(columns, row)) for row in cur.fetchall()]
                return orders
    except Exception as e:
        print("ERROR fetching user orders from Snowflake:", e)
        return []

# Can handle an incomplete transaction_id
def snowflake_fetch_item_category(transaction_id: str) -> str | None:
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    select p.category
                    from fact_sales f
                    join dim_products p
                    on f.product_id = p.product_id
                    where f.transaction_id like %s
                    limit 1
                    """
                cur.execute(query, (f"%{transaction_id}%",))
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        print("ERROR fetching item category from Snowflake:", e)
        return None

def snowflake_fetch_order(transaction_id: str) -> dict | None:
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    select f.*, p.product_name, p.category
                    from fact_sales f
                    join dim_products p
                    on f.product_id = p.product_id
                    where f.transaction_id like %s
                    limit 1
                    """
                cur.execute(query, (f"%{transaction_id}%",))
                row = cur.fetchone()
                if row:
                    columns = [col[0].lower() for col in cur.description]
                    return dict(zip(columns, row))
                return None
    except Exception as e:
        print("ERROR fetching order from Snowflake:", e)
        return None
    
def snowflake_update_order_refund(transaction_id: str) -> bool:
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    update fact_sales
                    set status = 'refunded'
                    where transaction_id like %s
                    """
                cur.execute(query, (f"%{transaction_id}%",))
                conn.commit()
                return cur.rowcount > 0
    except Exception as e:
        print("ERROR updating order refund in Snowflake:", e)
        return False
    
def snowflake_count_refunded(user_id: str) -> int:
    try:
        with get_snowflake_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    select count(*) 
                    from fact_sales
                    where user_id = %s and status = 'refunded'
                    """
                cur.execute(query, (user_id,))
                row = cur.fetchone()
                return row[0] if row else 0
    except Exception as e:
        print("ERROR counting refunded orders in Snowflake:", e)
        return 0
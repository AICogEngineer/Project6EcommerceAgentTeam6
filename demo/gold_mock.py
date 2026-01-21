# gold_mock.py
# Simulates Snowflake Gold-layer outputs for demos

GOLD_ORDER_LOOKUP = {
    "order_456": {
        "item_category": "electronics",
        "order_date": "2024-12-01",
        "item_price_usd": 249.99,
    }
}

GOLD_RISK_LOOKUP = {
    "user_123": {
        "refund_count_window": 1,
        "returnless_refund_count_window": 0,
        "chargeback_flag": False,
        "address_distance_miles": 12.5,
    }
}

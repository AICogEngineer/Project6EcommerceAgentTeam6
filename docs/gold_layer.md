# Agentic E-Commerce Orchestrator  
## Gold Layer Data Requirements (ML / AI Interface)

**Owner:** ML / AI Engineering  
**Audience:** Data Engineering  
**Purpose:** Define agent-safe, decision-ready Gold views for the LangGraph agent  
**Scope:** v1 Demo / Grading Submission

---

## 1. Design Principles

This agent **does not query raw tables**.  
It reasons only on **Gold-layer signals** that are:

- Deterministic
- Pre-aggregated
- Privacy-safe
- Stable across schema changes

The goal is to separate:
- **Data truth (DE)** from
- **Decision logic (ML/AI)**

---

## 2. Required Gold Views (Minimum Set)

### 🟨 Gold View 1: `gold_user_profile`

**Purpose:** Identity verification, trust assessment, and routing

| Column | Type | Notes |
|-----|-----|-----|
| user_id | STRING (PK) | Stable user identifier |
| account_created_at | DATE | Account creation date |
| account_age_days | INTEGER | Derived |
| trust_tier | STRING | `standard` \| `vip` \| `restricted` |
| vip_status | BOOLEAN | True if VIP |
| pii_verified | BOOLEAN | Identity verification completed |
| account_status | STRING | `active` \| `suspended` \| `review` |

**Used By Agent For**
- Identity gate (hard stop)
- Extended refund windows
- Blocking automation for suspended accounts

❌ No names, emails, phone numbers

---

### 🟨 Gold View 2: `gold_order_context`

**Purpose:** Refund eligibility and policy evaluation

| Column | Type | Notes |
|-----|-----|-----|
| order_id | STRING (PK) | Order identifier |
| user_id | STRING (FK) | Owning user |
| order_date | DATE | Used for 30/60-day window |
| item_category | STRING | Policy lookup key |
| item_price_usd | FLOAT | Automation ceiling logic |
| delivery_status | STRING | `delivered` \| `in_transit` \| `lost` |
| proof_of_delivery | BOOLEAN | POD available |
| shipping_address_hash | STRING | Hashed only |

**Used By Agent For**
- Refund window enforcement
- Pinecone policy retrieval
- Fraud detection

❌ No raw shipping addresses

---

### 🟨 Gold View 3: `gold_refund_risk_signals`

**Purpose:** Fraud, abuse, and escalation logic

| Column | Type | Notes |
|-----|-----|-----|
| user_id | STRING (PK) | User identifier |
| refunds_30d | INTEGER | Velocity check |
| returnless_refunds_90d | INTEGER | Abuse detection |
| return_rate_pct | FLOAT | Refund / order ratio |
| last_refund_date | DATE | Recency |
| chargeback_flag | BOOLEAN | Immediate escalation |

**Used By Agent For**
- Automated vs manual review
- Trust scoring
- Fraud flags

⚠️ Values should be **pre-aggregated**

---

### 🟨 Gold View 4: `gold_geo_risk_signal`

**Purpose:** Privacy-safe location discrepancy detection

| Column | Type | Notes |
|-----|-----|-----|
| user_id | STRING | User identifier |
| address_distance_miles | FLOAT | Shipping vs session |
| geo_mismatch_flag | BOOLEAN | Thresholded |

**Used By Agent For**
- Fraud escalation
- Manual review routing

❌ No IP addresses, cities, states, or lat/long

---

### 🟨 Gold View 5 (Optional): `gold_subscription_status`

**Purpose:** Subscription governance (FTC / ARL)

| Column | Type | Notes |
|-----|-----|-----|
| user_id | STRING | User identifier |
| subscription_status | STRING | `active` \| `paused` \| `cancelled` |
| plan_type | STRING | Plan name |
| billing_cycle | STRING | `monthly` \| `annual` |
| last_charge_date | DATE | Last billing event |
| next_renewal_date | DATE | Renewal notice logic |

---

## 3. Explicit Non-Requirements

The agent **must not** receive:

- Raw transaction events
- Raw session events
- IP addresses
- Billing or shipping addresses
- Payment method details
- Card tokens or financial PII

---

## 4. Integration Expectations

- Views may be tables or secure views
- Column names should remain stable (version if needed)
- Null values should be explicit
- Dates in ISO format

ML will integrate via **tool contracts**, not direct SQL.

---

## 5. Next Steps

Please confirm:
1. Which of these views already exist or are close  
2. Any required column adjustments  
3. Which 3 views can be delivered first (minimum for demo)

Once confirmed, ML will integrate immediately.

---

**End of Document**

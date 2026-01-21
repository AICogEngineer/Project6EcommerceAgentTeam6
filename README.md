# Agentic E-Commerce Orchestrator

## Overview

This project implements an agentic e-commerce refund orchestration system using LangGraph, Pydantic, and retrieval-augmented generation (RAG). The system demonstrates how large language models (LLMs) can be safely integrated into enterprise workflows by strictly separating deterministic decision logic from probabilistic language generation, while enforcing human-in-the-loop (HITL) safeguards.

The orchestrator evaluates refund requests by combining structured transactional data from Snowflake (Gold layer), unstructured policy documents retrieved via Pinecone, deterministic fraud detection rules, and LLM-generated customer-facing explanations. The LLM never makes financial decisions; it is used strictly for explanation and communication, mirroring real-world enterprise AI patterns.

## Key Concepts Demonstrated

- Agentic orchestration using LangGraph
- Strong schema enforcement via Pydantic
- Retrieval-augmented generation (RAG)
- Deterministic fraud and abuse detection
- Human-in-the-loop (HITL) interrupt handling
- Model-agnostic LLM integration
- Enterprise-style data separation (Gold data vs policy text)

## High-Level Architecture

User Request  
↓  
Intent Classification  
↓  
Identity Gate (HITL)  
↓  
Snowflake (Gold Data)  
↓  
Pinecone (Policy RAG)  
↓  
Fraud Detection (Rules)  
↓  
Decision Summary (Deterministic)  
↓  
LLM Explanation (Language Only)  
↓  
Human Review (If Required) → END  

## Project Structure

.
├── agent/
│   ├── graph.py
│   ├── state.py
│
├── adapters/
│   ├── snowflake_adapter.py
│   ├── pinecone_adapter.py
│
├── main.py
├── .env.example
└── README.md

## Agent State Design

The agent state is defined using Pydantic to ensure predictable state transitions, schema validation, and safe tool integration. The state contains routing metadata, identity verification flags, order context fields, fraud signals, retrieved policy clauses, deterministic decision summaries, and the final LLM-generated explanation.

This structured state design allows deterministic components to remain authoritative while enabling controlled language generation at the final stage of execution.

## Deterministic vs Probabilistic Logic

Deterministic (Non-LLM):
- Identity verification gating
- Fraud and abuse detection
- Refund eligibility checks
- Human review escalation
- State transitions

Probabilistic (LLM):
- Natural-language explanation of outcomes
- Customer-facing communication
- Policy interpretation phrasing

This strict separation prevents hallucinations and ensures compliance, auditability, and safety.

## Human-in-the-Loop (HITL)

The system raises LangGraph Interrupts when identity verification fails or fraud and abuse risk thresholds are triggered. These interrupts pause execution and require human approval before continuing, reflecting enterprise compliance and governance requirements.

## LLM Integration

The LLM is intentionally placed after all deterministic decisions have been finalized. Its sole responsibility is to generate a clear, professional, customer-safe explanation based on the current agent state and retrieved policy language.

The orchestration layer is model-agnostic and can be backed by OpenAI models for development or AWS Bedrock models for production deployments.

## Running the Demo

1. Create and activate a virtual environment
2. Install project dependencies
3. Create a `.env` file using `.env.example`
4. Run the demo with:

python main.py

The output includes a deterministic decision summary, a natural-language LLM explanation, and a flag indicating whether human review is required.

## Why LangGraph?

LangGraph enables explicit state transitions, deterministic control flow, safe interrupt handling, and clear separation of responsibilities. These properties make it well-suited for enterprise agentic systems compared to linear chains or prompt-only architectures.

## Production Considerations

In a production environment, Snowflake access would be strictly read-only and IAM-scoped, Pinecone would store versioned and audited policy embeddings, LLM calls would be routed through AWS Bedrock or a private endpoint, observability would be handled through LangSmith or OpenTelemetry, and human review actions would integrate with internal ticketing systems.

## Summary

This project demonstrates how to design a safe, explainable, and enterprise-ready agentic AI system by combining structured and unstructured data sources, enforcing deterministic decision boundaries, using LLMs responsibly for language generation, and supporting human oversight at critical decision points. It is designed to be educational, demo-safe, and production-inspired.

# Project6EcommerceAgentTeam6

## ML / AI Agent Overview

This project implements an agentic e-commerce support workflow using LangGraph.

Key features:
- Intent detection for refund requests
- Identity verification gate (Human-in-the-Loop interrupt)
- Policy grounding via Pinecone (RAG)
- Fraud and abuse detection using Gold-layer risk signals
- Conditional escalation to human review

The agent reasons only on pre-aggregated, privacy-safe Gold views provided by Data Engineering.


In .env, you must set the following values:
```export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_TRACING_V2="true" (for debugging)
export LANGCHAIN_PROJECT="my-langgraph-project"
```
For using the OpenAI models, you need:
```export OPENAI_API_KEY="sk-..."
```
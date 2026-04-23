"""
PHN Technology WhatsApp Agent — LangGraph Agent Graph

Defines the conversation flow as a directed graph.
Each node processes a step, and edges route between them based on intent.

Flow:
    START → classify_intent → [route_by_intent]
                                    │
                    ┌──────┬────────┼─────────────┬──────────────┐
                    ▼      ▼        ▼             ▼              ▼
             rag_retrieve  │  generate_response  human_handoff  handle_interested
                    │      │        │              │              │
                    ▼      │        ▼              ▼              ▼
           generate_response│   score_lead        END            END
                    │      ▼        │
                    ▼   handle_send_brochure
                score_lead  │
                    │       ▼
                    ▼    score_lead
                   END      │
                            ▼
                           END
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from app.config import get_settings

from app.agent.state import AgentState
from app.agent.nodes import (
    classify_intent,
    rag_retrieve,
    generate_response,
    score_lead,
    human_handoff,
    handle_interested,
    handle_send_brochure,
    route_by_intent,
)

logger = logging.getLogger(__name__)


def build_agent_graph():
    """
    Builds and compiles the LangGraph agent with memory checkpointing.

    Returns:
        Compiled LangGraph runnable with conversation memory.
    """

    # Create the state graph
    workflow = StateGraph(AgentState)

    # --- Add all nodes ---
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("rag_retrieve", rag_retrieve)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("score_lead", score_lead)
    workflow.add_node("human_handoff", human_handoff)
    workflow.add_node("handle_interested", handle_interested)
    workflow.add_node("handle_send_brochure", handle_send_brochure)

    # --- Define edges ---

    # Entry point: always classify intent first
    workflow.add_edge(START, "classify_intent")

    # Conditional routing based on intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "rag_retrieve": "rag_retrieve",
            "generate_response": "generate_response",
            "human_handoff": "human_handoff",
            "handle_interested": "handle_interested",
            "handle_send_brochure": "handle_send_brochure",
        }
    )

    # After RAG retrieval, generate response
    workflow.add_edge("rag_retrieve", "generate_response")

    # After generating response, score the lead
    workflow.add_edge("generate_response", "score_lead")

    # After sending brochure, score the lead
    workflow.add_edge("handle_send_brochure", "score_lead")

    # Terminal edges
    workflow.add_edge("score_lead", END)
    workflow.add_edge("human_handoff", END)
    workflow.add_edge("handle_interested", END)  # Direct to END — no scoring needed

    # --- Compile with memory checkpointer ---
    # SqliteSaver keeps conversation state permanently in SQLite (per thread_id)
    # Thread ID = WhatsApp phone number → each user gets their own conversation memory
    settings = get_settings()
    conn = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("✅ LangGraph agent compiled successfully")
    return graph


# Singleton agent instance
_agent = None


def get_agent():
    """Get or create the singleton agent instance."""
    global _agent
    if _agent is None:
        _agent = build_agent_graph()
    return _agent

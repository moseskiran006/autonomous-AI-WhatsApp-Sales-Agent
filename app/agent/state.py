"""
PHN Technology WhatsApp Agent — Agent State Schema

Defines the conversation state that flows through the LangGraph agent.
Each node reads and updates this state as the conversation progresses.
"""

from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State schema for the LangGraph WhatsApp agent.

    Attributes:
        messages: Chat message history (auto-accumulated via add_messages reducer)
        user_phone: WhatsApp phone number of the sender
        user_name: User's name (if known)
        intent: Classified intent of the current message
        language: Detected language ('en' for English, 'hi' for Hindi)
        retrieved_docs: Documents retrieved from RAG pipeline
        lead_score: Current lead score (hot / warm / cold)
        interested_courses: Courses the user has shown interest in
        needs_handoff: Whether to escalate to a human agent
        response_text: Final response text to send back to the user
    """
    messages: Annotated[list[BaseMessage], add_messages]
    user_phone: str
    user_name: str
    intent: str
    language: str
    retrieved_docs: list[str]
    lead_score: str
    interested_courses: str
    city: str
    occupation: str
    interest_field: str
    is_interested: str
    extracted_name: str
    needs_handoff: bool
    response_text: str

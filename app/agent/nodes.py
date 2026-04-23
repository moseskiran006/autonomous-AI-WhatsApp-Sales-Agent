"""
PHN Technology WhatsApp Agent — Agent Nodes

Each function is a node in the LangGraph agent graph.
Nodes read from and write to AgentState as the conversation flows.
"""

import logging
import re
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.agent.state import AgentState
from app.agent.prompts import (
    INTENT_CLASSIFICATION_PROMPT_V2,
    RAG_RESPONSE_PROMPT_V2,
    GENERAL_RESPONSE_PROMPT_V2,
    LEAD_SCORING_PROMPT_V2,
    HANDOFF_MESSAGE_EN,
    HANDOFF_MESSAGE_HI,
)
from app.rag.retriever import retrieve_documents
from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_llm() -> ChatOllama:
    """Create a ChatOllama instance with configured settings."""
    settings = get_settings()
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


def _get_last_user_message(state: AgentState) -> str:
    """Extract the last user message from the conversation."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _format_chat_history(state: AgentState, max_messages: int = 10) -> str:
    """Format recent chat history for prompt context."""
    settings = get_settings()
    history_lines = []
    recent = state["messages"][-(max_messages):]
    for msg in recent:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"Student: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_lines.append(f"Counselor: {msg.content}")
    return "\n".join(history_lines) if history_lines else "No previous conversation."


def _detect_interested_keyword(message: str) -> bool:
    """
    Detect if the student is saying they're interested — using keyword matching
    so we DON'T rely on the LLM to detect this correctly.
    """
    msg_lower = message.lower().strip()
    
    # Exact matches or phrases that clearly mean "interested"
    interested_patterns = [
        r'\binterested\b',
        r'\byes\s+interested\b',
        r'\bim\s+interested\b',
        r'\bi\s*am\s+interested\b',
        r'\bi\'m\s+interested\b',
        r'\bhaan\s+interested\b',
        r'\bjoin\s+karna\b',
        r'\benroll\b',
        r'\bregister\b',
        r'\badmission\b',
        r'\bseat\s+book\b',
        r'\block\s+seat\b',
        r'\block\s+my\s+seat\b',
        r'\bi\s+want\s+to\s+join\b',
        r'\bmujhe\s+join\b',
    ]
    
    for pattern in interested_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False


def _detect_brochure_request(message: str) -> bool:
    """
    Detect if the student is asking for a PDF/brochure.
    """
    msg_lower = message.lower().strip()
    
    brochure_patterns = [
        r'\bsend\s+pdf\b',
        r'\bsend\s+brochure\b',
        r'\bpdf\s+bhejo\b',
        r'\bpdf\s+send\b',
        r'\bbrochure\s+send\b',
        r'\bgive\s+pdf\b',
        r'\bgive\s+brochure\b',
        r'\bsend\s+details\b',
        r'\bsend\s+syllabus\b',
        r'\bsyllabus\s+send\b',
        r'\bwant\s+pdf\b',
        r'\bwant\s+brochure\b',
        r'\bshare\s+pdf\b',
        r'\bshare\s+brochure\b',
    ]
    
    for pattern in brochure_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False


def _detect_program_context(state: AgentState) -> str:
    """
    Figure out which program the student has been talking about
    by looking at chat history and interested_courses.
    """
    interested = state.get("interested_courses", "").lower()
    
    # Check explicit program interest
    if "summer" in interested or "online" in interested:
        return "summer_internship"
    if "edge" in interested or "offline" in interested or "iit" in interested or "nit" in interested:
        return "edge_ai"
    if "aiml" in interested or "ai/ml" in interested or "ai ml" in interested:
        return "aiml"
    
    # Scan recent messages for program mentions
    for msg in reversed(state["messages"][-10:]):
        content = msg.content.lower() if hasattr(msg, 'content') else ""
        if "summer" in content or "online internship" in content:
            return "summer_internship"
        if "offline" in content or "edge ai" in content or "iit" in content or "nit" in content or "campus" in content:
            return "edge_ai"
        if "aiml" in content or "ai/ml" in content or "ai ml" in content:
            return "aiml"
    
    return "summer_internship"  # Default to summer internship


# ============================================
# Node 1: Classify Intent
# ============================================
def classify_intent(state: AgentState) -> dict:
    """
    Classifies the user's message into an intent category
    and detects their preferred language.
    
    CRITICAL: Uses keyword detection FIRST for "interested" and "send_brochure"
    to avoid LLM unreliability on these critical actions.
    """
    user_message = _get_last_user_message(state)
    
    # ---- KEYWORD-BASED DETECTION FIRST (bypass LLM for critical intents) ----
    
    # Check for "interested" FIRST — this is the #1 bug
    if _detect_interested_keyword(user_message):
        logger.info(f"📋 Intent: interested (KEYWORD MATCH) | Message: {user_message[:50]}...")
        # Detect language quickly
        language = "hi" if any(c in user_message for c in "हिंदी") or "karna" in user_message.lower() or "haan" in user_message.lower() else "en"
        return {
            "intent": "interested",
            "language": language,
        }
    
    # Check for brochure/PDF request
    if _detect_brochure_request(user_message):
        logger.info(f"📋 Intent: send_brochure (KEYWORD MATCH) | Message: {user_message[:50]}...")
        language = "hi" if "bhejo" in user_message.lower() else "en"
        return {
            "intent": "send_brochure",
            "language": language,
        }
    
    # ---- LLM-BASED CLASSIFICATION for everything else ----
    llm = _get_llm()

    prompt = INTENT_CLASSIFICATION_PROMPT_V2.format(message=user_message)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Parse structured output
        intent = "general"
        language = "en"

        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                intent = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("LANGUAGE:"):
                language = line.split(":", 1)[1].strip().lower()

        # Validate intent
        valid_intents = [
            "course_query", "bootcamp_query", "pricing_query",
            "schedule_query", "placement_query", "certificate_query",
            "company_query", "policy_query", "greeting",
            "general", "support", "escalation",
            "interested", "send_brochure",
        ]
        if intent not in valid_intents:
            intent = "general"

        # Validate language
        if language not in ["en", "hi"]:
            language = "en"

        logger.info(f"📋 Intent: {intent} | Language: {language} | Message: {user_message[:50]}...")

        return {
            "intent": intent,
            "language": language,
        }

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {
            "intent": "general",
            "language": "en",
        }


# ============================================
# Node 2: RAG Retrieve
# ============================================
def rag_retrieve(state: AgentState) -> dict:
    """
    Retrieves relevant documents from ChromaDB based on the user's query.
    Filters by category based on classified intent.
    """
    user_message = _get_last_user_message(state)
    intent = state.get("intent", "")

    # Map intents to knowledge base categories for filtering
    category_map = {
        "course_query": None,  # Search all
        "bootcamp_query": "bootcamp",
        "pricing_query": None,  # Pricing spans all categories
        "schedule_query": None,
        "placement_query": "placement",
        "certificate_query": "certificate",
    }

    category_filter = category_map.get(intent)

    try:
        docs = retrieve_documents(
            query=user_message,
            k=get_settings().rag_top_k,
            category=category_filter,
        )

        context_texts = []
        for doc in docs:
            context_texts.append(doc.page_content)

        logger.info(f"📚 Retrieved {len(docs)} documents for query: {user_message[:50]}...")

        return {
            "retrieved_docs": context_texts,
        }

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return {
            "retrieved_docs": [],
        }


# ============================================
# Node 3: Generate Response (with RAG context)
# ============================================
def generate_response(state: AgentState) -> dict:
    """
    Generates a response using the LLM with RAG context (if available).
    Responds in the detected language (English or Hindi).
    """
    llm = _get_llm()
    user_message = _get_last_user_message(state)
    chat_history = _format_chat_history(state)
    retrieved_docs = state.get("retrieved_docs", [])
    intent = state.get("intent", "general")
    language = state.get("language", "en")
    
    lang_name = "Hindi/Hinglish" if language == "hi" else "English"

    # Build current student profile knowledge
    interested_courses = state.get("interested_courses", "unknown")
    if not interested_courses:
        interested_courses = "unknown"
    extracted_name = state.get("extracted_name", "unknown")
    city = state.get("city", "unknown")
    occupation = state.get("occupation", "unknown")
    
    student_profile = f"- Name: {extracted_name}\n- City: {city}\n- Interested Program: {interested_courses}\n- Occupation: {occupation}"

    try:
        # Use RAG prompt if we have retrieved context, otherwise general prompt
        if retrieved_docs:
            context = "\n\n---\n\n".join(retrieved_docs)
            prompt = RAG_RESPONSE_PROMPT_V2.format(
                context=context,
                chat_history=chat_history,
                message=user_message,
                detected_language=lang_name,
                student_profile=student_profile,
            )
        else:
            prompt = GENERAL_RESPONSE_PROMPT_V2.format(
                chat_history=chat_history,
                message=user_message,
                detected_language=lang_name,
                student_profile=student_profile,
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        logger.info(f"💬 Generated response ({len(response_text)} chars)")

        return {
            "response_text": response_text,
            "messages": [AIMessage(content=response_text)],
        }

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        error_msg = (
            "I'm having trouble processing your request right now. "
            "Please try again in a moment! 🙏"
        )
        return {
            "response_text": error_msg,
            "messages": [AIMessage(content=error_msg)],
        }


# ============================================
# Node 4: Score Lead
# ============================================
def score_lead(state: AgentState) -> dict:
    """
    Scores the lead based on conversation signals.
    Uses LLM to analyze buying intent and interest level.
    """
    llm = _get_llm()
    chat_history = _format_chat_history(state)

    try:
        prompt = LEAD_SCORING_PROMPT_V2.format(conversation=chat_history)
        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Parse structured output
        score = "cold"
        courses = ""
        extracted_name = ""
        city = ""
        occupation = ""
        interest_field = ""
        is_interested = ""

        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE:"):
                score = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("INTEREST_PROGRAMS:"):
                courses = line.split(":", 1)[1].strip()
            elif line.upper().startswith("COURSES:"):  # backward compat
                if not courses:
                    courses = line.split(":", 1)[1].strip()
            elif line.upper().startswith("NAME:"):
                extracted_name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CITY:"):
                city = line.split(":", 1)[1].strip()
            elif line.upper().startswith("OCCUPATION:"):
                occupation = line.split(":", 1)[1].strip()
            elif line.upper().startswith("INTEREST_FIELDS:"):
                interest_field = line.split(":", 1)[1].strip()
            elif line.upper().startswith("INTEREST_FIELD:"):  # backward compat
                if not interest_field:
                    interest_field = line.split(":", 1)[1].strip()
            elif line.upper().startswith("IS_INTERESTED:"):
                is_interested = line.split(":", 1)[1].strip()

        # Prevent overwriting existing known state with "unknown"
        if extracted_name.lower() in ["", "unknown"] and state.get("extracted_name"):
            extracted_name = state.get("extracted_name")
        if city.lower() in ["", "unknown"] and state.get("city"):
            city = state.get("city")
        if occupation.lower() in ["", "unknown"] and state.get("occupation"):
            occupation = state.get("occupation")
        if interest_field.lower() in ["", "unknown"] and state.get("interest_field"):
            interest_field = state.get("interest_field")
        if is_interested.lower() in ["", "unknown"] and state.get("is_interested"):
            is_interested = state.get("is_interested")

        # Validate score
        if score not in ["hot", "warm", "cold"]:
            score = "cold"

        if courses.lower() in ["none", "unknown"]:
            courses = ""

        logger.info(
            f"🎯 Lead Score: {score} | Programs: {courses or 'none'} | "
            f"Name: {extracted_name} | City: {city} | Interested: {is_interested}"
        )

        return {
            "lead_score": score,
            "interested_courses": courses,
            "extracted_name": extracted_name,
            "city": city,
            "occupation": occupation,
            "interest_field": interest_field,
            "is_interested": is_interested,
        }

    except Exception as e:
        logger.error(f"Lead scoring failed: {e}")
        return {
            "lead_score": "cold",
            "interested_courses": "",
            "extracted_name": "",
            "city": "",
            "occupation": "",
            "interest_field": "",
            "is_interested": "",
        }


# ============================================
# Node 5: Human Handoff
# ============================================
def human_handoff(state: AgentState) -> dict:
    """
    Handles escalation to a human agent.
    Sends a handoff message in the user's language.
    """
    language = state.get("language", "en")

    if language == "hi":
        handoff_msg = HANDOFF_MESSAGE_HI
    else:
        handoff_msg = HANDOFF_MESSAGE_EN

    logger.info(f"🤝 Human handoff triggered for {state.get('user_phone', 'unknown')}")

    return {
        "response_text": handoff_msg,
        "messages": [AIMessage(content=handoff_msg)],
        "needs_handoff": True,
        "lead_score": "hot",  # Escalations are always hot leads
    }


# ============================================
# Node 6: Handle "INTERESTED" — Direct Counselor Connect
# ============================================
def handle_interested(state: AgentState) -> dict:
    """
    When student says "interested" — immediately connect with counselor.
    NO more sales pitch. Just confirm and trigger handoff.
    """
    language = state.get("language", "en")
    extracted_name = state.get("extracted_name", "")
    name_greeting = f" {extracted_name}" if extracted_name and extracted_name.lower() != "unknown" else ""

    if language == "hi":
        response_text = (
            f"बढ़िया{name_greeting}! 🎉\n\n"
            f"मैं अभी Pragati Karad को तुम्हारी details भेज रहा हूं। "
            f"वह तुम्हें shortly WhatsApp करेगी batch dates और enrollment के लिए!\n\n"
            f"[NOTIFY_COUNSELOR]"
        )
    else:
        response_text = (
            f"Awesome{name_greeting}! 🎉\n\n"
            f"I'm sending your details to Pragati Karad right now. "
            f"She'll WhatsApp you shortly to help with batch dates and get you enrolled!\n\n"
            f"[NOTIFY_COUNSELOR]"
        )

    logger.info(f"🎯 INTERESTED detected for {state.get('user_phone', 'unknown')} — triggering counselor handoff")

    return {
        "response_text": response_text,
        "messages": [AIMessage(content=response_text)],
        "needs_handoff": True,
        "lead_score": "hot",
        "is_interested": "Yes",
    }


# ============================================
# Node 7: Handle "SEND_BROCHURE" — Direct PDF Send
# ============================================
def handle_send_brochure(state: AgentState) -> dict:
    """
    When student explicitly asks for PDF/brochure — send it directly.
    Detect which program they're asking about and send the right PDF.
    """
    language = state.get("language", "en")
    program = _detect_program_context(state)
    extracted_name = state.get("extracted_name", "")
    name_greeting = f" {extracted_name}" if extracted_name and extracted_name.lower() != "unknown" else ""
    
    # Map program to brochure tag
    brochure_tags = {
        "summer_internship": "[SEND_SUMMER_INTERNSHIP_BROCHURE]",
        "edge_ai": "[SEND_EDGE_AI_BROCHURE]",
        "aiml": "[SEND_AIML_BROCHURE]",
    }
    
    program_names = {
        "summer_internship": "Summer Internship",
        "edge_ai": "Edge AI & IoT",
        "aiml": "AI/ML & IoT",
    }
    
    brochure_tag = brochure_tags.get(program, "[SEND_SUMMER_INTERNSHIP_BROCHURE]")
    program_name = program_names.get(program, "Summer Internship")
    
    if language == "hi":
        response_text = (
            f"ज़रूर{name_greeting}! यह रहा {program_name} program का brochure 📄\n\n"
            f"इसमें syllabus, projects, और सब details हैं। Check कर लो!\n\n"
            f"अगर join करना है तो बस INTERESTED लिखो, मैं Pragati से connect करा दूंगा 🚀\n\n"
            f"{brochure_tag}"
        )
    else:
        response_text = (
            f"Sure{name_greeting}! Here's the {program_name} brochure 📄\n\n"
            f"It has the full syllabus, projects, and all the details. Check it out!\n\n"
            f"If you wanna go ahead, just type INTERESTED and I'll connect you with Pragati 🚀\n\n"
            f"{brochure_tag}"
        )

    logger.info(f"📄 Sending {program} brochure to {state.get('user_phone', 'unknown')}")

    return {
        "response_text": response_text,
        "messages": [AIMessage(content=response_text)],
        "lead_score": "warm",
    }


# ============================================
# Router: Route by Intent
# ============================================
def route_by_intent(state: AgentState) -> str:
    """
    Conditional edge function that routes to the appropriate node
    based on the classified intent.
    
    KEY CHANGE: "interested" and "send_brochure" now have dedicated nodes
    that bypass the LLM entirely for reliable execution.
    """
    intent = state.get("intent", "general")

    # CRITICAL: "interested" goes to dedicated handler (NOT LLM)
    if intent == "interested":
        return "handle_interested"

    # CRITICAL: "send_brochure" goes to dedicated handler (NOT LLM)
    if intent == "send_brochure":
        return "handle_send_brochure"

    # Escalation intents — skip RAG, go to human handoff
    if intent in ["support", "escalation"]:
        return "human_handoff"

    # Pure greetings — no RAG needed
    if intent == "greeting":
        return "generate_response"

    # EVERYTHING ELSE goes through RAG to avoid hallucination
    return "rag_retrieve"

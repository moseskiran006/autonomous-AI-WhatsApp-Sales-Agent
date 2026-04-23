"""
PHN Technology WhatsApp Agent — Webhook Handler

FastAPI router for handling Meta WhatsApp Business API webhooks.
- GET  /webhook → Verification (Meta challenge-response)
- POST /webhook → Incoming message processing
"""

import logging
import re
from fastapi import APIRouter, Request, Query, BackgroundTasks, Response
from langchain_core.messages import HumanMessage
import asyncio

from app.agent.graph import get_agent
from app.whatsapp.client import send_text_message, mark_as_read, send_media_message, send_document_file, send_image_file
from app.db.models import save_lead, update_lead_score, log_conversation, update_extracted_info, get_lead_info
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["WhatsApp Webhook"])


# ============================================
# GET /webhook — Meta Verification
# ============================================
@router.get("")
async def verify_webhook(
    request: Request,
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge"),
):
    """
    Meta sends a GET request to verify the webhook URL.
    We must return the challenge if the verify token matches.
    """
    settings = get_settings()

    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("✅ Webhook verified successfully")
        return Response(content=hub_challenge, media_type="text/plain")
    else:
        logger.warning(f"❌ Webhook verification failed — token mismatch")
        return Response(content="Forbidden", status_code=403)


# ============================================
# POST /webhook — Incoming Messages
# ============================================
@router.post("")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives incoming WhatsApp messages from Meta.

    Flow:
    1. Parse the incoming webhook payload
    2. Return 200 immediately (Meta requires fast response)
    3. Process the message in the background via LangGraph agent
    4. Send reply back via WhatsApp API
    """
    try:
        body = await request.json()
    except Exception:
        return {"status": "error", "message": "Invalid JSON"}

    # Extract message data from Meta's webhook format
    try:
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})

        # Check if this is a message (not a status update)
        messages = value.get("messages", [])
        if not messages:
            # Status update (delivered, read, etc.) — ignore
            return {"status": "ok"}

        message = messages[0]
        message_type = message.get("type", "")
        sender_phone = message.get("from", "")
        message_id = message.get("id", "")

        # Extract message text
        if message_type == "text":
            message_text = message.get("text", {}).get("body", "")
        elif message_type == "interactive":
            # Button or list reply
            interactive = message.get("interactive", {})
            if "button_reply" in interactive:
                message_text = interactive["button_reply"].get("title", "")
            elif "list_reply" in interactive:
                message_text = interactive["list_reply"].get("title", "")
            else:
                message_text = ""
        else:
            # Unsupported message type (image, audio, etc.)
            message_text = ""

        if not message_text or not sender_phone:
            return {"status": "ok"}

        # Get sender profile name if available
        contacts = value.get("contacts", [])
        sender_name = ""
        if contacts:
            profile = contacts[0].get("profile", {})
            sender_name = profile.get("name", "")

        logger.info(
            f"📩 Incoming message from {sender_phone} ({sender_name}): {message_text[:50]}..."
        )

        # Mark message as read (blue ticks)
        background_tasks.add_task(mark_as_read, message_id)

        # Process message in the background
        base_url = str(request.base_url)
        background_tasks.add_task(
            process_message,
            sender_phone,
            sender_name,
            message_text,
            base_url
        )

    except Exception as e:
        logger.error(f"❌ Error parsing webhook: {e}")

    # Always return 200 quickly to Meta
    return {"status": "ok"}


async def _send_posters(phone: str):
    """
    Send both posters (poster1.jpg and poster2.jpg) to a student.
    Uploads directly to WhatsApp Media API — no public URL needed.
    """
    logger.info(f"🖼️ Sending welcome posters to {phone}...")
    await send_image_file(
        phone,
        "/app/app/static/poster1.jpg",
        "📌 PHN Technology — Edge AI, ML & IoT Programs"
    )
    await send_image_file(
        phone,
        "/app/app/static/poster2.jpg",
        "📌 Placement Support & Program Fees"
    )


async def process_message(phone: str, name: str, message: str, base_url: str = ""):
    """
    Process an incoming message through the LangGraph agent.
    Runs as a background task after the webhook returns 200.
    """
    try:
        agent = get_agent()

        # Use phone number as thread_id for conversation memory
        config = {"configurable": {"thread_id": phone}}

        # Run agent synchronously in a thread (SqliteSaver is sync-only)
        def run_agent():
            return agent.invoke(
                {
                    "messages": [HumanMessage(content=message)],
                    "user_phone": phone,
                    "user_name": name,
                },
                config=config,
            )

        result = await asyncio.to_thread(run_agent)

        # Extract all fields from the result
        response_text = result.get("response_text", "")
        intent = result.get("intent", "")
        lead_score = result.get("lead_score", "cold")
        language = result.get("language", "en")
        interested_courses = result.get("interested_courses", "")
        extracted_name = result.get("extracted_name", "")
        city = result.get("city", "")
        occupation = result.get("occupation", "")
        interest_field = result.get("interest_field", "")
        is_interested = result.get("is_interested", "")

        # Save base lead info to database
        save_lead(phone, name, language)

        # Update newly extracted student data
        update_extracted_info(
            phone=phone,
            extracted_name=extracted_name,
            city=city,
            occupation=occupation,
            interest_field=interest_field,
            is_interested=is_interested
        )

        # Update lead score
        if lead_score:
            update_lead_score(phone, lead_score, interested_courses)

        # Log the conversation
        log_conversation(
            phone=phone,
            direction="incoming",
            message=message,
            intent=intent,
            response=response_text,
            lead_score=lead_score,
        )

        # Send response via WhatsApp
        if response_text:
            # --- HARD SANITIZATION (ZERO TOLERANCE FOR EMAIL SIGNATURES) ---
            response_text = re.sub(r'(?i)(Best regards|Warm regards|Sincerely|Thanks and regards|Best,|Regards,).*', '', response_text, flags=re.DOTALL)
            response_text = response_text.replace("[Your Name]", "").replace("[Counselor Name]", "")
            response_text = response_text.strip()

            # ============================================
            # AUTO-SEND POSTERS ON GREETING (first contact)
            # ============================================
            if intent == "greeting":
                await _send_posters(phone)

            # ============================================
            # Process Media & Brochure Triggers from LLM
            # ============================================

            # --- PDF BROCHURES (uploaded directly to WhatsApp servers) ---
            if "[SEND_EDGE_AI_BROCHURE]" in response_text:
                response_text = response_text.replace("[SEND_EDGE_AI_BROCHURE]", "").strip()
                logger.info(f"📄 Sending Edge AI brochure to {phone}...")
                await send_document_file(
                    phone,
                    "/app/app/static/edge_ai_brochure.pdf",
                    "PHN_Edge_AI_IoT_Brochure.pdf",
                    "📄 Edge AI & IoT Program Brochure"
                )

            if "[SEND_AIML_BROCHURE]" in response_text:
                response_text = response_text.replace("[SEND_AIML_BROCHURE]", "").strip()
                logger.info(f"📄 Sending AI/ML brochure to {phone}...")
                await send_document_file(
                    phone,
                    "/app/app/static/aiml_brochure.pdf",
                    "PHN_AIML_IoT_Brochure.pdf",
                    "📄 AI/ML & IoT Program Brochure"
                )

            if "[SEND_SUMMER_INTERNSHIP_BROCHURE]" in response_text:
                response_text = response_text.replace("[SEND_SUMMER_INTERNSHIP_BROCHURE]", "").strip()
                logger.info(f"📄 Sending Summer Internship brochure to {phone}...")
                await send_document_file(
                    phone,
                    "/app/app/static/summer_internship_brochure.pdf",
                    "PHN_Summer_Internship_Brochure.pdf",
                    "📄 Online Summer Internship Brochure"
                )

            # Backward compat: catch old generic tags
            if "[SEND_ONLINE_INTERNSHIP_BROCHURE]" in response_text:
                response_text = response_text.replace("[SEND_ONLINE_INTERNSHIP_BROCHURE]", "").strip()
                logger.info(f"📄 Sending Online Internship brochure (compat) to {phone}...")
                await send_document_file(
                    phone,
                    "/app/app/static/summer_internship_brochure.pdf",
                    "PHN_Summer_Internship_Brochure.pdf",
                    "📄 Online Summer Internship Brochure"
                )

            if "[SEND_OFFLINE_INTERNSHIP_BROCHURE]" in response_text:
                response_text = response_text.replace("[SEND_OFFLINE_INTERNSHIP_BROCHURE]", "").strip()
                logger.info(f"📄 Sending Offline Internship brochure (compat) to {phone}...")
                await send_document_file(
                    phone,
                    "/app/app/static/edge_ai_brochure.pdf",
                    "PHN_Edge_AI_IoT_Brochure.pdf",
                    "📄 Offline IIT/NIT Internship Brochure"
                )

            # --- POSTERS (uploaded directly — no URL needed) ---
            if "[SEND_PHOTO]" in response_text:
                response_text = response_text.replace("[SEND_PHOTO]", "").strip()
                await _send_posters(phone)

            # --- COUNSELOR HANDOFF ---
            if "[NOTIFY_COUNSELOR]" in response_text or "[NOTIFY_COUNSELOR:PRAGATI_KARAD]" in response_text:
                response_text = response_text.replace("[NOTIFY_COUNSELOR]", "").replace("[NOTIFY_COUNSELOR:PRAGATI_KARAD]", "").strip()

                # Read FULL student profile from database (has all data from previous messages)
                db_lead = get_lead_info(phone) or {}
                
                # Use DB data first, then fall back to current result, then WhatsApp profile name
                student_name = (
                    db_lead.get("extracted_name") or db_lead.get("name") or 
                    extracted_name or name or "Not provided"
                )
                if student_name.lower() == "unknown":
                    student_name = name or "Not provided"
                
                student_city = db_lead.get("city") or city or "Not provided"
                if student_city.lower() == "unknown":
                    student_city = "Not provided"
                
                student_occupation = db_lead.get("occupation") or occupation or "Not provided"
                if student_occupation.lower() == "unknown":
                    student_occupation = "Not provided"
                
                student_interest = (
                    db_lead.get("interested_courses") or db_lead.get("interest_field") or
                    interested_courses or interest_field or "Not specified"
                )
                if student_interest.lower() == "unknown":
                    student_interest = "Not specified"
                
                student_score = db_lead.get("lead_score") or lead_score or "hot"
                total_msgs = db_lead.get("total_messages", 0)

                # Build detailed alert for counselor with FULL info
                alert_message = (
                    f"🚨 HIGH PRIORITY LEAD 🚨\n"
                    f"Name: {student_name}\n"
                    f"Phone: +{phone}\n"
                    f"City: {student_city}\n"
                    f"Occupation: {student_occupation}\n"
                    f"Interested In: {student_interest}\n"
                    f"Lead Score: {student_score.upper()}\n"
                    f"Total Messages: {total_msgs}\n"
                    f"Status: Urgent Callback Requested. They just downloaded the brochure and agreed to speak with a counselor."
                )
                await send_text_message("918600586005", alert_message)
                logger.info(f"🚨 Sent FULL counselor alert for {phone} — Name: {student_name}, City: {student_city}")

            # Clean up any remaining tags that might have been missed
            response_text = re.sub(r'\[SEND_\w+\]', '', response_text)
            response_text = re.sub(r'\[NOTIFY_\w+(?::\w+)*\]', '', response_text)
            response_text = response_text.strip()

            # Send the text message
            if response_text:
                await send_text_message(phone, response_text)

            logger.info(
                f"✅ Replied to {phone} | Intent: {intent} | "
                f"Score: {lead_score} | Language: {language}"
            )

    except Exception as e:
        logger.error(f"❌ Error processing message from {phone}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Send a fallback error message
        await send_text_message(
            phone,
            "Apologies, I'm having trouble right now. Please try again in a moment! 🙏"
        )

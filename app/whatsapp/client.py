"""
PHN Technology WhatsApp Agent — WhatsApp Client

Handles sending messages via Meta WhatsApp Business Cloud API.
Supports text messages and interactive messages (buttons/lists).
"""

import logging
import httpx
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# Meta WhatsApp API base URL
WHATSAPP_API_URL = "https://graph.facebook.com/v21.0"


async def send_text_message(phone_number: str, message: str) -> bool:
    """
    Send a text message via WhatsApp Business API.

    Args:
        phone_number: Recipient's phone number (with country code, no +)
        message: Text message to send

    Returns:
        True if message was sent successfully, False otherwise.
    """
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        logger.warning("⚠️ WhatsApp API not configured — message not sent (this is fine in CLI test mode)")
        return False

    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info(f"✅ Message sent to {phone_number}")
            return True
        else:
            logger.error(
                f"❌ Failed to send message: {response.status_code} — {response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ WhatsApp API error: {e}")
        return False


async def send_media_message(
    phone_number: str, media_type: str, media_url: str, caption: Optional[str] = None
) -> bool:
    """
    Send a media message (image, document, video) via WhatsApp Business API.

    Args:
        phone_number: Recipient's phone number
        media_type: 'image', 'document', or 'video'
        media_url: The public URL of the media file
        caption: Optional text caption (not supported for audio)

    Returns:
        True if sent successfully.
    """
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        logger.warning("⚠️ WhatsApp API not configured — media not sent")
        return False

    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    media_obj = {"link": media_url}
    if caption:
        media_obj["caption"] = caption

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": media_type,
        media_type: media_obj,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info(f"✅ Media ({media_type}) sent to {phone_number}")
            return True
        else:
            logger.error(
                f"❌ Failed to send media: {response.status_code} — {response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ WhatsApp API error: {e}")
        return False


async def send_document_file(
    phone_number: str, file_path: str, filename: str, caption: Optional[str] = None
) -> bool:
    """
    Upload a local file to WhatsApp Media API and send it as a document.

    This uploads the file directly to Meta's servers (no public URL needed),
    gets a media_id back, and then sends the document using that media_id.

    Args:
        phone_number: Recipient's phone number (with country code, no +)
        file_path: Absolute path to the local file (e.g. /app/app/static/brochure.pdf)
        filename: The display filename the recipient will see (e.g. "PHN_Brochure.pdf")
        caption: Optional caption text

    Returns:
        True if sent successfully.
    """
    import os
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        logger.warning("⚠️ WhatsApp API not configured — document not sent")
        return False

    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        return False

    # Step 1: Upload the file to WhatsApp Media API
    upload_url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/media"

    try:
        # Determine MIME type
        if file_path.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        elif file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif file_path.lower().endswith(".png"):
            mime_type = "image/png"
        else:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            file_data = f.read()

        logger.info(f"📤 Uploading {filename} ({len(file_data)} bytes) to WhatsApp Media API...")

        async with httpx.AsyncClient(timeout=120.0) as client:
            upload_response = await client.post(
                upload_url,
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                },
                data={
                    "messaging_product": "whatsapp",
                    "type": mime_type,
                },
                files={
                    "file": (filename, file_data, mime_type),
                },
            )

        if upload_response.status_code != 200:
            logger.error(f"❌ Media upload failed: {upload_response.status_code} — {upload_response.text}")
            return False

        media_id = upload_response.json().get("id")
        if not media_id:
            logger.error(f"❌ No media_id in upload response: {upload_response.text}")
            return False

        logger.info(f"✅ File uploaded successfully. Media ID: {media_id}")

        # Step 2: Send the document using the media_id
        send_url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

        doc_obj = {
            "id": media_id,
            "filename": filename,
        }
        if caption:
            doc_obj["caption"] = caption

        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "document",
            "document": doc_obj,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            send_response = await client.post(
                send_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                    "Content-Type": "application/json",
                },
            )

        if send_response.status_code == 200:
            logger.info(f"✅ Document '{filename}' sent to {phone_number}")
            return True
        else:
            logger.error(f"❌ Failed to send document: {send_response.status_code} — {send_response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error sending document file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def send_image_file(
    phone_number: str, file_path: str, caption: Optional[str] = None
) -> bool:
    """
    Upload a local image file to WhatsApp Media API and send it as an image.

    This uploads the image directly to Meta's servers (no public URL needed),
    gets a media_id back, and then sends the image using that media_id.

    Args:
        phone_number: Recipient's phone number (with country code, no +)
        file_path: Absolute path to the local image (e.g. /app/app/static/poster1.jpg)
        caption: Optional caption text

    Returns:
        True if sent successfully.
    """
    import os
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        logger.warning("⚠️ WhatsApp API not configured — image not sent")
        return False

    if not os.path.exists(file_path):
        logger.error(f"❌ Image file not found: {file_path}")
        return False

    # Step 1: Upload the image to WhatsApp Media API
    upload_url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/media"

    try:
        # Determine MIME type
        if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
            mime_type = "image/jpeg"
        elif file_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif file_path.lower().endswith(".webp"):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        filename = os.path.basename(file_path)

        with open(file_path, "rb") as f:
            file_data = f.read()

        logger.info(f"📤 Uploading image {filename} ({len(file_data)} bytes) to WhatsApp Media API...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            upload_response = await client.post(
                upload_url,
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                },
                data={
                    "messaging_product": "whatsapp",
                    "type": mime_type,
                },
                files={
                    "file": (filename, file_data, mime_type),
                },
            )

        if upload_response.status_code != 200:
            logger.error(f"❌ Image upload failed: {upload_response.status_code} — {upload_response.text}")
            return False

        media_id = upload_response.json().get("id")
        if not media_id:
            logger.error(f"❌ No media_id in upload response: {upload_response.text}")
            return False

        logger.info(f"✅ Image uploaded successfully. Media ID: {media_id}")

        # Step 2: Send the image using the media_id
        send_url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

        img_obj = {"id": media_id}
        if caption:
            img_obj["caption"] = caption

        payload = {
            "messaging_product": "whatsapp",
            "to": phone_number,
            "type": "image",
            "image": img_obj,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            send_response = await client.post(
                send_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.whatsapp_access_token}",
                    "Content-Type": "application/json",
                },
            )

        if send_response.status_code == 200:
            logger.info(f"✅ Image '{filename}' sent to {phone_number}")
            return True
        else:
            logger.error(f"❌ Failed to send image: {send_response.status_code} — {send_response.text}")
            return False

    except Exception as e:
        logger.error(f"❌ Error sending image file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def send_interactive_buttons(
    phone_number: str,
    body_text: str,
    buttons: list[dict],
    header: Optional[str] = None,
    footer: Optional[str] = None,
) -> bool:
    """
    Send an interactive button message.

    Args:
        phone_number: Recipient's phone number
        body_text: Main message body
        buttons: List of button dicts [{"id": "btn_1", "title": "Option 1"}, ...]
        header: Optional header text
        footer: Optional footer text

    Returns:
        True if sent successfully.
    """
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        logger.warning("⚠️ WhatsApp API not configured — message not sent")
        return False

    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    # Build button list (max 3 buttons per WhatsApp API rules)
    button_list = []
    for btn in buttons[:3]:
        button_list.append({
            "type": "reply",
            "reply": {
                "id": btn["id"],
                "title": btn["title"][:20],  # Max 20 chars
            }
        })

    interactive = {
        "type": "button",
        "body": {"text": body_text},
        "action": {"buttons": button_list},
    }

    if header:
        interactive["header"] = {"type": "text", "text": header}
    if footer:
        interactive["footer"] = {"text": footer}

    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "interactive",
        "interactive": interactive,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            logger.info(f"✅ Interactive message sent to {phone_number}")
            return True
        else:
            logger.error(
                f"❌ Failed to send interactive message: {response.status_code} — {response.text}"
            )
            return False

    except Exception as e:
        logger.error(f"❌ WhatsApp API error: {e}")
        return False


async def mark_as_read(message_id: str) -> bool:
    """
    Mark a received message as read (blue ticks).

    Args:
        message_id: The WhatsApp message ID to mark as read.
    """
    settings = get_settings()

    if not settings.whatsapp_access_token or settings.whatsapp_access_token == "your_meta_access_token_here":
        return False

    url = f"{WHATSAPP_API_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload, headers=headers)
        return response.status_code == 200
    except Exception:
        return False

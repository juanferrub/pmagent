"""
WhatsApp integration tools using Meta Cloud API.

Provides:
- send_whatsapp_message: Send a free-form text reply (within 24h window)
- send_whatsapp_template: Send a template message (proactive reports)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import httpx
from langchain_core.tools import tool

from src.config import get_settings
from src.utils import CircuitBreaker, logger

_circuit = CircuitBreaker(threshold=3, reset_timeout=120)

GRAPH_API_VERSION = "v21.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"


def _get_whatsapp_config():
    """Get and validate WhatsApp configuration."""
    settings = get_settings()
    if not settings.whatsapp_access_token:
        raise ValueError(
            "WHATSAPP_ACCESS_TOKEN not configured. "
            "Set it in .env to enable WhatsApp messaging."
        )
    if not settings.whatsapp_phone_number_id:
        raise ValueError(
            "WHATSAPP_PHONE_NUMBER_ID not configured. "
            "Set it in .env to enable WhatsApp messaging."
        )
    return settings


async def _send_whatsapp_api(phone_number_id: str, access_token: str, payload: dict) -> dict:
    """Send a request to the Meta WhatsApp Cloud API."""
    url = f"{GRAPH_API_BASE}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


@tool
def send_whatsapp_message(to: str, text: str) -> str:
    """Send a free-form WhatsApp text message to a phone number.

    Use this within the 24-hour messaging window (after the user messages you first).
    The 'to' parameter should be the recipient's phone number in international format
    (e.g., '1234567890' without the '+' prefix).

    Args:
        to: Recipient phone number in international format (no '+' prefix)
        text: The text message to send
    """
    if _circuit.is_open:
        return json.dumps({"error": "WhatsApp circuit breaker is open, service temporarily unavailable"})
    try:
        import asyncio
        settings = _get_whatsapp_config()

        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to.lstrip("+"),
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }

        # Run async send in sync context
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    _send_whatsapp_api(
                        settings.whatsapp_phone_number_id,
                        settings.whatsapp_access_token,
                        payload,
                    ),
                ).result()
        else:
            result = asyncio.run(
                _send_whatsapp_api(
                    settings.whatsapp_phone_number_id,
                    settings.whatsapp_access_token,
                    payload,
                )
            )

        _circuit.record_success()
        logger.info("whatsapp_message_sent", to=to)
        return json.dumps({"success": True, "message_id": result.get("messages", [{}])[0].get("id", "unknown")})
    except Exception as e:
        _circuit.record_failure()
        logger.error("whatsapp_message_error", error=str(e), to=to)
        return json.dumps({"error": str(e)})


@tool
def send_whatsapp_template(to: str, template_name: str, language_code: str = "en_US", parameters: Optional[str] = None) -> str:
    """Send a WhatsApp template message. Use this for proactive reports sent outside the 24-hour window.

    Template messages must be pre-approved in the Meta Business dashboard.

    Args:
        to: Recipient phone number in international format (no '+' prefix)
        template_name: The name of the pre-approved template
        language_code: Language code for the template (default: en_US)
        parameters: Optional JSON string of template parameters, e.g. '[{"type":"text","text":"Hello"}]'
    """
    if _circuit.is_open:
        return json.dumps({"error": "WhatsApp circuit breaker is open, service temporarily unavailable"})
    try:
        import asyncio
        settings = _get_whatsapp_config()

        template_obj: Dict = {
            "name": template_name,
            "language": {"code": language_code},
        }

        if parameters:
            params = json.loads(parameters)
            template_obj["components"] = [
                {
                    "type": "body",
                    "parameters": params,
                }
            ]

        payload = {
            "messaging_product": "whatsapp",
            "to": to.lstrip("+"),
            "type": "template",
            "template": template_obj,
        }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    _send_whatsapp_api(
                        settings.whatsapp_phone_number_id,
                        settings.whatsapp_access_token,
                        payload,
                    ),
                ).result()
        else:
            result = asyncio.run(
                _send_whatsapp_api(
                    settings.whatsapp_phone_number_id,
                    settings.whatsapp_access_token,
                    payload,
                )
            )

        _circuit.record_success()
        logger.info("whatsapp_template_sent", to=to, template=template_name)
        return json.dumps({"success": True, "message_id": result.get("messages", [{}])[0].get("id", "unknown")})
    except Exception as e:
        _circuit.record_failure()
        logger.error("whatsapp_template_error", error=str(e), to=to)
        return json.dumps({"error": str(e)})


# Export tool list
whatsapp_tools = [send_whatsapp_message, send_whatsapp_template]

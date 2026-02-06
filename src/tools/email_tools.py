"""
Email integration tools using Python's built-in smtplib.

Provides:
- send_email_report: Send an HTML email report via SMTP
"""

from __future__ import annotations

import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import CircuitBreaker, logger

_circuit = CircuitBreaker(threshold=3, reset_timeout=120)


def _get_smtp_config():
    """Get and validate SMTP configuration."""
    settings = get_settings()
    if not settings.smtp_host:
        raise ValueError(
            "SMTP_HOST not configured. "
            "Set SMTP_HOST, SMTP_PORT, SMTP_USERNAME, and SMTP_PASSWORD in .env."
        )
    if not settings.smtp_username or not settings.smtp_password:
        raise ValueError(
            "SMTP_USERNAME and SMTP_PASSWORD are required. "
            "For Gmail, use an App Password (not your regular password)."
        )
    return settings


@tool
def send_email_report(to: str, subject: str, body_html: str) -> str:
    """Send an HTML email report via SMTP.

    Use this when the user asks you to email them a report, summary, or any content.
    The body should be formatted as HTML for rich presentation.

    Args:
        to: Recipient email address (e.g. 'user@example.com').
            Leave empty to use the default recipient from config.
        subject: Email subject line
        body_html: HTML content of the email body. Use proper HTML tags
            like <h2>, <p>, <ul>, <li>, <table>, <strong> for formatting.
    """
    if _circuit.is_open:
        return json.dumps({"error": "Email circuit breaker is open, service temporarily unavailable"})

    try:
        settings = _get_smtp_config()

        # Use default recipient if 'to' is empty
        recipient = to.strip() if to.strip() else settings.email_recipient
        if not recipient:
            return json.dumps({"error": "No recipient specified and no default EMAIL_RECIPIENT configured."})

        sender = settings.smtp_from_email or settings.smtp_username

        # Build the email
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"PM Agent <{sender}>"
        msg["To"] = recipient

        # Create a plain-text fallback by stripping HTML tags
        import re
        plain_text = re.sub(r"<[^>]+>", "", body_html)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()

        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        # Send via SMTP
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.ehlo()
            if settings.smtp_port in (587, 465):
                server.starttls()
                server.ehlo()
            server.login(settings.smtp_username, settings.smtp_password)
            server.sendmail(sender, [recipient], msg.as_string())

        _circuit.record_success()
        logger.info("email_sent", to=recipient, subject=subject)
        return json.dumps({"success": True, "to": recipient, "subject": subject})

    except Exception as e:
        _circuit.record_failure()
        logger.error("email_send_error", error=str(e), to=to, subject=subject)
        return json.dumps({"error": str(e)})


# Export tool list
email_tools = [send_email_report]

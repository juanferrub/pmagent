"""
Email integration tools using Python's built-in smtplib.

Provides:
- send_email_report: Send an HTML email report via SMTP (with evidence gating)
- send_email_report_force: Send without evidence validation (requires explicit approval)

The send_email_report tool integrates with the Evidence Ledger and Safety Gate
to prevent sending reports with fabricated or unverified claims.
"""

from __future__ import annotations

import json
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

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


def _extract_sections_from_html(html: str) -> List[str]:
    """Extract section names from HTML headers."""
    header_pattern = re.compile(r'<h[23][^>]*>([^<]+)</h[23]>', re.IGNORECASE)
    return [match.group(1).strip() for match in header_pattern.finditer(html)]


def _send_email_internal(to: str, subject: str, body_html: str) -> str:
    """Internal function to actually send the email."""
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


@tool
def send_email_report(to: str, subject: str, body_html: str) -> str:
    """Send an HTML email report via SMTP with evidence validation.

    IMPORTANT: This tool validates the report against the Evidence Ledger before sending.
    If the report contains unverified claims or lacks evidence from required data sources,
    the email will NOT be sent. Instead, you'll receive a draft report with:
    - Which sources are missing
    - Which claims couldn't be verified
    - A "Needs Human Check" list

    To send a report successfully:
    1. First gather data using the appropriate agents (jira_agent, github_agent, slack_agent, market_research_agent)
    2. Ensure tool calls succeeded and returned data
    3. Only include claims that are supported by the gathered evidence

    Args:
        to: Recipient email address (e.g. 'user@example.com').
            Leave empty to use the default recipient from config.
        subject: Email subject line
        body_html: HTML content of the email body. Use proper HTML tags
            like <h2>, <h3>, <p>, <ul>, <li>, <table>, <strong> for formatting.
            Include section headers (<h2>, <h3>) for validation.
    """
    # Import here to avoid circular imports
    from src.evidence import get_ledger, get_safety_gate
    from src.source_validation import validate_report_sources
    
    try:
        ledger = get_ledger()
        safety_gate = get_safety_gate()
        
        # Extract sections from HTML
        sections = _extract_sections_from_html(body_html)
        if not sections:
            sections = ["General Report"]
        
        # Run safety gate check
        gate_result = safety_gate.check(body_html, sections)
        
        # Run source validation
        source_valid, validation_results, escalations = validate_report_sources(body_html, ledger)
        
        # If source validation failed, add to gate result
        if not source_valid:
            gate_result.can_send = False
            gate_result.needs_human_check.extend(escalations)
            if gate_result.rejection_reason:
                gate_result.rejection_reason += "; Source validation failed"
            else:
                gate_result.rejection_reason = "Source validation failed"
        
        # Log the check
        logger.info(
            "email_safety_gate",
            can_send=gate_result.can_send,
            tool_success_rate=gate_result.tool_success_rate,
            evidence_coverage=gate_result.evidence_coverage_ratio,
            sections=sections,
            source_valid=source_valid,
        )
        
        if not gate_result.can_send:
            # Generate incomplete draft instead of sending
            draft = safety_gate.generate_incomplete_draft(body_html, gate_result)
            
            return json.dumps({
                "blocked": True,
                "reason": gate_result.rejection_reason,
                "tool_success_rate": gate_result.tool_success_rate,
                "evidence_coverage": gate_result.evidence_coverage_ratio,
                "missing_sources": gate_result.missing_sources,
                "needs_human_check": gate_result.needs_human_check[:10],  # Limit for readability
                "unverified_claims_count": len(gate_result.unverified_claims),
                "draft_report": draft,
                "action_required": (
                    "Email was NOT sent due to insufficient evidence. "
                    "Either: (1) Re-run data collection with proper tool calls, "
                    "(2) Use send_email_report_force to send anyway with explicit approval, "
                    "or (3) Review and address the 'needs_human_check' items."
                ),
            })
        
        # Safety gate passed - send the email
        return _send_email_internal(to, subject, body_html)
        
    except Exception as e:
        logger.error("email_safety_check_error", error=str(e))
        # On safety check error, block sending to be safe
        return json.dumps({
            "blocked": True,
            "reason": f"Safety check error: {str(e)}",
            "action_required": "Fix the safety check error or use send_email_report_force with explicit approval.",
        })


@tool
def send_email_report_force(
    to: str,
    subject: str,
    body_html: str,
    approval_reason: str,
) -> str:
    """Force-send an HTML email report, bypassing evidence validation.

    ⚠️ WARNING: This tool bypasses the evidence safety gate. Only use when:
    1. You have explicit user approval to send an incomplete/unverified report
    2. The user understands the report may contain unverified claims
    3. You've documented why the normal validation couldn't be satisfied

    The approval_reason will be logged for audit purposes.

    Args:
        to: Recipient email address (e.g. 'user@example.com').
            Leave empty to use the default recipient from config.
        subject: Email subject line
        body_html: HTML content of the email body.
        approval_reason: Explanation of why force-send is being used.
            This is logged for audit purposes.
    """
    if not approval_reason or len(approval_reason) < 10:
        return json.dumps({
            "error": "approval_reason is required and must explain why force-send is needed (min 10 chars)",
        })
    
    logger.warning(
        "email_force_send",
        to=to,
        subject=subject,
        approval_reason=approval_reason,
    )
    
    # Add a warning banner to the email
    warning_banner = """
    <div style="background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px; margin-bottom: 20px; border-radius: 4px;">
        <strong>⚠️ Note:</strong> This report was sent with reduced verification. 
        Some claims may not have been validated against source data.
    </div>
    """
    
    # Insert warning after opening body tag or at start
    if "<body" in body_html.lower():
        body_html = re.sub(
            r'(<body[^>]*>)',
            r'\1' + warning_banner,
            body_html,
            flags=re.IGNORECASE,
        )
    else:
        body_html = warning_banner + body_html
    
    return _send_email_internal(to, subject, body_html)


# Export tool list
email_tools = [send_email_report, send_email_report_force]

"""
Proactive alert tools for the PM Agent.

Provides: check_critical_jira_tickets, check_github_trending_issues, send_alert.
Monitors for P0/P1 tickets, trending GitHub issues, and other urgent items.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_jira_circuit = CircuitBreaker(threshold=5, reset_timeout=60)
_github_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


@tool
def check_critical_jira_tickets(hours_back: int = 1) -> str:
    """
    Check for new P0/P1 (critical/high priority) Jira tickets created recently.
    
    Args:
        hours_back: Number of hours to look back (default 1).
    
    Returns:
        JSON with critical tickets that may need immediate attention.
    """
    if _jira_circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    
    try:
        from atlassian import Jira
        
        settings = get_settings()
        if not settings.jira_url or not settings.jira_api_token:
            return '{"error": "Jira credentials not configured"}'
        
        jira = Jira(
            url=settings.jira_url,
            username=settings.jira_user_email,
            password=settings.jira_api_token,
            cloud=True,
        )
        
        # Calculate time threshold
        threshold = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        threshold_str = threshold.strftime("%Y-%m-%d %H:%M")
        
        # JQL for critical tickets
        project_filter = ""
        if settings.jira_project_keys:
            projects = settings.jira_project_keys.split(",")
            project_filter = f"project IN ({','.join(projects)}) AND "
        
        jql = f'{project_filter}priority IN ("Highest", "High", "Critical", "Blocker") AND created >= "{threshold_str}" ORDER BY priority DESC, created DESC'
        
        results = jira.jql(jql, limit=20)
        
        critical_tickets = []
        for issue in results.get("issues", []):
            fields = issue.get("fields", {})
            critical_tickets.append({
                "key": issue.get("key"),
                "summary": fields.get("summary"),
                "priority": fields.get("priority", {}).get("name"),
                "status": fields.get("status", {}).get("name"),
                "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else "Unknown",
                "created": fields.get("created"),
                "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                "labels": fields.get("labels", []),
            })
        
        _jira_circuit.record_success()
        logger.info("critical_jira_check", hours_back=hours_back, found=len(critical_tickets))
        
        return json.dumps({
            "hours_checked": hours_back,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "critical_count": len(critical_tickets),
            "tickets": critical_tickets,
            "alert_needed": len(critical_tickets) > 0,
        }, ensure_ascii=False)
        
    except Exception as e:
        _jira_circuit.record_failure()
        logger.error("critical_jira_check_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def check_github_trending_issues(repo_name: str = "", upvote_threshold: int = 10, days_back: int = 7) -> str:
    """
    Check for GitHub issues that are gaining traction (high reactions/comments).
    
    Args:
        repo_name: Repository to check (default: from settings).
        upvote_threshold: Minimum reactions to consider "trending" (default 10).
        days_back: Days to look back (default 7).
    
    Returns:
        JSON with trending issues that may need attention.
    """
    if _github_circuit.is_open:
        return '{"error": "GitHub circuit breaker is open"}'
    
    try:
        from github import Github
        
        settings = get_settings()
        if not settings.github_token:
            return '{"error": "GITHUB_TOKEN not configured"}'
        
        client = Github(settings.github_token)
        
        # Use provided repo or first from settings
        if not repo_name:
            repos = settings.github_repos.split(",") if settings.github_repos else []
            repo_name = repos[0].strip() if repos else ""
        
        if not repo_name:
            return '{"error": "No repository specified"}'
        
        repo = client.get_repo(repo_name)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        trending_issues = []
        
        for issue in repo.get_issues(state="open", sort="reactions", direction="desc")[:50]:
            if issue.pull_request:
                continue
            
            # Check if created within timeframe
            if issue.created_at and issue.created_at.replace(tzinfo=timezone.utc) < cutoff:
                continue
            
            # Get reaction count
            reactions = issue.reactions
            total_reactions = (
                reactions.get("+1", 0) + 
                reactions.get("heart", 0) + 
                reactions.get("hooray", 0) +
                reactions.get("rocket", 0)
            ) if isinstance(reactions, dict) else 0
            
            # Also consider comments as engagement
            engagement_score = total_reactions + (issue.comments * 2)
            
            if engagement_score >= upvote_threshold:
                trending_issues.append({
                    "number": issue.number,
                    "title": issue.title,
                    "url": issue.html_url,
                    "reactions": total_reactions,
                    "comments": issue.comments,
                    "engagement_score": engagement_score,
                    "labels": [l.name for l in issue.labels],
                    "created": issue.created_at.isoformat() if issue.created_at else "",
                    "author": issue.user.login if issue.user else "",
                })
        
        # Sort by engagement
        trending_issues.sort(key=lambda x: x["engagement_score"], reverse=True)
        
        _github_circuit.record_success()
        logger.info("github_trending_check", repo=repo_name, found=len(trending_issues))
        
        return json.dumps({
            "repo": repo_name,
            "days_checked": days_back,
            "upvote_threshold": upvote_threshold,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "trending_count": len(trending_issues),
            "issues": trending_issues[:10],  # Top 10
            "alert_needed": len(trending_issues) > 0,
        }, ensure_ascii=False)
        
    except Exception as e:
        _github_circuit.record_failure()
        logger.error("github_trending_check_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def check_blocked_tickets(days_stale: int = 5) -> str:
    """
    Check for Jira tickets that appear blocked or stale (no updates in N days).
    
    Args:
        days_stale: Number of days without update to consider stale (default 5).
    
    Returns:
        JSON with blocked/stale tickets that may need attention.
    """
    if _jira_circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    
    try:
        from atlassian import Jira
        
        settings = get_settings()
        if not settings.jira_url or not settings.jira_api_token:
            return '{"error": "Jira credentials not configured"}'
        
        jira = Jira(
            url=settings.jira_url,
            username=settings.jira_user_email,
            password=settings.jira_api_token,
            cloud=True,
        )
        
        # Calculate threshold
        threshold = datetime.now(timezone.utc) - timedelta(days=days_stale)
        threshold_str = threshold.strftime("%Y-%m-%d")
        
        project_filter = ""
        if settings.jira_project_keys:
            projects = settings.jira_project_keys.split(",")
            project_filter = f"project IN ({','.join(projects)}) AND "
        
        # JQL for stale in-progress tickets
        jql = f'{project_filter}status IN ("In Progress", "In Review", "In Development") AND updated < "{threshold_str}" ORDER BY updated ASC'
        
        results = jira.jql(jql, limit=20)
        
        stale_tickets = []
        for issue in results.get("issues", []):
            fields = issue.get("fields", {})
            
            # Check for blocker label or linked blockers
            labels = fields.get("labels", [])
            is_blocked = any("block" in l.lower() for l in labels)
            
            stale_tickets.append({
                "key": issue.get("key"),
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else "Unassigned",
                "updated": fields.get("updated"),
                "priority": fields.get("priority", {}).get("name"),
                "url": f"{settings.jira_url}/browse/{issue.get('key')}",
                "is_blocked": is_blocked,
                "labels": labels,
            })
        
        _jira_circuit.record_success()
        logger.info("blocked_tickets_check", days_stale=days_stale, found=len(stale_tickets))
        
        return json.dumps({
            "days_stale_threshold": days_stale,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "stale_count": len(stale_tickets),
            "tickets": stale_tickets,
            "alert_needed": len(stale_tickets) > 0,
        }, ensure_ascii=False)
        
    except Exception as e:
        _jira_circuit.record_failure()
        logger.error("blocked_tickets_check_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def send_urgent_alert(subject: str, message: str, priority: str = "high") -> str:
    """
    Send an urgent alert via email (and WhatsApp if configured).
    
    Args:
        subject: Alert subject line.
        message: Alert message body (can be HTML).
        priority: Alert priority (high, critical).
    
    Returns:
        JSON with alert delivery status.
    """
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        
        settings = get_settings()
        results = {"email": None, "whatsapp": None}
        
        # Send email alert
        if settings.smtp_host and settings.smtp_username and settings.email_recipient:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"[{priority.upper()} ALERT] {subject}"
                msg["From"] = settings.smtp_from_email or settings.smtp_username
                msg["To"] = settings.email_recipient
                
                # Add priority headers
                msg["X-Priority"] = "1" if priority == "critical" else "2"
                msg["Importance"] = "high"
                
                html_body = f"""
                <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="background: {'#dc3545' if priority == 'critical' else '#fd7e14'}; color: white; padding: 10px; border-radius: 5px;">
                        <strong>⚠️ {priority.upper()} PRIORITY ALERT</strong>
                    </div>
                    <div style="padding: 15px;">
                        {message}
                    </div>
                    <div style="color: #666; font-size: 12px; padding-top: 10px; border-top: 1px solid #eee;">
                        Sent by PM Agent at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
                    </div>
                </body>
                </html>
                """
                
                msg.attach(MIMEText(html_body, "html"))
                
                with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                    server.starttls()
                    server.login(settings.smtp_username, settings.smtp_password)
                    server.send_message(msg)
                
                results["email"] = "sent"
                logger.info("urgent_alert_email_sent", subject=subject, priority=priority)
                
            except Exception as e:
                results["email"] = f"failed: {str(e)}"
                logger.error("urgent_alert_email_failed", error=str(e))
        
        # Send WhatsApp alert if configured
        if settings.whatsapp_phone_number_id and settings.whatsapp_access_token and settings.whatsapp_recipient_phone:
            try:
                import httpx
                
                payload = {
                    "messaging_product": "whatsapp",
                    "to": settings.whatsapp_recipient_phone,
                    "type": "text",
                    "text": {
                        "body": f"⚠️ {priority.upper()} ALERT: {subject}\n\n{message[:500]}"
                    }
                }
                
                resp = httpx.post(
                    f"https://graph.facebook.com/v21.0/{settings.whatsapp_phone_number_id}/messages",
                    headers={
                        "Authorization": f"Bearer {settings.whatsapp_access_token}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                
                if resp.status_code == 200:
                    results["whatsapp"] = "sent"
                    logger.info("urgent_alert_whatsapp_sent", subject=subject)
                else:
                    results["whatsapp"] = f"failed: {resp.text}"
                    
            except Exception as e:
                results["whatsapp"] = f"failed: {str(e)}"
        
        return json.dumps({
            "subject": subject,
            "priority": priority,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "delivery": results,
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error("urgent_alert_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


alert_tools = [
    check_critical_jira_tickets,
    check_github_trending_issues,
    check_blocked_tickets,
    send_urgent_alert,
]

"""
Capabilities tools for the PM Agent.

Provides: check_capabilities - reports which integrations are configured and working.

This tool helps the agent give truthful responses about what it can and cannot do,
rather than making generic claims about capabilities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger


def _check_jira_status() -> Dict[str, Any]:
    """Check Jira integration status."""
    settings = get_settings()
    
    status = {
        "name": "Jira",
        "configured": False,
        "can_read": False,
        "can_write": False,
        "projects": [],
        "issues": [],
    }
    
    if not settings.jira_url or not settings.jira_api_token:
        status["issues"].append("JIRA_URL or JIRA_API_TOKEN not configured")
        return status
    
    status["configured"] = True
    status["projects"] = settings.jira_projects
    
    # Try a simple API call to verify access
    try:
        from atlassian import Jira
        client = Jira(
            url=settings.jira_url,
            username=settings.jira_user_email,
            password=settings.jira_api_token,
            cloud=True,
        )
        # Quick check - get server info
        client.get_server_info()
        status["can_read"] = True
        status["can_write"] = True  # If we can connect, we likely have write access
    except Exception as e:
        status["issues"].append(f"Connection test failed: {str(e)[:100]}")
    
    # Check customer field configuration
    if settings.jira_customer_field_id:
        status["customer_field_configured"] = True
        status["customer_field_id"] = settings.jira_customer_field_id
    else:
        status["customer_field_configured"] = False
        status["issues"].append("JIRA_CUSTOMER_FIELD_ID not set - customer attribution unavailable")
    
    return status


def _check_github_status() -> Dict[str, Any]:
    """Check GitHub integration status."""
    settings = get_settings()
    
    status = {
        "name": "GitHub",
        "configured": False,
        "can_read": False,
        "can_write": False,
        "repositories": [],
        "issues": [],
    }
    
    if not settings.github_token:
        status["issues"].append("GITHUB_TOKEN not configured")
        return status
    
    status["configured"] = True
    status["repositories"] = settings.github_repo_list
    
    # Try a simple API call
    try:
        from github import Github
        g = Github(settings.github_token)
        user = g.get_user()
        _ = user.login  # Trigger API call
        status["can_read"] = True
        
        # Check if we have push access to any repo
        for repo_name in settings.github_repo_list[:1]:  # Check first repo only
            try:
                repo = g.get_repo(repo_name)
                status["can_write"] = repo.permissions.push if repo.permissions else False
            except Exception:
                pass
    except Exception as e:
        status["issues"].append(f"Connection test failed: {str(e)[:100]}")
    
    return status


def _check_slack_status() -> Dict[str, Any]:
    """Check Slack integration status."""
    settings = get_settings()
    
    status = {
        "name": "Slack",
        "configured": False,
        "can_read": False,
        "can_search": False,
        "can_post": False,
        "bot_user": settings.slack_bot_user or "@pm-agent",
        "issues": [],
    }
    
    if not settings.slack_bot_token:
        status["issues"].append("SLACK_BOT_TOKEN not configured")
        return status
    
    status["configured"] = True
    
    # Try API calls to check capabilities
    try:
        from slack_sdk import WebClient
        client = WebClient(token=settings.slack_bot_token)
        
        # Check auth
        auth = client.auth_test()
        if auth.get("ok"):
            status["can_read"] = True
            status["can_post"] = True
            status["bot_id"] = auth.get("user_id")
            status["team"] = auth.get("team")
        
        # Check if search is available (requires User token, not Bot token)
        try:
            client.search_messages(query="test", count=1)
            status["can_search"] = True
        except Exception as search_err:
            if "not_allowed_token_type" in str(search_err):
                status["issues"].append("Search unavailable - requires User token (Bot token configured)")
                status["can_search"] = False
    except Exception as e:
        status["issues"].append(f"Connection test failed: {str(e)[:100]}")
    
    return status


def _check_notion_status() -> Dict[str, Any]:
    """Check Notion integration status."""
    settings = get_settings()
    
    status = {
        "name": "Notion",
        "configured": False,
        "can_read": False,
        "can_write": False,
        "issues": [],
    }
    
    if not settings.notion_api_key:
        status["issues"].append("NOTION_API_KEY not configured")
        return status
    
    status["configured"] = True
    
    try:
        from notion_client import Client
        client = Client(auth=settings.notion_api_key)
        # Try to list users (simple API check)
        client.users.list()
        status["can_read"] = True
        status["can_write"] = True
    except Exception as e:
        status["issues"].append(f"Connection test failed: {str(e)[:100]}")
    
    return status


def _check_web_research_status() -> Dict[str, Any]:
    """Check web research (Tavily) integration status."""
    settings = get_settings()
    
    status = {
        "name": "Web Research (Tavily)",
        "configured": False,
        "can_search": False,
        "issues": [],
    }
    
    if not settings.tavily_api_key:
        status["issues"].append("TAVILY_API_KEY not configured")
        return status
    
    status["configured"] = True
    status["can_search"] = True  # If key is set, assume it works
    
    return status


def _check_email_status() -> Dict[str, Any]:
    """Check email (SMTP) integration status."""
    settings = get_settings()
    
    status = {
        "name": "Email (SMTP)",
        "configured": False,
        "can_send": False,
        "issues": [],
    }
    
    if not settings.smtp_host or not settings.smtp_username:
        status["issues"].append("SMTP_HOST or SMTP_USERNAME not configured")
        return status
    
    status["configured"] = True
    status["can_send"] = True
    status["from_email"] = settings.smtp_from_email
    
    return status


def _check_whatsapp_status() -> Dict[str, Any]:
    """Check WhatsApp integration status."""
    settings = get_settings()
    
    status = {
        "name": "WhatsApp",
        "configured": False,
        "can_send": False,
        "issues": [],
    }
    
    if not settings.whatsapp_access_token or not settings.whatsapp_phone_number_id:
        status["issues"].append("WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID not configured")
        return status
    
    status["configured"] = True
    status["can_send"] = True
    
    return status


@tool
def check_capabilities() -> str:
    """
    Check which integrations are configured and working.
    
    Use this tool when:
    - User asks "what can you do?" or "what integrations do you have?"
    - User asks about specific capabilities (e.g., "can you access Jira?")
    - Before making claims about what data sources are available
    
    Returns:
        JSON string with status of each integration including:
        - configured: Whether credentials are set
        - can_read/can_write/can_search: Specific capabilities
        - issues: Any problems detected
        - summary: Human-readable capability summary
    """
    integrations = {
        "jira": _check_jira_status(),
        "github": _check_github_status(),
        "slack": _check_slack_status(),
        "notion": _check_notion_status(),
        "web_research": _check_web_research_status(),
        "email": _check_email_status(),
        "whatsapp": _check_whatsapp_status(),
    }
    
    # Build summary
    working = []
    partial = []
    not_configured = []
    
    for name, status in integrations.items():
        if not status["configured"]:
            not_configured.append(status["name"])
        elif status.get("issues"):
            partial.append(status["name"])
        else:
            working.append(status["name"])
    
    summary_parts = []
    if working:
        summary_parts.append(f"Fully working: {', '.join(working)}")
    if partial:
        summary_parts.append(f"Partial/issues: {', '.join(partial)}")
    if not_configured:
        summary_parts.append(f"Not configured: {', '.join(not_configured)}")
    
    result = {
        "integrations": integrations,
        "summary": {
            "fully_working": working,
            "partial_or_issues": partial,
            "not_configured": not_configured,
            "description": "; ".join(summary_parts),
        },
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    
    logger.info(
        "capabilities_check",
        working=len(working),
        partial=len(partial),
        not_configured=len(not_configured),
    )
    
    return json.dumps(result, ensure_ascii=False)


@tool
def get_configured_projects() -> str:
    """
    Get the list of configured projects and repositories.
    
    Use this to know which Jira projects and GitHub repos are available
    before making queries.
    
    Returns:
        JSON string with configured project keys and repo names.
    """
    settings = get_settings()
    
    result = {
        "jira_projects": settings.jira_projects,
        "github_repositories": settings.github_repo_list,
        "slack_channels": {
            "alert_channel": settings.slack_alert_channel,
            "summary_channel": settings.slack_summary_channel,
        },
        "timezone": settings.timezone,
    }
    
    return json.dumps(result, ensure_ascii=False)


# Exported list for agent binding
capabilities_tools = [check_capabilities, get_configured_projects]

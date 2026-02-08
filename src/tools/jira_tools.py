"""
Jira integration tools for the PM Agent.

Provides: search_jira_issues, get_jira_issue, create_jira_issue, update_jira_issue.
Uses atlassian-python-api under the hood.

Customer Attribution:
- Set JIRA_CUSTOMER_FIELD_ID to the custom field ID for customer/account data
- If not configured, customer information will not be included
- Never fabricate customer names - only use data from Jira fields
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


def _get_customer_field_value(fields: Dict[str, Any]) -> Optional[str]:
    """
    Extract customer/account value from Jira fields if configured.
    
    Returns None if:
    - JIRA_CUSTOMER_FIELD_ID is not configured
    - The field doesn't exist in the issue
    - The field value is empty
    
    Never fabricates customer names.
    """
    settings = get_settings()
    customer_field_id = settings.jira_customer_field_id
    
    if not customer_field_id:
        return None
    
    customer_value = fields.get(customer_field_id)
    
    if customer_value is None:
        return None
    
    # Handle different field formats (string, object with name/value, array)
    if isinstance(customer_value, str):
        return customer_value if customer_value.strip() else None
    elif isinstance(customer_value, dict):
        # Common patterns: {"name": "..."}, {"value": "..."}, {"displayName": "..."}
        return (
            customer_value.get("name") or 
            customer_value.get("value") or 
            customer_value.get("displayName") or
            None
        )
    elif isinstance(customer_value, list) and customer_value:
        # Multi-select or array field - take first value
        first = customer_value[0]
        if isinstance(first, str):
            return first
        elif isinstance(first, dict):
            return first.get("name") or first.get("value") or None
    
    return None


def _extract_text_from_adf(node) -> str:
    """Extract plain text from Atlassian Document Format (ADF) used by Jira Cloud v3."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    text = node.get("text", "")
    for child in node.get("content", []):
        text += _extract_text_from_adf(child)
    return text


def _get_description_text(desc) -> str:
    """Get description text whether it's a string or ADF dict."""
    if desc is None:
        return ""
    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        return _extract_text_from_adf(desc)
    return str(desc)


def _get_jira_client():
    """Lazy-init Jira client."""
    from atlassian import Jira

    settings = get_settings()
    if not settings.jira_url or not settings.jira_api_token:
        raise ValueError("Jira credentials not configured (JIRA_URL, JIRA_API_TOKEN)")
    return Jira(
        url=settings.jira_url,
        username=settings.jira_user_email,
        password=settings.jira_api_token,
        cloud=True,
    )


@tool
def search_jira_issues(jql: str, max_results: int = 50, fields: str = "") -> str:
    """
    Search Jira issues using JQL (Jira Query Language).

    Args:
        jql: JQL query string (e.g., "project = PROD AND status = Open").
        max_results: Maximum issues to return.
        fields: Comma-separated list of fields to return. If empty, returns default fields.
                Available: summary, status, priority, issuetype, assignee, reporter, 
                created, updated, labels, description, resolution, components.

    Returns:
        JSON string with:
        - issues: List of issues with requested fields
        - query_info: Details about the query executed (JQL, count, timestamp)
        - customer_field_status: Whether customer field is configured
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open - too many recent failures"}'
    
    try:
        client = _get_jira_client()
        settings = get_settings()
        
        # Build fields list for API call
        api_fields = [
            "summary", "status", "priority", "issuetype", "assignee", 
            "reporter", "created", "updated", "labels", "description"
        ]
        
        # Add customer field if configured
        customer_field_id = settings.jira_customer_field_id
        if customer_field_id:
            api_fields.append(customer_field_id)
        
        results = client.jql(jql, limit=min(max_results, 100), fields=api_fields)
        issues = []
        
        for issue in results.get("issues", []):
            issue_fields = issue.get("fields", {})
            
            issue_data = {
                "key": issue.get("key", ""),
                "summary": issue_fields.get("summary", ""),
                "status": (issue_fields.get("status") or {}).get("name", ""),
                "priority": (issue_fields.get("priority") or {}).get("name", ""),
                "issue_type": (issue_fields.get("issuetype") or {}).get("name", ""),
                "assignee": (issue_fields.get("assignee") or {}).get("displayName", "Unassigned"),
                "reporter": (issue_fields.get("reporter") or {}).get("displayName", ""),
                "created": issue_fields.get("created", ""),
                "updated": issue_fields.get("updated", ""),
                "labels": issue_fields.get("labels", []),
                "description": _get_description_text(issue_fields.get("description"))[:500],
            }
            
            # Add customer if configured and available (never fabricate)
            customer = _get_customer_field_value(issue_fields)
            if customer_field_id:
                issue_data["customer"] = customer  # Will be None if not found
            
            issues.append(issue_data)
        
        _circuit.record_success()
        
        # Build response with query metadata
        from datetime import datetime, timezone
        response = {
            "issues": issues,
            "query_info": {
                "jql": jql,
                "total_results": len(issues),
                "max_results": max_results,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "customer_field_status": {
                "configured": bool(customer_field_id),
                "field_id": customer_field_id if customer_field_id else None,
                "note": "Customer field not configured - set JIRA_CUSTOMER_FIELD_ID" if not customer_field_id else None,
            }
        }
        
        logger.info("jira_search", jql=jql, count=len(issues))
        return json.dumps(response, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_search_error", jql=jql, error=str(e))
        return json.dumps({
            "error": str(e),
            "query_info": {"jql": jql, "failed": True},
        })


@tool
def get_jira_issue(issue_key: str) -> str:
    """
    Get full details of a specific Jira issue.

    Args:
        issue_key: The issue key (e.g., "PROD-123").

    Returns:
        JSON string with full issue details including comments and customer info (if configured).
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open - too many recent failures"}'
    try:
        client = _get_jira_client()
        settings = get_settings()
        issue = client.issue(issue_key)
        fields = issue.get("fields", {})
        
        comments = []
        for c in (fields.get("comment", {}).get("comments", []))[:20]:
            comments.append({
                "author": (c.get("author") or {}).get("displayName", ""),
                "body": _get_description_text(c.get("body"))[:500],
                "created": c.get("created", ""),
            })
        
        result = {
            "key": issue.get("key", ""),
            "summary": fields.get("summary", ""),
            "description": _get_description_text(fields.get("description")),
            "status": (fields.get("status") or {}).get("name", ""),
            "priority": (fields.get("priority") or {}).get("name", ""),
            "issue_type": (fields.get("issuetype") or {}).get("name", ""),
            "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
            "reporter": (fields.get("reporter") or {}).get("displayName", ""),
            "labels": fields.get("labels", []),
            "created": fields.get("created", ""),
            "updated": fields.get("updated", ""),
            "resolution": (fields.get("resolution") or {}).get("name") if fields.get("resolution") else None,
            "comments": comments,
            "url": f"{settings.jira_url}/browse/{issue.get('key', '')}",
        }
        
        # Add customer if configured (never fabricate)
        customer_field_id = settings.jira_customer_field_id
        if customer_field_id:
            result["customer"] = _get_customer_field_value(fields)
            result["customer_field_configured"] = True
        else:
            result["customer_field_configured"] = False
            result["customer_note"] = "Customer field not configured - showing Reporter instead"
        
        _circuit.record_success()
        logger.info("jira_get_issue", key=issue_key)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_get_issue_error", key=issue_key, error=str(e))
        return json.dumps({"error": str(e), "issue_key": issue_key, "failed": True})


@tool
def create_jira_issue(
    project_key: str,
    summary: str,
    issue_type: str = "Task",
    description: str = "",
    priority: str = "Medium",
    labels: str = "",
) -> str:
    """
    Create a new Jira issue.

    Args:
        project_key: Project key (e.g., "PROD").
        summary: Issue title/summary.
        issue_type: Type of issue (Task, Bug, Story, Epic).
        description: Detailed description.
        priority: Priority level (Highest, High, Medium, Low, Lowest).
        labels: Comma-separated labels.

    Returns:
        JSON string with the created issue key and URL.
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    try:
        client = _get_jira_client()
        fields = {
            "project": {"key": project_key},
            "summary": summary,
            "issuetype": {"name": issue_type},
            "description": description,
            "priority": {"name": priority},
        }
        if labels:
            fields["labels"] = [l.strip() for l in labels.split(",")]
        result = client.create_issue(fields=fields)
        _circuit.record_success()
        issue_key = result.get("key", "")
        settings = get_settings()
        url = f"{settings.jira_url}/browse/{issue_key}"
        logger.info("jira_create_issue", key=issue_key)
        return json.dumps({"key": issue_key, "url": url})
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_create_issue_error", error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def update_jira_issue(issue_key: str, fields_json: str) -> str:
    """
    Update an existing Jira issue.

    Args:
        issue_key: The issue key (e.g., "PROD-123").
        fields_json: JSON string of fields to update (e.g., '{"summary": "New title", "priority": {"name": "High"}}').

    Returns:
        JSON with update result.
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    try:
        client = _get_jira_client()
        fields = json.loads(fields_json)
        client.update_issue_field(issue_key, fields)
        _circuit.record_success()
        logger.info("jira_update_issue", key=issue_key)
        return json.dumps({"ok": True, "key": issue_key})
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_update_issue_error", key=issue_key, error=str(e))
        return f'{{"error": "{str(e)}"}}'


jira_tools = [search_jira_issues, get_jira_issue, create_jira_issue, update_jira_issue]

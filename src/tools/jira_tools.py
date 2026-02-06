"""
Jira integration tools for the PM Agent.

Provides: search_jira_issues, get_jira_issue, create_jira_issue, update_jira_issue.
Uses atlassian-python-api under the hood.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


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
def search_jira_issues(jql: str, max_results: int = 50) -> str:
    """
    Search Jira issues using JQL (Jira Query Language).

    Args:
        jql: JQL query string (e.g., "project = PROD AND status = Open").
        max_results: Maximum issues to return.

    Returns:
        JSON string with list of issues (key, summary, status, priority, assignee, created, updated).
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    try:
        client = _get_jira_client()
        results = client.jql(jql, limit=min(max_results, 100))
        issues = []
        for issue in results.get("issues", []):
            fields = issue.get("fields", {})
            issues.append({
                "key": issue.get("key", ""),
                "summary": fields.get("summary", ""),
                "status": (fields.get("status") or {}).get("name", ""),
                "priority": (fields.get("priority") or {}).get("name", ""),
                "issue_type": (fields.get("issuetype") or {}).get("name", ""),
                "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
                "reporter": (fields.get("reporter") or {}).get("displayName", ""),
                "created": fields.get("created", ""),
                "updated": fields.get("updated", ""),
                "labels": fields.get("labels", []),
                "description": _get_description_text(fields.get("description"))[:500],
            })
        _circuit.record_success()
        logger.info("jira_search", jql=jql, count=len(issues))
        return json.dumps(issues, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_search_error", jql=jql, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def get_jira_issue(issue_key: str) -> str:
    """
    Get full details of a specific Jira issue.

    Args:
        issue_key: The issue key (e.g., "PROD-123").

    Returns:
        JSON string with full issue details including comments.
    """
    if _circuit.is_open:
        return '{"error": "Jira circuit breaker is open"}'
    try:
        client = _get_jira_client()
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
        }
        _circuit.record_success()
        logger.info("jira_get_issue", key=issue_key)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("jira_get_issue_error", key=issue_key, error=str(e))
        return f'{{"error": "{str(e)}"}}'


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

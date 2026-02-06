"""
Slack integration tools for the PM Agent.

Provides: read_channel_history, search_messages, post_message.
Uses slack_sdk.WebClient under the hood.
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, retry_with_backoff, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


def _get_slack_client():
    """Lazy-init Slack WebClient."""
    from slack_sdk import WebClient

    settings = get_settings()
    if not settings.slack_bot_token:
        raise ValueError("SLACK_BOT_TOKEN not configured")
    return WebClient(token=settings.slack_bot_token)


def _resolve_channel_id(client, channel: str) -> str:
    """Resolve a channel name (e.g. '#general' or 'general') to a Slack channel ID.
    If the input already looks like an ID (starts with C/G/D), return it as-is."""
    ch = channel.lstrip("#").strip()
    if ch and ch[0] in ("C", "G", "D") and len(ch) >= 9 and ch.isalnum():
        return ch  # already an ID
    # List channels the bot has access to and find a match
    cursor = None
    while True:
        kwargs = {"types": "public_channel", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_list(**kwargs)
        for c in resp.get("channels", []):
            if c.get("name") == ch or c.get("name_normalized") == ch:
                return c["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    raise ValueError(f"Slack channel '{channel}' not found. Make sure the bot is invited to it.")


@tool
def read_channel_history(channel: str, limit: int = 50) -> str:
    """
    Read recent messages from a Slack channel.

    Args:
        channel: Channel ID or name (e.g., C01234567 or #product).
        limit: Number of messages to retrieve (max 1000).

    Returns:
        JSON string with list of messages including user, text, timestamp.
    """
    if _circuit.is_open:
        return '{"error": "Slack circuit breaker is open, service temporarily unavailable"}'

    try:
        client = _get_slack_client()
        channel_id = _resolve_channel_id(client, channel)
        # Try to join the channel first (no-op if already a member)
        try:
            client.conversations_join(channel=channel_id)
        except Exception:
            pass  # may lack channels:join scope or channel is private
        try:
            result = client.conversations_history(channel=channel_id, limit=min(limit, 1000))
        except Exception as join_err:
            if "not_in_channel" in str(join_err):
                return json.dumps({"error": f"Bot is not a member of {channel}. Please invite it by typing /invite @pm_agent in the channel."})
            raise
        messages = []
        for msg in result.get("messages", []):
            messages.append({
                "user": msg.get("user", "unknown"),
                "text": msg.get("text", ""),
                "ts": msg.get("ts", ""),
                "thread_ts": msg.get("thread_ts"),
                "reactions": [r["name"] for r in msg.get("reactions", [])],
            })
        _circuit.record_success()
        logger.info("slack_read_history", channel=channel, count=len(messages))
        return json.dumps(messages, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("slack_read_history_error", channel=channel, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def search_slack_messages(query: str, count: int = 20) -> str:
    """
    Search Slack messages across all channels the bot has access to.

    Args:
        query: Search query string.
        count: Max number of results.

    Returns:
        JSON string with matching messages.
    """
    if _circuit.is_open:
        return '{"error": "Slack circuit breaker is open"}'

    try:
        client = _get_slack_client()
        result = client.search_messages(query=query, count=min(count, 100))
        matches = []
        for match in result.get("messages", {}).get("matches", []):
            matches.append({
                "channel": match.get("channel", {}).get("name", ""),
                "user": match.get("username", ""),
                "text": match.get("text", ""),
                "ts": match.get("ts", ""),
                "permalink": match.get("permalink", ""),
            })
        _circuit.record_success()
        logger.info("slack_search", query=query, results=len(matches))
        return json.dumps(matches, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("slack_search_error", query=query, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def post_slack_message(channel: str, text: str, blocks: Optional[str] = None) -> str:
    """
    Post a message to a Slack channel.

    Args:
        channel: Channel ID or name.
        text: Plain text message (fallback for notifications).
        blocks: Optional JSON string of Slack Block Kit blocks for rich formatting.

    Returns:
        JSON string with result (ok/error and message ts).
    """
    if _circuit.is_open:
        return '{"error": "Slack circuit breaker is open"}'

    try:
        client = _get_slack_client()
        channel_id = _resolve_channel_id(client, channel)
        kwargs = {"channel": channel_id, "text": text}
        if blocks:
            kwargs["blocks"] = json.loads(blocks)
        result = client.chat_postMessage(**kwargs)
        _circuit.record_success()
        logger.info("slack_post", channel=channel, ok=result.get("ok"))
        return json.dumps({"ok": result.get("ok"), "ts": result.get("ts")})
    except Exception as e:
        _circuit.record_failure()
        logger.error("slack_post_error", channel=channel, error=str(e))
        return f'{{"error": "{str(e)}"}}'


# Exported list for agent binding
slack_tools = [read_channel_history, search_slack_messages, post_slack_message]

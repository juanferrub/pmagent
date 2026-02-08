"""
Slack integration tools for the PM Agent.

Provides: read_channel_history, search_messages, post_message, list_channels.
Uses slack_sdk.WebClient under the hood.

Permission Errors:
- If the bot is not in a channel, errors will include the configured bot handle
- Set SLACK_BOT_USER to customize the invite instruction (default: @pm-agent)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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


def _get_bot_user_handle() -> str:
    """Get the configured bot user handle for invite instructions."""
    settings = get_settings()
    return settings.slack_bot_user or "@pm-agent"


def _format_permission_error(channel: str, error_type: str) -> Dict[str, Any]:
    """Format a helpful permission error with invite instructions."""
    bot_handle = _get_bot_user_handle()
    
    error_messages = {
        "not_in_channel": {
            "error": f"Bot is not a member of channel '{channel}'",
            "error_type": "permission_error",
            "resolution": f"Invite the bot by typing: /invite {bot_handle}",
            "channel": channel,
        },
        "channel_not_found": {
            "error": f"Channel '{channel}' not found or bot doesn't have access",
            "error_type": "not_found",
            "resolution": f"Verify the channel name/ID is correct and invite the bot: /invite {bot_handle}",
            "channel": channel,
        },
        "not_allowed_token_type": {
            "error": "Slack token doesn't have required permissions for this operation",
            "error_type": "token_scope_error",
            "resolution": "The Slack app needs additional OAuth scopes. Contact your admin to update the app permissions.",
            "required_scopes": ["channels:history", "channels:read", "search:read"],
        },
        "missing_scope": {
            "error": "Missing required OAuth scope for this operation",
            "error_type": "token_scope_error",
            "resolution": "Update the Slack app's OAuth scopes in the Slack API dashboard.",
        },
    }
    
    return error_messages.get(error_type, {
        "error": f"Slack API error: {error_type}",
        "error_type": "unknown",
        "resolution": f"Try inviting the bot to the channel: /invite {bot_handle}",
    })


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
def read_channel_history(channel: str, limit: int = 50, oldest: str = "", latest: str = "") -> str:
    """
    Read recent messages from a Slack channel.

    Args:
        channel: Channel ID or name (e.g., C01234567 or #product).
        limit: Number of messages to retrieve (max 1000).
        oldest: Unix timestamp - only messages after this time (for time-bounded queries).
        latest: Unix timestamp - only messages before this time.

    Returns:
        JSON string with:
        - messages: List of messages with user, text, timestamp
        - query_info: Channel, time range, and count metadata
        - error: If failed, includes resolution steps
    """
    if _circuit.is_open:
        return json.dumps({
            "error": "Slack circuit breaker is open - too many recent failures",
            "error_type": "circuit_breaker",
            "resolution": "Wait a minute and try again",
        })

    try:
        client = _get_slack_client()
        channel_id = _resolve_channel_id(client, channel)
        
        # Try to join the channel first (no-op if already a member)
        try:
            client.conversations_join(channel=channel_id)
        except Exception:
            pass  # may lack channels:join scope or channel is private
        
        # Build API call kwargs
        kwargs = {"channel": channel_id, "limit": min(limit, 1000)}
        if oldest:
            kwargs["oldest"] = oldest
        if latest:
            kwargs["latest"] = latest
        
        try:
            result = client.conversations_history(**kwargs)
        except Exception as api_err:
            error_str = str(api_err).lower()
            if "not_in_channel" in error_str:
                return json.dumps(_format_permission_error(channel, "not_in_channel"))
            elif "channel_not_found" in error_str:
                return json.dumps(_format_permission_error(channel, "channel_not_found"))
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
        
        # Build response with query metadata
        response = {
            "messages": messages,
            "query_info": {
                "channel": channel,
                "channel_id": channel_id,
                "message_count": len(messages),
                "oldest": oldest or "not specified",
                "latest": latest or "not specified",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }
        
        return json.dumps(response, ensure_ascii=False)
    except ValueError as ve:
        # Channel resolution errors
        _circuit.record_failure()
        logger.error("slack_read_history_error", channel=channel, error=str(ve))
        return json.dumps(_format_permission_error(channel, "channel_not_found"))
    except Exception as e:
        _circuit.record_failure()
        error_str = str(e).lower()
        logger.error("slack_read_history_error", channel=channel, error=str(e))
        
        # Detect specific error types
        if "not_allowed_token_type" in error_str:
            return json.dumps(_format_permission_error(channel, "not_allowed_token_type"))
        elif "missing_scope" in error_str:
            return json.dumps(_format_permission_error(channel, "missing_scope"))
        
        return json.dumps({
            "error": str(e),
            "channel": channel,
            "resolution": f"Check bot permissions and try: /invite {_get_bot_user_handle()}",
        })


@tool
def search_slack_messages(query: str, count: int = 20) -> str:
    """
    Search Slack messages across all channels the bot has access to.

    Args:
        query: Search query string (supports Slack search modifiers like 'in:#channel from:@user').
        count: Max number of results.

    Returns:
        JSON string with:
        - matches: List of matching messages with channel, user, text, permalink
        - query_info: Search query and result count
        - error: If failed, includes resolution steps (search requires User token, not Bot token)
    """
    if _circuit.is_open:
        return json.dumps({
            "error": "Slack circuit breaker is open - too many recent failures",
            "error_type": "circuit_breaker",
            "resolution": "Wait a minute and try again",
        })

    try:
        client = _get_slack_client()
        result = client.search_messages(query=query, count=min(count, 100))
        matches = []
        for match in result.get("messages", {}).get("matches", []):
            matches.append({
                "channel": match.get("channel", {}).get("name", ""),
                "channel_id": match.get("channel", {}).get("id", ""),
                "user": match.get("username", ""),
                "text": match.get("text", ""),
                "ts": match.get("ts", ""),
                "permalink": match.get("permalink", ""),
            })
        _circuit.record_success()
        logger.info("slack_search", query=query, results=len(matches))
        
        response = {
            "matches": matches,
            "query_info": {
                "query": query,
                "result_count": len(matches),
                "max_requested": count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }
        
        return json.dumps(response, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        error_str = str(e).lower()
        logger.error("slack_search_error", query=query, error=str(e))
        
        # Search requires User token, not Bot token
        if "not_allowed_token_type" in error_str:
            return json.dumps({
                "error": "Slack search requires a User OAuth token, not a Bot token",
                "error_type": "token_type_error",
                "resolution": "Use read_channel_history instead to read specific channels, or configure a User token with search:read scope",
                "alternative": "Call read_channel_history(channel='#channel-name') to read messages from a specific channel",
                "query": query,
            })
        
        return json.dumps({
            "error": str(e),
            "query": query,
            "resolution": "Check Slack app permissions or use read_channel_history for specific channels",
        })


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


@tool
def list_slack_channels(limit: int = 100) -> str:
    """
    List Slack channels the bot has access to.
    
    Use this to discover available channels before reading history.

    Args:
        limit: Maximum number of channels to return.

    Returns:
        JSON string with:
        - channels: List of channels with id, name, is_member status
        - query_info: Count and timestamp
    """
    if _circuit.is_open:
        return json.dumps({
            "error": "Slack circuit breaker is open - too many recent failures",
            "error_type": "circuit_breaker",
        })

    try:
        client = _get_slack_client()
        channels = []
        cursor = None
        
        while len(channels) < limit:
            kwargs = {"types": "public_channel,private_channel", "limit": min(200, limit - len(channels))}
            if cursor:
                kwargs["cursor"] = cursor
            
            resp = client.conversations_list(**kwargs)
            
            for ch in resp.get("channels", []):
                channels.append({
                    "id": ch.get("id", ""),
                    "name": ch.get("name", ""),
                    "is_member": ch.get("is_member", False),
                    "is_private": ch.get("is_private", False),
                    "num_members": ch.get("num_members", 0),
                    "topic": ch.get("topic", {}).get("value", "")[:100],
                })
            
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        
        _circuit.record_success()
        logger.info("slack_list_channels", count=len(channels))
        
        # Separate member vs non-member channels
        member_channels = [c for c in channels if c["is_member"]]
        
        return json.dumps({
            "channels": channels[:limit],
            "query_info": {
                "total_returned": len(channels[:limit]),
                "bot_is_member_of": len(member_channels),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "note": "Use channel 'id' or '#name' with read_channel_history",
            }
        }, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("slack_list_channels_error", error=str(e))
        return json.dumps({
            "error": str(e),
            "resolution": "Check SLACK_BOT_TOKEN has channels:read scope",
        })


# Exported list for agent binding
slack_tools = [read_channel_history, search_slack_messages, post_slack_message, list_slack_channels]

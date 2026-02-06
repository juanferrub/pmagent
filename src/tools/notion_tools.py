"""
Notion integration tools for the PM Agent.

Provides: query_notion_database, read_notion_page, append_notion_blocks, create_notion_page.
Uses notion-client under the hood.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.config import get_settings
from src.utils import logger, CircuitBreaker

_circuit = CircuitBreaker(threshold=5, reset_timeout=60)


def _get_notion_client():
    """Lazy-init Notion client."""
    from notion_client import Client

    settings = get_settings()
    if not settings.notion_api_key:
        raise ValueError("NOTION_API_KEY not configured")
    return Client(auth=settings.notion_api_key)


def _extract_text(rich_text_list) -> str:
    """Extract plain text from Notion rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in (rich_text_list or []))


@tool
def query_notion_database(database_id: str, filter_json: str = "", limit: int = 50) -> str:
    """
    Query a Notion database with optional filter.

    Args:
        database_id: Notion database ID.
        filter_json: Optional JSON filter object (Notion API filter format). Empty string = no filter.
        limit: Max results to return.

    Returns:
        JSON array of database entries with their properties.
    """
    if _circuit.is_open:
        return '{"error": "Notion circuit breaker is open"}'
    try:
        client = _get_notion_client()
        kwargs = {"database_id": database_id, "page_size": min(limit, 100)}
        if filter_json:
            kwargs["filter"] = json.loads(filter_json)
        results = client.databases.query(**kwargs)
        pages = []
        for page in results.get("results", []):
            props = {}
            for name, prop in page.get("properties", {}).items():
                ptype = prop.get("type", "")
                if ptype == "title":
                    props[name] = _extract_text(prop.get("title", []))
                elif ptype == "rich_text":
                    props[name] = _extract_text(prop.get("rich_text", []))
                elif ptype == "select":
                    props[name] = (prop.get("select") or {}).get("name", "")
                elif ptype == "multi_select":
                    props[name] = [s.get("name", "") for s in prop.get("multi_select", [])]
                elif ptype == "status":
                    props[name] = (prop.get("status") or {}).get("name", "")
                elif ptype == "date":
                    d = prop.get("date")
                    props[name] = d.get("start", "") if d else ""
                elif ptype == "number":
                    props[name] = prop.get("number")
                elif ptype == "checkbox":
                    props[name] = prop.get("checkbox", False)
                elif ptype == "url":
                    props[name] = prop.get("url", "")
                elif ptype == "people":
                    props[name] = [p.get("name", "") for p in prop.get("people", [])]
                else:
                    props[name] = str(prop.get(ptype, ""))
            pages.append({
                "id": page.get("id", ""),
                "url": page.get("url", ""),
                "last_edited": page.get("last_edited_time", ""),
                "properties": props,
            })
        _circuit.record_success()
        logger.info("notion_query_db", database_id=database_id, count=len(pages))
        return json.dumps(pages, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("notion_query_db_error", database_id=database_id, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def read_notion_page(page_id: str) -> str:
    """
    Read a Notion page's content (blocks).

    Args:
        page_id: Notion page ID.

    Returns:
        JSON with page title, properties, and text content of blocks.
    """
    if _circuit.is_open:
        return '{"error": "Notion circuit breaker is open"}'
    try:
        client = _get_notion_client()
        page = client.pages.retrieve(page_id=page_id)
        blocks = client.blocks.children.list(block_id=page_id, page_size=100)
        content_parts = []
        for block in blocks.get("results", []):
            btype = block.get("type", "")
            block_data = block.get(btype, {})
            if "rich_text" in block_data:
                text = _extract_text(block_data["rich_text"])
                if text:
                    content_parts.append(text)
            elif btype == "child_database":
                content_parts.append(f"[Database: {block_data.get('title', 'Untitled')}]")
        # Extract title
        title = ""
        for name, prop in page.get("properties", {}).items():
            if prop.get("type") == "title":
                title = _extract_text(prop.get("title", []))
                break
        result = {
            "id": page.get("id", ""),
            "title": title,
            "url": page.get("url", ""),
            "last_edited": page.get("last_edited_time", ""),
            "content": "\n".join(content_parts),
        }
        _circuit.record_success()
        logger.info("notion_read_page", page_id=page_id)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        _circuit.record_failure()
        logger.error("notion_read_page_error", page_id=page_id, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def create_notion_page(parent_page_id: str, title: str, content_markdown: str) -> str:
    """
    Create a new Notion page under a parent page.

    Args:
        parent_page_id: Parent page ID to create under.
        title: Title for the new page.
        content_markdown: Markdown-ish content. Each line becomes a paragraph block.

    Returns:
        JSON with the new page ID and URL.
    """
    if _circuit.is_open:
        return '{"error": "Notion circuit breaker is open"}'
    try:
        client = _get_notion_client()
        # Build children blocks from content lines
        children = []
        for line in content_markdown.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                children.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]},
                })
            elif line.startswith("## "):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]},
                })
            elif line.startswith("### "):
                children.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {"rich_text": [{"type": "text", "text": {"content": line[4:]}}]},
                })
            elif line.startswith("- "):
                children.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]},
                })
            else:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]},
                })

        result = client.pages.create(
            parent={"page_id": parent_page_id},
            properties={"title": [{"type": "text", "text": {"content": title}}]},
            children=children[:100],  # Notion API limit
        )
        _circuit.record_success()
        logger.info("notion_create_page", title=title, page_id=result.get("id"))
        return json.dumps({"id": result.get("id", ""), "url": result.get("url", "")})
    except Exception as e:
        _circuit.record_failure()
        logger.error("notion_create_page_error", title=title, error=str(e))
        return f'{{"error": "{str(e)}"}}'


@tool
def append_notion_blocks(page_id: str, content_markdown: str) -> str:
    """
    Append content blocks to an existing Notion page.

    Args:
        page_id: Notion page ID to append to.
        content_markdown: Content to append. Each line becomes a block.

    Returns:
        JSON with result status.
    """
    if _circuit.is_open:
        return '{"error": "Notion circuit breaker is open"}'
    try:
        client = _get_notion_client()
        children = []
        for line in content_markdown.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                children.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]},
                })
            elif line.startswith("## "):
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]},
                })
            elif line.startswith("- "):
                children.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]},
                })
            else:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]},
                })
        client.blocks.children.append(block_id=page_id, children=children[:100])
        _circuit.record_success()
        logger.info("notion_append_blocks", page_id=page_id, blocks=len(children))
        return json.dumps({"ok": True, "blocks_added": len(children)})
    except Exception as e:
        _circuit.record_failure()
        logger.error("notion_append_blocks_error", page_id=page_id, error=str(e))
        return f'{{"error": "{str(e)}"}}'


notion_tools = [query_notion_database, read_notion_page, create_notion_page, append_notion_blocks]

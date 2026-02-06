"""
API routes for the PM Agent.

Implements:
- POST /invoke     → sync invocation
- POST /stream     → SSE streaming responses
- POST /chat       → conversational interface
- POST /webhooks/slack   → Slack events
- POST /webhooks/jira    → Jira webhooks
- POST /webhooks/github  → GitHub webhooks
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graphs.main_graph import invoke_graph, stream_graph
from src.utils import logger

router = APIRouter()


# ── Auth dependency ──────────────────────────────────────────────

async def verify_auth(authorization: Optional[str] = Header(None)):
    """Simple token-based auth (AC-8.1)."""
    settings = get_settings()
    if not settings.api_auth_token:
        return  # No auth configured, skip
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.replace("Bearer ", "")
    if token != settings.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Request / Response models ────────────────────────────────────

class InvokeRequest(BaseModel):
    query: str = Field(..., description="The query or task to execute")
    thread_id: Optional[str] = Field(None, description="Thread ID for state continuity")
    trigger_type: str = Field("manual", description="Trigger type: manual, scheduled, webhook")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InvokeResponse(BaseModel):
    thread_id: str
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    thread_id: str
    reply: str


# ── Routes ───────────────────────────────────────────────────────

@router.post("/invoke", response_model=InvokeResponse, dependencies=[Depends(verify_auth)])
async def invoke_endpoint(request: InvokeRequest):
    """
    Synchronously invoke the PM Agent graph.
    Returns the final result after all agents complete.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        result = await invoke_graph(
            query=request.query,
            thread_id=thread_id,
            trigger_type=request.trigger_type,
            metadata=request.metadata,
        )
        # Extract final response from messages
        messages = result.get("messages", [])
        response_text = messages[-1].content if messages else "No response generated"
        return InvokeResponse(
            thread_id=thread_id,
            response=response_text,
            metadata={"trigger_type": request.trigger_type},
        )
    except Exception as e:
        logger.error("invoke_error", error=str(e), thread_id=thread_id)
        raise HTTPException(status_code=500, detail=str(e))


_NODE_LABELS = {
    "supervisor": "Thinking...",
    "jira_agent": "Searching Jira...",
    "github_agent": "Checking GitHub...",
    "market_research_agent": "Researching the web...",
    "notion_agent": "Reading Notion...",
    "slack_agent": "Checking Slack...",
}


@router.post("/stream", dependencies=[Depends(verify_auth)])
async def stream_endpoint(request: InvokeRequest):
    """
    Stream the PM Agent graph execution via Server-Sent Events.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator():
        try:
            last_node = None
            async for event in stream_graph(
                query=request.query,
                thread_id=thread_id,
                trigger_type=request.trigger_type,
            ):
                # event is a dict like {"node_name": {"messages": [...]}}
                for node_name, node_data in event.items():
                    # Emit a status event when the active agent changes
                    if node_name != last_node:
                        last_node = node_name
                        label = _NODE_LABELS.get(node_name, "Working...")
                        status = json.dumps({
                            "type": "status",
                            "node": node_name,
                            "label": label,
                        })
                        yield f"data: {status}\n\n"

                    messages = node_data.get("messages", [])
                    for msg in messages:
                        if hasattr(msg, 'content') and msg.content:
                            msg_type = getattr(msg, 'type', '')
                            if msg_type == 'human':
                                continue
                            data = json.dumps({
                                "node": node_name,
                                "content": msg.content,
                                "type": msg_type,
                            }, default=str)
                            yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'done': True, 'thread_id': thread_id})}\n\n"
        except Exception as e:
            logger.error("stream_error", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_auth)])
async def chat_endpoint(request: ChatRequest):
    """
    Conversational chat interface. Maintains context via thread_id.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    try:
        result = await invoke_graph(
            query=request.message,
            thread_id=thread_id,
            trigger_type="manual",
        )
        messages = result.get("messages", [])
        reply = messages[-1].content if messages else "I couldn't generate a response."
        return ChatResponse(thread_id=thread_id, reply=reply)
    except Exception as e:
        logger.error("chat_error", error=str(e), thread_id=thread_id)
        raise HTTPException(status_code=500, detail=str(e))


# ── Webhook Endpoints ────────────────────────────────────────────

@router.post("/webhooks/slack")
async def slack_webhook(request: Request):
    """
    Handle Slack Events API webhooks.
    Processes: URL verification, event callbacks (message, app_mention).
    """
    body = await request.json()

    # URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    # Event callback
    if body.get("type") == "event_callback":
        event = body.get("event", {})
        event_type = event.get("type", "")
        thread_id = f"slack-{event.get('channel', 'unknown')}-{event.get('ts', '')}"

        logger.info("slack_webhook", event_type=event_type, channel=event.get("channel"))

        if event_type in ("message", "app_mention"):
            text = event.get("text", "")
            if text:
                try:
                    await invoke_graph(
                        query=f"Slack event ({event_type}): {text}",
                        thread_id=thread_id,
                        trigger_type="webhook",
                        metadata={"source": "slack", "event": event},
                    )
                except Exception as e:
                    logger.error("slack_webhook_error", error=str(e))

    return {"ok": True}


@router.post("/webhooks/jira")
async def jira_webhook(request: Request):
    """
    Handle Jira webhooks.
    Processes: issue created, issue updated, comment added.
    """
    body = await request.json()
    webhook_event = body.get("webhookEvent", "")
    issue = body.get("issue", {})
    issue_key = issue.get("key", "unknown")
    thread_id = f"jira-{issue_key}-{webhook_event}"

    logger.info("jira_webhook", webhook_event=webhook_event, issue_key=issue_key)

    if webhook_event in ("jira:issue_created", "jira:issue_updated"):
        fields = issue.get("fields", {})
        summary = fields.get("summary", "")
        priority = (fields.get("priority") or {}).get("name", "")
        issue_type = (fields.get("issuetype") or {}).get("name", "")

        query = (
            f"Jira {webhook_event}: [{issue_key}] {summary} "
            f"(Type: {issue_type}, Priority: {priority}). "
            f"Analyze this issue, check for related market feedback, "
            f"and suggest prioritization."
        )

        try:
            await invoke_graph(
                query=query,
                thread_id=thread_id,
                trigger_type="webhook",
                metadata={"source": "jira", "issue_key": issue_key, "event": webhook_event},
            )
        except Exception as e:
            logger.error("jira_webhook_error", error=str(e), issue_key=issue_key)

    return {"ok": True}


@router.post("/webhooks/github")
async def github_webhook(request: Request):
    """
    Handle GitHub webhooks.
    Processes: issues opened, PR opened/merged.
    """
    body = await request.json()
    action = body.get("action", "")
    thread_id = f"github-{action}-{uuid.uuid4().hex[:8]}"

    # Pull Request events
    if "pull_request" in body:
        pr = body["pull_request"]
        pr_number = pr.get("number", 0)
        repo = body.get("repository", {}).get("full_name", "")
        title = pr.get("title", "")

        logger.info("github_webhook_pr", action=action, repo=repo, pr=pr_number)

        if action in ("opened", "synchronize", "closed"):
            query = (
                f"GitHub PR {action}: {repo}#{pr_number} - {title}. "
                f"Review this PR, analyze the diff for potential issues, "
                f"and check if any Jira tickets should be updated."
            )
            try:
                await invoke_graph(
                    query=query,
                    thread_id=thread_id,
                    trigger_type="webhook",
                    metadata={"source": "github", "repo": repo, "pr": pr_number},
                )
            except Exception as e:
                logger.error("github_webhook_error", error=str(e))

    # Issue events
    elif "issue" in body:
        issue = body["issue"]
        number = issue.get("number", 0)
        repo = body.get("repository", {}).get("full_name", "")
        title = issue.get("title", "")

        logger.info("github_webhook_issue", action=action, repo=repo, issue=number)

        if action in ("opened", "labeled"):
            query = (
                f"GitHub issue {action}: {repo}#{number} - {title}. "
                f"Analyze this issue and determine if it needs PM attention."
            )
            try:
                await invoke_graph(
                    query=query,
                    thread_id=thread_id,
                    trigger_type="webhook",
                    metadata={"source": "github", "repo": repo, "issue": number},
                )
            except Exception as e:
                logger.error("github_webhook_error", error=str(e))

    return {"ok": True}


# ── Scheduled Job Triggers ────────────────────────────────

@router.post("/trigger/{job_name}", dependencies=[Depends(verify_auth)])
async def trigger_scheduled_job(job_name: str):
    """
    Manually trigger a scheduled job for testing.
    
    Available jobs:
    - daily_digest: Full daily PM briefing
    - weekly_market_scan: Competitor and market analysis
    - hourly_check: Quick scan for urgent items
    - critical_alert_check: P0/P1 alert scan
    - weekly_customer_voice: Customer feedback aggregation
    - weekly_status_update: Team status report
    """
    from api.background import (
        run_daily_digest,
        run_weekly_market_scan,
        run_hourly_check,
        run_critical_alert_check,
        run_weekly_customer_voice,
        run_weekly_status_update,
    )
    
    jobs = {
        "daily_digest": run_daily_digest,
        "weekly_market_scan": run_weekly_market_scan,
        "hourly_check": run_hourly_check,
        "critical_alert_check": run_critical_alert_check,
        "weekly_customer_voice": run_weekly_customer_voice,
        "weekly_status_update": run_weekly_status_update,
    }
    
    if job_name not in jobs:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown job: {job_name}. Available: {list(jobs.keys())}"
        )
    
    logger.info("manual_job_trigger", job=job_name)
    
    try:
        result = await jobs[job_name]()
        return {"ok": True, "job": job_name, "status": "completed"}
    except Exception as e:
        logger.error("manual_job_error", job=job_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scheduler/status")
async def scheduler_status():
    """Get status of all scheduled jobs."""
    from api.background import get_scheduler
    
    scheduler = get_scheduler()
    if not scheduler:
        return {"running": False, "jobs": []}
    
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger": str(job.trigger),
        })
    
    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


# ── WhatsApp Webhooks ────────────────────────────────────

@router.get("/webhooks/whatsapp")
async def whatsapp_webhook_verify(request: Request):
    """
    Meta webhook verification endpoint.
    Facebook sends a GET with hub.mode, hub.verify_token, hub.challenge.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    settings = get_settings()
    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        logger.info("whatsapp_webhook_verified")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("whatsapp_webhook_verification_failed", mode=mode)
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhooks/whatsapp")
async def whatsapp_webhook_inbound(request: Request):
    """
    Receive inbound WhatsApp messages from Meta Cloud API.
    Extracts sender phone + message text, invokes the agent graph,
    and replies back via the WhatsApp API.
    """
    body = await request.json()
    logger.info("whatsapp_webhook_received")

    try:
        entries = body.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])

                for msg in messages:
                    if msg.get("type") != "text":
                        continue

                    sender_phone = msg.get("from", "")
                    text = msg.get("text", {}).get("body", "")
                    if not text:
                        continue

                    thread_id = f"whatsapp-{sender_phone}"
                    logger.info(
                        "whatsapp_inbound_message",
                        sender=sender_phone,
                        text_preview=text[:80],
                    )

                    try:
                        result = await invoke_graph(
                            query=text,
                            thread_id=thread_id,
                            trigger_type="webhook",
                            metadata={"source": "whatsapp", "sender": sender_phone},
                        )

                        result_messages = result.get("messages", [])
                        reply = (
                            result_messages[-1].content
                            if result_messages
                            else "Sorry, I couldn't process that."
                        )

                        # Send reply back via WhatsApp Cloud API
                        settings = get_settings()
                        if settings.whatsapp_access_token and settings.whatsapp_phone_number_id:
                            api_url = (
                                f"https://graph.facebook.com/v21.0/"
                                f"{settings.whatsapp_phone_number_id}/messages"
                            )
                            payload = {
                                "messaging_product": "whatsapp",
                                "to": sender_phone,
                                "type": "text",
                                "text": {"body": reply[:4096]},
                            }
                            async with httpx.AsyncClient(timeout=30) as client:
                                resp = await client.post(
                                    api_url,
                                    json=payload,
                                    headers={
                                        "Authorization": f"Bearer {settings.whatsapp_access_token}",
                                        "Content-Type": "application/json",
                                    },
                                )
                                resp.raise_for_status()
                                logger.info("whatsapp_reply_sent", to=sender_phone)

                    except Exception as e:
                        logger.error("whatsapp_agent_error", error=str(e), sender=sender_phone)

    except Exception as e:
        logger.error("whatsapp_webhook_error", error=str(e))

    # Always return 200 to Meta
    return {"ok": True}

"""
Tests for the FastAPI application and routes.

Validates:
- AC-7.3: API endpoints
- AC-8.1: Authentication
- AC-3.1: Webhook processing
- AC-11.2: Health check
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked graph."""
    with patch("api.routes.invoke_graph", new_callable=AsyncMock) as mock_invoke, \
         patch("api.routes.stream_graph", new_callable=AsyncMock) as mock_stream, \
         patch("api.background.start_scheduler"), \
         patch("api.background.stop_scheduler"):

        mock_invoke.return_value = {
            "messages": [MagicMock(content="Test response from PM Agent")]
        }

        from api.main import app
        with TestClient(app) as c:
            c._mock_invoke = mock_invoke
            c._mock_stream = mock_stream
            yield c


class TestHealthCheck:
    """Test health endpoint (AC-11.2)."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "pm-agent"
        assert data["version"] == "1.0.0"


class TestAuthentication:
    """Test auth (AC-8.1)."""

    def test_invoke_without_auth_returns_401(self, client):
        response = client.post("/invoke", json={"query": "test"})
        assert response.status_code == 401

    def test_invoke_with_invalid_token_returns_401(self, client):
        response = client.post(
            "/invoke",
            json={"query": "test"},
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_invoke_with_valid_token_succeeds(self, client):
        response = client.post(
            "/invoke",
            json={"query": "test"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200


class TestInvokeEndpoint:
    """Test POST /invoke (AC-7.3)."""

    def test_invoke_returns_response(self, client):
        response = client.post(
            "/invoke",
            json={"query": "Summarize latest Jira tickets"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert "response" in data
        assert data["response"] == "Test response from PM Agent"

    def test_invoke_with_custom_thread_id(self, client):
        response = client.post(
            "/invoke",
            json={"query": "Test", "thread_id": "my-thread-123"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "my-thread-123"

    def test_invoke_with_metadata(self, client):
        response = client.post(
            "/invoke",
            json={
                "query": "Test",
                "trigger_type": "scheduled",
                "metadata": {"workflow": "daily_digest"},
            },
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["trigger_type"] == "scheduled"


class TestChatEndpoint:
    """Test POST /chat (AC-7.3)."""

    def test_chat_returns_reply(self, client):
        response = client.post(
            "/chat",
            json={"message": "What are the top blockers?"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert "reply" in data

    def test_chat_with_thread_id(self, client):
        response = client.post(
            "/chat",
            json={"message": "Follow up question", "thread_id": "session-1"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == "session-1"


class TestStreamEndpoint:
    """Test POST /stream (AC-7.3)."""

    def test_stream_returns_sse(self, client):
        async def mock_stream_gen(*args, **kwargs):
            yield {"node": "supervisor", "data": "processing"}
            yield {"node": "jira_agent", "data": "analyzing"}

        client._mock_stream.return_value = mock_stream_gen()

        response = client.post(
            "/stream",
            json={"query": "Test stream"},
            headers={"Authorization": "Bearer test-auth-token"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


class TestWebhooks:
    """Test webhook endpoints (AC-3.1)."""

    def test_slack_url_verification(self, client):
        response = client.post(
            "/webhooks/slack",
            json={"type": "url_verification", "challenge": "abc123"},
        )
        assert response.status_code == 200
        assert response.json()["challenge"] == "abc123"

    def test_slack_event_callback(self, client):
        response = client.post(
            "/webhooks/slack",
            json={
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C01",
                    "text": "Test message",
                    "ts": "123456",
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_jira_webhook_issue_created(self, client):
        response = client.post(
            "/webhooks/jira",
            json={
                "webhookEvent": "jira:issue_created",
                "issue": {
                    "key": "PROD-123",
                    "fields": {
                        "summary": "Critical bug",
                        "priority": {"name": "Highest"},
                        "issuetype": {"name": "Bug"},
                    },
                },
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_github_webhook_pr_opened(self, client):
        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "pull_request": {
                    "number": 42,
                    "title": "Add new feature",
                    "body": "This PR adds...",
                },
                "repository": {"full_name": "org/repo"},
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_github_webhook_issue_opened(self, client):
        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "issue": {
                    "number": 99,
                    "title": "Bug in production",
                },
                "repository": {"full_name": "org/repo"},
            },
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True


class TestScheduler:
    """Test scheduler (AC-3.2)."""

    @pytest.mark.asyncio
    async def test_start_scheduler(self):
        from api.background import start_scheduler, stop_scheduler, get_scheduler

        start_scheduler()
        scheduler = get_scheduler()
        assert scheduler is not None
        assert scheduler.running is True

        # Check jobs are registered
        jobs = scheduler.get_jobs()
        job_ids = [j.id for j in jobs]
        assert "daily_digest" in job_ids
        assert "weekly_market_scan" in job_ids
        assert "hourly_check" in job_ids

        stop_scheduler()

    @pytest.mark.asyncio
    async def test_stop_scheduler(self):
        from api.background import start_scheduler, stop_scheduler, get_scheduler

        start_scheduler()
        stop_scheduler()
        scheduler = get_scheduler()
        assert scheduler is None

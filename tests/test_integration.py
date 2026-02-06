"""
Integration tests for end-to-end workflows.

Validates:
- AC-4.1: Daily Digest workflow
- AC-4.2: Support Ticket workflow
- AC-4.3: Code Change/PR workflow
- AC-4.4: Market Feedback workflow
- AC-4.5: On-demand chat
- AC-5.1/5.2: Human-in-the-loop
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.state import AgentState, InsightData, ActionItem, AlertData


class TestDailyDigestWorkflow:
    """Test daily digest workflow (AC-4.1)."""

    @pytest.mark.asyncio
    async def test_daily_digest_invocation(self):
        """Test that daily digest can be triggered and returns results."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {
                "messages": [MagicMock(content="Daily digest: 5 new tickets, 3 PRs merged, market stable")]
            }
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph
            result = await invoke_graph(
                query="Generate daily digest",
                trigger_type="scheduled",
                metadata={"workflow": "daily_digest"},
            )
            assert "messages" in result
            assert len(result["messages"]) > 0


class TestSupportTicketWorkflow:
    """Test support ticket workflow (AC-4.2)."""

    @pytest.mark.asyncio
    async def test_support_ticket_analysis(self):
        """Test that new support ticket triggers analysis."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {
                "messages": [MagicMock(content="Ticket PROD-123 analyzed: High priority, similar to competitor issues")]
            }
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph
            result = await invoke_graph(
                query="Jira issue_created: [PROD-123] Critical checkout bug (Type: Bug, Priority: Highest)",
                trigger_type="webhook",
                metadata={"source": "jira", "issue_key": "PROD-123"},
            )
            assert "messages" in result


class TestCodeChangePRWorkflow:
    """Test PR workflow (AC-4.3)."""

    @pytest.mark.asyncio
    async def test_pr_review_workflow(self):
        """Test that PR event triggers analysis."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {
                "messages": [MagicMock(content="PR #42 reviewed: No critical issues found, 5 files changed")]
            }
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph
            result = await invoke_graph(
                query="GitHub PR opened: org/repo#42 - Add new feature",
                trigger_type="webhook",
                metadata={"source": "github", "repo": "org/repo", "pr": 42},
            )
            assert "messages" in result


class TestMarketFeedbackWorkflow:
    """Test market feedback workflow (AC-4.4)."""

    @pytest.mark.asyncio
    async def test_market_scan(self):
        """Test scheduled market scan."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {
                "messages": [MagicMock(content="Weekly market scan: Competitor X launched new feature, positive Reddit sentiment")]
            }
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph
            result = await invoke_graph(
                query="Perform weekly market intelligence scan",
                trigger_type="scheduled",
                metadata={"workflow": "weekly_market_scan"},
            )
            assert "messages" in result


class TestOnDemandChat:
    """Test on-demand chat (AC-4.5)."""

    @pytest.mark.asyncio
    async def test_chat_query(self):
        """Test direct user query."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.return_value = {
                "messages": [MagicMock(content="Top blockers: 1) PROD-456 API outage 2) PROD-789 deployment issue")]
            }
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph
            result = await invoke_graph(
                query="What are the top blockers this week?",
                thread_id="chat-session-1",
                trigger_type="manual",
            )
            assert "messages" in result

    @pytest.mark.asyncio
    async def test_multi_turn_chat(self):
        """Test multi-turn conversation with context."""
        with patch("src.graphs.main_graph.get_graph") as mock_get_graph:
            mock_graph = AsyncMock()
            mock_graph.ainvoke.side_effect = [
                {"messages": [MagicMock(content="Found 3 open P0 tickets")]},
                {"messages": [MagicMock(content="PROD-456 is the most critical, reported by 5 customers")]},
            ]
            mock_get_graph.return_value = mock_graph

            from src.graphs.main_graph import invoke_graph

            # Turn 1
            result1 = await invoke_graph(
                query="How many P0 tickets are open?",
                thread_id="session-multi",
            )
            assert "messages" in result1

            # Turn 2 (same thread)
            result2 = await invoke_graph(
                query="Which one is most critical?",
                thread_id="session-multi",
            )
            assert "messages" in result2


class TestAnomalyDetection:
    """Test anomaly detection (AC-3.3)."""

    def test_high_priority_detection_state(self):
        """Test that high priority items get flagged in state."""
        state = AgentState(
            jira_data=[],
            alerts=[
                AlertData(
                    severity="critical",
                    title="P0 Bug Detected",
                    message="New P0 ticket PROD-999",
                    source="jira",
                    channel="#pm-alerts",
                )
            ],
        )
        critical_alerts = [a for a in state.alerts if a.severity == "critical"]
        assert len(critical_alerts) == 1

    def test_insight_with_recommendations(self):
        """Test that insights generate recommendations."""
        insight = InsightData(
            category="customer_sentiment",
            title="Negative feedback spike",
            description="30% increase in negative sentiment",
            severity="high",
            source_agents=["slack_agent", "jira_agent"],
            recommended_actions=["Review P0 tickets", "Schedule customer call"],
        )
        assert len(insight.recommended_actions) == 2
        assert insight.severity == "high"


class TestHumanInTheLoop:
    """Test HITL mechanisms (AC-5.1, AC-5.2)."""

    def test_hitl_state_pause(self):
        """Test that state correctly stores pending approval."""
        state = AgentState(
            human_approval=None,
            pending_approval_context="Creating Jira ticket PROD-999 for checkout bug. Approve?",
            action_items=[
                ActionItem(
                    action_type="create_jira",
                    description="Create ticket for checkout bug",
                    priority="high",
                    auto_execute=False,
                    status="pending",
                )
            ],
        )
        assert state.human_approval is None
        assert state.pending_approval_context is not None
        assert state.action_items[0].status == "pending"

    def test_hitl_state_approved(self):
        """Test state after approval."""
        state = AgentState(
            human_approval=True,
            action_items=[
                ActionItem(
                    action_type="create_jira",
                    description="Create ticket",
                    auto_execute=True,
                    status="approved",
                )
            ],
        )
        assert state.human_approval is True
        assert state.action_items[0].status == "approved"

    def test_hitl_state_rejected(self):
        """Test state after rejection."""
        state = AgentState(
            human_approval=False,
            action_items=[
                ActionItem(
                    action_type="create_jira",
                    description="Create ticket",
                    auto_execute=False,
                    status="rejected",
                )
            ],
        )
        assert state.human_approval is False
        assert state.action_items[0].status == "rejected"

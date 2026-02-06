"""
Tests for the AgentState schema and data models.

Validates:
- AC-1.2: StateGraph state schema
- AC-6.1: State persistence models
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.state import (
    AgentState,
    SlackMessageData,
    JiraIssueData,
    GitHubItemData,
    NotionPageData,
    WebContentData,
    InsightData,
    ActionItem,
    AlertData,
)


class TestDataModels:
    """Test individual data models."""

    def test_slack_message_data_defaults(self):
        msg = SlackMessageData()
        assert msg.message_id == ""
        assert msg.reactions == []
        assert msg.is_thread_reply is False

    def test_slack_message_data_full(self):
        msg = SlackMessageData(
            message_id="123",
            channel_id="C01",
            channel_name="product",
            user_id="U01",
            user_name="Alice",
            timestamp="2026-02-06T10:00:00",
            text="We need to prioritize the new feature",
            thread_ts="1234567890.123",
            reactions=["thumbsup", "eyes"],
            is_thread_reply=True,
        )
        assert msg.message_id == "123"
        assert msg.channel_name == "product"
        assert len(msg.reactions) == 2

    def test_jira_issue_data(self):
        issue = JiraIssueData(
            issue_key="PROD-123",
            project_key="PROD",
            summary="Critical bug in checkout",
            status="Open",
            priority="Highest",
            issue_type="Bug",
            labels=["critical", "checkout"],
        )
        assert issue.issue_key == "PROD-123"
        assert issue.priority == "Highest"
        assert "critical" in issue.labels

    def test_github_item_data_issue(self):
        item = GitHubItemData(
            item_type="issue",
            number=42,
            repository="org/repo",
            title="Feature request: dark mode",
            state="open",
            labels=["enhancement"],
            author="user1",
        )
        assert item.item_type == "issue"
        assert item.merged is False

    def test_github_item_data_pr(self):
        pr = GitHubItemData(
            item_type="pull_request",
            number=100,
            repository="org/repo",
            title="Add dark mode support",
            state="closed",
            changed_files=["src/theme.py", "src/ui.py"],
            additions=150,
            deletions=20,
            merged=True,
        )
        assert pr.item_type == "pull_request"
        assert pr.merged is True
        assert len(pr.changed_files) == 2

    def test_notion_page_data(self):
        page = NotionPageData(
            page_id="abc-123",
            title="Q1 Roadmap",
            url="https://notion.so/Q1-Roadmap",
            properties={"Status": "Active", "Priority": "High"},
        )
        assert page.title == "Q1 Roadmap"
        assert page.properties["Status"] == "Active"

    def test_web_content_data(self):
        content = WebContentData(
            source="tavily",
            url="https://example.com/article",
            title="Market Analysis 2026",
            content="The AI market is growing...",
            category="market_trend",
            sentiment="positive",
        )
        assert content.source == "tavily"
        assert content.sentiment == "positive"

    def test_insight_data(self):
        insight = InsightData(
            category="customer_sentiment",
            title="Negative feedback spike",
            description="Customer complaints increased 30% this week",
            severity="high",
            source_agents=["slack_agent", "jira_agent"],
            evidence=["PROD-123", "Slack thread #support"],
            recommended_actions=["Review P0 tickets", "Schedule customer call"],
        )
        assert insight.severity == "high"
        assert len(insight.source_agents) == 2
        assert len(insight.recommended_actions) == 2

    def test_action_item(self):
        action = ActionItem(
            action_type="create_jira",
            description="Create ticket for checkout bug",
            priority="critical",
            auto_execute=False,
            target="PROD",
            payload={"summary": "Checkout bug", "type": "Bug"},
        )
        assert action.auto_execute is False
        assert action.status == "pending"

    def test_alert_data(self):
        alert = AlertData(
            severity="critical",
            title="P0 Bug Detected",
            message="New critical bug in production",
            source="jira",
            channel="#pm-alerts",
        )
        assert alert.severity == "critical"


class TestAgentState:
    """Test the main AgentState."""

    def test_agent_state_defaults(self):
        state = AgentState()
        assert state.messages == []
        assert state.user_query is None
        assert state.trigger_type == "manual"
        assert state.slack_data == []
        assert state.jira_data == []
        assert state.github_data == []
        assert state.notion_data == []
        assert state.web_data == []
        assert state.insights == []
        assert state.action_items == []
        assert state.alerts == []
        assert state.human_approval is None
        assert state.final_output is None
        assert state.errors == []
        assert state.timestamp  # should have a default timestamp

    def test_agent_state_with_data(self):
        state = AgentState(
            user_query="What are the top blockers?",
            trigger_type="manual",
            thread_id="test-thread-1",
            current_agent="supervisor",
            slack_data=[SlackMessageData(text="Blocker: API is down")],
            jira_data=[JiraIssueData(issue_key="PROD-1", priority="Highest")],
            insights=[InsightData(category="internal_blocker", title="API outage")],
        )
        assert state.user_query == "What are the top blockers?"
        assert len(state.slack_data) == 1
        assert len(state.jira_data) == 1
        assert len(state.insights) == 1

    def test_agent_state_next_step_enum(self):
        state = AgentState(next_step="jira_agent")
        assert state.next_step == "jira_agent"

        state = AgentState(next_step="END")
        assert state.next_step == "END"

    def test_agent_state_results_dict(self):
        state = AgentState(results={"jira": [{"key": "PROD-1"}], "github": {"prs": 5}})
        assert "jira" in state.results
        assert "github" in state.results

    def test_agent_state_metadata(self):
        state = AgentState(metadata={"workflow": "daily_digest", "source": "scheduler"})
        assert state.metadata["workflow"] == "daily_digest"

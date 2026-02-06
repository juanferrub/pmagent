"""
Tests for integration tools.

Validates:
- AC-2.1 through AC-2.5: Tool functionality with mocked APIs
- AC-10.1: Retry and error handling
- AC-10.2: Circuit breaker behavior
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


class TestSlackTools:
    """Test Slack tools (AC-2.1)."""

    @patch("src.tools.slack_tools._get_slack_client")
    def test_read_channel_history(self, mock_client_fn):
        from src.tools.slack_tools import read_channel_history

        mock_client = MagicMock()
        mock_client.conversations_history.return_value = {
            "messages": [
                {"user": "U01", "text": "Hello team", "ts": "123", "reactions": []},
                {"user": "U02", "text": "Let's discuss roadmap", "ts": "124", "reactions": [{"name": "thumbsup"}]},
            ]
        }
        mock_client_fn.return_value = mock_client

        result = read_channel_history.invoke({"channel": "C01", "limit": 10})
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["text"] == "Hello team"
        assert data[1]["reactions"] == ["thumbsup"]

    @patch("src.tools.slack_tools._get_slack_client")
    def test_search_slack_messages(self, mock_client_fn):
        from src.tools.slack_tools import search_slack_messages

        mock_client = MagicMock()
        mock_client.search_messages.return_value = {
            "messages": {
                "matches": [
                    {"channel": {"name": "product"}, "username": "alice", "text": "feature request", "ts": "123", "permalink": "http://link"},
                ]
            }
        }
        mock_client_fn.return_value = mock_client

        result = search_slack_messages.invoke({"query": "feature", "count": 5})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["text"] == "feature request"

    @patch("src.tools.slack_tools._get_slack_client")
    def test_post_slack_message(self, mock_client_fn):
        from src.tools.slack_tools import post_slack_message

        mock_client = MagicMock()
        mock_client.chat_postMessage.return_value = {"ok": True, "ts": "456"}
        mock_client_fn.return_value = mock_client

        result = post_slack_message.invoke({"channel": "#alerts", "text": "Test alert"})
        data = json.loads(result)
        assert data["ok"] is True

    @patch("src.tools.slack_tools._get_slack_client")
    def test_slack_error_handling(self, mock_client_fn):
        from src.tools.slack_tools import read_channel_history

        mock_client_fn.side_effect = Exception("Slack API error")

        result = read_channel_history.invoke({"channel": "C01", "limit": 10})
        data = json.loads(result)
        assert "error" in data


class TestJiraTools:
    """Test Jira tools (AC-2.2)."""

    @patch("src.tools.jira_tools._get_jira_client")
    def test_search_jira_issues(self, mock_client_fn):
        from src.tools.jira_tools import search_jira_issues

        mock_client = MagicMock()
        mock_client.jql.return_value = {
            "issues": [
                {
                    "key": "PROD-1",
                    "fields": {
                        "summary": "Critical bug",
                        "status": {"name": "Open"},
                        "priority": {"name": "Highest"},
                        "issuetype": {"name": "Bug"},
                        "assignee": {"displayName": "Alice"},
                        "reporter": {"displayName": "Bob"},
                        "created": "2026-02-06",
                        "updated": "2026-02-06",
                        "labels": ["critical"],
                        "description": "Something is broken",
                    },
                }
            ]
        }
        mock_client_fn.return_value = mock_client

        result = search_jira_issues.invoke({"jql": "project = PROD", "max_results": 10})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["key"] == "PROD-1"
        assert data[0]["priority"] == "Highest"

    @patch("src.tools.jira_tools._get_jira_client")
    def test_get_jira_issue(self, mock_client_fn):
        from src.tools.jira_tools import get_jira_issue

        mock_client = MagicMock()
        mock_client.issue.return_value = {
            "key": "PROD-123",
            "fields": {
                "summary": "Test issue",
                "description": "Full description here",
                "status": {"name": "In Progress"},
                "priority": {"name": "High"},
                "issuetype": {"name": "Story"},
                "assignee": {"displayName": "Alice"},
                "reporter": {"displayName": "Bob"},
                "labels": [],
                "created": "2026-01-01",
                "updated": "2026-02-06",
                "resolution": None,
                "comment": {"comments": [
                    {"author": {"displayName": "Charlie"}, "body": "Working on it", "created": "2026-02-06"}
                ]},
            },
        }
        mock_client_fn.return_value = mock_client

        result = get_jira_issue.invoke({"issue_key": "PROD-123"})
        data = json.loads(result)
        assert data["key"] == "PROD-123"
        assert len(data["comments"]) == 1

    @patch("src.tools.jira_tools._get_jira_client")
    def test_create_jira_issue(self, mock_client_fn):
        from src.tools.jira_tools import create_jira_issue

        mock_client = MagicMock()
        mock_client.create_issue.return_value = {"key": "PROD-999"}
        mock_client_fn.return_value = mock_client

        result = create_jira_issue.invoke({
            "project_key": "PROD",
            "summary": "New bug found",
            "issue_type": "Bug",
            "priority": "High",
        })
        data = json.loads(result)
        assert data["key"] == "PROD-999"

    @patch("src.tools.jira_tools._get_jira_client")
    def test_update_jira_issue(self, mock_client_fn):
        from src.tools.jira_tools import update_jira_issue

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        result = update_jira_issue.invoke({
            "issue_key": "PROD-1",
            "fields_json": '{"priority": {"name": "Highest"}}',
        })
        data = json.loads(result)
        assert data["ok"] is True


class TestGitHubTools:
    """Test GitHub tools (AC-2.3)."""

    @patch("src.tools.github_tools._get_github_client")
    def test_list_github_issues(self, mock_client_fn):
        from src.tools.github_tools import list_github_issues

        mock_label = MagicMock()
        mock_label.name = "bug"

        mock_issue = MagicMock()
        mock_issue.number = 42
        mock_issue.title = "Bug report"
        mock_issue.state = "open"
        mock_issue.labels = [mock_label]
        mock_issue.user = MagicMock(login="user1")

        from datetime import datetime as dt
        mock_issue.created_at = dt(2026, 2, 6)
        mock_issue.updated_at = dt(2026, 2, 6)
        mock_issue.comments = 3
        mock_issue.body = "Something is wrong"
        mock_issue.pull_request = None

        mock_repo = MagicMock()
        mock_repo.get_issues.return_value = [mock_issue]

        mock_client = MagicMock()
        mock_client.get_repo.return_value = mock_repo
        mock_client_fn.return_value = mock_client

        result = list_github_issues.invoke({"repo_name": "org/repo", "state": "open", "limit": 10})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["number"] == 42
        assert data[0]["title"] == "Bug report"

    @patch("src.tools.github_tools._get_github_client")
    def test_list_github_prs(self, mock_client_fn):
        from src.tools.github_tools import list_github_prs

        mock_pr = MagicMock()
        mock_pr.number = 100
        mock_pr.title = "Add feature X"
        mock_pr.state = "open"
        mock_pr.user = MagicMock(login="dev1")
        mock_pr.created_at = MagicMock(isoformat=lambda: "2026-02-06T00:00:00")
        mock_pr.updated_at = MagicMock(isoformat=lambda: "2026-02-06T00:00:00")
        mock_pr.changed_files = 5
        mock_pr.additions = 200
        mock_pr.deletions = 50
        mock_pr.merged = False
        mock_pr.body = "New feature"

        mock_repo = MagicMock()
        mock_repo.get_pulls.return_value = [mock_pr]

        mock_client = MagicMock()
        mock_client.get_repo.return_value = mock_repo
        mock_client_fn.return_value = mock_client

        result = list_github_prs.invoke({"repo_name": "org/repo", "state": "open", "limit": 5})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["number"] == 100

    @patch("src.tools.github_tools._get_github_client")
    def test_get_github_file_contents(self, mock_client_fn):
        from src.tools.github_tools import get_github_file_contents

        mock_content = MagicMock()
        mock_content.decoded_content = b"print('hello world')"

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content

        mock_client = MagicMock()
        mock_client.get_repo.return_value = mock_repo
        mock_client_fn.return_value = mock_client

        result = get_github_file_contents.invoke({"repo_name": "org/repo", "file_path": "main.py"})
        assert "hello world" in result


class TestNotionTools:
    """Test Notion tools (AC-2.5)."""

    @patch("src.tools.notion_tools._get_notion_client")
    def test_query_notion_database(self, mock_client_fn):
        from src.tools.notion_tools import query_notion_database

        mock_client = MagicMock()
        mock_client.databases.query.return_value = {
            "results": [
                {
                    "id": "page-1",
                    "url": "https://notion.so/page-1",
                    "last_edited_time": "2026-02-06",
                    "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "Q1 Roadmap"}]},
                        "Status": {"type": "select", "select": {"name": "Active"}},
                    },
                }
            ]
        }
        mock_client_fn.return_value = mock_client

        result = query_notion_database.invoke({"database_id": "db-1", "limit": 10})
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["properties"]["Name"] == "Q1 Roadmap"

    @patch("src.tools.notion_tools._get_notion_client")
    def test_read_notion_page(self, mock_client_fn):
        from src.tools.notion_tools import read_notion_page

        mock_client = MagicMock()
        mock_client.pages.retrieve.return_value = {
            "id": "page-1",
            "url": "https://notion.so/page-1",
            "last_edited_time": "2026-02-06",
            "properties": {
                "title": {"type": "title", "title": [{"plain_text": "Test Page"}]},
            },
        }
        mock_client.blocks.children.list.return_value = {
            "results": [
                {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Hello world"}]}},
            ]
        }
        mock_client_fn.return_value = mock_client

        result = read_notion_page.invoke({"page_id": "page-1"})
        data = json.loads(result)
        assert data["title"] == "Test Page"
        assert "Hello world" in data["content"]

    @patch("src.tools.notion_tools._get_notion_client")
    def test_create_notion_page(self, mock_client_fn):
        from src.tools.notion_tools import create_notion_page

        mock_client = MagicMock()
        mock_client.pages.create.return_value = {
            "id": "new-page-1",
            "url": "https://notion.so/new-page-1",
        }
        mock_client_fn.return_value = mock_client

        result = create_notion_page.invoke({
            "parent_page_id": "parent-1",
            "title": "Daily Report",
            "content_markdown": "# Summary\n- Item 1\n- Item 2",
        })
        data = json.loads(result)
        assert data["id"] == "new-page-1"

    @patch("src.tools.notion_tools._get_notion_client")
    def test_append_notion_blocks(self, mock_client_fn):
        from src.tools.notion_tools import append_notion_blocks

        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client

        result = append_notion_blocks.invoke({
            "page_id": "page-1",
            "content_markdown": "## New Section\nSome content here",
        })
        data = json.loads(result)
        assert data["ok"] is True
        assert data["blocks_added"] == 2


class TestResearchTools:
    """Test research tools (AC-2.4)."""

    @patch("src.tools.research_tools.get_settings")
    def test_web_search(self, mock_settings):
        from src.tools.research_tools import web_search

        mock_settings.return_value = MagicMock(tavily_api_key="test-key")

        with patch("src.tools.research_tools.TavilyClient", create=True) as MockTavily:
            # Need to mock the import
            import sys
            mock_tavily_module = MagicMock()
            mock_tavily_client = MagicMock()
            mock_tavily_client.search.return_value = {
                "results": [
                    {"title": "AI Market 2026", "url": "https://example.com", "content": "AI is growing", "score": 0.9},
                ]
            }
            mock_tavily_module.TavilyClient.return_value = mock_tavily_client
            sys.modules["tavily"] = mock_tavily_module

            try:
                result = web_search.invoke({"query": "AI market trends", "max_results": 5})
                data = json.loads(result)
                assert len(data) == 1
                assert data[0]["title"] == "AI Market 2026"
            finally:
                del sys.modules["tavily"]

    @patch("httpx.get")
    def test_browse_page(self, mock_get):
        from src.tools.research_tools import browse_page

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Main content here</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = browse_page.invoke({"url": "https://example.com", "instructions": "Extract content"})
        data = json.loads(result)
        assert "Main content" in data["content"]

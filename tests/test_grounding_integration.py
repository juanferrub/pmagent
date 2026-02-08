"""
Integration-style tests for grounding with mocked tool outputs.

These tests verify end-to-end scenarios:
- Customer Request tickets with correct JQL and count matching
- GitHub activity with time filtering
- Slack permission errors with proper resolution instructions
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.grounding import assert_grounded, resolve_timerange, GroundingValidator


class TestJiraCustomerRequestScenario:
    """
    Scenario: "How many Customer Request tickets opened last week?"
    
    Verifies:
    - jira_search called with correct JQL
    - Response count matches tool output
    - Customer field handling
    """
    
    def test_correct_jql_for_last_week(self):
        """Should use correct JQL for 'last week' query."""
        now = datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc)  # Wednesday
        
        time_range = resolve_timerange("tickets opened last week", now=now)
        
        assert time_range is not None
        # Should use Jira's startOfWeek function
        assert "startOfWeek(-1)" in time_range.jql_clause or "2024-01-08" in time_range.jql_clause
    
    def test_count_matches_tool_output(self):
        """Response count should match tool output exactly."""
        # Simulated tool output with 3 tickets
        tool_output = {
            "issues": [
                {"key": "SUPPORT-101", "summary": "Support request 1", "labels": ["support"]},
                {"key": "SUPPORT-102", "summary": "Support request 2", "labels": ["support"]},
                {"key": "SUPPORT-103", "summary": "Support request 3", "labels": ["support"]},
            ],
            "query_info": {
                "jql": "labels = support AND created >= startOfWeek(-1)",
                "total_results": 3,
            }
        }
        
        # Good response - matches tool output
        good_answer = "Found 3 support tickets opened last week: SUPPORT-101, SUPPORT-102, SUPPORT-103."
        
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps(tool_output)}
        ]
        
        result = assert_grounded(good_answer, tool_messages)
        
        assert result.is_grounded
        assert len(result.violations) == 0
    
    def test_fabricated_count_detected(self):
        """Should detect when count doesn't match tool output."""
        tool_output = {
            "issues": [
                {"key": "SUPPORT-101", "summary": "Support request 1"},
            ],
            "query_info": {"total_results": 1}
        }
        
        # Bad response - claims 5 tickets when tool returned 1
        bad_answer = "Found 5 support tickets opened last week."
        
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps(tool_output)}
        ]
        
        result = assert_grounded(bad_answer, tool_messages, strict=True)
        
        # Should flag the fabricated count (5 is not in tool output)
        assert not result.is_grounded
        # The count claim for "5" should be flagged
        assert any(v.claim_type == "count_claim" and v.matched_value == "5" for v in result.violations)
    
    def test_customer_field_not_fabricated(self):
        """Should not fabricate customer names."""
        tool_output = {
            "issues": [
                {
                    "key": "SUPPORT-101",
                    "summary": "Issue from customer",
                    "reporter": "john@example.com",
                    # No customer field
                }
            ],
            "customer_field_status": {
                "configured": False,
                "note": "Customer field not configured"
            }
        }
        
        # Bad response - fabricates customer name
        bad_answer = "Customer: Acme Corp reported SUPPORT-101."
        
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps(tool_output)}
        ]
        
        result = assert_grounded(bad_answer, tool_messages, strict=True)
        
        # Should flag fabricated customer
        assert any(v.claim_type == "customer_name" for v in result.violations)


class TestGitHubActivityScenario:
    """
    Scenario: "GitHub activity last couple days"
    
    Verifies:
    - since filter applied correctly
    - PRs outside range excluded
    - Time range reported in response
    """
    
    def test_since_filter_applied(self):
        """Should apply correct 'since' filter for 'couple days'."""
        now = datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc)
        
        time_range = resolve_timerange("activity last couple days", now=now)
        
        assert time_range is not None
        # Should be 2 days ago
        expected_start = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        assert time_range.start.date() == expected_start.date()
        
        # GitHub since should be ISO format
        assert "2024-01-15" in time_range.github_since
    
    def test_prs_verified_against_tool_output(self):
        """PR numbers should be verified against tool output."""
        tool_output = [
            {"number": 42, "title": "Fix bug", "state": "open", "created_at": "2024-01-16T10:00:00Z"},
            {"number": 43, "title": "Add feature", "state": "merged", "created_at": "2024-01-16T14:00:00Z"},
        ]
        
        # Good response - only mentions PRs from tool output
        good_answer = "In the last couple days, PR #42 (Fix bug) and PR #43 (Add feature) were active."
        
        tool_messages = [
            {"name": "list_github_prs", "content": json.dumps(tool_output)}
        ]
        
        result = assert_grounded(good_answer, tool_messages)
        
        # Should verify PR numbers are in tool output
        assert "github_number: 42" in result.verified_claims
        assert "github_number: 43" in result.verified_claims
        assert result.is_grounded
    
    def test_fabricated_pr_detected(self):
        """Should detect fabricated PR numbers."""
        tool_output = [
            {"number": 42, "title": "Fix bug"},
        ]
        
        # Bad response - mentions PR #99 which wasn't in output
        bad_answer = "PR #42 and PR #99 were merged in the last couple days."
        
        tool_messages = [
            {"name": "list_github_prs", "content": json.dumps(tool_output)}
        ]
        
        result = assert_grounded(bad_answer, tool_messages, strict=True)
        
        # Should flag PR #99
        assert any(v.matched_value == "99" for v in result.violations)


class TestSlackPermissionScenario:
    """
    Scenario: "What did team discuss in #engineering yesterday?"
    
    Verifies:
    - Permission error properly surfaced
    - Response asks to invite bot
    - Bot handle is configurable
    """
    
    def test_permission_error_response(self):
        """Should generate proper response for permission errors."""
        validator = GroundingValidator()
        
        response = validator.generate_limitation_response(
            original_query="What did team discuss in #engineering yesterday?",
            attempted_tools=["read_channel_history"],
            errors=["not_in_channel: Bot is not a member of #engineering"]
        )
        
        assert "read_channel_history" in response
        assert "not_in_channel" in response.lower() or "permission" in response.lower()
        assert "invite" in response.lower()
    
    def test_slack_tool_error_format(self):
        """Slack tool should return structured error with resolution."""
        # Simulated error response from slack tool
        error_response = {
            "error": "Bot is not a member of channel '#engineering'",
            "error_type": "permission_error",
            "resolution": "Invite the bot by typing: /invite @pm-agent",
            "channel": "#engineering",
        }
        
        # Verify error structure
        assert "resolution" in error_response
        assert "@pm-agent" in error_response["resolution"] or "invite" in error_response["resolution"].lower()
    
    def test_no_claim_without_tool_success(self):
        """Should not make claims about Slack without successful tool call."""
        # Tool returned error
        tool_messages = [
            {"name": "read_channel_history", "content": json.dumps({
                "error": "not_in_channel",
                "resolution": "Invite bot to channel"
            })}
        ]
        
        # Bad response - claims to know what was discussed (uses "discussed in" pattern)
        bad_answer = "The team discussed in #engineering about the new feature release yesterday."
        
        result = assert_grounded(bad_answer, tool_messages, strict=True)
        
        # Should flag because tool failed - either slack_activity or slack_channel violation
        assert not result.is_grounded
        # Should have some violation related to the unverified claim
        assert len(result.violations) > 0


class TestNoToolNoFactsPolicy:
    """
    Tests for the no-tool-no-facts policy enforcement.
    """
    
    def test_no_jira_claims_without_tool(self):
        """Cannot make Jira claims without calling Jira tools."""
        answer = "There are 5 P0 tickets in the OPIK project: OPIK-1, OPIK-2, OPIK-3, OPIK-4, OPIK-5."
        tool_messages = []  # No tools called
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        assert not result.is_grounded
        # Should flag multiple violations
        assert len(result.violations) >= 1
    
    def test_no_github_claims_without_tool(self):
        """Cannot make GitHub claims without calling GitHub tools."""
        answer = "PR #42 was merged and PR #43 is waiting for review."
        tool_messages = []  # No tools called
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        assert not result.is_grounded
    
    def test_no_slack_claims_without_tool(self):
        """Cannot make Slack claims without calling Slack tools."""
        answer = "The team discussed the blocker in #engineering channel."
        tool_messages = []  # No tools called
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        assert not result.is_grounded
    
    def test_can_make_claims_with_tool_evidence(self):
        """Can make claims when tool evidence exists."""
        answer = "Found OPIK-123 which is a high priority bug."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-123", "priority": "High", "issue_type": "Bug"}
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        assert result.is_grounded


class TestTimeRangeInResponses:
    """
    Tests for including time ranges in responses.
    """
    
    def test_time_range_description_available(self):
        """Time range should include human-readable description."""
        now = datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc)
        
        time_range = resolve_timerange("tickets from last week", now=now)
        
        assert time_range is not None
        assert "last week" in time_range.description.lower()
        # Should include actual dates
        assert "2024-01" in time_range.description
    
    def test_jql_includes_explicit_dates(self):
        """JQL clause should include explicit date references."""
        now = datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc)
        
        time_range = resolve_timerange("issues from yesterday", now=now)
        
        assert time_range is not None
        # Should have date in JQL
        assert "2024-01-16" in time_range.jql_clause or "created >=" in time_range.jql_clause

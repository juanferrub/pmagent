"""
Tests for grounding utilities.

Tests:
- assert_grounded: Validates claims are supported by tool outputs
- resolve_timerange: Converts natural language to timestamps
- GroundingValidator: Runtime validation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta

import pytest

from src.grounding import (
    assert_grounded,
    resolve_timerange,
    GroundingValidator,
    GroundingResult,
    TimeRange,
)


class TestAssertGrounded:
    """Tests for the assert_grounded function."""
    
    def test_grounded_jira_keys(self):
        """Jira keys mentioned in answer should be verified against tool output."""
        answer = "Found ticket OPIK-123 which is a high priority bug."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-123", "summary": "Bug in tracing", "priority": "High"}
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        assert result.is_grounded
        assert len(result.violations) == 0
        assert "jira_key: OPIK-123" in result.verified_claims
    
    def test_ungrounded_jira_keys(self):
        """Jira keys not in tool output should be flagged as violations."""
        answer = "Found ticket OPIK-999 which needs attention."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-123", "summary": "Different ticket"}
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        assert not result.is_grounded
        assert len(result.violations) == 1
        assert result.violations[0].claim_type == "jira_key"
        assert result.violations[0].matched_value == "OPIK-999"
    
    def test_grounded_github_numbers(self):
        """GitHub PR/issue numbers should be verified."""
        answer = "PR #42 was merged successfully."
        tool_messages = [
            {"name": "list_github_prs", "content": json.dumps([
                {"number": 42, "title": "Fix bug", "state": "merged"}
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        assert result.is_grounded
        assert "github_number: 42" in result.verified_claims
    
    def test_ungrounded_github_numbers(self):
        """GitHub numbers not in tool output should be flagged."""
        answer = "Check PR #999 for the fix."
        tool_messages = [
            {"name": "list_github_prs", "content": json.dumps([
                {"number": 42, "title": "Different PR"}
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        assert not result.is_grounded
        assert any(v.claim_type == "github_number" for v in result.violations)
    
    def test_count_claims_verified(self):
        """Count claims should match tool output counts."""
        answer = "There are 5 tickets in the backlog."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps({
                "issues": [{"key": f"OPIK-{i}"} for i in range(5)],
                "query_info": {"total_results": 5}
            })}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        # Count should be verified since 5 appears in tool output
        assert result.is_grounded or len(result.violations) == 0
    
    def test_fabricated_count_flagged(self):
        """Fabricated counts should be flagged."""
        answer = "There are 100 tickets waiting for review."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-1"}, {"key": "OPIK-2"}  # Only 2 tickets
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        # 100 doesn't appear in tool output
        assert any(v.claim_type == "count_claim" for v in result.violations)
    
    def test_no_issues_claim_requires_tool_call(self):
        """'No P0 issues' claims require tool evidence."""
        answer = "There are no P0 issues currently."
        tool_messages = []  # No tool calls
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        # Should flag because no tool was called
        assert any(v.claim_type == "no_issues_claim" for v in result.violations)
    
    def test_no_issues_claim_with_tool_evidence(self):
        """'No P0 issues' is valid when tool returned empty results."""
        answer = "There are no P0 issues currently."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([])}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        # Should be grounded because tool was called (even if empty)
        assert "jira" in result.tool_sources_used
    
    def test_slack_channel_verification(self):
        """Slack channel references should be verified."""
        answer = "The discussion in #engineering mentioned a blocker."
        tool_messages = [
            {"name": "read_channel_history", "content": json.dumps({
                "messages": [{"text": "We have a blocker"}],
                "query_info": {"channel": "engineering"}
            })}
        ]
        
        result = assert_grounded(answer, tool_messages)
        
        assert result.is_grounded
    
    def test_fabricated_customer_name_flagged(self):
        """Fabricated customer names should be flagged."""
        answer = "Customer: Acme Corp reported this issue."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-1", "reporter": "john@example.com"}
                # No customer field
            ])}
        ]
        
        result = assert_grounded(answer, tool_messages, strict=True)
        
        # Customer name not in tool output
        assert any(v.claim_type == "customer_name" for v in result.violations)
    
    def test_empty_answer_is_grounded(self):
        """Empty or very short answers should pass."""
        result = assert_grounded("OK", [])
        assert result.is_grounded
    
    def test_non_strict_mode_allows_few_violations(self):
        """Non-strict mode allows a few violations."""
        answer = "Found OPIK-999 and OPIK-888."  # Two fabricated keys
        tool_messages = []
        
        result = assert_grounded(answer, tool_messages, strict=False)
        
        # Non-strict allows up to 2 violations
        assert len(result.violations) == 2
        assert result.is_grounded  # Still passes in non-strict


class TestResolveTimerange:
    """Tests for the resolve_timerange function."""
    
    def test_today(self):
        """'today' should resolve to start of current day."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("What happened today?", now=now)
        
        assert result is not None
        assert result.start.date() == now.date()
        assert "today" in result.description.lower()
        assert "2024-01-15" in result.jql_clause
    
    def test_yesterday(self):
        """'yesterday' should resolve to previous day."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Show me yesterday's tickets", now=now)
        
        assert result is not None
        assert result.start.date() == datetime(2024, 1, 14).date()
        assert "yesterday" in result.description.lower()
    
    def test_last_week(self):
        """'last week' should resolve to previous Monday-Sunday."""
        # Wednesday Jan 17, 2024
        now = datetime(2024, 1, 17, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Issues from last week", now=now)
        
        assert result is not None
        # Last week should be Jan 8-14 (Mon-Sun)
        assert result.start.weekday() == 0  # Monday
        assert "startOfWeek(-1)" in result.jql_clause
    
    def test_this_week(self):
        """'this week' should resolve to current week starting Monday."""
        # Wednesday Jan 17, 2024
        now = datetime(2024, 1, 17, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("This week's activity", now=now)
        
        assert result is not None
        # Should start on Monday Jan 15
        assert result.start.weekday() == 0
        assert result.start.day == 15
    
    def test_last_n_days(self):
        """'last N days' should resolve correctly."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Activity in the last 7 days", now=now)
        
        assert result is not None
        assert result.start.date() == datetime(2024, 1, 8).date()
        assert "last 7 days" in result.description
    
    def test_couple_days(self):
        """'couple days' should resolve to 2 days."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("What happened in the last couple days?", now=now)
        
        assert result is not None
        assert result.start.date() == datetime(2024, 1, 13).date()
    
    def test_this_month(self):
        """'this month' should resolve to start of current month."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("This month's tickets", now=now)
        
        assert result is not None
        assert result.start.day == 1
        assert result.start.month == 1
    
    def test_last_month(self):
        """'last month' should resolve to previous full month."""
        now = datetime(2024, 2, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Last month's summary", now=now)
        
        assert result is not None
        assert result.start.month == 1
        assert result.end.month == 1
    
    def test_current_sprint(self):
        """'current sprint' should resolve to approximately 2 weeks."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Current sprint status", now=now)
        
        assert result is not None
        assert "openSprints()" in result.jql_clause
    
    def test_no_time_reference(self):
        """Queries without time references should return None."""
        result = resolve_timerange("Show me all P0 bugs")
        
        assert result is None
    
    def test_jql_clause_format(self):
        """JQL clause should be properly formatted."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Yesterday's issues", now=now)
        
        assert result is not None
        assert "created >=" in result.jql_clause
        assert "2024-01-14" in result.jql_clause
    
    def test_github_since_format(self):
        """GitHub since should be ISO format."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("PRs from today", now=now)
        
        assert result is not None
        assert "2024-01-15" in result.github_since
        assert "T" in result.github_since  # ISO format
    
    def test_slack_oldest_format(self):
        """Slack oldest should be Unix timestamp."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange("Messages from today", now=now)
        
        assert result is not None
        # Should be a numeric string (Unix timestamp)
        assert result.slack_oldest.replace(".", "").isdigit()
    
    def test_timezone_handling(self):
        """Should handle different timezones."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        
        result = resolve_timerange(
            "Today's activity",
            now=now,
            user_tz="America/New_York"
        )
        
        assert result is not None
        # Result should be in user's timezone
        assert result.start is not None


class TestGroundingValidator:
    """Tests for the GroundingValidator class."""
    
    def test_validate_grounded_answer(self):
        """Grounded answers should pass validation."""
        validator = GroundingValidator(strict=True)
        
        answer = "Found ticket OPIK-123 with high priority."
        tool_messages = [
            {"name": "search_jira_issues", "content": json.dumps([
                {"key": "OPIK-123", "priority": "High"}
            ])}
        ]
        
        fixed, result = validator.validate_and_fix(answer, tool_messages)
        
        assert result.is_grounded
        assert fixed == answer  # No changes needed
    
    def test_fix_ungrounded_answer(self):
        """Ungrounded answers should be fixed with disclaimers."""
        validator = GroundingValidator(strict=True)
        
        answer = "Found ticket OPIK-999 which is critical."
        tool_messages = []  # No tool calls
        
        fixed, result = validator.validate_and_fix(answer, tool_messages)
        
        assert not result.is_grounded
        assert "Note:" in fixed or "Unverified" in fixed
        assert "No data sources" in fixed or "could not be verified" in fixed.lower()
    
    def test_generate_limitation_response(self):
        """Should generate helpful limitation responses."""
        validator = GroundingValidator()
        
        response = validator.generate_limitation_response(
            original_query="What are the P0 issues?",
            attempted_tools=["search_jira_issues"],
            errors=["permission denied: not_in_channel"]
        )
        
        assert "attempted" in response.lower()
        assert "search_jira_issues" in response
        assert "permission" in response.lower()
        assert "what you can do" in response.lower()
    
    def test_limitation_response_with_not_found(self):
        """Should suggest verification for not found errors."""
        validator = GroundingValidator()
        
        response = validator.generate_limitation_response(
            original_query="Show me OPIK project",
            attempted_tools=["search_jira_issues"],
            errors=["Project not found: 404"]
        )
        
        assert "verify" in response.lower() or "correct" in response.lower()
    
    def test_limitation_response_with_timeout(self):
        """Should suggest retry for timeout errors."""
        validator = GroundingValidator()
        
        response = validator.generate_limitation_response(
            original_query="GitHub activity",
            attempted_tools=["list_github_prs"],
            errors=["Connection timeout"]
        )
        
        assert "try again" in response.lower() or "unavailable" in response.lower()


class TestTimeRangeToDict:
    """Tests for TimeRange serialization."""
    
    def test_to_dict(self):
        """TimeRange should serialize to dict correctly."""
        now = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = resolve_timerange("Today's issues", now=now)
        
        assert result is not None
        
        d = result.to_dict()
        
        assert "start" in d
        assert "end" in d
        assert "description" in d
        assert "jql_clause" in d
        assert "github_since" in d
        assert "slack_oldest" in d
        
        # Should be ISO format strings
        assert "2024-01-15" in d["start"]


class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_tool_messages(self):
        """Should handle empty tool messages list."""
        result = assert_grounded("Some answer", [])
        assert isinstance(result, GroundingResult)
    
    def test_malformed_tool_content(self):
        """Should handle malformed JSON in tool content."""
        answer = "Found OPIK-123"
        tool_messages = [
            {"name": "search_jira_issues", "content": "not valid json"}
        ]
        
        result = assert_grounded(answer, tool_messages)
        # Should not crash, just not find evidence
        assert isinstance(result, GroundingResult)
    
    def test_none_tool_content(self):
        """Should handle None content in tool messages."""
        answer = "Some answer"
        tool_messages = [
            {"name": "search_jira_issues", "content": None}
        ]
        
        result = assert_grounded(answer, tool_messages)
        assert isinstance(result, GroundingResult)
    
    def test_mixed_message_formats(self):
        """Should handle different message formats."""
        answer = "Found OPIK-123"
        
        # Mix of dict and object-like messages
        class MockMessage:
            def __init__(self):
                self.content = json.dumps([{"key": "OPIK-123"}])
                self.name = "search_jira_issues"
        
        tool_messages = [
            {"name": "tool1", "content": "{}"},
            MockMessage(),
        ]
        
        result = assert_grounded(answer, tool_messages)
        assert isinstance(result, GroundingResult)

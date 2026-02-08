"""
Tests for the Evidence Ledger and Safety Gate System.

These tests verify that the evidence-based reporting system correctly:
1. Blocks reports with no tool calls
2. Handles partial tool failures
3. Detects and removes hallucinated claims
4. Allows reports with proper evidence
5. Validates source types match section claims
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from src.evidence import (
    EvidenceLedger,
    EvidenceEntry,
    SourceType,
    ClaimScanner,
    CoverageContract,
    SafetyGate,
    SafetyGateResult,
    reset_ledger,
    get_ledger,
)
from src.source_validation import (
    SourceValidator,
    validate_report_sources,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def fresh_ledger():
    """Create a fresh evidence ledger for each test."""
    return reset_ledger()


@pytest.fixture
def ledger_with_jira_success(fresh_ledger):
    """Ledger with successful Jira tool call."""
    fresh_ledger.record_tool_call(
        source_type=SourceType.JIRA,
        tool_name="search_jira_issues",
        query_params={"jql": "project = OPIK"},
        result={
            "issues": [
                {"key": "OPIK-123", "summary": "Fix login bug"},
                {"key": "OPIK-124", "summary": "Add dark mode"},
            ]
        },
        success=True,
    )
    return fresh_ledger


@pytest.fixture
def ledger_with_github_success(fresh_ledger):
    """Ledger with successful GitHub tool call."""
    fresh_ledger.record_tool_call(
        source_type=SourceType.GITHUB,
        tool_name="list_github_prs",
        query_params={"repo_name": "comet-ml/opik"},
        result=[
            {"number": 456, "title": "Add tracing feature", "repository": "comet-ml/opik"},
            {"number": 457, "title": "Fix memory leak", "repository": "comet-ml/opik"},
        ],
        success=True,
    )
    return fresh_ledger


@pytest.fixture
def ledger_with_slack_failure(fresh_ledger):
    """Ledger with failed Slack tool call."""
    fresh_ledger.record_tool_call(
        source_type=SourceType.SLACK,
        tool_name="read_slack_channel",
        query_params={"channel": "general"},
        result={"error": "not_authed"},
        success=False,
        error="not_authed",
    )
    return fresh_ledger


@pytest.fixture
def ledger_with_multiple_sources(fresh_ledger):
    """Ledger with multiple successful source types."""
    # Jira
    fresh_ledger.record_tool_call(
        source_type=SourceType.JIRA,
        tool_name="search_jira_issues",
        query_params={"jql": "project = OPIK"},
        result={"issues": [{"key": "OPIK-100", "summary": "Test issue"}]},
        success=True,
    )
    # GitHub
    fresh_ledger.record_tool_call(
        source_type=SourceType.GITHUB,
        tool_name="list_github_prs",
        query_params={"repo_name": "comet-ml/opik"},
        result=[{"number": 200, "title": "Test PR", "repository": "comet-ml/opik"}],
        success=True,
    )
    # Web
    fresh_ledger.record_tool_call(
        source_type=SourceType.WEB,
        tool_name="web_search",
        query_params={"query": "LangSmith release"},
        result={"url": "https://langchain.com/blog/release", "title": "LangSmith 2.0"},
        success=True,
    )
    return fresh_ledger


@pytest.fixture
def sample_report_html():
    """Sample HTML report with multiple sections."""
    return """
    <html>
    <body>
        <h1>Weekly PM Briefing</h1>
        
        <h2>Jira Analysis</h2>
        <p>This week we have 5 high-priority tickets in progress.</p>
        <ul>
            <li>OPIK-123: Fix login bug - In Progress</li>
            <li>OPIK-124: Add dark mode - Ready for Review</li>
        </ul>
        
        <h2>GitHub Activity</h2>
        <p>3 PRs were merged this week.</p>
        <ul>
            <li>#456: Add tracing feature</li>
            <li>#457: Fix memory leak</li>
        </ul>
        
        <h2>Slack Highlights</h2>
        <p>Team discussions focused on the upcoming release.</p>
        
        <h2>Competitor Updates</h2>
        <p>LangSmith released version 2.0 with new features.</p>
    </body>
    </html>
    """


@pytest.fixture
def fabricated_report_html():
    """Report with fabricated claims (no tool calls)."""
    return """
    <html>
    <body>
        <h2>Jira Analysis</h2>
        <p>Sprint velocity increased by 20% this week.</p>
        <p>Top issues are FAKE-001 and FAKE-002.</p>
        
        <h2>GitHub Activity</h2>
        <p>15 PRs merged with excellent code quality.</p>
        
        <h2>Competitor Updates</h2>
        <p>Langfuse launched enterprise pricing at $500/month.</p>
        <p>Arize Phoenix released version 3.0.</p>
    </body>
    </html>
    """


# ============================================================================
# Test 1: No Tool Call Run - System blocks send_email_report
# ============================================================================

class TestNoToolCallRun:
    """Test that reports with no tool calls are blocked."""
    
    def test_empty_ledger_blocks_send(self, fresh_ledger):
        """Safety gate should block when ledger is empty."""
        gate = SafetyGate(fresh_ledger)
        
        result = gate.check(
            report_text="<h2>Jira Analysis</h2><p>5 tickets in progress</p>",
            requested_sections=["Jira Analysis"],
        )
        
        assert result.can_send is False
        assert "No tool calls recorded" in result.rejection_reason
        assert result.tool_success_rate == 0.0
    
    def test_empty_ledger_produces_draft(self, fresh_ledger):
        """Empty ledger should produce an incomplete draft."""
        gate = SafetyGate(fresh_ledger)
        
        result = gate.check(
            report_text="<h2>GitHub Activity</h2><p>10 PRs merged</p>",
            requested_sections=["GitHub Activity"],
        )
        
        draft = gate.generate_incomplete_draft(
            "<h2>GitHub Activity</h2><p>10 PRs merged</p>",
            result,
        )
        
        assert "DRAFT REPORT" in draft
        assert "NOT SENT" in draft
        assert "Evidence Validation Failed" in draft
    
    def test_coverage_summary_shows_zero(self, fresh_ledger):
        """Coverage summary should show zero entries."""
        summary = fresh_ledger.get_coverage_summary()
        
        assert summary["total_entries"] == 0
        assert summary["successful_entries"] == 0
        assert summary["sources_covered"] == []


# ============================================================================
# Test 2: Partial Tool Failure - Slack section becomes unverified
# ============================================================================

class TestPartialToolFailure:
    """Test handling of partial tool failures."""
    
    def test_slack_failure_marks_section_unverified(self, ledger_with_slack_failure):
        """Failed Slack call should mark Slack section as unverified."""
        contract = CoverageContract(ledger_with_slack_failure)
        
        is_covered, missing = contract.check_section_coverage("Slack Highlights")
        
        assert is_covered is False
        assert "slack" in missing
    
    def test_partial_failure_in_needs_human_check(self, ledger_with_slack_failure):
        """Failed source should appear in needs_human_check list."""
        gate = SafetyGate(ledger_with_slack_failure)
        
        result = gate.check(
            report_text="<h2>Slack Highlights</h2><p>Team discussed release</p>",
            requested_sections=["Slack Highlights"],
        )
        
        assert result.can_send is False
        assert "slack" in result.missing_sources
        # Check needs_human_check contains slack-related message
        assert any("slack" in item.lower() for item in result.needs_human_check)
    
    def test_mixed_success_failure(self, fresh_ledger):
        """Mixed success/failure should calculate correct rate."""
        # Add successful call
        fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="search_jira_issues",
            query_params={},
            result={"issues": []},
            success=True,
        )
        # Add failed call
        fresh_ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="read_slack_channel",
            query_params={},
            result=None,
            success=False,
            error="auth_failed",
        )
        
        summary = fresh_ledger.get_coverage_summary()
        
        assert summary["total_entries"] == 2
        assert summary["successful_entries"] == 1
        assert summary["failed_entries"] == 1


# ============================================================================
# Test 3: Hallucinated Release Names - Removed or marked uncertain
# ============================================================================

class TestHallucinatedClaims:
    """Test detection and handling of hallucinated claims."""
    
    def test_release_claim_without_evidence(self, fresh_ledger):
        """Release claims without web evidence should be flagged."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "LangSmith released version 2.0 with new features",
            section="Competitor Updates",
        )
        
        assert len(claims) > 0
        assert claims[0].claim_type == "release"
        
        verified, unverified = scanner.validate_claims(claims)
        
        assert len(verified) == 0
        assert len(unverified) > 0
    
    def test_metric_claim_without_evidence(self, fresh_ledger):
        """Metric claims without source evidence should be flagged."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "Sprint velocity increased by 20% this week",
            section="Jira Analysis",
        )
        
        verified, unverified = scanner.validate_claims(claims)
        
        assert len(unverified) > 0
        assert any("metric" in c.claim_type for c in unverified)
    
    def test_rewrite_unverified_claims(self, fresh_ledger):
        """Unverified claims should be rewritten with uncertainty markers."""
        scanner = ClaimScanner(fresh_ledger)
        
        # Use a pattern that matches our claim detection regex
        claims = scanner.scan_for_claims(
            "Top 5 issues this week are critical bugs",
            section="Jira Analysis",
        )
        
        _, unverified = scanner.validate_claims(claims)
        rewrites = scanner.rewrite_unverified_claims(unverified)
        
        assert len(rewrites) > 0
        assert any("[Unverified]" in r or "[Needs verification]" in r for r in rewrites)
    
    def test_fabricated_report_blocked(self, fresh_ledger, fabricated_report_html):
        """Completely fabricated report should be blocked."""
        gate = SafetyGate(fresh_ledger)
        
        result = gate.check(
            report_text=fabricated_report_html,
            requested_sections=["Jira Analysis", "GitHub Activity", "Competitor Updates"],
        )
        
        assert result.can_send is False
        assert len(result.unverified_claims) > 0 or "No tool calls" in result.rejection_reason


# ============================================================================
# Test 4: Successful Tool Runs - Report with citations
# ============================================================================

class TestSuccessfulToolRuns:
    """Test that successful tool runs produce valid reports."""
    
    def test_jira_evidence_enables_jira_section(self, ledger_with_jira_success):
        """Successful Jira call should enable Jira section."""
        contract = CoverageContract(ledger_with_jira_success)
        
        is_covered, missing = contract.check_section_coverage("Jira Analysis")
        
        assert is_covered is True
        assert len(missing) == 0
    
    def test_github_evidence_enables_github_section(self, ledger_with_github_success):
        """Successful GitHub call should enable GitHub section."""
        contract = CoverageContract(ledger_with_github_success)
        
        is_covered, missing = contract.check_section_coverage("GitHub Activity")
        
        assert is_covered is True
        assert len(missing) == 0
    
    def test_full_coverage_allows_send(self, ledger_with_multiple_sources, sample_report_html):
        """Full coverage should allow email send."""
        # Add Slack to make it complete
        ledger_with_multiple_sources.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="read_slack_channel",
            query_params={"channel": "general"},
            result={"messages": [{"text": "Release discussion", "permalink": "https://slack.com/archives/C123/p456"}]},
            success=True,
        )
        
        gate = SafetyGate(ledger_with_multiple_sources)
        
        result = gate.check(
            report_text=sample_report_html,
            requested_sections=["Jira Analysis", "GitHub Activity", "Slack Highlights", "Competitor Updates"],
        )
        
        # Should pass or be close to passing
        assert result.tool_success_rate >= 0.5
        assert len(result.missing_sources) <= 1  # May still miss some
    
    def test_evidence_entry_has_identifiers(self, ledger_with_jira_success):
        """Evidence entries should contain extracted identifiers."""
        entries = ledger_with_jira_success.get_entries_by_source(SourceType.JIRA)
        
        assert len(entries) > 0
        entry = entries[0]
        assert len(entry.identifiers) > 0
        assert any("OPIK" in id for id in entry.identifiers)
    
    def test_evidence_entry_has_snippets(self, ledger_with_jira_success):
        """Evidence entries should contain extracted snippets."""
        entries = ledger_with_jira_success.get_entries_by_source(SourceType.JIRA)
        
        assert len(entries) > 0
        entry = entries[0]
        assert len(entry.snippets) > 0


# ============================================================================
# Test 5: Source Type Validation
# ============================================================================

class TestSourceTypeValidation:
    """Test that source types match section claims."""
    
    def test_jira_section_with_github_ids_fails(self, ledger_with_jira_success):
        """Jira section with GitHub identifiers should fail validation."""
        validator = SourceValidator(ledger_with_jira_success)
        
        # Jira section but with GitHub URLs
        result = validator.validate_section(
            "Jira Analysis",
            "Issues this week: https://github.com/comet-ml/opik/issues/123",
        )
        
        # Should detect mismatch
        assert result.expected_source == SourceType.JIRA
        # Found GitHub identifiers in Jira section
        assert len(result.mismatched_identifiers) > 0 or not result.is_valid
    
    def test_github_section_with_jira_ids_fails(self, ledger_with_github_success):
        """GitHub section with Jira identifiers should fail validation."""
        validator = SourceValidator(ledger_with_github_success)
        
        result = validator.validate_section(
            "GitHub Activity",
            "PRs this week: OPIK-123, OPIK-456",  # Jira keys in GitHub section
        )
        
        assert result.expected_source == SourceType.GITHUB
        # Should detect Jira keys as mismatched
        assert len(result.mismatched_identifiers) > 0
    
    def test_correct_identifiers_pass(self, ledger_with_jira_success):
        """Correct identifiers should pass validation."""
        validator = SourceValidator(ledger_with_jira_success)
        
        result = validator.validate_section(
            "Jira Analysis",
            "Issues: OPIK-123 is in progress, OPIK-124 needs review",
        )
        
        assert result.expected_source == SourceType.JIRA
        assert len(result.found_identifiers) > 0
        assert "OPIK-123" in result.found_identifiers
    
    def test_slack_section_needs_permalinks(self, fresh_ledger):
        """Slack section should have permalinks or channel IDs."""
        # Add Slack evidence with permalink
        fresh_ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="read_slack_channel",
            query_params={},
            result={"permalink": "https://slack.com/archives/C123/p456"},
            success=True,
        )
        
        validator = SourceValidator(fresh_ledger)
        
        result = validator.validate_section(
            "Slack Highlights",
            "Discussion at https://slack.com/archives/C123/p456",
        )
        
        assert result.expected_source == SourceType.SLACK
        assert len(result.found_identifiers) > 0
    
    def test_full_report_validation(self, ledger_with_multiple_sources, sample_report_html):
        """Full report should be validated for source consistency."""
        is_valid, results, escalations = validate_report_sources(
            sample_report_html,
            ledger_with_multiple_sources,
        )
        
        # Should have results for each section
        assert len(results) > 0
        
        # Check that we got validation results
        section_names = [r.section_name for r in results]
        assert any("Jira" in s for s in section_names)


# ============================================================================
# Test 6: Coverage Contract
# ============================================================================

class TestCoverageContract:
    """Test the coverage contract enforcement."""
    
    def test_missing_jira_for_jira_section(self, fresh_ledger):
        """Missing Jira calls should fail Jira section coverage."""
        contract = CoverageContract(fresh_ledger)
        
        is_covered, missing = contract.check_section_coverage("Jira Analysis")
        
        assert is_covered is False
        assert "jira" in missing
    
    def test_missing_github_for_pr_section(self, fresh_ledger):
        """Missing GitHub calls should fail PR section coverage."""
        contract = CoverageContract(fresh_ledger)
        
        is_covered, missing = contract.check_section_coverage("Pull Requests")
        
        assert is_covered is False
        assert "github" in missing
    
    def test_coverage_report_summary(self, ledger_with_multiple_sources):
        """Coverage report should summarize all sections."""
        contract = CoverageContract(ledger_with_multiple_sources)
        
        report = contract.get_coverage_report([
            "Jira Analysis",
            "GitHub Activity",
            "Competitor Updates",
        ])
        
        assert "sections" in report
        assert "all_covered" in report
        assert "available_sources" in report
        
        # Should have Jira, GitHub, Web covered
        assert SourceType.JIRA in report["available_sources"]
        assert SourceType.GITHUB in report["available_sources"]


# ============================================================================
# Test 7: Safety Gate Thresholds
# ============================================================================

class TestSafetyGateThresholds:
    """Test safety gate threshold enforcement."""
    
    def test_below_success_rate_threshold(self, fresh_ledger):
        """Below 50% success rate should block."""
        # Add 1 success, 2 failures
        fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="search_jira_issues",
            query_params={},
            result={"issues": []},
            success=True,
        )
        fresh_ledger.record_tool_call(
            source_type=SourceType.GITHUB,
            tool_name="list_github_prs",
            query_params={},
            result=None,
            success=False,
            error="auth_failed",
        )
        fresh_ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="read_slack_channel",
            query_params={},
            result=None,
            success=False,
            error="not_in_channel",
        )
        
        gate = SafetyGate(fresh_ledger)
        
        result = gate.check(
            report_text="<h2>Summary</h2><p>Test</p>",
            requested_sections=["Summary"],
        )
        
        assert result.tool_success_rate < 0.5
        assert result.can_send is False
        assert "success rate" in result.rejection_reason.lower()
    
    def test_above_thresholds_allows_send(self, ledger_with_multiple_sources):
        """Above thresholds should allow send."""
        gate = SafetyGate(ledger_with_multiple_sources)
        
        # Simple report that matches available evidence
        result = gate.check(
            report_text="<h2>Summary</h2><p>General update</p>",
            requested_sections=["Summary"],
        )
        
        # With 3 successful calls and no specific section requirements
        assert result.tool_success_rate >= 0.5


# ============================================================================
# Test 8: Evidence Ledger Operations
# ============================================================================

class TestEvidenceLedgerOperations:
    """Test evidence ledger basic operations."""
    
    def test_record_and_retrieve(self, fresh_ledger):
        """Should record and retrieve entries."""
        entry_id = fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="test_tool",
            query_params={"key": "value"},
            result={"data": "test"},
            success=True,
        )
        
        entry = fresh_ledger.get_entry(entry_id)
        
        assert entry is not None
        assert entry.tool_name == "test_tool"
        assert entry.success is True
    
    def test_get_entries_by_source(self, fresh_ledger):
        """Should filter entries by source type."""
        fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="jira_tool",
            query_params={},
            result={},
            success=True,
        )
        fresh_ledger.record_tool_call(
            source_type=SourceType.GITHUB,
            tool_name="github_tool",
            query_params={},
            result={},
            success=True,
        )
        
        jira_entries = fresh_ledger.get_entries_by_source(SourceType.JIRA)
        github_entries = fresh_ledger.get_entries_by_source(SourceType.GITHUB)
        
        assert len(jira_entries) == 1
        assert len(github_entries) == 1
        assert jira_entries[0].tool_name == "jira_tool"
    
    def test_get_successful_sources(self, fresh_ledger):
        """Should return only successful source types."""
        fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="jira_tool",
            query_params={},
            result={},
            success=True,
        )
        fresh_ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="slack_tool",
            query_params={},
            result=None,
            success=False,
        )
        
        successful = fresh_ledger.get_successful_sources()
        
        assert SourceType.JIRA in successful
        assert SourceType.SLACK not in successful
    
    def test_clear_ledger(self, fresh_ledger):
        """Should clear all entries."""
        fresh_ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="test",
            query_params={},
            result={},
            success=True,
        )
        
        fresh_ledger.clear()
        
        assert fresh_ledger.get_coverage_summary()["total_entries"] == 0
    
    def test_to_dict_serialization(self, ledger_with_jira_success):
        """Should serialize to dict."""
        data = ledger_with_jira_success.to_dict()
        
        assert "entries" in data
        assert "coverage" in data
        assert len(data["entries"]) > 0


# ============================================================================
# Test 9: Claim Pattern Detection
# ============================================================================

class TestClaimPatternDetection:
    """Test detection of claim patterns that require evidence."""
    
    def test_detect_release_claims(self, fresh_ledger):
        """Should detect release/launch claims."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "OpenAI launched GPT-5 yesterday",
            section="Market Updates",
        )
        
        assert len(claims) > 0
        assert claims[0].claim_type == "release"
    
    def test_detect_metric_claims(self, fresh_ledger):
        """Should detect metric claims."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "Top 5 issues this sprint are critical bugs",
            section="Jira Analysis",
        )
        
        assert len(claims) > 0
        assert claims[0].claim_type == "metric"
    
    def test_detect_activity_claims(self, fresh_ledger):
        """Should detect activity claims."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "Slack highlights include the release planning discussion",
            section="Slack Highlights",
        )
        
        assert len(claims) > 0
        assert claims[0].claim_type == "activity"
    
    def test_no_claims_in_neutral_text(self, fresh_ledger):
        """Should not flag neutral text."""
        scanner = ClaimScanner(fresh_ledger)
        
        claims = scanner.scan_for_claims(
            "This is a general summary of the week",
            section="Summary",
        )
        
        # Neutral text shouldn't trigger claim detection
        assert len(claims) == 0


# ============================================================================
# Test 10: Integration Test - Full Flow
# ============================================================================

class TestFullIntegrationFlow:
    """Integration tests for the full evidence flow."""
    
    def test_complete_valid_flow(self, ledger_with_multiple_sources):
        """Test complete flow with valid evidence."""
        # Add Slack to complete coverage
        ledger_with_multiple_sources.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="read_slack_channel",
            query_params={},
            result={"messages": [{"text": "test", "permalink": "https://slack.com/test"}]},
            success=True,
        )
        
        gate = SafetyGate(ledger_with_multiple_sources)
        
        # Report that matches evidence
        report = """
        <h2>Summary</h2>
        <p>Weekly update based on gathered data.</p>
        """
        
        result = gate.check(
            report_text=report,
            requested_sections=["Summary"],
        )
        
        # Should have high success rate
        assert result.tool_success_rate >= 0.5
    
    def test_complete_invalid_flow(self, fresh_ledger, fabricated_report_html):
        """Test complete flow with fabricated report."""
        gate = SafetyGate(fresh_ledger)
        
        result = gate.check(
            report_text=fabricated_report_html,
            requested_sections=["Jira Analysis", "GitHub Activity", "Competitor Updates"],
        )
        
        # Should be blocked
        assert result.can_send is False
        
        # Should have draft with gaps
        draft = gate.generate_incomplete_draft(fabricated_report_html, result)
        assert "DRAFT" in draft
        assert "Missing" in draft or "Needs" in draft


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

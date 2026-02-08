"""
Tests for Trust-Critical PM Agent Operating Instructions.

These tests verify the strict execution flow, alerting rules,
language constraints, and trust scoring system.

Version: Trust-Critical / Production
"""

import pytest
from unittest.mock import MagicMock, patch

from src.execution_state import (
    CheckStatus,
    CheckType,
    ExecutionState,
    CheckResult,
    JiraFinding,
    GitHubFinding,
    SlackFinding,
    get_execution_state,
    reset_execution_state,
    record_check_start,
    record_check_success,
    record_check_failure,
)
from src.alerting import (
    AlertGate,
    AlertPayload,
    AlertSeverity,
    AlertDecision,
    validate_language,
    PROHIBITED_PHRASES,
    format_safe_summary,
)
from src.trust_score import (
    TrustScorer,
    TrustScoreResult,
    TrustViolation,
    calculate_trust_score,
)
from src.evidence import reset_ledger, get_ledger, SourceType


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def fresh_state():
    """Reset execution state before each test."""
    reset_execution_state()
    reset_ledger()
    return get_execution_state()


@pytest.fixture
def complete_success_state(fresh_state):
    """Execution state with all checks successful."""
    state = fresh_state
    
    # Jira check
    state.start_check(CheckType.JIRA)
    state.complete_check_success(
        CheckType.JIRA,
        findings=[{
            "source": "jira",
            "issue_id": "OPIK-123",
            "priority": "P1",
            "status": "Open",
            "summary": "Critical bug in tracing",
        }],
    )
    
    # GitHub check
    state.start_check(CheckType.GITHUB)
    state.complete_check_success(
        CheckType.GITHUB,
        findings=[{
            "source": "github",
            "issue_id": "comet-ml/opik#456",
            "labels": ["bug", "critical"],
            "state": "open",
            "repository": "comet-ml/opik",
            "title": "Regression in evaluation module",
        }],
    )
    
    # Slack check
    state.start_check(CheckType.SLACK)
    state.complete_check_success(
        CheckType.SLACK,
        findings=[{
            "source": "slack",
            "channel": "#incidents",
            "timestamp": "1234567890.123456",
            "message_excerpt": "prod down - investigating",
        }],
    )
    
    return state


@pytest.fixture
def partial_failure_state(fresh_state):
    """Execution state with one check failed."""
    state = fresh_state
    
    # Jira success
    state.start_check(CheckType.JIRA)
    state.complete_check_success(CheckType.JIRA, findings=[])
    
    # GitHub success
    state.start_check(CheckType.GITHUB)
    state.complete_check_success(CheckType.GITHUB, findings=[])
    
    # Slack failure
    state.start_check(CheckType.SLACK)
    state.complete_check_failed(
        CheckType.SLACK,
        reason="Missing channel access permissions",
    )
    
    return state


@pytest.fixture
def incomplete_state(fresh_state):
    """Execution state with checks not completed."""
    state = fresh_state
    
    # Only Jira started
    state.start_check(CheckType.JIRA)
    
    return state


# ============================================================================
# EXECUTION STATE TESTS
# ============================================================================

class TestExecutionState:
    """Test the execution state machine."""
    
    def test_initial_state_all_not_started(self, fresh_state):
        """All checks should start as NOT_STARTED."""
        for check_type in fresh_state.REQUIRED_CHECKS:
            assert fresh_state.get_check_status(check_type) == CheckStatus.NOT_STARTED
    
    def test_start_check_changes_status(self, fresh_state):
        """Starting a check should change status to IN_PROGRESS."""
        fresh_state.start_check(CheckType.JIRA)
        assert fresh_state.get_check_status(CheckType.JIRA) == CheckStatus.IN_PROGRESS
    
    def test_complete_success_changes_status(self, fresh_state):
        """Completing a check successfully should change status to SUCCESS."""
        fresh_state.start_check(CheckType.JIRA)
        fresh_state.complete_check_success(CheckType.JIRA, findings=[])
        assert fresh_state.get_check_status(CheckType.JIRA) == CheckStatus.SUCCESS
    
    def test_complete_failure_changes_status(self, fresh_state):
        """Completing a check with failure should change status to FAILED_WITH_REASON."""
        fresh_state.start_check(CheckType.JIRA)
        fresh_state.complete_check_failed(CheckType.JIRA, reason="API timeout")
        assert fresh_state.get_check_status(CheckType.JIRA) == CheckStatus.FAILED_WITH_REASON
    
    def test_is_complete_false_when_not_started(self, fresh_state):
        """is_complete should be False when checks haven't started."""
        assert fresh_state.is_complete() is False
    
    def test_is_complete_false_when_in_progress(self, incomplete_state):
        """is_complete should be False when checks are in progress."""
        assert incomplete_state.is_complete() is False
    
    def test_is_complete_true_when_all_finished(self, complete_success_state):
        """is_complete should be True when all checks finished."""
        assert complete_success_state.is_complete() is True
    
    def test_is_complete_true_with_failures(self, partial_failure_state):
        """is_complete should be True even with failures (they're finished)."""
        assert partial_failure_state.is_complete() is True
    
    def test_is_all_success_true(self, complete_success_state):
        """is_all_success should be True when all checks succeeded."""
        assert complete_success_state.is_all_success() is True
    
    def test_is_all_success_false_with_failure(self, partial_failure_state):
        """is_all_success should be False when any check failed."""
        assert partial_failure_state.is_all_success() is False
    
    def test_alert_eligible_only_when_all_success(self, complete_success_state):
        """Alert eligibility requires all checks to succeed."""
        assert complete_success_state.is_alert_eligible() is True
    
    def test_alert_not_eligible_with_failure(self, partial_failure_state):
        """Alert should not be eligible when any check failed."""
        assert partial_failure_state.is_alert_eligible() is False
    
    def test_get_failed_checks(self, partial_failure_state):
        """get_failed_checks should return failed check details."""
        failed = partial_failure_state.get_failed_checks()
        assert len(failed) == 1
        assert failed[0].check_type == CheckType.SLACK
        assert "permissions" in failed[0].failure_reason
    
    def test_get_incomplete_checks(self, fresh_state):
        """get_incomplete_checks should return checks not finished."""
        # Start all three checks so they become required
        fresh_state.start_check(CheckType.JIRA)
        fresh_state.start_check(CheckType.GITHUB)
        fresh_state.start_check(CheckType.SLACK)
        # Only complete SLACK
        fresh_state.complete_check_success(CheckType.SLACK, findings=[])
        incomplete = fresh_state.get_incomplete_checks()
        # JIRA and GITHUB are still in progress
        assert CheckType.JIRA in incomplete
        assert CheckType.GITHUB in incomplete
        assert CheckType.SLACK not in incomplete


class TestExecutionStateReports:
    """Test execution state report generation."""
    
    def test_incomplete_report_format(self, incomplete_state):
        """Incomplete state should generate proper report."""
        report = incomplete_state.generate_status_report()
        assert "STATUS: CHECK INCOMPLETE" in report
        assert "UNKNOWN STATE" in report
        assert "could not be verified" in report
    
    def test_partial_success_report_format(self, partial_failure_state):
        """Partial success should generate proper report."""
        report = partial_failure_state.generate_status_report()
        assert "PARTIAL CHECK" in report or "FAILED CHECKS" in report
        assert "SLACK" in report
        assert "permissions" in report.lower()
    
    def test_complete_success_report_format(self, complete_success_state):
        """Complete success should generate proper report."""
        report = complete_success_state.generate_status_report()
        assert "ALL CHECKS COMPLETE" in report
        assert "CRITICAL FINDINGS" in report or "ALERT STATUS" in report


# ============================================================================
# ALERTING TESTS
# ============================================================================

class TestAlertGate:
    """Test the strict alerting rules."""
    
    def test_no_alert_when_incomplete(self, incomplete_state):
        """No alert should be sent when checks are incomplete."""
        gate = AlertGate(incomplete_state)
        decision = gate.check_alert_eligibility()
        
        assert decision.should_alert is False
        assert "incomplete" in decision.reason.lower() or "not finished" in decision.reason.lower()
    
    def test_no_alert_when_failed_checks(self, partial_failure_state):
        """No alert should be sent when checks failed."""
        gate = AlertGate(partial_failure_state)
        decision = gate.check_alert_eligibility()
        
        assert decision.should_alert is False
        assert len(decision.blocked_reasons) > 0
    
    def test_no_alert_without_critical_findings(self, fresh_state):
        """No alert should be sent without P0/P1 findings."""
        # Complete all checks with no findings
        for check_type in fresh_state.REQUIRED_CHECKS:
            fresh_state.start_check(check_type)
            fresh_state.complete_check_success(check_type, findings=[])
        
        gate = AlertGate(fresh_state)
        decision = gate.check_alert_eligibility()
        
        assert decision.should_alert is False
        assert "No verified P0/P1" in decision.reason or "critical" in decision.reason.lower()
    
    def test_alert_with_p0_finding(self, complete_success_state):
        """Alert should be allowed with verified P0 finding."""
        # Add a P0 finding
        complete_success_state._checks[CheckType.JIRA].findings.append({
            "source": "jira",
            "issue_id": "OPIK-999",
            "priority": "P0",
            "status": "Open",
            "summary": "Production database down - all users affected",
        })
        
        gate = AlertGate(complete_success_state)
        decision = gate.check_alert_eligibility()
        
        assert decision.should_alert is True
        assert decision.payload is not None
        assert decision.payload.severity == AlertSeverity.P0


class TestAlertPayload:
    """Test alert payload validation."""
    
    def test_valid_payload(self):
        """Valid payload should pass validation."""
        payload = AlertPayload(
            source="jira",
            identifier="OPIK-123",
            severity=AlertSeverity.P0,
            impact="Production database is down. All users cannot access the platform.",
            recommended_action="Contact on-call DBA immediately. Check database cluster status.",
            url="https://jira.example.com/OPIK-123",
        )
        
        is_valid, error = payload.validate()
        assert is_valid is True
        assert error is None
    
    def test_invalid_payload_missing_identifier(self):
        """Payload without identifier should fail."""
        payload = AlertPayload(
            source="jira",
            identifier="",
            severity=AlertSeverity.P0,
            impact="Something is wrong",
            recommended_action="Fix it",
        )
        
        is_valid, error = payload.validate()
        assert is_valid is False
        assert "identifier" in error.lower()
    
    def test_invalid_payload_p2_severity(self):
        """P2 severity should not be alertable."""
        payload = AlertPayload(
            source="jira",
            identifier="OPIK-123",
            severity=AlertSeverity.P2,
            impact="Minor UI issue affecting some users",
            recommended_action="Add to backlog",
        )
        
        is_valid, error = payload.validate()
        assert is_valid is False
        assert "P0/P1" in error or "not alertable" in error.lower()


# ============================================================================
# LANGUAGE CONSTRAINT TESTS
# ============================================================================

class TestLanguageConstraints:
    """Test language validation rules."""
    
    def test_prohibited_phrases_detected(self):
        """Prohibited phrases should be detected."""
        for phrase in PROHIBITED_PHRASES:
            text = f"The system {phrase} today."
            is_valid, violations = validate_language(text)
            assert is_valid is False, f"Should detect prohibited phrase: {phrase}"
            assert len(violations) > 0
    
    def test_approved_phrases_allowed(self):
        """Approved phrases should be allowed."""
        approved_texts = [
            "No verified critical issues detected in checked sources",
            "Data unavailable for Slack channel",
            "Unable to verify GitHub status",
            "Status: Unknown",
        ]
        
        for text in approved_texts:
            is_valid, violations = validate_language(text)
            assert is_valid is True, f"Should allow: {text}"
    
    def test_mixed_content_detects_prohibited(self):
        """Mixed content should still detect prohibited phrases."""
        text = """
        STATUS: CHECK COMPLETE
        
        Jira: No verified critical issues detected in checked sources
        GitHub: All good
        Slack: Data unavailable
        """
        
        is_valid, violations = validate_language(text)
        assert is_valid is False
        assert any("all good" in v.lower() for v in violations)


class TestSafeSummaryFormat:
    """Test safe summary formatting."""
    
    def test_incomplete_state_format(self, incomplete_state):
        """Incomplete state should format safely."""
        summary = format_safe_summary(incomplete_state)
        
        assert "INCOMPLETE" in summary
        assert "could not be verified" in summary
        # Should NOT contain prohibited phrases
        is_valid, _ = validate_language(summary)
        assert is_valid is True
    
    def test_partial_failure_format(self, partial_failure_state):
        """Partial failure should format safely."""
        summary = format_safe_summary(partial_failure_state)
        
        assert "PARTIAL" in summary or "SLACK" in summary
        # Should NOT contain prohibited phrases
        is_valid, _ = validate_language(summary)
        assert is_valid is True
    
    def test_complete_success_format(self, complete_success_state):
        """Complete success should format safely."""
        summary = format_safe_summary(complete_success_state)
        
        assert "COMPLETE" in summary
        # Should NOT contain prohibited phrases
        is_valid, _ = validate_language(summary)
        assert is_valid is True


# ============================================================================
# TRUST SCORE TESTS
# ============================================================================

class TestTrustScore:
    """Test trust score calculation."""
    
    def test_zero_score_no_tool_calls(self, fresh_state):
        """No tool calls should result in low evidence score."""
        scorer = TrustScorer(
            ledger=get_ledger(),
            execution_state=fresh_state,
        )
        result = scorer.calculate_score()
        
        # Evidence score should be 0 (no tool calls)
        assert result.evidence_score == 0.0
        assert not result.is_trustworthy()
    
    def test_high_score_complete_success(self, complete_success_state):
        """Complete success should result in high score."""
        # Add evidence to ledger
        ledger = get_ledger()
        ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="jira_search",
            query_params={"jql": "priority = P0"},
            result={"issues": [{"key": "OPIK-123"}]},
            success=True,
        )
        ledger.record_tool_call(
            source_type=SourceType.GITHUB,
            tool_name="github_list_issues",
            query_params={"labels": "bug"},
            result={"items": []},
            success=True,
        )
        ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="slack_search",
            query_params={"query": "incident"},
            result={"messages": []},
            success=True,
        )
        
        scorer = TrustScorer(
            ledger=ledger,
            execution_state=complete_success_state,
        )
        result = scorer.calculate_score()
        
        # Should have high execution score
        assert result.execution_score >= 0.8
        assert result.is_trustworthy()
    
    def test_language_violations_reduce_score(self, complete_success_state):
        """Language violations should reduce trust score."""
        # Add evidence
        ledger = get_ledger()
        for source in [SourceType.JIRA, SourceType.GITHUB, SourceType.SLACK]:
            ledger.record_tool_call(
                source_type=source,
                tool_name=f"{source.value}_tool",
                query_params={},
                result={},
                success=True,
            )
        
        scorer = TrustScorer(
            ledger=ledger,
            execution_state=complete_success_state,
        )
        
        # Test with prohibited phrase
        result = scorer.calculate_score(
            output_text="Everything looks fine and all good!",
        )
        
        assert result.language_score < 1.0
        assert any(v.category == "language" for v in result.violations)
    
    def test_trust_score_grades(self):
        """Test trust score grade calculation."""
        result = TrustScoreResult(
            overall_score=0.95,
            evidence_score=1.0,
            execution_score=1.0,
            language_score=1.0,
            alerting_score=1.0,
        )
        assert result.get_grade() == "A+"
        
        result.overall_score = 0.85
        assert result.get_grade() == "B+"
        
        result.overall_score = 0.65
        assert result.get_grade() == "D"
        
        result.overall_score = 0.50
        assert result.get_grade() == "F"


class TestTrustScoreReport:
    """Test trust score report formatting."""
    
    def test_report_format(self, complete_success_state):
        """Trust score report should be properly formatted."""
        ledger = get_ledger()
        ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="jira_search",
            query_params={},
            result={},
            success=True,
        )
        
        scorer = TrustScorer(
            ledger=ledger,
            execution_state=complete_success_state,
        )
        result = scorer.calculate_score()
        report = result.format_report()
        
        assert "TRUST SCORE REPORT" in report
        assert "Overall Score" in report
        assert "Evidence Coverage" in report
        assert "Execution Complete" in report


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestTrustCriticalIntegration:
    """Integration tests for the full trust-critical system."""
    
    def test_full_flow_no_tools(self, fresh_state):
        """Full flow with no tool calls should be untrusted."""
        # Don't run any checks
        
        # Generate report
        report = fresh_state.generate_status_report()
        assert "INCOMPLETE" in report
        
        # Check alert eligibility
        gate = AlertGate(fresh_state)
        decision = gate.check_alert_eligibility()
        assert decision.should_alert is False
        
        # Calculate trust score
        result = calculate_trust_score(output_text=report)
        assert result.is_trustworthy() is False
        assert result.get_grade() == "F"
    
    def test_full_flow_partial_failure(self, partial_failure_state):
        """Full flow with partial failure should be partially trusted."""
        # Add some evidence
        ledger = get_ledger()
        ledger.record_tool_call(
            source_type=SourceType.JIRA,
            tool_name="jira_search",
            query_params={},
            result={},
            success=True,
        )
        ledger.record_tool_call(
            source_type=SourceType.GITHUB,
            tool_name="github_list_issues",
            query_params={},
            result={},
            success=True,
        )
        ledger.record_tool_call(
            source_type=SourceType.SLACK,
            tool_name="slack_search",
            query_params={},
            result=None,
            success=False,
            error="Permission denied",
        )
        
        # Generate report
        report = partial_failure_state.generate_status_report()
        assert "SLACK" in report
        
        # Check alert eligibility
        gate = AlertGate(partial_failure_state)
        decision = gate.check_alert_eligibility()
        assert decision.should_alert is False
        
        # Calculate trust score
        scorer = TrustScorer(
            ledger=ledger,
            execution_state=partial_failure_state,
        )
        result = scorer.calculate_score(output_text=report)
        
        # Should have some violations but not completely untrusted
        assert len(result.violations) > 0
        assert result.execution_score < 1.0
    
    def test_full_flow_success_with_alert(self, complete_success_state):
        """Full flow with success and critical finding should allow alert."""
        # Add P0 finding
        complete_success_state._checks[CheckType.JIRA].findings.append({
            "source": "jira",
            "issue_id": "OPIK-CRITICAL",
            "priority": "P0",
            "status": "Open",
            "summary": "Production down - all users blocked",
            "url": "https://jira.example.com/OPIK-CRITICAL",
        })
        
        # Add evidence
        ledger = get_ledger()
        for source in [SourceType.JIRA, SourceType.GITHUB, SourceType.SLACK]:
            ledger.record_tool_call(
                source_type=source,
                tool_name=f"{source.value}_tool",
                query_params={},
                result={"issues": [{"key": "OPIK-CRITICAL", "priority": "P0"}]} if source == SourceType.JIRA else {},
                success=True,
            )
        
        # Check alert eligibility
        gate = AlertGate(complete_success_state)
        decision = gate.check_alert_eligibility()
        
        assert decision.should_alert is True
        assert decision.payload is not None
        assert decision.payload.severity == AlertSeverity.P0
        
        # Calculate trust score
        scorer = TrustScorer(
            ledger=ledger,
            execution_state=complete_success_state,
        )
        result = scorer.calculate_score(
            output_text="No verified critical issues detected in checked sources",
            alert_was_sent=True,
        )
        
        assert result.is_trustworthy()


# ============================================================================
# GLOBAL FUNCTION TESTS
# ============================================================================

class TestGlobalFunctions:
    """Test global helper functions."""
    
    def test_reset_execution_state(self):
        """reset_execution_state should create fresh state."""
        state1 = reset_execution_state()
        state1.start_check(CheckType.JIRA)
        
        state2 = reset_execution_state()
        assert state2.get_check_status(CheckType.JIRA) == CheckStatus.NOT_STARTED
    
    def test_record_check_functions(self):
        """Test convenience recording functions."""
        reset_execution_state()
        
        record_check_start(CheckType.JIRA)
        state = get_execution_state()
        assert state.get_check_status(CheckType.JIRA) == CheckStatus.IN_PROGRESS
        
        record_check_success(CheckType.JIRA, findings=[{"key": "TEST-1"}])
        assert state.get_check_status(CheckType.JIRA) == CheckStatus.SUCCESS
        
        record_check_start(CheckType.GITHUB)
        record_check_failure(CheckType.GITHUB, "API error")
        assert state.get_check_status(CheckType.GITHUB) == CheckStatus.FAILED_WITH_REASON

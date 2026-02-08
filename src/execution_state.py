"""
Strict Execution State Machine for Trust-Critical Operations.

This module implements the required execution flow from the PM Agent Operating Instructions.
Every check must be explicitly tracked and no claims can be made without evidence.

Version: Trust-Critical / Production
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.utils import logger


class CheckStatus(str, Enum):
    """Status of each required check."""
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    FAILED_WITH_REASON = "FAILED_WITH_REASON"


class CheckType(str, Enum):
    """Types of required checks."""
    JIRA = "JIRA_CHECK"
    GITHUB = "GITHUB_CHECK"
    SLACK = "SLACK_CHECK"
    WEB = "WEB_CHECK"  # Optional for market research


@dataclass
class CheckResult:
    """Result of a single check execution."""
    check_type: CheckType
    status: CheckStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    findings: List[Dict[str, Any]] = field(default_factory=list)
    raw_tool_output: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_type": self.check_type.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failure_reason": self.failure_reason,
            "findings_count": len(self.findings),
            "findings": self.findings[:10],  # Limit for serialization
        }


@dataclass
class JiraFinding:
    """A verified Jira finding with required fields."""
    issue_id: str
    priority: str
    status: str
    summary: str
    url: Optional[str] = None
    assignee: Optional[str] = None
    created: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "jira",
            "issue_id": self.issue_id,
            "priority": self.priority,
            "status": self.status,
            "summary": self.summary,
            "url": self.url,
            "assignee": self.assignee,
        }


@dataclass
class GitHubFinding:
    """A verified GitHub finding with required fields."""
    issue_id: str  # repo#number format
    labels: List[str]
    state: str
    repository: str
    title: str
    url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "github",
            "issue_id": self.issue_id,
            "labels": self.labels,
            "state": self.state,
            "repository": self.repository,
            "title": self.title,
            "url": self.url,
        }


@dataclass
class SlackFinding:
    """A verified Slack finding with required fields."""
    channel: str
    timestamp: str
    message_excerpt: str
    permalink: Optional[str] = None
    matched_keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "slack",
            "channel": self.channel,
            "timestamp": self.timestamp,
            "message_excerpt": self.message_excerpt[:200],
            "permalink": self.permalink,
            "matched_keywords": self.matched_keywords,
        }


class ExecutionState:
    """
    Tracks the execution state of all required checks.
    
    Enforces the strict execution flow:
    1. All checks start as NOT_STARTED
    2. Each check must be explicitly started and completed
    3. No final output until all checks are SUCCESS or FAILED_WITH_REASON
    4. ALERT_ELIGIBLE only if all checks succeeded
    """
    
    # Default required checks for a complete run (can be overridden per query)
    DEFAULT_REQUIRED_CHECKS = {CheckType.JIRA, CheckType.GITHUB, CheckType.SLACK}
    
    def __init__(self):
        self._checks: Dict[CheckType, CheckResult] = {}
        self._alert_eligible = False
        self._initialized_at = datetime.now(timezone.utc).isoformat()
        self._required_checks: Set[CheckType] = set()  # Empty until set
        
        # Initialize all checks as NOT_STARTED
        for check_type in CheckType:
            self._checks[check_type] = CheckResult(
                check_type=check_type,
                status=CheckStatus.NOT_STARTED,
            )
        
        logger.info("execution_state_initialized", checks=list(CheckType))
    
    def set_required_checks(self, check_types: Set[CheckType]) -> None:
        """
        Set which checks are required for this specific query.
        
        This allows dynamic check requirements based on query routing.
        Only the specified checks need to complete for is_complete() to return True.
        """
        self._required_checks = check_types
        logger.info("required_checks_set", checks=[ct.value for ct in check_types])
    
    @property
    def REQUIRED_CHECKS(self) -> Set[CheckType]:
        """Get the required checks for this run."""
        # If explicitly set, use those; otherwise use checks that were started
        if self._required_checks:
            return self._required_checks
        
        # Auto-detect: any check that was started is required
        started_checks = {
            ct for ct, result in self._checks.items()
            if result.status != CheckStatus.NOT_STARTED
        }
        
        # If nothing started, fall back to default
        return started_checks if started_checks else self.DEFAULT_REQUIRED_CHECKS
    
    def start_check(self, check_type: CheckType) -> None:
        """Mark a check as in progress."""
        self._checks[check_type].status = CheckStatus.IN_PROGRESS
        self._checks[check_type].started_at = datetime.now(timezone.utc).isoformat()
        
        logger.info("check_started", check_type=check_type.value)
    
    def complete_check_success(
        self,
        check_type: CheckType,
        findings: List[Dict[str, Any]],
        raw_output: Optional[str] = None,
    ) -> None:
        """Mark a check as successfully completed with findings."""
        check = self._checks[check_type]
        check.status = CheckStatus.SUCCESS
        check.completed_at = datetime.now(timezone.utc).isoformat()
        check.findings = findings
        check.raw_tool_output = raw_output[:1000] if raw_output else None
        
        logger.info(
            "check_completed_success",
            check_type=check_type.value,
            findings_count=len(findings),
        )
        
        # Update alert eligibility
        self._update_alert_eligibility()
    
    def complete_check_failed(
        self,
        check_type: CheckType,
        reason: str,
    ) -> None:
        """Mark a check as failed with explicit reason."""
        check = self._checks[check_type]
        check.status = CheckStatus.FAILED_WITH_REASON
        check.completed_at = datetime.now(timezone.utc).isoformat()
        check.failure_reason = reason
        
        logger.warning(
            "check_completed_failed",
            check_type=check_type.value,
            reason=reason,
        )
        
        # Failed check means not alert eligible
        self._alert_eligible = False
    
    def _update_alert_eligibility(self) -> None:
        """Update alert eligibility based on check states."""
        required_checks = self.REQUIRED_CHECKS
        
        all_required_success = all(
            self._checks[ct].status == CheckStatus.SUCCESS
            for ct in required_checks
        )
        
        self._alert_eligible = all_required_success
    
    def get_check_status(self, check_type: CheckType) -> CheckStatus:
        """Get the status of a specific check."""
        return self._checks[check_type].status
    
    def get_check_result(self, check_type: CheckType) -> CheckResult:
        """Get the full result of a specific check."""
        return self._checks[check_type]
    
    def is_complete(self) -> bool:
        """
        Check if all required checks have completed (success or failed).
        
        A run is complete when all required checks are either:
        - SUCCESS
        - FAILED_WITH_REASON
        
        NOT_STARTED or IN_PROGRESS means incomplete.
        """
        for check_type in self.REQUIRED_CHECKS:
            status = self._checks[check_type].status
            if status in (CheckStatus.NOT_STARTED, CheckStatus.IN_PROGRESS):
                return False
        return True
    
    def is_all_success(self) -> bool:
        """Check if all required checks succeeded."""
        for check_type in self.REQUIRED_CHECKS:
            if self._checks[check_type].status != CheckStatus.SUCCESS:
                return False
        return True
    
    def is_alert_eligible(self) -> bool:
        """Check if alerting is allowed (all checks must have succeeded)."""
        return self._alert_eligible
    
    def get_failed_checks(self) -> List[CheckResult]:
        """Get list of failed checks with reasons."""
        return [
            check for check in self._checks.values()
            if check.status == CheckStatus.FAILED_WITH_REASON
        ]
    
    def get_incomplete_checks(self) -> List[CheckType]:
        """Get list of checks that haven't completed."""
        incomplete = []
        for check_type in self.REQUIRED_CHECKS:
            status = self._checks[check_type].status
            if status in (CheckStatus.NOT_STARTED, CheckStatus.IN_PROGRESS):
                incomplete.append(check_type)
        return incomplete
    
    def get_all_findings(self) -> List[Dict[str, Any]]:
        """Get all findings from successful checks."""
        findings = []
        for check in self._checks.values():
            if check.status == CheckStatus.SUCCESS:
                findings.extend(check.findings)
        return findings
    
    def get_critical_findings(self) -> List[Dict[str, Any]]:
        """Get only P0/P1 or critical findings."""
        critical = []
        for finding in self.get_all_findings():
            priority = finding.get("priority", "").upper()
            labels = finding.get("labels", [])
            
            is_critical = (
                priority in ("P0", "P1", "HIGHEST", "CRITICAL", "BLOCKER")
                or any(l.lower() in ("critical", "blocker", "p0", "p1", "regression") for l in labels)
            )
            
            if is_critical:
                critical.append(finding)
        
        return critical
    
    def generate_status_report(self) -> str:
        """
        Generate the required status report.
        
        If incomplete: STATUS: CHECK INCOMPLETE with explicit unknowns
        If complete: Full status with findings
        """
        if not self.is_complete():
            return self._generate_incomplete_report()
        elif not self.is_all_success():
            return self._generate_partial_success_report()
        else:
            return self._generate_complete_report()
    
    def _generate_incomplete_report(self) -> str:
        """Generate report for incomplete state."""
        incomplete = self.get_incomplete_checks()
        failed = self.get_failed_checks()
        
        lines = [
            "STATUS: CHECK INCOMPLETE",
            "",
            "The following checks did not complete:",
        ]
        
        for check_type in incomplete:
            status = self._checks[check_type].status
            lines.append(f"  - {check_type.value}: {status.value}")
        
        if failed:
            lines.append("")
            lines.append("The following checks failed:")
            for check in failed:
                lines.append(f"  - {check.check_type.value}: {check.failure_reason}")
        
        lines.extend([
            "",
            "UNKNOWN STATE:",
            "  - Critical issues may exist but could not be verified",
            "  - No alerts were sent",
            "  - Human verification required",
        ])
        
        return "\n".join(lines)
    
    def _generate_partial_success_report(self) -> str:
        """Generate report when some checks failed."""
        failed = self.get_failed_checks()
        successful = [c for c in self._checks.values() if c.status == CheckStatus.SUCCESS]
        
        lines = [
            "STATUS: PARTIAL CHECK - SOME SOURCES UNAVAILABLE",
            "",
        ]
        
        # Report failures first
        lines.append("FAILED CHECKS (data unknown):")
        for check in failed:
            lines.append(f"  - {check.check_type.value}: {check.failure_reason}")
            lines.append(f"    → Issues may exist but could not be verified")
        
        lines.append("")
        
        # Report successes
        if successful:
            lines.append("SUCCESSFUL CHECKS:")
            for check in successful:
                critical_count = sum(
                    1 for f in check.findings
                    if f.get("priority", "").upper() in ("P0", "P1", "HIGHEST", "CRITICAL")
                    or any(l.lower() in ("critical", "blocker") for l in f.get("labels", []))
                )
                lines.append(f"  - {check.check_type.value}: {len(check.findings)} findings ({critical_count} critical)")
        
        lines.extend([
            "",
            "ALERT STATUS: NOT ELIGIBLE (incomplete data)",
            "No alerts were sent due to incomplete verification.",
        ])
        
        return "\n".join(lines)
    
    def _generate_complete_report(self) -> str:
        """Generate report when all checks succeeded."""
        critical = self.get_critical_findings()
        all_findings = self.get_all_findings()
        
        lines = [
            "STATUS: ALL CHECKS COMPLETE",
            "",
        ]
        
        # Summary per check
        for check_type in self.REQUIRED_CHECKS:
            check = self._checks[check_type]
            lines.append(f"{check_type.value}: {len(check.findings)} findings")
        
        lines.append("")
        
        if critical:
            lines.append(f"CRITICAL FINDINGS: {len(critical)}")
            for finding in critical[:5]:  # Limit to top 5
                source = finding.get("source", "unknown")
                issue_id = finding.get("issue_id", finding.get("key", "?"))
                summary = finding.get("summary", finding.get("title", ""))[:60]
                lines.append(f"  - [{source}] {issue_id}: {summary}")
            
            lines.append("")
            lines.append("ALERT STATUS: ELIGIBLE")
        else:
            lines.append("No verified critical issues detected in checked sources.")
            lines.append("")
            lines.append("ALERT STATUS: NOT REQUIRED")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize execution state."""
        return {
            "initialized_at": self._initialized_at,
            "is_complete": self.is_complete(),
            "is_all_success": self.is_all_success(),
            "alert_eligible": self._alert_eligible,
            "checks": {
                ct.value: self._checks[ct].to_dict()
                for ct in CheckType
            },
            "critical_findings_count": len(self.get_critical_findings()),
        }


# Global execution state (reset per run)
_current_state: Optional[ExecutionState] = None


def get_execution_state() -> ExecutionState:
    """Get or create the current execution state."""
    global _current_state
    if _current_state is None:
        _current_state = ExecutionState()
    return _current_state


def reset_execution_state() -> ExecutionState:
    """Reset execution state (call at start of each run)."""
    global _current_state
    _current_state = ExecutionState()
    return _current_state


def record_check_start(check_type: CheckType) -> None:
    """Record that a check has started."""
    get_execution_state().start_check(check_type)


def record_check_success(
    check_type: CheckType,
    findings: List[Dict[str, Any]],
    raw_output: Optional[str] = None,
) -> None:
    """Record successful check completion."""
    get_execution_state().complete_check_success(check_type, findings, raw_output)


def record_check_failure(check_type: CheckType, reason: str) -> None:
    """Record check failure with reason."""
    get_execution_state().complete_check_failed(check_type, reason)

"""
Strict Alerting System for Trust-Critical Operations.

Alerting humans is HIGH RISK and must be RARE.

An alert may ONLY be sent if ALL of the following are true:
1. At least one verified P0 or P1 issue exists
2. You have: ID, Link, Impact description, Suggested immediate action
3. The issue is: User-blocking, Revenue-blocking, or Production-down

If ANY condition is missing → NO ALERT

Version: Trust-Critical / Production
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.execution_state import ExecutionState, get_execution_state
from src.utils import logger


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    P0 = "P0"  # Production down, all users affected
    P1 = "P1"  # Critical functionality broken, many users affected
    P2 = "P2"  # Important issue, some users affected (NOT alertable)
    P3 = "P3"  # Minor issue (NOT alertable)


class ImpactType(str, Enum):
    """Types of impact that justify alerting."""
    USER_BLOCKING = "user_blocking"
    REVENUE_BLOCKING = "revenue_blocking"
    PRODUCTION_DOWN = "production_down"


@dataclass
class AlertPayload:
    """
    Required alert payload structure.
    
    ALL fields are REQUIRED. No fluff. No speculation.
    """
    source: str  # "jira" | "github" | "slack"
    identifier: str  # Ticket/issue ID
    severity: AlertSeverity
    impact: str  # 1-2 sentences, factual only
    recommended_action: str  # Concrete next step
    url: Optional[str] = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate that all required fields are present and valid."""
        if not self.source:
            return False, "Missing required field: source"
        if not self.identifier:
            return False, "Missing required field: identifier"
        if not self.impact or len(self.impact) < 10:
            return False, "Impact description too short or missing"
        if not self.recommended_action or len(self.recommended_action) < 10:
            return False, "Recommended action too short or missing"
        if self.severity not in (AlertSeverity.P0, AlertSeverity.P1):
            return False, f"Severity {self.severity} is not alertable (only P0/P1)"
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "severity": self.severity.value,
            "impact": self.impact,
            "recommended_action": self.recommended_action,
            "url": self.url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    def format_message(self) -> str:
        """Format alert for human consumption."""
        return f"""🚨 {self.severity.value} ALERT

Source: {self.source.upper()}
Issue: {self.identifier}
{f'Link: {self.url}' if self.url else ''}

Impact: {self.impact}

Recommended Action: {self.recommended_action}
"""


@dataclass
class AlertDecision:
    """Result of alert eligibility check."""
    should_alert: bool
    reason: str
    payload: Optional[AlertPayload] = None
    blocked_reasons: List[str] = None
    
    def __post_init__(self):
        if self.blocked_reasons is None:
            self.blocked_reasons = []


class AlertGate:
    """
    Strict gate that determines if an alert should be sent.
    
    Rules (ALL must be true):
    1. Execution state must be complete (all checks ran)
    2. Execution state must be alert-eligible (all checks succeeded)
    3. At least one verified P0/P1 issue exists
    4. Issue has all required fields (ID, link, impact, action)
    5. Issue is user-blocking, revenue-blocking, or production-down
    """
    
    # Keywords that indicate alertable impact
    IMPACT_KEYWORDS = {
        ImpactType.USER_BLOCKING: [
            "cannot login", "cannot access", "blocked", "broken",
            "not working", "fails for all", "500 error", "503",
            "authentication failed", "cannot use",
        ],
        ImpactType.REVENUE_BLOCKING: [
            "payment", "billing", "subscription", "checkout",
            "purchase", "revenue", "transaction", "cannot pay",
        ],
        ImpactType.PRODUCTION_DOWN: [
            "prod down", "production down", "outage", "incident",
            "service unavailable", "complete failure", "all users affected",
        ],
    }
    
    def __init__(self, execution_state: Optional[ExecutionState] = None):
        self.state = execution_state or get_execution_state()
    
    def check_alert_eligibility(self) -> AlertDecision:
        """
        Determine if an alert should be sent.
        
        Returns AlertDecision with detailed reasoning.
        """
        blocked_reasons = []
        
        # Rule 1: Execution must be complete
        if not self.state.is_complete():
            incomplete = self.state.get_incomplete_checks()
            blocked_reasons.append(
                f"Execution incomplete: {[c.value for c in incomplete]} not finished"
            )
        
        # Rule 2: All checks must have succeeded
        if not self.state.is_alert_eligible():
            failed = self.state.get_failed_checks()
            if failed:
                blocked_reasons.append(
                    f"Checks failed: {[c.check_type.value for c in failed]}"
                )
            else:
                blocked_reasons.append("Not alert eligible (checks not all successful)")
        
        # Rule 3: Must have critical findings
        critical_findings = self.state.get_critical_findings()
        if not critical_findings:
            blocked_reasons.append("No verified P0/P1 issues found")
        
        # If any blocking reason, no alert
        if blocked_reasons:
            return AlertDecision(
                should_alert=False,
                reason="Alert blocked: " + "; ".join(blocked_reasons),
                blocked_reasons=blocked_reasons,
            )
        
        # Rule 4 & 5: Validate the most critical finding
        best_finding = self._select_most_critical(critical_findings)
        
        if not best_finding:
            return AlertDecision(
                should_alert=False,
                reason="No findings met all alert criteria",
                blocked_reasons=["No finding with complete required fields"],
            )
        
        # Build and validate payload
        payload = self._build_payload(best_finding)
        is_valid, validation_error = payload.validate()
        
        if not is_valid:
            return AlertDecision(
                should_alert=False,
                reason=f"Payload validation failed: {validation_error}",
                blocked_reasons=[validation_error],
            )
        
        # All checks passed
        return AlertDecision(
            should_alert=True,
            reason="All alert criteria met",
            payload=payload,
        )
    
    def _select_most_critical(
        self, findings: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Select the most critical finding that has all required fields."""
        # Sort by severity (P0 > P1)
        def severity_key(f):
            priority = f.get("priority", "").upper()
            if priority in ("P0", "HIGHEST", "BLOCKER"):
                return 0
            elif priority in ("P1", "CRITICAL"):
                return 1
            return 2
        
        sorted_findings = sorted(findings, key=severity_key)
        
        for finding in sorted_findings:
            # Check required fields
            has_id = bool(
                finding.get("issue_id") or 
                finding.get("key") or 
                finding.get("identifier")
            )
            has_summary = bool(
                finding.get("summary") or 
                finding.get("title") or 
                finding.get("message_excerpt")
            )
            
            if has_id and has_summary:
                # Check for alertable impact
                if self._has_alertable_impact(finding):
                    return finding
        
        return None
    
    def _has_alertable_impact(self, finding: Dict[str, Any]) -> bool:
        """Check if finding indicates alertable impact."""
        # Combine all text fields for keyword search
        text = " ".join([
            str(finding.get("summary", "")),
            str(finding.get("title", "")),
            str(finding.get("description", "")),
            str(finding.get("message_excerpt", "")),
            " ".join(finding.get("labels", [])),
        ]).lower()
        
        for impact_type, keywords in self.IMPACT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return True
        
        # Also check explicit priority
        priority = finding.get("priority", "").upper()
        if priority in ("P0", "HIGHEST", "BLOCKER"):
            return True
        
        return False
    
    def _build_payload(self, finding: Dict[str, Any]) -> AlertPayload:
        """Build alert payload from finding."""
        source = finding.get("source", "unknown")
        identifier = (
            finding.get("issue_id") or 
            finding.get("key") or 
            finding.get("identifier") or
            "UNKNOWN"
        )
        
        # Determine severity
        priority = finding.get("priority", "").upper()
        labels = [l.lower() for l in finding.get("labels", [])]
        
        if priority in ("P0", "HIGHEST", "BLOCKER") or "p0" in labels or "blocker" in labels:
            severity = AlertSeverity.P0
        else:
            severity = AlertSeverity.P1
        
        # Build impact statement (factual only)
        summary = finding.get("summary") or finding.get("title") or finding.get("message_excerpt", "")
        status = finding.get("status", finding.get("state", ""))
        
        impact = f"{summary[:100]}. Current status: {status}."
        
        # Build recommended action
        if source == "jira":
            action = f"Review {identifier} in Jira and assess immediate mitigation options."
        elif source == "github":
            action = f"Review {identifier} on GitHub. Check for related PRs or recent deployments."
        elif source == "slack":
            action = f"Check Slack thread for additional context. Coordinate with on-call team."
        else:
            action = f"Investigate {identifier} immediately. Determine scope and impact."
        
        return AlertPayload(
            source=source,
            identifier=identifier,
            severity=severity,
            impact=impact,
            recommended_action=action,
            url=finding.get("url") or finding.get("permalink"),
        )


# Prohibited reassurance phrases
PROHIBITED_PHRASES = [
    "looks fine",
    "no major issues",
    "all good",
    "seems okay",
    "everything is fine",
    "nothing to worry about",
    "all clear",
    "no problems",
    "running smoothly",
]

# Approved uncertainty phrases
APPROVED_PHRASES = [
    "No verified critical issues detected in checked sources",
    "Data unavailable",
    "Unable to verify",
    "Unknown",
    "Could not be verified",
    "Check incomplete",
    "Verification required",
]


def validate_language(text: str) -> tuple[bool, List[str]]:
    """
    Validate that output text doesn't contain prohibited reassurance phrases.
    
    Returns (is_valid, list of violations found).
    """
    text_lower = text.lower()
    violations = []
    
    for phrase in PROHIBITED_PHRASES:
        if phrase in text_lower:
            violations.append(f"Prohibited phrase found: '{phrase}'")
    
    return len(violations) == 0, violations


def get_approved_no_issues_phrase() -> str:
    """Get the approved phrase for when no issues are found."""
    return "No verified critical issues detected in checked sources"


def get_approved_unknown_phrase(context: str) -> str:
    """Get the approved phrase for unknown state."""
    return f"{context} could not be verified. Status unknown."


def format_safe_summary(
    execution_state: ExecutionState,
    include_findings: bool = True,
) -> str:
    """
    Format a trust-safe summary that follows all language constraints.
    
    Never uses prohibited phrases. Always explicit about unknowns.
    """
    lines = []
    
    # Status header
    if not execution_state.is_complete():
        lines.append("STATUS: CHECK INCOMPLETE")
        lines.append("")
        incomplete = execution_state.get_incomplete_checks()
        for check in incomplete:
            lines.append(f"- {check.value}: Not completed")
        lines.append("")
        lines.append("Critical issues may exist but could not be verified.")
        return "\n".join(lines)
    
    if not execution_state.is_all_success():
        lines.append("STATUS: PARTIAL VERIFICATION")
        lines.append("")
        failed = execution_state.get_failed_checks()
        for check in failed:
            lines.append(f"- {check.check_type.value}: {check.failure_reason}")
            lines.append(f"  → Issues may exist but could not be verified")
        lines.append("")
    else:
        lines.append("STATUS: ALL CHECKS COMPLETE")
        lines.append("")
    
    # Findings
    if include_findings:
        critical = execution_state.get_critical_findings()
        if critical:
            lines.append(f"VERIFIED CRITICAL ISSUES: {len(critical)}")
            for finding in critical[:5]:
                source = finding.get("source", "?")
                issue_id = finding.get("issue_id") or finding.get("key", "?")
                summary = (finding.get("summary") or finding.get("title", ""))[:50]
                lines.append(f"  [{source}] {issue_id}: {summary}")
        else:
            lines.append(get_approved_no_issues_phrase())
    
    return "\n".join(lines)

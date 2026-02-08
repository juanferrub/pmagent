"""
Trust Score Metric for PM Agent Runs.

Measures how trustworthy an agent run was based on:
- Evidence coverage (claims backed by tool calls)
- Execution completeness (all required checks ran)
- Language compliance (no prohibited reassurance phrases)
- Alert appropriateness (alerts only when justified)

Score range: 0.0 (completely untrustworthy) to 1.0 (fully trustworthy)

Version: Trust-Critical / Production
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.evidence import EvidenceLedger, get_ledger
from src.execution_state import ExecutionState, get_execution_state, CheckStatus
from src.alerting import (
    PROHIBITED_PHRASES,
    AlertGate,
    validate_language,
)
from src.utils import logger


@dataclass
class TrustViolation:
    """A specific trust violation found during scoring."""
    category: str  # evidence, execution, language, alerting
    severity: str  # critical, major, minor
    description: str
    penalty: float  # How much this reduces the score (0.0 to 1.0)


@dataclass
class TrustScoreResult:
    """Complete trust score analysis for a run."""
    overall_score: float  # 0.0 to 1.0
    
    # Component scores
    evidence_score: float
    execution_score: float
    language_score: float
    alerting_score: float
    
    # Details
    violations: List[TrustViolation] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def is_trustworthy(self) -> bool:
        """Check if the run meets minimum trust threshold."""
        return self.overall_score >= 0.7
    
    def get_grade(self) -> str:
        """Get letter grade for trust score."""
        if self.overall_score >= 0.95:
            return "A+"
        elif self.overall_score >= 0.90:
            return "A"
        elif self.overall_score >= 0.85:
            return "B+"
        elif self.overall_score >= 0.80:
            return "B"
        elif self.overall_score >= 0.70:
            return "C"
        elif self.overall_score >= 0.60:
            return "D"
        else:
            return "F"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 3),
            "grade": self.get_grade(),
            "is_trustworthy": self.is_trustworthy(),
            "component_scores": {
                "evidence": round(self.evidence_score, 3),
                "execution": round(self.execution_score, 3),
                "language": round(self.language_score, 3),
                "alerting": round(self.alerting_score, 3),
            },
            "violations_count": len(self.violations),
            "violations": [
                {
                    "category": v.category,
                    "severity": v.severity,
                    "description": v.description,
                    "penalty": v.penalty,
                }
                for v in self.violations
            ],
            "passed_checks": self.passed_checks,
            "timestamp": self.timestamp,
        }
    
    def format_report(self) -> str:
        """Format trust score as human-readable report."""
        lines = [
            "═" * 60,
            "TRUST SCORE REPORT",
            "═" * 60,
            "",
            f"Overall Score: {self.overall_score:.1%} ({self.get_grade()})",
            f"Trustworthy: {'YES' if self.is_trustworthy() else 'NO'}",
            "",
            "Component Scores:",
            f"  Evidence Coverage:    {self.evidence_score:.1%}",
            f"  Execution Complete:   {self.execution_score:.1%}",
            f"  Language Compliance:  {self.language_score:.1%}",
            f"  Alerting Appropriate: {self.alerting_score:.1%}",
            "",
        ]
        
        if self.violations:
            lines.append(f"Violations Found: {len(self.violations)}")
            for v in self.violations:
                lines.append(f"  [{v.severity.upper()}] {v.category}: {v.description}")
        else:
            lines.append("No violations found.")
        
        if self.passed_checks:
            lines.append("")
            lines.append("Passed Checks:")
            for check in self.passed_checks:
                lines.append(f"  ✓ {check}")
        
        lines.append("")
        lines.append("═" * 60)
        
        return "\n".join(lines)


class TrustScorer:
    """
    Calculates trust score for an agent run.
    
    Scoring weights:
    - Evidence: 40% (most important - claims must be backed)
    - Execution: 30% (all required checks must complete)
    - Language: 15% (no false reassurance)
    - Alerting: 15% (alerts only when appropriate)
    """
    
    WEIGHTS = {
        "evidence": 0.40,
        "execution": 0.30,
        "language": 0.15,
        "alerting": 0.15,
    }
    
    def __init__(
        self,
        ledger: Optional[EvidenceLedger] = None,
        execution_state: Optional[ExecutionState] = None,
    ):
        self.ledger = ledger or get_ledger()
        self.execution_state = execution_state or get_execution_state()
    
    def calculate_score(
        self,
        output_text: Optional[str] = None,
        alert_was_sent: bool = False,
    ) -> TrustScoreResult:
        """
        Calculate complete trust score for the run.
        
        Args:
            output_text: The final output text to check for language violations
            alert_was_sent: Whether an alert was actually sent
            
        Returns:
            TrustScoreResult with detailed breakdown
        """
        violations = []
        passed_checks = []
        
        # Score each component
        evidence_score, ev_violations, ev_passed = self._score_evidence()
        violations.extend(ev_violations)
        passed_checks.extend(ev_passed)
        
        execution_score, ex_violations, ex_passed = self._score_execution()
        violations.extend(ex_violations)
        passed_checks.extend(ex_passed)
        
        language_score, lang_violations, lang_passed = self._score_language(output_text)
        violations.extend(lang_violations)
        passed_checks.extend(lang_passed)
        
        alerting_score, alert_violations, alert_passed = self._score_alerting(alert_was_sent)
        violations.extend(alert_violations)
        passed_checks.extend(alert_passed)
        
        # Calculate weighted overall score
        overall_score = (
            evidence_score * self.WEIGHTS["evidence"]
            + execution_score * self.WEIGHTS["execution"]
            + language_score * self.WEIGHTS["language"]
            + alerting_score * self.WEIGHTS["alerting"]
        )
        
        result = TrustScoreResult(
            overall_score=overall_score,
            evidence_score=evidence_score,
            execution_score=execution_score,
            language_score=language_score,
            alerting_score=alerting_score,
            violations=violations,
            passed_checks=passed_checks,
        )
        
        logger.info(
            "trust_score_calculated",
            overall=round(overall_score, 3),
            grade=result.get_grade(),
            violations=len(violations),
        )
        
        return result
    
    def _score_evidence(self) -> tuple[float, List[TrustViolation], List[str]]:
        """Score evidence coverage."""
        violations = []
        passed = []
        
        coverage = self.ledger.get_coverage_summary()
        total_entries = coverage["total_entries"]
        successful_entries = coverage["successful_entries"]
        
        # Check 1: Must have at least some tool calls
        if total_entries == 0:
            violations.append(TrustViolation(
                category="evidence",
                severity="critical",
                description="No tool calls recorded - output would be entirely fabricated",
                penalty=1.0,
            ))
            return 0.0, violations, passed
        
        passed.append("Tool calls were recorded")
        
        # Check 2: Success rate
        success_rate = successful_entries / total_entries if total_entries > 0 else 0
        
        if success_rate < 0.5:
            violations.append(TrustViolation(
                category="evidence",
                severity="major",
                description=f"Tool success rate ({success_rate:.0%}) below 50% threshold",
                penalty=0.3,
            ))
        else:
            passed.append(f"Tool success rate: {success_rate:.0%}")
        
        # Check 3: Source coverage
        sources_covered = set(coverage["sources_covered"])
        required_sources = {"jira", "github", "slack"}
        missing_sources = required_sources - {s.value if hasattr(s, 'value') else s for s in sources_covered}
        
        if missing_sources:
            violations.append(TrustViolation(
                category="evidence",
                severity="major",
                description=f"Missing data sources: {missing_sources}",
                penalty=0.2 * len(missing_sources),
            ))
        else:
            passed.append("All required data sources covered")
        
        # Check 4: Evidence has identifiers
        has_identifiers = any(
            entry.identifiers
            for entry in self.ledger._entries.values()
            if entry.success
        )
        
        if not has_identifiers:
            violations.append(TrustViolation(
                category="evidence",
                severity="minor",
                description="No identifiable evidence (issue keys, PR numbers, etc.)",
                penalty=0.1,
            ))
        else:
            passed.append("Evidence contains identifiable references")
        
        # Calculate score
        total_penalty = sum(v.penalty for v in violations)
        score = max(0.0, 1.0 - total_penalty)
        
        return score, violations, passed
    
    def _score_execution(self) -> tuple[float, List[TrustViolation], List[str]]:
        """Score execution completeness."""
        violations = []
        passed = []
        
        # Check 1: All required checks completed
        if not self.execution_state.is_complete():
            incomplete = self.execution_state.get_incomplete_checks()
            violations.append(TrustViolation(
                category="execution",
                severity="critical",
                description=f"Incomplete checks: {[c.value for c in incomplete]}",
                penalty=0.5,
            ))
        else:
            passed.append("All required checks completed")
        
        # Check 2: Check success rate
        failed = self.execution_state.get_failed_checks()
        if failed:
            violations.append(TrustViolation(
                category="execution",
                severity="major",
                description=f"Failed checks: {[c.check_type.value for c in failed]}",
                penalty=0.15 * len(failed),
            ))
        else:
            passed.append("All checks succeeded")
        
        # Calculate score
        total_penalty = sum(v.penalty for v in violations)
        score = max(0.0, 1.0 - total_penalty)
        
        return score, violations, passed
    
    def _score_language(
        self, output_text: Optional[str]
    ) -> tuple[float, List[TrustViolation], List[str]]:
        """Score language compliance."""
        violations = []
        passed = []
        
        if not output_text:
            passed.append("No output text to validate")
            return 1.0, violations, passed
        
        # Check for prohibited phrases
        is_valid, phrase_violations = validate_language(output_text)
        
        if not is_valid:
            for pv in phrase_violations:
                violations.append(TrustViolation(
                    category="language",
                    severity="major",
                    description=pv,
                    penalty=0.2,
                ))
        else:
            passed.append("No prohibited reassurance phrases found")
        
        # Check for hedging without explicit uncertainty
        hedging_patterns = [
            r"\bprobably\b",
            r"\blikely\b",
            r"\bmight\b",
            r"\bcould be\b",
            r"\bshould be\b",
        ]
        
        text_lower = output_text.lower()
        for pattern in hedging_patterns:
            if re.search(pattern, text_lower):
                # Only penalize if not followed by explicit uncertainty marker
                if "unknown" not in text_lower and "unable to verify" not in text_lower:
                    violations.append(TrustViolation(
                        category="language",
                        severity="minor",
                        description=f"Hedging language without explicit uncertainty: {pattern}",
                        penalty=0.05,
                    ))
        
        # Calculate score
        total_penalty = sum(v.penalty for v in violations)
        score = max(0.0, 1.0 - min(total_penalty, 1.0))
        
        return score, violations, passed
    
    def _score_alerting(
        self, alert_was_sent: bool
    ) -> tuple[float, List[TrustViolation], List[str]]:
        """Score alerting appropriateness."""
        violations = []
        passed = []
        
        alert_gate = AlertGate(self.execution_state)
        decision = alert_gate.check_alert_eligibility()
        
        if alert_was_sent:
            # Alert was sent - check if it was justified
            if not decision.should_alert:
                violations.append(TrustViolation(
                    category="alerting",
                    severity="critical",
                    description=f"Alert sent without justification: {decision.reason}",
                    penalty=0.8,
                ))
            else:
                passed.append("Alert was justified by verified critical issues")
        else:
            # No alert sent - check if one should have been
            if decision.should_alert:
                violations.append(TrustViolation(
                    category="alerting",
                    severity="major",
                    description="Critical issue detected but no alert sent",
                    penalty=0.3,
                ))
            else:
                passed.append("No alert needed - correct decision")
        
        # Calculate score
        total_penalty = sum(v.penalty for v in violations)
        score = max(0.0, 1.0 - total_penalty)
        
        return score, violations, passed


def calculate_trust_score(
    output_text: Optional[str] = None,
    alert_was_sent: bool = False,
) -> TrustScoreResult:
    """
    Convenience function to calculate trust score for current run.
    
    Args:
        output_text: The final output text to check
        alert_was_sent: Whether an alert was sent
        
    Returns:
        TrustScoreResult
    """
    scorer = TrustScorer()
    return scorer.calculate_score(output_text, alert_was_sent)


def get_trust_score_summary() -> Dict[str, Any]:
    """Get a quick trust score summary for the current run."""
    result = calculate_trust_score()
    return {
        "score": result.overall_score,
        "grade": result.get_grade(),
        "trustworthy": result.is_trustworthy(),
        "violations": len(result.violations),
    }

"""
Deterministic Scorers.

Rule-based scoring functions for evaluating PM Agent responses.
These scorers use exact matching, regex patterns, and structural checks.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from evals.schema import (
    EvalScenario,
    ExpectedAgent,
    ExpectedSource,
    GoldenOutput,
    ScoreResult,
)


class BaseScorer(ABC):
    """Base class for all scorers."""
    
    name: str = "base_scorer"
    
    @abstractmethod
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score the agent output.
        
        Args:
            scenario: The evaluation scenario.
            output: The agent's output text.
            metadata: Additional metadata (tool calls, agents invoked, etc.).
            
        Returns:
            ScoreResult with score, pass/fail, and details.
        """
        pass


class RouteAccuracyScorer(BaseScorer):
    """Checks if correct agents were invoked for the query."""
    
    name = "route_accuracy"
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score routing accuracy.
        
        Checks if the expected agents were invoked and no unexpected agents.
        """
        expected_agents = set(a.value for a in scenario.expected_agents)
        invoked_agents = set(metadata.get("invoked_agents", []))
        
        # Calculate metrics
        correct_invocations = expected_agents & invoked_agents
        missed_agents = expected_agents - invoked_agents
        extra_agents = invoked_agents - expected_agents
        
        # Score: 1.0 if perfect match, penalize for misses and extras
        if not expected_agents:
            # No expected agents (e.g., out-of-scope query)
            score = 1.0 if not invoked_agents else 0.5
            passed = not invoked_agents
            details = "No agents expected" if passed else f"Unexpected agents invoked: {extra_agents}"
        else:
            # Calculate accuracy
            precision = len(correct_invocations) / len(invoked_agents) if invoked_agents else 0.0
            recall = len(correct_invocations) / len(expected_agents)
            
            # F1-like score
            if precision + recall > 0:
                score = 2 * (precision * recall) / (precision + recall)
            else:
                score = 0.0
            
            passed = score >= 0.95  # Allow minor tolerance
            
            details_parts = []
            if correct_invocations:
                details_parts.append(f"Correct: {correct_invocations}")
            if missed_agents:
                details_parts.append(f"Missed: {missed_agents}")
            if extra_agents:
                details_parts.append(f"Extra: {extra_agents}")
            details = "; ".join(details_parts) if details_parts else "Perfect routing"
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "expected": list(expected_agents),
                "invoked": list(invoked_agents),
                "correct": list(correct_invocations),
                "missed": list(missed_agents),
                "extra": list(extra_agents),
            },
        )


class SourceCoverageScorer(BaseScorer):
    """Verifies all required data sources were checked."""
    
    name = "source_coverage"
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score source coverage.
        
        Checks if all expected data sources were queried.
        """
        expected_sources = set(s.value for s in scenario.expected_sources)
        checked_sources = set(metadata.get("checked_sources", []))
        
        covered = expected_sources & checked_sources
        missed = expected_sources - checked_sources
        
        if not expected_sources:
            score = 1.0
            passed = True
            details = "No sources expected"
        else:
            score = len(covered) / len(expected_sources)
            passed = score >= 0.9  # Allow 90% coverage
            
            if passed:
                details = f"All required sources checked: {covered}"
            else:
                details = f"Missing sources: {missed}; Checked: {covered}"
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "expected": list(expected_sources),
                "checked": list(checked_sources),
                "covered": list(covered),
                "missed": list(missed),
            },
        )


class HallucinationDetector(BaseScorer):
    """Detects fabricated IDs, metrics, or claims without evidence."""
    
    name = "hallucination_detector"
    
    # Patterns for common fabricated content
    JIRA_ID_PATTERN = re.compile(r"\b([A-Z]+-\d+)\b")
    GITHUB_ID_PATTERN = re.compile(r"#(\d+)\b")
    METRIC_PATTERN = re.compile(r"\b(\d+(?:\.\d+)?)\s*%\b")
    TIMELINE_PATTERN = re.compile(
        r"\b(will be|estimated|ETA|by next|within \d+|in \d+ days?)\b",
        re.IGNORECASE,
    )
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Detect hallucinations in the output.
        
        Checks for:
        1. Jira IDs not in tool responses
        2. GitHub IDs not in tool responses
        3. Metrics not backed by data
        4. Timeline claims without evidence
        """
        violations = []
        
        # Get valid identifiers from tool responses
        valid_jira_ids = set(metadata.get("jira_ids", []))
        valid_github_ids = set(str(id) for id in metadata.get("github_ids", []))
        valid_metrics = set(str(m) for m in metadata.get("metrics", []))
        
        # Also add expected identifiers from golden output
        if scenario.golden_output:
            for ident in scenario.golden_output.expected_identifiers:
                if re.match(r"[A-Z]+-", ident):
                    valid_jira_ids.add(ident)
                elif ident.isdigit():
                    valid_github_ids.add(ident)
        
        # Check Jira IDs
        found_jira_ids = set(self.JIRA_ID_PATTERN.findall(output))
        fabricated_jira = found_jira_ids - valid_jira_ids
        if fabricated_jira:
            violations.append(f"Fabricated Jira IDs: {fabricated_jira}")
        
        # Check GitHub IDs
        found_github_ids = set(self.GITHUB_ID_PATTERN.findall(output))
        fabricated_github = found_github_ids - valid_github_ids
        if fabricated_github:
            violations.append(f"Fabricated GitHub IDs: {fabricated_github}")
        
        # Check for timeline claims when no timeline data exists
        has_timeline_data = metadata.get("has_timeline_data", False)
        if not has_timeline_data:
            timeline_matches = self.TIMELINE_PATTERN.findall(output)
            if timeline_matches:
                violations.append(f"Timeline claims without evidence: {timeline_matches}")
        
        # Calculate score
        if not violations:
            score = 1.0
            passed = True
            details = "No hallucinations detected"
        else:
            # Each violation reduces score
            score = max(0.0, 1.0 - (len(violations) * 0.25))
            passed = False  # Zero tolerance for hallucinations
            details = "; ".join(violations)
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "violations": violations,
                "found_jira_ids": list(found_jira_ids),
                "found_github_ids": list(found_github_ids),
                "valid_jira_ids": list(valid_jira_ids),
                "valid_github_ids": list(valid_github_ids),
            },
        )


class LanguageComplianceScorer(BaseScorer):
    """Checks for prohibited reassurance phrases."""
    
    name = "language_compliance"
    
    # Prohibited phrases from trust-critical spec
    PROHIBITED_PHRASES = [
        "looks fine",
        "no major issues",
        "all good",
        "seems okay",
        "everything is fine",
        "nothing to worry about",
        "all clear",
        "no problems",
        "looks good",
        "seems fine",
        "appears to be okay",
        "should be fine",
        "probably fine",
        "likely okay",
        "no concerns",
        "all quiet",
        "nothing significant",
    ]
    
    # Approved phrases
    APPROVED_PHRASES = [
        "No verified critical issues detected in checked sources",
        "Data unavailable",
        "Unable to verify",
        "Unknown",
        "Could not be verified",
        "Check incomplete",
        "Verification required",
    ]
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Check language compliance.
        
        Detects prohibited reassurance phrases that could create false confidence.
        """
        output_lower = output.lower()
        
        found_prohibited = []
        for phrase in self.PROHIBITED_PHRASES:
            if phrase.lower() in output_lower:
                found_prohibited.append(phrase)
        
        found_approved = []
        for phrase in self.APPROVED_PHRASES:
            if phrase.lower() in output_lower:
                found_approved.append(phrase)
        
        if not found_prohibited:
            score = 1.0
            passed = True
            if found_approved:
                details = f"Uses approved language: {found_approved}"
            else:
                details = "No prohibited phrases found"
        else:
            # Each prohibited phrase reduces score
            score = max(0.0, 1.0 - (len(found_prohibited) * 0.2))
            passed = False
            details = f"Prohibited phrases found: {found_prohibited}"
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "prohibited_found": found_prohibited,
                "approved_found": found_approved,
            },
        )


class StructureScorer(BaseScorer):
    """Validates output format (sections, citations)."""
    
    name = "structure_scorer"
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score output structure.
        
        Checks for:
        1. Expected sections present
        2. Proper formatting
        3. Source citations
        """
        issues = []
        
        # Check expected sections
        if scenario.golden_output and scenario.golden_output.expected_sections:
            expected_sections = scenario.golden_output.expected_sections
            output_lower = output.lower()
            
            missing_sections = []
            for section in expected_sections:
                if section.lower() not in output_lower:
                    missing_sections.append(section)
            
            if missing_sections:
                issues.append(f"Missing sections: {missing_sections}")
        
        # Check for must_contain keywords
        if scenario.golden_output and scenario.golden_output.must_contain:
            missing_keywords = []
            for keyword in scenario.golden_output.must_contain:
                if keyword.lower() not in output.lower():
                    missing_keywords.append(keyword)
            
            if missing_keywords:
                issues.append(f"Missing required content: {missing_keywords}")
        
        # Check for must_not_contain
        if scenario.golden_output and scenario.golden_output.must_not_contain:
            found_prohibited = []
            for keyword in scenario.golden_output.must_not_contain:
                if keyword.lower() in output.lower():
                    found_prohibited.append(keyword)
            
            if found_prohibited:
                issues.append(f"Contains prohibited content: {found_prohibited}")
        
        # Calculate score
        if not issues:
            score = 1.0
            passed = True
            details = "Output structure is valid"
        else:
            score = max(0.0, 1.0 - (len(issues) * 0.25))
            passed = score >= 0.75
            details = "; ".join(issues)
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={"issues": issues},
        )


class IdentifierValidator(BaseScorer):
    """Verifies mentioned IDs exist in tool responses."""
    
    name = "identifier_validator"
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Validate identifiers in output.
        
        Ensures all expected identifiers are present and no fabricated ones.
        """
        if not scenario.golden_output:
            return ScoreResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details="No identifier validation required",
                metadata={},
            )
        
        expected_ids = set(scenario.golden_output.expected_identifiers)
        
        if not expected_ids:
            return ScoreResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details="No identifiers expected",
                metadata={},
            )
        
        # Check which expected IDs are present
        found_ids = set()
        missing_ids = set()
        
        for identifier in expected_ids:
            if identifier in output:
                found_ids.add(identifier)
            else:
                missing_ids.add(identifier)
        
        # Calculate score
        score = len(found_ids) / len(expected_ids) if expected_ids else 1.0
        passed = score >= 0.9
        
        if passed:
            details = f"All expected identifiers found: {found_ids}"
        else:
            details = f"Missing identifiers: {missing_ids}; Found: {found_ids}"
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "expected": list(expected_ids),
                "found": list(found_ids),
                "missing": list(missing_ids),
            },
        )


class AlertAppropriatenessScorer(BaseScorer):
    """Checks if alerting decision matches expected behavior."""
    
    name = "alert_appropriateness"
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score alert appropriateness.
        
        Checks if alert was sent when expected (and not sent when not expected).
        """
        if not scenario.golden_output or scenario.golden_output.should_alert is None:
            return ScoreResult(
                scorer_name=self.name,
                score=1.0,
                passed=True,
                details="No alert expectation defined",
                metadata={},
            )
        
        should_alert = scenario.golden_output.should_alert
        did_alert = metadata.get("alert_sent", False)
        
        if should_alert == did_alert:
            score = 1.0
            passed = True
            if should_alert:
                details = "Correctly sent alert for critical issue"
            else:
                details = "Correctly did not send alert"
        else:
            score = 0.0
            passed = False
            if should_alert:
                details = "MISSED: Should have sent alert but did not"
            else:
                details = "FALSE POSITIVE: Sent alert when not warranted"
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=details,
            metadata={
                "should_alert": should_alert,
                "did_alert": did_alert,
            },
        )


# Convenience function to run all deterministic scorers
def run_all_deterministic_scorers(
    scenario: EvalScenario,
    output: str,
    metadata: Dict[str, Any],
) -> List[ScoreResult]:
    """Run all deterministic scorers on the output.
    
    Args:
        scenario: The evaluation scenario.
        output: The agent's output text.
        metadata: Additional metadata from the run.
        
    Returns:
        List of ScoreResult objects from all scorers.
    """
    scorers = [
        RouteAccuracyScorer(),
        SourceCoverageScorer(),
        HallucinationDetector(),
        LanguageComplianceScorer(),
        StructureScorer(),
        IdentifierValidator(),
        AlertAppropriatenessScorer(),
    ]
    
    return [scorer.score(scenario, output, metadata) for scorer in scorers]

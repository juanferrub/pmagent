"""
Trust Scorer Wrapper.

Wrapper around the existing trust score system for use in evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from evals.schema import EvalScenario, ScoreResult


class TrustScorerWrapper:
    """Wrapper around the existing TrustScorer for evaluation use."""
    
    name = "trust_score"
    
    def __init__(self):
        """Initialize the trust scorer wrapper."""
        self._trust_scorer = None
    
    @property
    def trust_scorer(self):
        """Lazy-load the trust scorer."""
        if self._trust_scorer is None:
            try:
                from src.trust_score import TrustScorer
                self._trust_scorer = TrustScorer
            except ImportError:
                self._trust_scorer = None
        return self._trust_scorer
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Calculate trust score for the output.
        
        Uses the existing TrustScorer from src/trust_score.py.
        Falls back to a simple heuristic-based score if the module isn't available.
        
        Args:
            scenario: The evaluation scenario.
            output: The agent's output text.
            metadata: Additional metadata including execution state.
            
        Returns:
            ScoreResult with trust score and breakdown.
        """
        # Check if trust scorer is available
        if self.trust_scorer is None:
            # Fall back to simple heuristic scoring
            return self._fallback_score(scenario, output, metadata)
        
        try:
            # Get execution state from metadata if available
            execution_state = metadata.get("execution_state")
            evidence_ledger = metadata.get("evidence_ledger")
            alert_sent = metadata.get("alert_sent", False)
            
            # Create trust scorer instance
            # TrustScorer expects 'ledger' not 'evidence_ledger'
            scorer = self.trust_scorer(
                ledger=evidence_ledger,
            )
            
            # Calculate score
            result = scorer.calculate_score(
                output_text=output,
                alert_was_sent=alert_sent,
            )
            
            # Check against minimum threshold if specified
            min_score = None
            if scenario.golden_output and scenario.golden_output.min_trust_score:
                min_score = scenario.golden_output.min_trust_score
            
            passed = result.is_trustworthy()
            if min_score is not None:
                passed = passed and result.overall_score >= min_score
            
            return ScoreResult(
                scorer_name=self.name,
                score=result.overall_score,
                passed=passed,
                details=f"Grade: {result.get_grade()}; {len(result.violations)} violations",
                metadata={
                    "overall_score": result.overall_score,
                    "evidence_score": result.evidence_score,
                    "execution_score": result.execution_score,
                    "language_score": result.language_score,
                    "alerting_score": result.alerting_score,
                    "grade": result.get_grade(),
                    "violations": [
                        {"category": v.category, "description": v.description}
                        for v in result.violations
                    ],
                    "min_threshold": min_score,
                },
            )
        except Exception as e:
            return ScoreResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                details=f"Trust score calculation failed: {str(e)}",
                metadata={"error": str(e)},
            )


    def _fallback_score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Fallback scoring when TrustScorer module isn't available.
        
        Uses simple heuristics based on:
        - Language compliance (no prohibited phrases)
        - Identifier presence (expected IDs found)
        - Structure (expected sections present)
        """
        score = 1.0
        violations = []
        
        output_lower = output.lower()
        
        # Check for prohibited phrases
        prohibited = [
            "looks fine", "no major issues", "all good", "seems okay",
            "everything is fine", "all clear", "no problems",
        ]
        for phrase in prohibited:
            if phrase in output_lower:
                score -= 0.15
                violations.append(f"Prohibited phrase: '{phrase}'")
        
        # Check for expected identifiers
        if scenario.golden_output and scenario.golden_output.expected_identifiers:
            found = 0
            for ident in scenario.golden_output.expected_identifiers:
                if ident in output:
                    found += 1
            if scenario.golden_output.expected_identifiers:
                id_ratio = found / len(scenario.golden_output.expected_identifiers)
                if id_ratio < 1.0:
                    score -= (1.0 - id_ratio) * 0.2
                    if id_ratio < 0.5:
                        violations.append(f"Missing identifiers: {id_ratio:.0%} found")
        
        # Check for must_not_contain violations
        if scenario.golden_output and scenario.golden_output.must_not_contain:
            for forbidden in scenario.golden_output.must_not_contain:
                if forbidden.lower() in output_lower:
                    score -= 0.1
                    violations.append(f"Contains forbidden content: '{forbidden}'")
        
        # Ensure score stays in valid range
        score = max(0.0, min(1.0, score))
        
        # Determine pass/fail
        min_score = 0.7  # Default threshold
        if scenario.golden_output and scenario.golden_output.min_trust_score:
            min_score = scenario.golden_output.min_trust_score
        
        passed = score >= min_score and len(violations) == 0
        
        return ScoreResult(
            scorer_name=self.name,
            score=score,
            passed=passed,
            details=f"Fallback score: {score:.2f}; {len(violations)} issues" if violations else f"Fallback score: {score:.2f}",
            metadata={
                "fallback_mode": True,
                "violations": violations,
                "min_threshold": min_score,
            },
        )


def calculate_trust_score(
    scenario: EvalScenario,
    output: str,
    metadata: Dict[str, Any],
) -> ScoreResult:
    """Convenience function to calculate trust score.
    
    Args:
        scenario: The evaluation scenario.
        output: The agent's output text.
        metadata: Additional metadata.
        
    Returns:
        ScoreResult with trust score.
    """
    wrapper = TrustScorerWrapper()
    return wrapper.score(scenario, output, metadata)

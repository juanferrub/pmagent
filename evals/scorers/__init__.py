"""
Evaluation Scorers.

This module provides scoring functions for evaluating PM Agent responses:
- Deterministic scorers: Rule-based checks (route accuracy, hallucination, language)
- LLM-as-judge scorers: Quality scoring using LLM evaluation
- Trust scorer: Wrapper around the existing trust score system
"""

from evals.scorers.deterministic import (
    RouteAccuracyScorer,
    SourceCoverageScorer,
    HallucinationDetector,
    LanguageComplianceScorer,
    StructureScorer,
    IdentifierValidator,
)
from evals.scorers.llm_judge import (
    RelevanceScorer,
    CompletenessScorer,
    AccuracyScorer,
    ClarityScorer,
    ActionabilityScorer,
)
from evals.scorers.trust_scorer import TrustScorerWrapper

__all__ = [
    # Deterministic
    "RouteAccuracyScorer",
    "SourceCoverageScorer",
    "HallucinationDetector",
    "LanguageComplianceScorer",
    "StructureScorer",
    "IdentifierValidator",
    # LLM-as-judge
    "RelevanceScorer",
    "CompletenessScorer",
    "AccuracyScorer",
    "ClarityScorer",
    "ActionabilityScorer",
    # Trust
    "TrustScorerWrapper",
]

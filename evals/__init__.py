"""
PM Agent Evaluation Suite.

A comprehensive evaluation framework for testing regression, quality benchmarks,
and trust-critical validation across all agent scenarios.

Features:
- 43 test scenarios across 6 categories
- Deterministic scorers (route accuracy, hallucination detection, language compliance)
- LLM-as-judge scorers (relevance, completeness, accuracy, clarity, actionability)
- Trust score integration
- Opik integration for tracking and visualization

Usage:
    # Run all evaluations
    python -m evals --all
    
    # Run specific category
    python -m evals --category critical-issues
    
    # Run single scenario
    python -m evals --scenario daily_digest_001
    
    # List scenarios
    python -m evals --list
"""

__version__ = "1.0.0"

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "EvaluationRunner":
        from evals.runner import EvaluationRunner
        return EvaluationRunner
    elif name == "run_evaluation":
        from evals.runner import run_evaluation
        return run_evaluation
    elif name == "EvaluationReport":
        from evals.report import EvaluationReport
        return EvaluationReport
    elif name == "generate_report":
        from evals.report import generate_report
        return generate_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "EvaluationRunner",
    "run_evaluation",
    "EvaluationReport",
    "generate_report",
]

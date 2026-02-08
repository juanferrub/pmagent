"""
Evaluation Report Generation.

Generates reports from evaluation results for analysis and comparison.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from evals.schema import EvalResult, EvalSummary, ScenarioCategory


logger = structlog.get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing two evaluation runs."""
    
    baseline_name: str
    comparison_name: str
    baseline_summary: EvalSummary
    comparison_summary: EvalSummary
    
    # Deltas
    pass_rate_delta: float = 0.0
    trust_score_delta: float = 0.0
    latency_delta: float = 0.0
    
    # Category-level changes
    category_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Regressions and improvements
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    
    def is_regression(self) -> bool:
        """Check if this comparison shows a regression."""
        return (
            self.pass_rate_delta < -0.05 or  # 5% drop in pass rate
            self.trust_score_delta < -0.05 or  # 5% drop in trust score
            len(self.regressions) > 0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline": self.baseline_name,
            "comparison": self.comparison_name,
            "pass_rate_delta": self.pass_rate_delta,
            "trust_score_delta": self.trust_score_delta,
            "latency_delta": self.latency_delta,
            "category_changes": self.category_changes,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "is_regression": self.is_regression(),
        }


class EvaluationReport:
    """Generates and formats evaluation reports."""
    
    def __init__(
        self,
        results: List[EvalResult],
        summary: Optional[EvalSummary] = None,
    ):
        """Initialize report generator.
        
        Args:
            results: List of evaluation results.
            summary: Optional pre-computed summary.
        """
        self.results = results
        self._summary = summary
    
    @property
    def summary(self) -> EvalSummary:
        """Get or compute summary."""
        if self._summary is None:
            self._summary = self._compute_summary()
        return self._summary
    
    def _compute_summary(self) -> EvalSummary:
        """Compute summary from results."""
        if not self.results:
            return EvalSummary(
                total_scenarios=0,
                passed=0,
                failed=0,
                errors=0,
                pass_rate=0.0,
                avg_trust_score=0.0,
                avg_latency_ms=0.0,
                by_category={},
            )
        
        passed = sum(1 for r in self.results if r.overall_passed and not r.error)
        failed = sum(1 for r in self.results if not r.overall_passed and not r.error)
        errors = sum(1 for r in self.results if r.error)
        
        trust_scores = [r.trust_score for r in self.results if r.trust_score is not None]
        latencies = [r.latency_ms for r in self.results if r.latency_ms is not None]
        
        by_category = {}
        for category in ScenarioCategory:
            cat_results = [r for r in self.results if r.category == category]
            if cat_results:
                cat_passed = sum(1 for r in cat_results if r.overall_passed)
                cat_trust = [r.trust_score for r in cat_results if r.trust_score]
                by_category[category.value] = {
                    "total": len(cat_results),
                    "passed": cat_passed,
                    "pass_rate": cat_passed / len(cat_results),
                    "avg_trust_score": sum(cat_trust) / len(cat_trust) if cat_trust else 0.0,
                }
        
        return EvalSummary(
            total_scenarios=len(self.results),
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=passed / len(self.results) if self.results else 0.0,
            avg_trust_score=sum(trust_scores) / len(trust_scores) if trust_scores else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            by_category=by_category,
        )
    
    def generate_markdown(self) -> str:
        """Generate markdown report.
        
        Returns:
            Markdown formatted report string.
        """
        lines = []
        
        # Header
        lines.append("# PM Agent Evaluation Report")
        lines.append(f"\n**Generated:** {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        
        # Overall Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Scenarios | {self.summary.total_scenarios} |")
        lines.append(f"| Passed | {self.summary.passed} |")
        lines.append(f"| Failed | {self.summary.failed} |")
        lines.append(f"| Errors | {self.summary.errors} |")
        lines.append(f"| Pass Rate | {self.summary.pass_rate:.1%} |")
        lines.append(f"| Avg Trust Score | {self.summary.avg_trust_score:.3f} |")
        lines.append(f"| Avg Latency | {self.summary.avg_latency_ms:.0f}ms |")
        lines.append("")
        
        # Thresholds check
        lines.append("## Threshold Check")
        lines.append("")
        thresholds = [
            ("Route Accuracy", 0.95, self._get_avg_score("route_accuracy")),
            ("Source Coverage", 0.90, self._get_avg_score("source_coverage")),
            ("Hallucination Detection", 1.00, self._get_avg_score("hallucination_detector")),
            ("Trust Score", 0.80, self.summary.avg_trust_score),
            ("LLM Relevance", 4.0, self._get_avg_score("llm_relevance")),
            ("LLM Completeness", 4.0, self._get_avg_score("llm_completeness")),
        ]
        
        lines.append("| Metric | Threshold | Actual | Status |")
        lines.append("|--------|-----------|--------|--------|")
        for name, threshold, actual in thresholds:
            status = "✅" if actual >= threshold else "❌"
            lines.append(f"| {name} | {threshold:.2f} | {actual:.2f} | {status} |")
        lines.append("")
        
        # By Category
        lines.append("## Results by Category")
        lines.append("")
        lines.append("| Category | Total | Passed | Pass Rate | Avg Trust |")
        lines.append("|----------|-------|--------|-----------|-----------|")
        for cat_name, cat_data in self.summary.by_category.items():
            lines.append(
                f"| {cat_name} | {cat_data['total']} | {cat_data['passed']} | "
                f"{cat_data['pass_rate']:.1%} | {cat_data['avg_trust_score']:.3f} |"
            )
        lines.append("")
        
        # Failed Scenarios
        failed_results = [r for r in self.results if not r.overall_passed]
        if failed_results:
            lines.append("## Failed Scenarios")
            lines.append("")
            for result in failed_results[:10]:  # Limit to 10
                lines.append(f"### {result.scenario_id}: {result.scenario_name}")
                lines.append(f"- **Category:** {result.category.value}")
                lines.append(f"- **Query:** {result.query[:100]}...")
                if result.error:
                    lines.append(f"- **Error:** {result.error}")
                else:
                    failed_scores = [s for s in result.scores if not s.passed]
                    for score in failed_scores:
                        lines.append(f"- **{score.scorer_name}:** {score.details}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _get_avg_score(self, scorer_name: str) -> float:
        """Get average score for a specific scorer."""
        scores = []
        for result in self.results:
            for score in result.scores:
                if score.scorer_name == scorer_name:
                    scores.append(score.score)
        return sum(scores) / len(scores) if scores else 0.0
    
    def generate_json(self) -> Dict[str, Any]:
        """Generate JSON report.
        
        Returns:
            Dictionary with full report data.
        """
        return {
            "summary": self.summary.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def save(self, output_dir: Path, format: str = "both") -> List[Path]:
        """Save report to files.
        
        Args:
            output_dir: Directory to save reports.
            format: "json", "markdown", or "both".
            
        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        saved_files = []
        
        if format in ("json", "both"):
            json_path = output_dir / f"report_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(self.generate_json(), f, indent=2)
            saved_files.append(json_path)
        
        if format in ("markdown", "both"):
            md_path = output_dir / f"report_{timestamp}.md"
            with open(md_path, "w") as f:
                f.write(self.generate_markdown())
            saved_files.append(md_path)
        
        logger.info("report_saved", files=[str(f) for f in saved_files])
        return saved_files


def compare_runs(
    baseline_results: List[EvalResult],
    comparison_results: List[EvalResult],
    baseline_name: str = "baseline",
    comparison_name: str = "comparison",
) -> ComparisonResult:
    """Compare two evaluation runs.
    
    Args:
        baseline_results: Results from baseline run.
        comparison_results: Results from comparison run.
        baseline_name: Name for baseline run.
        comparison_name: Name for comparison run.
        
    Returns:
        ComparisonResult with deltas and analysis.
    """
    baseline_report = EvaluationReport(baseline_results)
    comparison_report = EvaluationReport(comparison_results)
    
    baseline_summary = baseline_report.summary
    comparison_summary = comparison_report.summary
    
    # Calculate deltas
    pass_rate_delta = comparison_summary.pass_rate - baseline_summary.pass_rate
    trust_score_delta = comparison_summary.avg_trust_score - baseline_summary.avg_trust_score
    latency_delta = comparison_summary.avg_latency_ms - baseline_summary.avg_latency_ms
    
    # Category-level changes
    category_changes = {}
    for cat_name in set(baseline_summary.by_category.keys()) | set(comparison_summary.by_category.keys()):
        baseline_cat = baseline_summary.by_category.get(cat_name, {})
        comparison_cat = comparison_summary.by_category.get(cat_name, {})
        
        category_changes[cat_name] = {
            "pass_rate_delta": (
                comparison_cat.get("pass_rate", 0) - baseline_cat.get("pass_rate", 0)
            ),
            "trust_delta": (
                comparison_cat.get("avg_trust_score", 0) - baseline_cat.get("avg_trust_score", 0)
            ),
        }
    
    # Identify regressions and improvements
    regressions = []
    improvements = []
    
    # Compare individual scenarios
    baseline_by_id = {r.scenario_id: r for r in baseline_results}
    comparison_by_id = {r.scenario_id: r for r in comparison_results}
    
    for scenario_id in set(baseline_by_id.keys()) & set(comparison_by_id.keys()):
        baseline_result = baseline_by_id[scenario_id]
        comparison_result = comparison_by_id[scenario_id]
        
        if baseline_result.overall_passed and not comparison_result.overall_passed:
            regressions.append(f"{scenario_id}: was passing, now failing")
        elif not baseline_result.overall_passed and comparison_result.overall_passed:
            improvements.append(f"{scenario_id}: was failing, now passing")
        
        # Check trust score regression
        if baseline_result.trust_score and comparison_result.trust_score:
            trust_diff = comparison_result.trust_score - baseline_result.trust_score
            if trust_diff < -0.1:  # 10% drop
                regressions.append(
                    f"{scenario_id}: trust score dropped {trust_diff:.2f}"
                )
    
    return ComparisonResult(
        baseline_name=baseline_name,
        comparison_name=comparison_name,
        baseline_summary=baseline_summary,
        comparison_summary=comparison_summary,
        pass_rate_delta=pass_rate_delta,
        trust_score_delta=trust_score_delta,
        latency_delta=latency_delta,
        category_changes=category_changes,
        regressions=regressions,
        improvements=improvements,
    )


def load_results_from_file(filepath: Path) -> List[EvalResult]:
    """Load evaluation results from a JSON file.
    
    Args:
        filepath: Path to the results JSON file.
        
    Returns:
        List of EvalResult objects.
    """
    from evals.schema import ScoreResult
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    results = []
    for item in data:
        scores = [
            ScoreResult(
                scorer_name=s["scorer_name"],
                score=s["score"],
                passed=s["passed"],
                details=s["details"],
                metadata=s.get("metadata", {}),
            )
            for s in item.get("scores", [])
        ]
        
        result = EvalResult(
            scenario_id=item["scenario_id"],
            scenario_name=item["scenario_name"],
            category=ScenarioCategory(item["category"]),
            query=item["query"],
            output=item.get("output", ""),
            scores=scores,
            overall_passed=item["overall_passed"],
            trust_score=item.get("trust_score"),
            latency_ms=item.get("latency_ms"),
            timestamp=item.get("timestamp", ""),
            error=item.get("error"),
        )
        results.append(result)
    
    return results


def generate_report(
    results: List[EvalResult],
    output_dir: Optional[Path] = None,
    format: str = "both",
) -> EvaluationReport:
    """Convenience function to generate a report.
    
    Args:
        results: List of evaluation results.
        output_dir: Optional directory to save report.
        format: Output format ("json", "markdown", or "both").
        
    Returns:
        EvaluationReport instance.
    """
    report = EvaluationReport(results)
    
    if output_dir:
        report.save(output_dir, format)
    
    return report

"""
PM Agent Evaluation Suite CLI.

Usage:
    python -m evals --all                    # Run all evaluations
    python -m evals --category critical-issues  # Run specific category
    python -m evals --scenario daily_digest_001  # Run single scenario
    python -m evals --live                   # Run with live tools
    python -m evals report --compare v1.0 v1.1  # Compare runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

from evals.schema import ScenarioCategory
from evals.runner import EvaluationRunner, RunConfig
from evals.report import EvaluationReport, compare_runs, load_results_from_file
from evals.datasets import get_scenario_count, get_all_scenario_ids


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PM Agent Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m evals --all                          Run all evaluations
    python -m evals --category critical-issues     Run specific category
    python -m evals --scenario daily_digest_001    Run single scenario
    python -m evals --live                         Run with live tools (not mocked)
    python -m evals --list                         List all scenarios
    python -m evals report --compare v1.0 v1.1     Compare two evaluation runs
        """,
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command (default)
    run_parser = subparsers.add_parser("run", help="Run evaluations")
    _add_run_args(run_parser)
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate or compare reports")
    report_parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "COMPARISON"),
        help="Compare two evaluation runs by file path",
    )
    report_parser.add_argument(
        "--input",
        type=Path,
        help="Input results file for report generation",
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results"),
        help="Output directory for reports",
    )
    report_parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format",
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List scenarios")
    list_parser.add_argument(
        "--category",
        type=str,
        help="Filter by category",
    )
    list_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed scenario info",
    )
    
    # Also add run args to main parser for convenience
    _add_run_args(parser)
    
    return parser.parse_args()


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add run-related arguments to parser."""
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all evaluation scenarios",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="Run scenarios from specific category",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Run a specific scenario by ID",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with live tools instead of mocks",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring",
    )
    parser.add_argument(
        "--no-trust-score",
        action="store_true",
        help="Skip trust score calculation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=5,
        help="Maximum parallel scenarios",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for LLM-as-judge",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        help="Filter scenarios by tags",
    )


def list_scenarios(category: Optional[str] = None, verbose: bool = False) -> None:
    """List available scenarios."""
    from evals.datasets import load_all_datasets, load_dataset
    
    print("\n📋 PM Agent Evaluation Scenarios\n")
    
    counts = get_scenario_count()
    print(f"Total scenarios: {counts['total']}\n")
    
    if category:
        try:
            cat = ScenarioCategory(category.replace("-", "_"))
            scenarios = load_dataset(cat)
            print(f"Category: {cat.value} ({len(scenarios)} scenarios)\n")
            for s in scenarios:
                if verbose:
                    print(f"  {s.id}")
                    print(f"    Name: {s.name}")
                    print(f"    Query: {s.query[:60]}...")
                    print(f"    Tags: {', '.join(s.tags)}")
                    print()
                else:
                    print(f"  {s.id}: {s.name}")
        except ValueError:
            print(f"Unknown category: {category}")
            print(f"Available: {[c.value for c in ScenarioCategory]}")
    else:
        for cat in ScenarioCategory:
            print(f"  {cat.value}: {counts[cat.value]} scenarios")
        print()
        print("Use --category <name> to see scenarios in a category")


async def run_evaluations(args: argparse.Namespace) -> int:
    """Run evaluations based on arguments."""
    # Build config
    config = RunConfig(
        mode="live" if args.live else "mock",
        run_llm_judges=not args.no_llm_judge,
        run_trust_score=not args.no_trust_score,
        judge_model=args.judge_model,
        max_parallel=args.parallel,
        output_dir=args.output,
        save_results=True,
    )
    
    # Determine what to run
    categories = None
    scenario_ids = None
    
    if args.scenario:
        scenario_ids = [args.scenario]
    elif args.category:
        try:
            cat = ScenarioCategory(args.category.replace("-", "_"))
            categories = [cat]
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Available: {[c.value for c in ScenarioCategory]}")
            return 1
    elif not args.all:
        print("Specify --all, --category, or --scenario")
        return 1
    
    if categories:
        config.categories = categories
    if scenario_ids:
        config.scenario_ids = scenario_ids
    if args.tags:
        config.tags = args.tags
    
    # Run evaluations
    print("\n🚀 Starting PM Agent Evaluation Suite\n")
    print(f"Mode: {'Live' if args.live else 'Mock'}")
    print(f"LLM Judge: {'Enabled' if config.run_llm_judges else 'Disabled'}")
    print(f"Trust Score: {'Enabled' if config.run_trust_score else 'Disabled'}")
    print(f"Output: {config.output_dir}")
    print()
    
    runner = EvaluationRunner(config)
    await runner.run_all()
    
    # Print summary
    summary = runner.get_summary()
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal Scenarios: {summary.total_scenarios}")
    print(f"Passed: {summary.passed} ✅")
    print(f"Failed: {summary.failed} ❌")
    print(f"Errors: {summary.errors} ⚠️")
    print(f"\nPass Rate: {summary.pass_rate:.1%}")
    print(f"Avg Trust Score: {summary.avg_trust_score:.3f}")
    print(f"Avg Latency: {summary.avg_latency_ms:.0f}ms")
    
    print("\n📁 By Category:")
    for cat_name, cat_data in summary.by_category.items():
        status = "✅" if cat_data["pass_rate"] >= 0.9 else "❌"
        print(f"  {cat_name}: {cat_data['passed']}/{cat_data['total']} ({cat_data['pass_rate']:.0%}) {status}")
    
    # Check thresholds
    print("\n🎯 Threshold Check:")
    thresholds_met = True
    
    if summary.pass_rate < 0.95:
        print(f"  ❌ Pass Rate: {summary.pass_rate:.1%} < 95%")
        thresholds_met = False
    else:
        print(f"  ✅ Pass Rate: {summary.pass_rate:.1%} >= 95%")
    
    if summary.avg_trust_score < 0.80:
        print(f"  ❌ Trust Score: {summary.avg_trust_score:.3f} < 0.80")
        thresholds_met = False
    else:
        print(f"  ✅ Trust Score: {summary.avg_trust_score:.3f} >= 0.80")
    
    print("\n" + "=" * 60)
    
    if thresholds_met:
        print("✅ All thresholds met!")
        return 0
    else:
        print("❌ Some thresholds not met - review failures")
        return 1


def run_report(args: argparse.Namespace) -> int:
    """Generate or compare reports."""
    if args.compare:
        baseline_path, comparison_path = args.compare
        
        try:
            baseline_results = load_results_from_file(Path(baseline_path))
            comparison_results = load_results_from_file(Path(comparison_path))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        
        comparison = compare_runs(
            baseline_results,
            comparison_results,
            baseline_name=baseline_path,
            comparison_name=comparison_path,
        )
        
        print("\n📊 EVALUATION COMPARISON")
        print("=" * 60)
        print(f"\nBaseline: {comparison.baseline_name}")
        print(f"Comparison: {comparison.comparison_name}")
        print()
        print(f"Pass Rate Delta: {comparison.pass_rate_delta:+.1%}")
        print(f"Trust Score Delta: {comparison.trust_score_delta:+.3f}")
        print(f"Latency Delta: {comparison.latency_delta:+.0f}ms")
        
        if comparison.regressions:
            print(f"\n❌ Regressions ({len(comparison.regressions)}):")
            for r in comparison.regressions[:5]:
                print(f"  - {r}")
        
        if comparison.improvements:
            print(f"\n✅ Improvements ({len(comparison.improvements)}):")
            for i in comparison.improvements[:5]:
                print(f"  - {i}")
        
        print("\n" + "=" * 60)
        if comparison.is_regression():
            print("⚠️  REGRESSION DETECTED")
            return 1
        else:
            print("✅ No regression detected")
            return 0
    
    elif args.input:
        try:
            results = load_results_from_file(args.input)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        
        report = EvaluationReport(results)
        saved = report.save(args.output, args.format)
        
        print(f"\n📄 Report generated:")
        for path in saved:
            print(f"  - {path}")
        
        return 0
    
    else:
        print("Specify --compare or --input for report generation")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Handle list command
    if args.command == "list" or getattr(args, "list", False):
        category = getattr(args, "category", None)
        verbose = getattr(args, "verbose", False)
        list_scenarios(category, verbose)
        return 0
    
    # Handle report command
    if args.command == "report":
        return run_report(args)
    
    # Handle run command (default)
    if args.command == "run" or args.all or args.category or args.scenario:
        return asyncio.run(run_evaluations(args))
    
    # No command specified
    print("PM Agent Evaluation Suite")
    print("Use --help for usage information")
    print("\nQuick start:")
    print("  python -m evals --all        Run all evaluations")
    print("  python -m evals --list       List scenarios")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Evaluation Runner.

Executes evaluation scenarios against the PM Agent with mock or live tools.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from evals.schema import (
    EvalResult,
    EvalScenario,
    EvalSummary,
    MockToolResponse,
    ScenarioCategory,
    ScoreResult,
)
from evals.datasets import load_all_datasets, load_dataset, load_scenario_by_id
from evals.scorers.deterministic import run_all_deterministic_scorers
from evals.scorers.llm_judge import run_all_llm_judges, JudgeConfig
from evals.scorers.trust_scorer import TrustScorerWrapper


logger = structlog.get_logger(__name__)


@dataclass
class RunConfig:
    """Configuration for evaluation runs."""
    
    # Mode: mock uses mock responses, live calls actual tools
    mode: str = "mock"  # "mock" or "live"
    
    # Scoring options
    run_deterministic: bool = True
    run_llm_judges: bool = True
    run_trust_score: bool = True
    
    # LLM judge config
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    
    # Parallel execution
    max_parallel: int = 5
    
    # Output options
    output_dir: Optional[Path] = None
    save_results: bool = True
    
    # Filtering
    categories: Optional[List[ScenarioCategory]] = None
    scenario_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class MockToolExecutor:
    """Executes scenarios with mock tool responses."""
    
    def __init__(self, mock_responses: List[MockToolResponse]):
        """Initialize with mock responses.
        
        Args:
            mock_responses: List of mock tool responses for this scenario.
        """
        self.mock_responses = {m.tool_name: m for m in mock_responses}
        self.called_tools: List[str] = []
        self.tool_outputs: Dict[str, Any] = {}
    
    def get_response(self, tool_name: str) -> Dict[str, Any]:
        """Get mock response for a tool.
        
        Args:
            tool_name: Name of the tool being called.
            
        Returns:
            Mock response dictionary.
        """
        self.called_tools.append(tool_name)
        
        if tool_name in self.mock_responses:
            mock = self.mock_responses[tool_name]
            if mock.success:
                self.tool_outputs[tool_name] = mock.response
                return {"success": True, "data": mock.response}
            else:
                return {"success": False, "error": mock.error}
        
        # No mock defined - return empty success
        return {"success": True, "data": {}}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the mock execution.
        
        Returns:
            Dictionary with execution metadata.
        """
        # Extract identifiers from mock responses
        jira_ids = set()
        github_ids = set()
        
        for tool_name, output in self.tool_outputs.items():
            if isinstance(output, dict):
                # Extract Jira IDs
                if "issues" in output:
                    for issue in output.get("issues", []):
                        if "key" in issue:
                            jira_ids.add(issue["key"])
                
                # Extract GitHub IDs
                if "items" in output:
                    for item in output.get("items", []):
                        if "number" in item:
                            github_ids.add(str(item["number"]))
        
        return {
            "called_tools": self.called_tools,
            "tool_outputs": self.tool_outputs,
            "jira_ids": list(jira_ids),
            "github_ids": list(github_ids),
        }


class MockAgentRunner:
    """Runs scenarios with mock agent responses."""
    
    def __init__(self, scenario: EvalScenario):
        """Initialize with scenario.
        
        Args:
            scenario: The evaluation scenario to run.
        """
        self.scenario = scenario
        self.executor = MockToolExecutor(scenario.mock_responses)
        self._evidence_ledger = None
    
    def _get_evidence_ledger(self):
        """Lazy-load and create evidence ledger."""
        if self._evidence_ledger is None:
            try:
                from src.evidence import EvidenceLedger, SourceType
                self._evidence_ledger = EvidenceLedger()
                self._source_type_class = SourceType
            except ImportError:
                self._evidence_ledger = None
                self._source_type_class = None
        return self._evidence_ledger
    
    def _tool_name_to_source_type(self, tool_name: str):
        """Map tool name to SourceType enum."""
        if self._source_type_class is None:
            return None
        
        tool_lower = tool_name.lower()
        if "jira" in tool_lower:
            return self._source_type_class.JIRA
        elif "github" in tool_lower:
            return self._source_type_class.GITHUB
        elif "slack" in tool_lower:
            return self._source_type_class.SLACK
        elif "tavily" in tool_lower or "web" in tool_lower or "search" in tool_lower:
            return self._source_type_class.WEB
        elif "notion" in tool_lower:
            return self._source_type_class.NOTION
        elif "email" in tool_lower:
            return self._source_type_class.EMAIL
        else:
            return self._source_type_class.OTHER
    
    async def run(self) -> tuple[str, Dict[str, Any]]:
        """Run the scenario with mocks.
        
        Returns:
            Tuple of (output_text, metadata).
        """
        # Get evidence ledger for recording
        ledger = self._get_evidence_ledger()
        
        # Simulate tool calls based on mock responses
        for mock in self.scenario.mock_responses:
            result = self.executor.get_response(mock.tool_name)
            
            # Record in evidence ledger if available
            if ledger is not None:
                source_type = self._tool_name_to_source_type(mock.tool_name)
                if source_type:
                    ledger.record_tool_call(
                        source_type=source_type,
                        tool_name=mock.tool_name,
                        query_params={"scenario": self.scenario.id},
                        result=mock.response if mock.success else None,
                        success=mock.success,
                        error=mock.error if not mock.success else None,
                    )
        
        # Generate mock output based on scenario expectations
        output = self._generate_mock_output()
        
        # Build metadata
        metadata = self.executor.get_metadata()
        metadata["invoked_agents"] = [a.value for a in self.scenario.expected_agents]
        metadata["checked_sources"] = [s.value for s in self.scenario.expected_sources]
        metadata["alert_sent"] = (
            self.scenario.golden_output.should_alert
            if self.scenario.golden_output
            else False
        )
        
        # Add evidence ledger to metadata for trust scorer
        if ledger is not None:
            metadata["evidence_ledger"] = ledger
        
        return output, metadata
    
    def _generate_mock_output(self) -> str:
        """Generate a mock output that matches expected patterns.
        
        This creates a realistic-looking output based on the mock responses.
        """
        parts = []
        
        # Add header
        parts.append(f"# Response to: {self.scenario.query}\n")
        
        # Process each mock response
        for mock in self.scenario.mock_responses:
            if not mock.success:
                parts.append(f"\n## {mock.tool_name}\nData unavailable: {mock.error}")
                continue
            
            if mock.response:
                # Format based on tool type
                if "jira" in mock.tool_name.lower():
                    parts.append(self._format_jira_output(mock.response))
                elif "github" in mock.tool_name.lower():
                    parts.append(self._format_github_output(mock.response))
                elif "slack" in mock.tool_name.lower():
                    parts.append(self._format_slack_output(mock.response))
                elif "tavily" in mock.tool_name.lower() or "search" in mock.tool_name.lower():
                    parts.append(self._format_web_output(mock.response))
        
        # Add expected content if specified
        if self.scenario.golden_output:
            if self.scenario.golden_output.must_contain:
                # Ensure key content is present
                for keyword in self.scenario.golden_output.must_contain[:3]:
                    if keyword not in "\n".join(parts):
                        parts.append(f"\n{keyword}")
        
        return "\n".join(parts)
    
    def _format_jira_output(self, data: Dict[str, Any]) -> str:
        """Format Jira response data."""
        lines = ["\n## Jira"]
        
        issues = data.get("issues", [])
        if not issues:
            if isinstance(data, dict) and "key" in data:
                # Single issue response
                issues = [data]
        
        for issue in issues:
            key = issue.get("key", "UNKNOWN")
            summary = issue.get("summary", "No summary")
            status = issue.get("status", "Unknown")
            priority = issue.get("priority", "Unknown")
            lines.append(f"- **{key}**: {summary} (Status: {status}, Priority: {priority})")
        
        if not issues:
            lines.append("No Jira issues found.")
        
        return "\n".join(lines)
    
    def _format_github_output(self, data: Dict[str, Any]) -> str:
        """Format GitHub response data."""
        lines = ["\n## GitHub"]
        
        items = data.get("items", [])
        if not items:
            if isinstance(data, dict) and "number" in data:
                items = [data]
        
        for item in items:
            number = item.get("number", 0)
            title = item.get("title", "No title")
            state = item.get("state", "unknown")
            labels = item.get("labels", [])
            label_str = ", ".join(labels) if isinstance(labels, list) else str(labels)
            lines.append(f"- **#{number}**: {title} ({state}) [{label_str}]")
        
        if not items:
            lines.append("No GitHub items found.")
        
        return "\n".join(lines)
    
    def _format_slack_output(self, data: Dict[str, Any]) -> str:
        """Format Slack response data."""
        lines = ["\n## Slack"]
        
        messages = data.get("messages", [])
        for msg in messages:
            channel = msg.get("channel", "#unknown")
            text = msg.get("text", "")
            lines.append(f"- {channel}: \"{text}\"")
        
        if not messages:
            lines.append("No Slack messages found.")
        
        return "\n".join(lines)
    
    def _format_web_output(self, data: Dict[str, Any]) -> str:
        """Format web search response data."""
        lines = ["\n## Web Search Results"]
        
        results = data.get("results", [])
        for result in results:
            title = result.get("title", "No title")
            content = result.get("content", "")[:200]
            url = result.get("url", "")
            lines.append(f"- **{title}**\n  {content}\n  Source: {url}")
        
        if not results:
            lines.append("No web search results found.")
        
        return "\n".join(lines)


class EvaluationRunner:
    """Main evaluation runner."""
    
    def __init__(self, config: Optional[RunConfig] = None):
        """Initialize the evaluation runner.
        
        Args:
            config: Optional run configuration.
        """
        self.config = config or RunConfig()
        self.results: List[EvalResult] = []
        self._judge_config = JudgeConfig(
            model=self.config.judge_model,
            temperature=self.config.judge_temperature,
        )
    
    async def run_scenario(self, scenario: EvalScenario) -> EvalResult:
        """Run a single evaluation scenario.
        
        Args:
            scenario: The scenario to evaluate.
            
        Returns:
            EvalResult with all scores.
        """
        start_time = time.time()
        
        try:
            # Run the scenario
            if self.config.mode == "mock":
                runner = MockAgentRunner(scenario)
                output, metadata = await runner.run()
            else:
                # Live mode - call actual agent
                output, metadata = await self._run_live(scenario)
            
            # Collect scores
            scores: List[ScoreResult] = []
            
            # Run deterministic scorers
            if self.config.run_deterministic:
                det_scores = run_all_deterministic_scorers(scenario, output, metadata)
                scores.extend(det_scores)
            
            # Run LLM judges
            if self.config.run_llm_judges:
                llm_scores = run_all_llm_judges(
                    scenario, output, metadata, self._judge_config
                )
                scores.extend(llm_scores)
            
            # Run trust scorer
            trust_score = None
            if self.config.run_trust_score:
                trust_wrapper = TrustScorerWrapper()
                trust_result = trust_wrapper.score(scenario, output, metadata)
                scores.append(trust_result)
                trust_score = trust_result.score
            
            # Calculate overall pass/fail
            overall_passed = all(s.passed for s in scores if s.passed is not None)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return EvalResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                category=scenario.category,
                query=scenario.query,
                output=output,
                scores=scores,
                overall_passed=overall_passed,
                trust_score=trust_score,
                latency_ms=latency_ms,
            )
        
        except Exception as e:
            logger.error("scenario_failed", scenario_id=scenario.id, error=str(e))
            return EvalResult(
                scenario_id=scenario.id,
                scenario_name=scenario.name,
                category=scenario.category,
                query=scenario.query,
                output="",
                scores=[],
                overall_passed=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
    
    async def _run_live(self, scenario: EvalScenario) -> tuple[str, Dict[str, Any]]:
        """Run scenario with live agent.
        
        Args:
            scenario: The scenario to run.
            
        Returns:
            Tuple of (output_text, metadata).
        """
        try:
            from src.graphs.main_graph import invoke_graph
            from src.evidence import get_ledger
            from src.execution_state import get_execution_state
            
            # Run the actual agent
            result = await invoke_graph(scenario.query)
            
            # Extract output
            output = ""
            if result and "messages" in result:
                messages = result["messages"]
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        output = last_msg.content
            
            # Build metadata from actual execution
            ledger = get_ledger()
            exec_state = get_execution_state()
            
            metadata = {
                "evidence_ledger": ledger,
                "execution_state": exec_state,
                "invoked_agents": [],  # Would need to track this
                "checked_sources": [],  # Would need to track this
                "alert_sent": False,  # Would need to track this
            }
            
            return output, metadata
        
        except ImportError as e:
            logger.warning("live_mode_unavailable", error=str(e))
            raise RuntimeError(f"Live mode requires src module: {e}")
    
    async def run_category(
        self,
        category: ScenarioCategory,
    ) -> List[EvalResult]:
        """Run all scenarios in a category.
        
        Args:
            category: The category to run.
            
        Returns:
            List of EvalResult objects.
        """
        scenarios = load_dataset(category)
        
        # Filter by tags if specified
        if self.config.tags:
            scenarios = [
                s for s in scenarios
                if any(t in s.tags for t in self.config.tags)
            ]
        
        # Filter by specific IDs if specified
        if self.config.scenario_ids:
            scenarios = [
                s for s in scenarios
                if s.id in self.config.scenario_ids
            ]
        
        logger.info(
            "running_category",
            category=category.value,
            scenario_count=len(scenarios),
        )
        
        # Run scenarios with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_parallel)
        
        async def run_with_semaphore(scenario: EvalScenario) -> EvalResult:
            async with semaphore:
                return await self.run_scenario(scenario)
        
        results = await asyncio.gather(
            *[run_with_semaphore(s) for s in scenarios]
        )
        
        self.results.extend(results)
        return list(results)
    
    async def run_all(self) -> List[EvalResult]:
        """Run all evaluation scenarios.
        
        Returns:
            List of all EvalResult objects.
        """
        categories = self.config.categories or list(ScenarioCategory)
        
        logger.info("starting_evaluation_run", categories=[c.value for c in categories])
        
        for category in categories:
            await self.run_category(category)
        
        logger.info(
            "evaluation_run_complete",
            total_scenarios=len(self.results),
            passed=sum(1 for r in self.results if r.overall_passed),
            failed=sum(1 for r in self.results if not r.overall_passed),
        )
        
        # Save results if configured
        if self.config.save_results and self.config.output_dir:
            self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """Save results to output directory."""
        if not self.config.output_dir:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = output_dir / f"eval_results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
            )
        
        # Save summary
        summary = self.get_summary()
        summary_file = output_dir / f"eval_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info("results_saved", output_dir=str(output_dir))
    
    def get_summary(self) -> EvalSummary:
        """Get summary of evaluation results.
        
        Returns:
            EvalSummary with aggregated metrics.
        """
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
        
        # Group by category
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


async def run_evaluation(
    config: Optional[RunConfig] = None,
    categories: Optional[List[ScenarioCategory]] = None,
    scenario_ids: Optional[List[str]] = None,
) -> EvalSummary:
    """Convenience function to run evaluations.
    
    Args:
        config: Optional run configuration.
        categories: Optional list of categories to run.
        scenario_ids: Optional list of specific scenario IDs.
        
    Returns:
        EvalSummary with results.
    """
    if config is None:
        config = RunConfig()
    
    if categories:
        config.categories = categories
    if scenario_ids:
        config.scenario_ids = scenario_ids
    
    runner = EvaluationRunner(config)
    await runner.run_all()
    return runner.get_summary()

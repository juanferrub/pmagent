"""
Opik Integration.

Integration with Opik for evaluation tracking and visualization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import structlog

from evals.schema import EvalResult, EvalScenario, EvalSummary, ScoreResult


logger = structlog.get_logger(__name__)


@dataclass
class OpikConfig:
    """Configuration for Opik integration."""
    
    project_name: str = "pm-agent-evals"
    workspace: Optional[str] = None
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("OPIK_API_KEY")
        if self.workspace is None:
            self.workspace = os.getenv("OPIK_WORKSPACE")


class OpikEvaluator:
    """Opik evaluation integration."""
    
    def __init__(self, config: Optional[OpikConfig] = None):
        """Initialize Opik evaluator.
        
        Args:
            config: Optional Opik configuration.
        """
        self.config = config or OpikConfig()
        self._client = None
        self._dataset = None
    
    @property
    def client(self):
        """Lazy-load Opik client."""
        if self._client is None:
            try:
                from opik import Opik
                self._client = Opik(
                    project_name=self.config.project_name,
                    workspace=self.config.workspace,
                )
                logger.info("opik_client_initialized", project=self.config.project_name)
            except ImportError:
                logger.warning("opik_not_installed")
                self._client = None
            except Exception as e:
                logger.error("opik_init_failed", error=str(e))
                self._client = None
        return self._client
    
    def create_dataset(self, name: str, scenarios: List[EvalScenario]) -> Optional[str]:
        """Create an Opik dataset from scenarios.
        
        Args:
            name: Dataset name.
            scenarios: List of evaluation scenarios.
            
        Returns:
            Dataset ID if successful, None otherwise.
        """
        if self.client is None:
            logger.warning("opik_unavailable_for_dataset")
            return None
        
        try:
            # Convert scenarios to Opik dataset format
            items = []
            for scenario in scenarios:
                item = {
                    "id": scenario.id,
                    "input": {
                        "query": scenario.query,
                        "category": scenario.category.value,
                        "expected_agents": [a.value for a in scenario.expected_agents],
                        "expected_sources": [s.value for s in scenario.expected_sources],
                    },
                    "expected_output": {
                        "should_alert": scenario.golden_output.should_alert if scenario.golden_output else None,
                        "must_contain": scenario.golden_output.must_contain if scenario.golden_output else [],
                        "min_trust_score": scenario.golden_output.min_trust_score if scenario.golden_output else None,
                    },
                    "metadata": {
                        "name": scenario.name,
                        "description": scenario.description,
                        "tags": scenario.tags,
                    },
                }
                items.append(item)
            
            # Create dataset in Opik
            dataset = self.client.create_dataset(
                name=name,
                description=f"PM Agent evaluation dataset - {len(items)} scenarios",
            )
            
            # Add items to dataset
            for item in items:
                dataset.insert(item)
            
            logger.info("opik_dataset_created", name=name, items=len(items))
            return dataset.id
        
        except Exception as e:
            logger.error("opik_dataset_creation_failed", error=str(e))
            return None
    
    def log_evaluation_result(
        self,
        result: EvalResult,
        experiment_name: str,
    ) -> bool:
        """Log a single evaluation result to Opik.
        
        Args:
            result: The evaluation result to log.
            experiment_name: Name of the experiment/run.
            
        Returns:
            True if successful, False otherwise.
        """
        if self.client is None:
            return False
        
        try:
            # Create trace for this evaluation
            trace = self.client.trace(
                name=f"eval_{result.scenario_id}",
                input={"query": result.query},
                output={"response": result.output[:1000] if result.output else ""},
                metadata={
                    "scenario_id": result.scenario_id,
                    "scenario_name": result.scenario_name,
                    "category": result.category.value,
                    "experiment": experiment_name,
                    "overall_passed": result.overall_passed,
                    "trust_score": result.trust_score,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                },
            )
            
            # Log individual scores as spans
            for score in result.scores:
                trace.span(
                    name=score.scorer_name,
                    input={"criteria": score.metadata.get("criteria", "")},
                    output={
                        "score": score.score,
                        "passed": score.passed,
                        "details": score.details,
                    },
                    metadata=score.metadata,
                )
            
            trace.end()
            return True
        
        except Exception as e:
            logger.error("opik_log_failed", scenario_id=result.scenario_id, error=str(e))
            return False
    
    def log_evaluation_run(
        self,
        results: List[EvalResult],
        summary: EvalSummary,
        experiment_name: str,
    ) -> bool:
        """Log a complete evaluation run to Opik.
        
        Args:
            results: List of evaluation results.
            summary: Summary of the evaluation run.
            experiment_name: Name of the experiment.
            
        Returns:
            True if successful, False otherwise.
        """
        if self.client is None:
            logger.warning("opik_unavailable_for_logging")
            return False
        
        try:
            # Log summary as experiment metadata
            self.client.log_experiment(
                name=experiment_name,
                metadata={
                    "total_scenarios": summary.total_scenarios,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "errors": summary.errors,
                    "pass_rate": summary.pass_rate,
                    "avg_trust_score": summary.avg_trust_score,
                    "avg_latency_ms": summary.avg_latency_ms,
                    "by_category": summary.by_category,
                    "timestamp": summary.timestamp,
                },
            )
            
            # Log individual results
            logged = 0
            for result in results:
                if self.log_evaluation_result(result, experiment_name):
                    logged += 1
            
            logger.info(
                "opik_run_logged",
                experiment=experiment_name,
                total=len(results),
                logged=logged,
            )
            return True
        
        except Exception as e:
            logger.error("opik_run_log_failed", error=str(e))
            return False
    
    def create_scoring_metrics(self) -> Dict[str, Callable]:
        """Create Opik-compatible scoring metrics.
        
        Returns:
            Dictionary of metric name to scoring function.
        """
        from evals.scorers.deterministic import (
            RouteAccuracyScorer,
            SourceCoverageScorer,
            HallucinationDetector,
            LanguageComplianceScorer,
        )
        from evals.scorers.llm_judge import (
            RelevanceScorer,
            CompletenessScorer,
            AccuracyScorer,
        )
        
        def make_metric(scorer_class):
            """Create a metric function from a scorer class."""
            def metric(input_data: Dict, output_data: Dict, metadata: Dict) -> float:
                from evals.schema import EvalScenario, ScenarioCategory, ExpectedAgent, ExpectedSource
                
                # Reconstruct scenario from input
                scenario = EvalScenario(
                    id=metadata.get("scenario_id", "unknown"),
                    name=metadata.get("scenario_name", ""),
                    description="",
                    category=ScenarioCategory(input_data.get("category", "daily_digest")),
                    query=input_data.get("query", ""),
                    expected_agents=[ExpectedAgent(a) for a in input_data.get("expected_agents", [])],
                    expected_sources=[ExpectedSource(s) for s in input_data.get("expected_sources", [])],
                )
                
                scorer = scorer_class()
                result = scorer.score(scenario, output_data.get("response", ""), metadata)
                return result.score
            
            return metric
        
        return {
            "route_accuracy": make_metric(RouteAccuracyScorer),
            "source_coverage": make_metric(SourceCoverageScorer),
            "hallucination_score": make_metric(HallucinationDetector),
            "language_compliance": make_metric(LanguageComplianceScorer),
        }


def run_opik_evaluation(
    scenarios: List[EvalScenario],
    experiment_name: str,
    task_fn: Callable[[EvalScenario], str],
    config: Optional[OpikConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Run evaluation using Opik's evaluate API.
    
    Args:
        scenarios: List of scenarios to evaluate.
        experiment_name: Name for this evaluation run.
        task_fn: Function that takes a scenario and returns agent output.
        config: Optional Opik configuration.
        
    Returns:
        Evaluation results dictionary if successful, None otherwise.
    """
    try:
        from opik import Opik
        from opik.evaluation import evaluate
        
        opik_config = config or OpikConfig()
        client = Opik(
            project_name=opik_config.project_name,
            workspace=opik_config.workspace,
        )
        
        # Create dataset
        dataset_items = [
            {
                "id": s.id,
                "input": {"query": s.query, "category": s.category.value},
                "metadata": {"name": s.name, "tags": s.tags},
            }
            for s in scenarios
        ]
        
        dataset = client.create_dataset(f"{experiment_name}_dataset")
        for item in dataset_items:
            dataset.insert(item)
        
        # Create evaluator
        evaluator = OpikEvaluator(opik_config)
        metrics = evaluator.create_scoring_metrics()
        
        # Define task wrapper
        def task(item: Dict) -> Dict:
            scenario_id = item["id"]
            scenario = next((s for s in scenarios if s.id == scenario_id), None)
            if scenario:
                output = task_fn(scenario)
                return {"response": output}
            return {"response": ""}
        
        # Run evaluation
        results = evaluate(
            experiment_name=experiment_name,
            dataset=dataset,
            task=task,
            scoring_metrics=list(metrics.values()),
        )
        
        logger.info("opik_evaluation_complete", experiment=experiment_name)
        return results
    
    except ImportError:
        logger.warning("opik_not_installed_for_evaluation")
        return None
    except Exception as e:
        logger.error("opik_evaluation_failed", error=str(e))
        return None

"""
Tests for the PM Agent Evaluation Suite.

Tests cover:
- Dataset loading and schema validation
- Deterministic scorers
- LLM-as-judge scorers (mocked)
- Evaluation runner
- Report generation
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from evals.schema import (
    EvalScenario,
    EvalResult,
    EvalSummary,
    GoldenOutput,
    MockToolResponse,
    QualityCriteria,
    ScenarioCategory,
    ScoreResult,
    ExpectedAgent,
    ExpectedSource,
)
from evals.datasets import (
    load_dataset,
    load_all_datasets,
    load_scenario_by_id,
    get_scenario_count,
    get_all_scenario_ids,
)
from evals.scorers.deterministic import (
    RouteAccuracyScorer,
    SourceCoverageScorer,
    HallucinationDetector,
    LanguageComplianceScorer,
    StructureScorer,
    IdentifierValidator,
    AlertAppropriatenessScorer,
    run_all_deterministic_scorers,
)
from evals.runner import (
    EvaluationRunner,
    RunConfig,
    MockToolExecutor,
    MockAgentRunner,
)
from evals.report import (
    EvaluationReport,
    compare_runs,
    ComparisonResult,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_scenario() -> EvalScenario:
    """Create a sample evaluation scenario."""
    return EvalScenario(
        id="test_001",
        name="Test Scenario",
        description="A test scenario for unit tests",
        category=ScenarioCategory.DAILY_DIGEST,
        query="Give me a daily digest",
        expected_agents=[ExpectedAgent.JIRA, ExpectedAgent.GITHUB],
        expected_sources=[ExpectedSource.JIRA, ExpectedSource.GITHUB],
        mock_responses=[
            MockToolResponse(
                tool_name="jira_search",
                response={"issues": [{"key": "TEST-123", "summary": "Test issue"}]},
                success=True,
            ),
            MockToolResponse(
                tool_name="github_list_issues",
                response={"items": [{"number": 456, "title": "Test PR"}]},
                success=True,
            ),
        ],
        golden_output=GoldenOutput(
            must_contain=["TEST-123", "#456"],
            must_not_contain=["all good", "looks fine"],
            expected_sections=["Jira", "GitHub"],
            expected_identifiers=["TEST-123", "456"],
            should_alert=False,
            min_trust_score=0.8,
        ),
        quality_criteria=QualityCriteria(
            relevance="Should summarize activity",
            completeness="Should cover both sources",
        ),
        tags=["test", "unit-test"],
    )


@pytest.fixture
def sample_output_good() -> str:
    """Sample good output that should pass most checks."""
    return """
# Daily Digest

## Jira
- **TEST-123**: Test issue (Status: Open)

## GitHub
- **#456**: Test PR (open)

No verified critical issues detected in checked sources.
"""


@pytest.fixture
def sample_output_bad() -> str:
    """Sample bad output with issues."""
    return """
# Daily Digest

Everything looks fine! All good here.

## Jira
- **FAKE-999**: Made up issue

The team is doing great!
"""


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Sample metadata from a run."""
    return {
        "invoked_agents": ["jira_agent", "github_agent"],
        "checked_sources": ["jira", "github"],
        "jira_ids": ["TEST-123"],
        "github_ids": ["456"],
        "alert_sent": False,
    }


# =============================================================================
# Dataset Tests
# =============================================================================

class TestDatasetLoading:
    """Tests for dataset loading functionality."""
    
    def test_load_daily_digest_dataset(self):
        """Test loading daily digest scenarios."""
        scenarios = load_dataset(ScenarioCategory.DAILY_DIGEST)
        assert len(scenarios) == 6
        assert all(s.category == ScenarioCategory.DAILY_DIGEST for s in scenarios)
    
    def test_load_critical_issues_dataset(self):
        """Test loading critical issues scenarios."""
        scenarios = load_dataset(ScenarioCategory.CRITICAL_ISSUES)
        assert len(scenarios) == 8
        assert all(s.category == ScenarioCategory.CRITICAL_ISSUES for s in scenarios)
    
    def test_load_all_datasets(self):
        """Test loading all datasets."""
        all_datasets = load_all_datasets()
        assert len(all_datasets) == 6  # 6 categories
        
        total = sum(len(scenarios) for scenarios in all_datasets.values())
        assert total == 43  # Total scenarios as per plan
    
    def test_load_scenario_by_id(self):
        """Test loading a specific scenario by ID."""
        scenario = load_scenario_by_id("daily_digest_001")
        assert scenario is not None
        assert scenario.id == "daily_digest_001"
        assert scenario.category == ScenarioCategory.DAILY_DIGEST
    
    def test_load_nonexistent_scenario(self):
        """Test loading a scenario that doesn't exist."""
        scenario = load_scenario_by_id("nonexistent_scenario")
        assert scenario is None
    
    def test_get_scenario_count(self):
        """Test getting scenario counts."""
        counts = get_scenario_count()
        assert counts["total"] == 43
        assert counts["daily_digest"] == 6
        assert counts["critical_issues"] == 8
        assert counts["competitor_research"] == 5
        assert counts["routing"] == 8
        assert counts["hallucination"] == 10
        assert counts["alerts"] == 6
    
    def test_get_all_scenario_ids(self):
        """Test getting all scenario IDs."""
        ids = get_all_scenario_ids()
        assert len(ids) == 43
        assert "daily_digest_001" in ids
        assert "critical_001" in ids


class TestScenarioSchema:
    """Tests for scenario schema and serialization."""
    
    def test_scenario_to_dict(self, sample_scenario: EvalScenario):
        """Test converting scenario to dictionary."""
        data = sample_scenario.to_dict()
        
        assert data["id"] == "test_001"
        assert data["category"] == "daily_digest"
        assert data["expected_agents"] == ["jira_agent", "github_agent"]
        assert data["golden_output"]["must_contain"] == ["TEST-123", "#456"]
    
    def test_scenario_from_dict(self, sample_scenario: EvalScenario):
        """Test creating scenario from dictionary."""
        data = sample_scenario.to_dict()
        restored = EvalScenario.from_dict(data)
        
        assert restored.id == sample_scenario.id
        assert restored.category == sample_scenario.category
        assert restored.expected_agents == sample_scenario.expected_agents


# =============================================================================
# Deterministic Scorer Tests
# =============================================================================

class TestRouteAccuracyScorer:
    """Tests for route accuracy scoring."""
    
    def test_perfect_routing(self, sample_scenario: EvalScenario, sample_metadata: Dict):
        """Test scoring when routing is perfect."""
        scorer = RouteAccuracyScorer()
        result = scorer.score(sample_scenario, "", sample_metadata)
        
        assert result.score == 1.0
        assert result.passed is True
        assert "Perfect routing" in result.details or "Correct" in result.details
    
    def test_missing_agent(self, sample_scenario: EvalScenario):
        """Test scoring when an agent is missing."""
        scorer = RouteAccuracyScorer()
        metadata = {"invoked_agents": ["jira_agent"]}  # Missing github_agent
        result = scorer.score(sample_scenario, "", metadata)
        
        assert result.score < 1.0
        assert result.passed is False
        assert "Missed" in result.details
    
    def test_extra_agent(self, sample_scenario: EvalScenario):
        """Test scoring when extra agents are invoked."""
        scorer = RouteAccuracyScorer()
        metadata = {"invoked_agents": ["jira_agent", "github_agent", "slack_agent"]}
        result = scorer.score(sample_scenario, "", metadata)
        
        assert result.score < 1.0
        assert "Extra" in result.details


class TestSourceCoverageScorer:
    """Tests for source coverage scoring."""
    
    def test_full_coverage(self, sample_scenario: EvalScenario, sample_metadata: Dict):
        """Test scoring when all sources are covered."""
        scorer = SourceCoverageScorer()
        result = scorer.score(sample_scenario, "", sample_metadata)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_partial_coverage(self, sample_scenario: EvalScenario):
        """Test scoring with partial source coverage."""
        scorer = SourceCoverageScorer()
        metadata = {"checked_sources": ["jira"]}  # Missing github
        result = scorer.score(sample_scenario, "", metadata)
        
        assert result.score == 0.5
        assert result.passed is False


class TestHallucinationDetector:
    """Tests for hallucination detection."""
    
    def test_no_hallucinations(
        self,
        sample_scenario: EvalScenario,
        sample_output_good: str,
        sample_metadata: Dict,
    ):
        """Test when output has no hallucinations."""
        scorer = HallucinationDetector()
        result = scorer.score(sample_scenario, sample_output_good, sample_metadata)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_fabricated_jira_id(self, sample_scenario: EvalScenario, sample_metadata: Dict):
        """Test detection of fabricated Jira IDs."""
        scorer = HallucinationDetector()
        output = "Found issue FAKE-999 which is critical"
        result = scorer.score(sample_scenario, output, sample_metadata)
        
        assert result.passed is False
        assert "Fabricated Jira IDs" in result.details
    
    def test_fabricated_github_id(self, sample_scenario: EvalScenario, sample_metadata: Dict):
        """Test detection of fabricated GitHub IDs."""
        scorer = HallucinationDetector()
        output = "PR #99999 needs review"
        result = scorer.score(sample_scenario, output, sample_metadata)
        
        assert result.passed is False
        assert "Fabricated GitHub IDs" in result.details


class TestLanguageComplianceScorer:
    """Tests for language compliance scoring."""
    
    def test_compliant_language(self, sample_scenario: EvalScenario, sample_output_good: str):
        """Test when language is compliant."""
        scorer = LanguageComplianceScorer()
        result = scorer.score(sample_scenario, sample_output_good, {})
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_prohibited_phrases(self, sample_scenario: EvalScenario, sample_output_bad: str):
        """Test detection of prohibited phrases."""
        scorer = LanguageComplianceScorer()
        result = scorer.score(sample_scenario, sample_output_bad, {})
        
        assert result.passed is False
        assert "Prohibited phrases found" in result.details


class TestStructureScorer:
    """Tests for structure scoring."""
    
    def test_valid_structure(
        self,
        sample_scenario: EvalScenario,
        sample_output_good: str,
    ):
        """Test when output has valid structure."""
        scorer = StructureScorer()
        result = scorer.score(sample_scenario, sample_output_good, {})
        
        assert result.score >= 0.75
        assert result.passed is True
    
    def test_missing_content(self, sample_scenario: EvalScenario):
        """Test when required content is missing."""
        scorer = StructureScorer()
        output = "Some output without required identifiers"
        result = scorer.score(sample_scenario, output, {})
        
        assert result.passed is False
        assert "Missing" in result.details


class TestIdentifierValidator:
    """Tests for identifier validation."""
    
    def test_all_identifiers_present(
        self,
        sample_scenario: EvalScenario,
        sample_output_good: str,
    ):
        """Test when all expected identifiers are present."""
        scorer = IdentifierValidator()
        result = scorer.score(sample_scenario, sample_output_good, {})
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_missing_identifiers(self, sample_scenario: EvalScenario):
        """Test when identifiers are missing."""
        scorer = IdentifierValidator()
        output = "No identifiers here"
        result = scorer.score(sample_scenario, output, {})
        
        assert result.passed is False
        assert "Missing identifiers" in result.details


class TestAlertAppropriatenessScorer:
    """Tests for alert appropriateness scoring."""
    
    def test_correct_no_alert(self, sample_scenario: EvalScenario, sample_metadata: Dict):
        """Test when correctly not alerting."""
        scorer = AlertAppropriatenessScorer()
        result = scorer.score(sample_scenario, "", sample_metadata)
        
        assert result.score == 1.0
        assert result.passed is True
    
    def test_false_positive_alert(self, sample_scenario: EvalScenario):
        """Test when alert sent but shouldn't have been."""
        scorer = AlertAppropriatenessScorer()
        metadata = {"alert_sent": True}
        result = scorer.score(sample_scenario, "", metadata)
        
        assert result.score == 0.0
        assert result.passed is False
        assert "FALSE POSITIVE" in result.details


class TestRunAllDeterministicScorers:
    """Tests for running all deterministic scorers."""
    
    def test_run_all_scorers(
        self,
        sample_scenario: EvalScenario,
        sample_output_good: str,
        sample_metadata: Dict,
    ):
        """Test running all deterministic scorers."""
        results = run_all_deterministic_scorers(
            sample_scenario,
            sample_output_good,
            sample_metadata,
        )
        
        assert len(results) == 7  # 7 deterministic scorers
        assert all(isinstance(r, ScoreResult) for r in results)


# =============================================================================
# Runner Tests
# =============================================================================

class TestMockToolExecutor:
    """Tests for mock tool executor."""
    
    def test_get_mock_response(self):
        """Test getting mock responses."""
        mocks = [
            MockToolResponse(
                tool_name="jira_search",
                response={"issues": []},
                success=True,
            ),
        ]
        executor = MockToolExecutor(mocks)
        
        response = executor.get_response("jira_search")
        assert response["success"] is True
        assert response["data"] == {"issues": []}
    
    def test_get_failed_response(self):
        """Test getting failed mock response."""
        mocks = [
            MockToolResponse(
                tool_name="jira_search",
                response=None,
                success=False,
                error="API error",
            ),
        ]
        executor = MockToolExecutor(mocks)
        
        response = executor.get_response("jira_search")
        assert response["success"] is False
        assert response["error"] == "API error"
    
    def test_get_metadata(self):
        """Test getting execution metadata."""
        mocks = [
            MockToolResponse(
                tool_name="jira_search",
                response={"issues": [{"key": "TEST-1"}]},
                success=True,
            ),
        ]
        executor = MockToolExecutor(mocks)
        executor.get_response("jira_search")
        
        metadata = executor.get_metadata()
        assert "jira_search" in metadata["called_tools"]
        assert "TEST-1" in metadata["jira_ids"]


class TestMockAgentRunner:
    """Tests for mock agent runner."""
    
    @pytest.mark.asyncio
    async def test_run_scenario(self, sample_scenario: EvalScenario):
        """Test running a scenario with mocks."""
        runner = MockAgentRunner(sample_scenario)
        output, metadata = await runner.run()
        
        assert len(output) > 0
        assert "invoked_agents" in metadata
        assert "checked_sources" in metadata


class TestEvaluationRunner:
    """Tests for the main evaluation runner."""
    
    @pytest.mark.asyncio
    async def test_run_single_scenario(self, sample_scenario: EvalScenario):
        """Test running a single scenario."""
        config = RunConfig(
            mode="mock",
            run_llm_judges=False,  # Skip LLM judges for unit tests
            run_trust_score=False,
            save_results=False,
        )
        runner = EvaluationRunner(config)
        
        result = await runner.run_scenario(sample_scenario)
        
        assert result.scenario_id == sample_scenario.id
        assert len(result.scores) > 0
    
    @pytest.mark.asyncio
    async def test_run_category(self):
        """Test running a category of scenarios."""
        config = RunConfig(
            mode="mock",
            run_llm_judges=False,
            run_trust_score=False,
            save_results=False,
        )
        runner = EvaluationRunner(config)
        
        results = await runner.run_category(ScenarioCategory.DAILY_DIGEST)
        
        assert len(results) == 6
        assert all(r.category == ScenarioCategory.DAILY_DIGEST for r in results)
    
    def test_get_summary(self):
        """Test getting evaluation summary."""
        config = RunConfig(save_results=False)
        runner = EvaluationRunner(config)
        
        # Add some mock results
        runner.results = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=True,
                trust_score=0.9,
                latency_ms=100,
            ),
            EvalResult(
                scenario_id="test_2",
                scenario_name="Test 2",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=False,
                trust_score=0.7,
                latency_ms=150,
            ),
        ]
        
        summary = runner.get_summary()
        
        assert summary.total_scenarios == 2
        assert summary.passed == 1
        assert summary.failed == 1
        assert summary.pass_rate == 0.5
        assert summary.avg_trust_score == 0.8


# =============================================================================
# Report Tests
# =============================================================================

class TestEvaluationReport:
    """Tests for evaluation report generation."""
    
    def test_generate_markdown(self):
        """Test generating markdown report."""
        results = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test query",
                output="test output",
                scores=[
                    ScoreResult(
                        scorer_name="route_accuracy",
                        score=1.0,
                        passed=True,
                        details="Perfect",
                    ),
                ],
                overall_passed=True,
                trust_score=0.9,
                latency_ms=100,
            ),
        ]
        
        report = EvaluationReport(results)
        markdown = report.generate_markdown()
        
        assert "# PM Agent Evaluation Report" in markdown
        assert "Summary" in markdown
        assert "Pass Rate" in markdown
    
    def test_generate_json(self):
        """Test generating JSON report."""
        results = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=True,
                trust_score=0.9,
                latency_ms=100,
            ),
        ]
        
        report = EvaluationReport(results)
        json_data = report.generate_json()
        
        assert "summary" in json_data
        assert "results" in json_data
        assert json_data["summary"]["total_scenarios"] == 1


class TestCompareRuns:
    """Tests for comparing evaluation runs."""
    
    def test_compare_runs_no_regression(self):
        """Test comparing runs with no regression."""
        baseline = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=True,
                trust_score=0.8,
                latency_ms=100,
            ),
        ]
        comparison = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=True,
                trust_score=0.9,  # Improved
                latency_ms=90,
            ),
        ]
        
        result = compare_runs(baseline, comparison)
        
        assert result.is_regression() is False
        assert result.trust_score_delta > 0
    
    def test_compare_runs_with_regression(self):
        """Test comparing runs with regression."""
        baseline = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=True,
                trust_score=0.9,
                latency_ms=100,
            ),
        ]
        comparison = [
            EvalResult(
                scenario_id="test_1",
                scenario_name="Test 1",
                category=ScenarioCategory.DAILY_DIGEST,
                query="test",
                output="output",
                scores=[],
                overall_passed=False,  # Regressed
                trust_score=0.7,
                latency_ms=100,
            ),
        ]
        
        result = compare_runs(baseline, comparison)
        
        assert result.is_regression() is True
        assert len(result.regressions) > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestEvalIntegration:
    """Integration tests for the evaluation suite."""
    
    @pytest.mark.asyncio
    async def test_full_evaluation_flow(self):
        """Test complete evaluation flow with mock mode."""
        config = RunConfig(
            mode="mock",
            run_llm_judges=False,
            run_trust_score=False,
            save_results=False,
            categories=[ScenarioCategory.DAILY_DIGEST],
        )
        
        runner = EvaluationRunner(config)
        await runner.run_all()
        
        summary = runner.get_summary()
        
        assert summary.total_scenarios == 6
        assert summary.passed + summary.failed + summary.errors == 6
    
    def test_scenario_coverage(self):
        """Test that all categories have scenarios."""
        for category in ScenarioCategory:
            scenarios = load_dataset(category)
            assert len(scenarios) > 0, f"Category {category} has no scenarios"
    
    def test_all_scenarios_have_required_fields(self):
        """Test that all scenarios have required fields."""
        all_datasets = load_all_datasets()
        
        for category, scenarios in all_datasets.items():
            for scenario in scenarios:
                assert scenario.id, f"Scenario missing ID in {category}"
                assert scenario.name, f"Scenario {scenario.id} missing name"
                assert scenario.query, f"Scenario {scenario.id} missing query"
                assert scenario.category, f"Scenario {scenario.id} missing category"

"""
Pytest configuration and fixtures for evaluation tests.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from evals.schema import (
    EvalScenario,
    EvalResult,
    GoldenOutput,
    MockToolResponse,
    QualityCriteria,
    ScenarioCategory,
    ScoreResult,
    ExpectedAgent,
    ExpectedSource,
)


@pytest.fixture
def jira_mock_response() -> MockToolResponse:
    """Standard Jira mock response."""
    return MockToolResponse(
        tool_name="jira_search",
        response={
            "issues": [
                {
                    "key": "OPIK-123",
                    "summary": "Test issue",
                    "status": "Open",
                    "priority": "High",
                },
                {
                    "key": "OPIK-124",
                    "summary": "Another issue",
                    "status": "In Progress",
                    "priority": "Medium",
                },
            ],
            "total": 2,
        },
        success=True,
    )


@pytest.fixture
def github_mock_response() -> MockToolResponse:
    """Standard GitHub mock response."""
    return MockToolResponse(
        tool_name="github_list_issues",
        response={
            "items": [
                {
                    "number": 456,
                    "title": "Bug fix",
                    "state": "open",
                    "labels": ["bug"],
                },
                {
                    "number": 457,
                    "title": "Feature request",
                    "state": "open",
                    "labels": ["enhancement"],
                },
            ],
        },
        success=True,
    )


@pytest.fixture
def slack_mock_response() -> MockToolResponse:
    """Standard Slack mock response."""
    return MockToolResponse(
        tool_name="slack_search_messages",
        response={
            "messages": [
                {
                    "channel": "#dev",
                    "text": "Deployed to staging",
                    "ts": "1234567890.123",
                },
            ],
        },
        success=True,
    )


@pytest.fixture
def failed_mock_response() -> MockToolResponse:
    """Failed tool response."""
    return MockToolResponse(
        tool_name="jira_search",
        response=None,
        success=False,
        error="API connection failed",
    )


@pytest.fixture
def basic_scenario(
    jira_mock_response: MockToolResponse,
    github_mock_response: MockToolResponse,
) -> EvalScenario:
    """Basic evaluation scenario with Jira and GitHub."""
    return EvalScenario(
        id="basic_001",
        name="Basic Test Scenario",
        description="A basic scenario for testing",
        category=ScenarioCategory.DAILY_DIGEST,
        query="Give me a daily digest",
        expected_agents=[ExpectedAgent.JIRA, ExpectedAgent.GITHUB],
        expected_sources=[ExpectedSource.JIRA, ExpectedSource.GITHUB],
        mock_responses=[jira_mock_response, github_mock_response],
        golden_output=GoldenOutput(
            must_contain=["OPIK-123", "#456"],
            must_not_contain=["all good"],
            expected_sections=["Jira", "GitHub"],
            expected_identifiers=["OPIK-123", "OPIK-124", "456", "457"],
            should_alert=False,
            min_trust_score=0.8,
        ),
        quality_criteria=QualityCriteria(
            relevance="Should summarize daily activity",
            completeness="Should cover both sources",
        ),
        tags=["basic", "test"],
    )


@pytest.fixture
def critical_scenario() -> EvalScenario:
    """Critical issue scenario that should trigger alert."""
    return EvalScenario(
        id="critical_001",
        name="Critical Issue Scenario",
        description="A P0 critical issue scenario",
        category=ScenarioCategory.CRITICAL_ISSUES,
        query="Check for critical issues",
        expected_agents=[ExpectedAgent.JIRA, ExpectedAgent.GITHUB, ExpectedAgent.SLACK],
        expected_sources=[ExpectedSource.JIRA, ExpectedSource.GITHUB, ExpectedSource.SLACK],
        mock_responses=[
            MockToolResponse(
                tool_name="jira_search",
                response={
                    "issues": [
                        {
                            "key": "OPIK-P0",
                            "summary": "PRODUCTION DOWN",
                            "status": "Open",
                            "priority": "Highest",
                        },
                    ],
                    "total": 1,
                },
                success=True,
            ),
        ],
        golden_output=GoldenOutput(
            must_contain=["OPIK-P0", "critical", "P0"],
            must_not_contain=["no critical issues"],
            expected_sections=["Critical"],
            expected_identifiers=["OPIK-P0"],
            should_alert=True,
            min_trust_score=0.9,
        ),
        tags=["critical", "p0", "alert"],
    )


@pytest.fixture
def passing_result() -> EvalResult:
    """A passing evaluation result."""
    return EvalResult(
        scenario_id="test_pass",
        scenario_name="Passing Test",
        category=ScenarioCategory.DAILY_DIGEST,
        query="test query",
        output="test output with OPIK-123 and #456",
        scores=[
            ScoreResult(
                scorer_name="route_accuracy",
                score=1.0,
                passed=True,
                details="Perfect routing",
            ),
            ScoreResult(
                scorer_name="hallucination_detector",
                score=1.0,
                passed=True,
                details="No hallucinations",
            ),
        ],
        overall_passed=True,
        trust_score=0.9,
        latency_ms=100,
    )


@pytest.fixture
def failing_result() -> EvalResult:
    """A failing evaluation result."""
    return EvalResult(
        scenario_id="test_fail",
        scenario_name="Failing Test",
        category=ScenarioCategory.DAILY_DIGEST,
        query="test query",
        output="everything looks fine",
        scores=[
            ScoreResult(
                scorer_name="language_compliance",
                score=0.0,
                passed=False,
                details="Prohibited phrases found: ['looks fine']",
            ),
        ],
        overall_passed=False,
        trust_score=0.5,
        latency_ms=150,
    )


@pytest.fixture
def sample_results(passing_result: EvalResult, failing_result: EvalResult) -> List[EvalResult]:
    """List of sample results for testing."""
    return [passing_result, failing_result]

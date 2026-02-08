"""
Evaluation Dataset Schema.

Defines the structure for evaluation scenarios and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ScenarioCategory(str, Enum):
    """Categories of evaluation scenarios."""
    DAILY_DIGEST = "daily_digest"
    CRITICAL_ISSUES = "critical_issues"
    COMPETITOR_RESEARCH = "competitor_research"
    ROUTING = "routing"
    HALLUCINATION = "hallucination"
    ALERTS = "alerts"


class ExpectedAgent(str, Enum):
    """Agents that can be invoked."""
    SUPERVISOR = "supervisor"
    SLACK = "slack_agent"
    JIRA = "jira_agent"
    GITHUB = "github_agent"
    MARKET_RESEARCH = "market_research_agent"
    NOTION = "notion_agent"


class ExpectedSource(str, Enum):
    """Data sources that should be checked."""
    JIRA = "jira"
    GITHUB = "github"
    SLACK = "slack"
    WEB = "web"
    NOTION = "notion"


@dataclass
class MockToolResponse:
    """Mock response for a tool call."""
    tool_name: str
    response: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


@dataclass
class QualityCriteria:
    """Criteria for LLM-as-judge scoring."""
    relevance: Optional[str] = None  # What makes response relevant
    completeness: Optional[str] = None  # What aspects should be covered
    accuracy: Optional[str] = None  # What facts should be accurate
    clarity: Optional[str] = None  # Structure/clarity expectations
    actionability: Optional[str] = None  # What actions should be recommended


@dataclass
class GoldenOutput:
    """Expected output patterns for validation."""
    must_contain: List[str] = field(default_factory=list)  # Required keywords/phrases
    must_not_contain: List[str] = field(default_factory=list)  # Prohibited content
    expected_sections: List[str] = field(default_factory=list)  # Required sections
    expected_identifiers: List[str] = field(default_factory=list)  # IDs that should appear
    should_alert: Optional[bool] = None  # Whether alert should be triggered
    min_trust_score: Optional[float] = None  # Minimum acceptable trust score


@dataclass
class EvalScenario:
    """A single evaluation scenario."""
    id: str
    name: str
    description: str
    category: ScenarioCategory
    query: str
    expected_agents: List[ExpectedAgent]
    expected_sources: List[ExpectedSource]
    mock_responses: List[MockToolResponse] = field(default_factory=list)
    golden_output: Optional[GoldenOutput] = None
    quality_criteria: Optional[QualityCriteria] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "query": self.query,
            "expected_agents": [a.value for a in self.expected_agents],
            "expected_sources": [s.value for s in self.expected_sources],
            "mock_responses": [
                {
                    "tool_name": m.tool_name,
                    "response": m.response,
                    "success": m.success,
                    "error": m.error,
                }
                for m in self.mock_responses
            ],
            "golden_output": {
                "must_contain": self.golden_output.must_contain,
                "must_not_contain": self.golden_output.must_not_contain,
                "expected_sections": self.golden_output.expected_sections,
                "expected_identifiers": self.golden_output.expected_identifiers,
                "should_alert": self.golden_output.should_alert,
                "min_trust_score": self.golden_output.min_trust_score,
            } if self.golden_output else None,
            "quality_criteria": {
                "relevance": self.quality_criteria.relevance,
                "completeness": self.quality_criteria.completeness,
                "accuracy": self.quality_criteria.accuracy,
                "clarity": self.quality_criteria.clarity,
                "actionability": self.quality_criteria.actionability,
            } if self.quality_criteria else None,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvalScenario":
        """Create from dictionary (JSON deserialization)."""
        mock_responses = [
            MockToolResponse(
                tool_name=m["tool_name"],
                response=m["response"],
                success=m.get("success", True),
                error=m.get("error"),
            )
            for m in data.get("mock_responses", [])
        ]
        
        golden_output = None
        if data.get("golden_output"):
            go = data["golden_output"]
            golden_output = GoldenOutput(
                must_contain=go.get("must_contain", []),
                must_not_contain=go.get("must_not_contain", []),
                expected_sections=go.get("expected_sections", []),
                expected_identifiers=go.get("expected_identifiers", []),
                should_alert=go.get("should_alert"),
                min_trust_score=go.get("min_trust_score"),
            )
        
        quality_criteria = None
        if data.get("quality_criteria"):
            qc = data["quality_criteria"]
            quality_criteria = QualityCriteria(
                relevance=qc.get("relevance"),
                completeness=qc.get("completeness"),
                accuracy=qc.get("accuracy"),
                clarity=qc.get("clarity"),
                actionability=qc.get("actionability"),
            )
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=ScenarioCategory(data["category"]),
            query=data["query"],
            expected_agents=[ExpectedAgent(a) for a in data.get("expected_agents", [])],
            expected_sources=[ExpectedSource(s) for s in data.get("expected_sources", [])],
            mock_responses=mock_responses,
            golden_output=golden_output,
            quality_criteria=quality_criteria,
            tags=data.get("tags", []),
        )


@dataclass
class ScoreResult:
    """Result from a single scorer."""
    scorer_name: str
    score: float  # 0.0 to 1.0 for deterministic, 1-5 for LLM-as-judge
    passed: bool
    details: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation result for a scenario."""
    scenario_id: str
    scenario_name: str
    category: ScenarioCategory
    query: str
    output: str
    scores: List[ScoreResult]
    overall_passed: bool
    trust_score: Optional[float] = None
    latency_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "category": self.category.value,
            "query": self.query,
            "output": self.output[:500] if self.output else None,  # Truncate for storage
            "scores": [
                {
                    "scorer_name": s.scorer_name,
                    "score": s.score,
                    "passed": s.passed,
                    "details": s.details,
                    "metadata": s.metadata,
                }
                for s in self.scores
            ],
            "overall_passed": self.overall_passed,
            "trust_score": self.trust_score,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class EvalSummary:
    """Summary of evaluation run."""
    total_scenarios: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_trust_score: float
    avg_latency_ms: float
    by_category: Dict[str, Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "avg_trust_score": self.avg_trust_score,
            "avg_latency_ms": self.avg_latency_ms,
            "by_category": self.by_category,
            "timestamp": self.timestamp,
        }

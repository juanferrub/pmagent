"""
LLM-as-Judge Scorers.

Quality scoring using LLM evaluation for subjective criteria
like relevance, completeness, accuracy, clarity, and actionability.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from evals.schema import EvalScenario, QualityCriteria, ScoreResult


# Default model for LLM-as-judge
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


@dataclass
class JudgeConfig:
    """Configuration for LLM judge."""
    model: str = DEFAULT_JUDGE_MODEL
    temperature: float = 0.0
    max_tokens: int = 500


class BaseLLMJudge(ABC):
    """Base class for LLM-as-judge scorers."""
    
    name: str = "base_llm_judge"
    dimension: str = "quality"
    
    def __init__(self, config: Optional[JudgeConfig] = None):
        """Initialize the LLM judge.
        
        Args:
            config: Optional configuration for the judge model.
        """
        self.config = config or JudgeConfig()
        self._llm = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-load the LLM."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        return self._llm
    
    @abstractmethod
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        """Generate the evaluation prompt for this dimension.
        
        Args:
            scenario: The evaluation scenario.
            output: The agent's output text.
            criteria: Optional specific criteria from the scenario.
            
        Returns:
            The evaluation prompt string.
        """
        pass
    
    def score(
        self,
        scenario: EvalScenario,
        output: str,
        metadata: Dict[str, Any],
    ) -> ScoreResult:
        """Score the output using LLM-as-judge.
        
        Args:
            scenario: The evaluation scenario.
            output: The agent's output text.
            metadata: Additional metadata.
            
        Returns:
            ScoreResult with 1-5 score and explanation.
        """
        # Get criteria from scenario if available
        criteria = None
        if scenario.quality_criteria:
            criteria = getattr(scenario.quality_criteria, self.dimension, None)
        
        # Generate evaluation prompt
        eval_prompt = self.get_evaluation_prompt(scenario, output, criteria)
        
        # System prompt for consistent scoring
        system_prompt = """You are an expert evaluator for AI agent responses.
Your task is to score the response on a scale of 1-5 based on the given criteria.

Scoring Guidelines:
- 5: Excellent - Fully meets all criteria with no issues
- 4: Good - Meets most criteria with minor issues
- 3: Acceptable - Meets basic criteria but has notable gaps
- 2: Poor - Fails to meet several important criteria
- 1: Very Poor - Fails to meet most criteria

You MUST respond with a JSON object containing:
{
    "score": <1-5>,
    "reasoning": "<brief explanation of the score>"
}

Be objective and consistent. Focus on the specific evaluation criteria provided."""
        
        try:
            # Call LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=eval_prompt),
            ]
            response = self.llm.invoke(messages)
            
            # Parse response
            result = json.loads(response.content)
            score = float(result["score"])
            reasoning = result["reasoning"]
            
            # Normalize score to 0-1 for consistency
            normalized_score = (score - 1) / 4  # Maps 1-5 to 0-1
            
            return ScoreResult(
                scorer_name=self.name,
                score=score,  # Keep 1-5 scale for LLM judges
                passed=score >= 4.0,
                details=reasoning,
                metadata={
                    "raw_score": score,
                    "normalized_score": normalized_score,
                    "criteria": criteria,
                },
            )
        except Exception as e:
            return ScoreResult(
                scorer_name=self.name,
                score=0.0,
                passed=False,
                details=f"LLM judge error: {str(e)}",
                metadata={"error": str(e)},
            )


class RelevanceScorer(BaseLLMJudge):
    """Scores how relevant the response is to the query."""
    
    name = "llm_relevance"
    dimension = "relevance"
    
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        base_criteria = criteria or "The response should directly address the user's query."
        
        return f"""Evaluate the RELEVANCE of this AI agent response.

USER QUERY:
{scenario.query}

AGENT RESPONSE:
{output}

EVALUATION CRITERIA:
{base_criteria}

Consider:
1. Does the response directly address what was asked?
2. Is the information provided related to the query?
3. Does it avoid irrelevant tangents?
4. Does it focus on the most important aspects?

Provide your evaluation as JSON with "score" (1-5) and "reasoning"."""


class CompletenessScorer(BaseLLMJudge):
    """Scores how complete the response is."""
    
    name = "llm_completeness"
    dimension = "completeness"
    
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        base_criteria = criteria or "The response should cover all aspects of the query."
        
        # Include expected sources/agents for context
        expected_context = ""
        if scenario.expected_sources:
            sources = [s.value for s in scenario.expected_sources]
            expected_context += f"\nExpected data sources: {sources}"
        
        return f"""Evaluate the COMPLETENESS of this AI agent response.

USER QUERY:
{scenario.query}

AGENT RESPONSE:
{output}

EVALUATION CRITERIA:
{base_criteria}
{expected_context}

Consider:
1. Does it address all parts of the query?
2. Are all relevant data sources covered?
3. Is important information missing?
4. Does it acknowledge what it couldn't find or verify?

Provide your evaluation as JSON with "score" (1-5) and "reasoning"."""


class AccuracyScorer(BaseLLMJudge):
    """Scores the factual accuracy of the response."""
    
    name = "llm_accuracy"
    dimension = "accuracy"
    
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        base_criteria = criteria or "All facts should be accurate based on available data."
        
        # Include mock responses as ground truth
        ground_truth = ""
        if scenario.mock_responses:
            ground_truth = "\nGROUND TRUTH DATA (from tool responses):\n"
            for mock in scenario.mock_responses:
                if mock.success:
                    ground_truth += f"- {mock.tool_name}: {json.dumps(mock.response, indent=2)[:500]}\n"
        
        return f"""Evaluate the ACCURACY of this AI agent response.

USER QUERY:
{scenario.query}

AGENT RESPONSE:
{output}

EVALUATION CRITERIA:
{base_criteria}
{ground_truth}

Consider:
1. Are all stated facts correct based on the ground truth?
2. Are IDs, numbers, and names accurate?
3. Are there any fabricated or hallucinated details?
4. Does it correctly represent the status of items?

Provide your evaluation as JSON with "score" (1-5) and "reasoning"."""


class ClarityScorer(BaseLLMJudge):
    """Scores how clear and well-structured the response is."""
    
    name = "llm_clarity"
    dimension = "clarity"
    
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        base_criteria = criteria or "The response should be clear, organized, and easy to understand."
        
        return f"""Evaluate the CLARITY of this AI agent response.

USER QUERY:
{scenario.query}

AGENT RESPONSE:
{output}

EVALUATION CRITERIA:
{base_criteria}

Consider:
1. Is the response well-organized with clear sections?
2. Is the language clear and professional?
3. Is it easy to find key information?
4. Does it use appropriate formatting (lists, headers)?
5. Is the length appropriate (not too verbose or too brief)?

Provide your evaluation as JSON with "score" (1-5) and "reasoning"."""


class ActionabilityScorer(BaseLLMJudge):
    """Scores how actionable the recommendations are."""
    
    name = "llm_actionability"
    dimension = "actionability"
    
    def get_evaluation_prompt(
        self,
        scenario: EvalScenario,
        output: str,
        criteria: Optional[str],
    ) -> str:
        base_criteria = criteria or "The response should provide actionable next steps when appropriate."
        
        return f"""Evaluate the ACTIONABILITY of this AI agent response.

USER QUERY:
{scenario.query}

AGENT RESPONSE:
{output}

EVALUATION CRITERIA:
{base_criteria}

Consider:
1. Does it provide clear next steps when appropriate?
2. Are recommendations specific and achievable?
3. Does it prioritize actions by importance?
4. Does it identify who should take action?
5. For informational queries, does it suggest follow-up if relevant?

Provide your evaluation as JSON with "score" (1-5) and "reasoning"."""


# Convenience function to run all LLM judges
def run_all_llm_judges(
    scenario: EvalScenario,
    output: str,
    metadata: Dict[str, Any],
    config: Optional[JudgeConfig] = None,
) -> List[ScoreResult]:
    """Run all LLM-as-judge scorers on the output.
    
    Args:
        scenario: The evaluation scenario.
        output: The agent's output text.
        metadata: Additional metadata from the run.
        config: Optional configuration for judge models.
        
    Returns:
        List of ScoreResult objects from all LLM judges.
    """
    judges = [
        RelevanceScorer(config),
        CompletenessScorer(config),
        AccuracyScorer(config),
        ClarityScorer(config),
        ActionabilityScorer(config),
    ]
    
    return [judge.score(scenario, output, metadata) for judge in judges]


def run_selected_llm_judges(
    scenario: EvalScenario,
    output: str,
    metadata: Dict[str, Any],
    dimensions: List[str],
    config: Optional[JudgeConfig] = None,
) -> List[ScoreResult]:
    """Run selected LLM-as-judge scorers.
    
    Args:
        scenario: The evaluation scenario.
        output: The agent's output text.
        metadata: Additional metadata from the run.
        dimensions: List of dimensions to evaluate (relevance, completeness, etc.).
        config: Optional configuration for judge models.
        
    Returns:
        List of ScoreResult objects from selected judges.
    """
    judge_map = {
        "relevance": RelevanceScorer,
        "completeness": CompletenessScorer,
        "accuracy": AccuracyScorer,
        "clarity": ClarityScorer,
        "actionability": ActionabilityScorer,
    }
    
    results = []
    for dim in dimensions:
        if dim in judge_map:
            judge = judge_map[dim](config)
            results.append(judge.score(scenario, output, metadata))
    
    return results

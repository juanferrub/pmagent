"""
Evaluation Datasets.

This module provides utilities for loading and managing evaluation datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from evals.schema import EvalScenario, ScenarioCategory


DATASETS_DIR = Path(__file__).parent


def load_dataset(category: ScenarioCategory) -> List[EvalScenario]:
    """Load all scenarios for a given category.
    
    Args:
        category: The scenario category to load.
        
    Returns:
        List of EvalScenario objects.
    """
    file_map = {
        ScenarioCategory.DAILY_DIGEST: "daily_digest.json",
        ScenarioCategory.CRITICAL_ISSUES: "critical_issues.json",
        ScenarioCategory.COMPETITOR_RESEARCH: "competitor_research.json",
        ScenarioCategory.ROUTING: "routing.json",
        ScenarioCategory.HALLUCINATION: "hallucination.json",
        ScenarioCategory.ALERTS: "alerts.json",
    }
    
    filepath = DATASETS_DIR / file_map[category]
    with open(filepath, "r") as f:
        data = json.load(f)
    
    return [EvalScenario.from_dict(s) for s in data["scenarios"]]


def load_all_datasets() -> Dict[ScenarioCategory, List[EvalScenario]]:
    """Load all evaluation datasets.
    
    Returns:
        Dictionary mapping categories to lists of scenarios.
    """
    return {category: load_dataset(category) for category in ScenarioCategory}


def load_scenario_by_id(scenario_id: str) -> Optional[EvalScenario]:
    """Load a specific scenario by its ID.
    
    Args:
        scenario_id: The unique scenario identifier.
        
    Returns:
        The matching EvalScenario or None if not found.
    """
    for category in ScenarioCategory:
        scenarios = load_dataset(category)
        for scenario in scenarios:
            if scenario.id == scenario_id:
                return scenario
    return None


def get_scenario_count() -> Dict[str, int]:
    """Get count of scenarios per category.
    
    Returns:
        Dictionary mapping category names to scenario counts.
    """
    counts = {}
    total = 0
    for category in ScenarioCategory:
        scenarios = load_dataset(category)
        counts[category.value] = len(scenarios)
        total += len(scenarios)
    counts["total"] = total
    return counts


def get_all_scenario_ids() -> List[str]:
    """Get all scenario IDs across all categories.
    
    Returns:
        List of all scenario IDs.
    """
    ids = []
    for category in ScenarioCategory:
        scenarios = load_dataset(category)
        ids.extend([s.id for s in scenarios])
    return ids

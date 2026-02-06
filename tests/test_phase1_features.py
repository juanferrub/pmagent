"""
Phase 1 Feature Tests - Acceptance Criteria Verification

Tests for:
1. Competitor Release Monitor
2. Proactive Alert System
3. Customer Voice Report
4. Status Update Generator
5. Integrated Daily Briefing
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta


# ============================================================
# 1. COMPETITOR RELEASE MONITOR TESTS
# ============================================================

class TestCompetitorReleaseMonitor:
    """
    Acceptance Criteria:
    - AC-1.1: Monitor GitHub releases from LangSmith, Langfuse, Arize Phoenix, W&B Weave
    - AC-1.2: Check releases within configurable time window (default 7 days)
    - AC-1.3: Return structured data with version, date, release notes
    - AC-1.4: Handle API failures gracefully with circuit breaker
    """
    
    def test_check_github_releases_returns_structured_data(self):
        """AC-1.1, AC-1.3: Returns releases from competitors with proper structure."""
        from src.tools.competitor_tools import check_github_releases
        
        result = check_github_releases.invoke({"days_back": 7})
        data = json.loads(result)
        
        # Should have required fields
        assert "period_days" in data
        assert "checked_at" in data
        assert "releases" in data
        
        # If releases found, verify structure
        for competitor, info in data.get("releases", {}).items():
            assert "repo" in info
            assert "releases" in info
            for release in info["releases"]:
                assert "tag" in release
                assert "published" in release
                assert "url" in release
    
    def test_check_github_releases_configurable_days(self):
        """AC-1.2: Days back parameter is respected."""
        from src.tools.competitor_tools import check_github_releases
        
        result = check_github_releases.invoke({"days_back": 14})
        data = json.loads(result)
        
        assert data["period_days"] == 14
    
    def test_competitor_repos_coverage(self):
        """AC-1.1: All major competitors are monitored."""
        from src.tools.competitor_tools import COMPETITOR_REPOS
        
        expected_competitors = ["langsmith", "langfuse", "arize_phoenix", "weave"]
        for competitor in expected_competitors:
            assert competitor in COMPETITOR_REPOS
    
    def test_check_competitor_changelogs(self):
        """AC-1.1: Can fetch competitor changelog pages."""
        from src.tools.competitor_tools import check_competitor_changelogs
        
        result = check_competitor_changelogs.invoke({"competitor": "langfuse"})
        data = json.loads(result)
        
        assert "checked_at" in data
        assert "changelogs" in data
    
    def test_get_competitor_github_activity(self):
        """AC-1.1: Can get competitor GitHub activity metrics."""
        from src.tools.competitor_tools import get_competitor_github_activity
        
        result = get_competitor_github_activity.invoke({"days_back": 7})
        data = json.loads(result)
        
        assert "activity" in data
        # Should have metrics for each competitor
        for competitor, activity in data.get("activity", {}).items():
            if "error" not in activity:
                assert "stars" in activity
                assert "open_issues" in activity
    
    def test_compare_competitor_features(self):
        """AC-1.1: Feature comparison matrix is available."""
        from src.tools.competitor_tools import compare_competitor_features
        
        result = compare_competitor_features.invoke({})
        data = json.loads(result)
        
        assert "features" in data
        assert "pricing_models" in data
        assert "opik" in data["features"].get("tracing", {})


# ============================================================
# 2. PROACTIVE ALERT SYSTEM TESTS
# ============================================================

class TestProactiveAlertSystem:
    """
    Acceptance Criteria:
    - AC-2.1: Detect P0/P1 Jira tickets within configurable time window
    - AC-2.2: Detect blocked/stale tickets
    - AC-2.3: Detect trending GitHub issues
    - AC-2.4: Send urgent alerts via email
    - AC-2.5: Return alert_needed flag for automation
    """
    
    def test_check_critical_jira_tickets_structure(self):
        """AC-2.1: Returns critical tickets with proper structure."""
        from src.tools.alert_tools import check_critical_jira_tickets
        
        result = check_critical_jira_tickets.invoke({"hours_back": 24})
        data = json.loads(result)
        
        assert "hours_checked" in data
        assert "critical_count" in data
        assert "tickets" in data
        assert "alert_needed" in data
        
        # alert_needed should be boolean
        assert isinstance(data["alert_needed"], bool)
    
    def test_check_blocked_tickets_structure(self):
        """AC-2.2: Returns blocked/stale tickets."""
        from src.tools.alert_tools import check_blocked_tickets
        
        result = check_blocked_tickets.invoke({"days_stale": 5})
        data = json.loads(result)
        
        assert "days_stale_threshold" in data
        assert "stale_count" in data
        assert "tickets" in data
        assert "alert_needed" in data
    
    def test_check_github_trending_issues_structure(self):
        """AC-2.3: Returns trending GitHub issues."""
        from src.tools.alert_tools import check_github_trending_issues
        
        result = check_github_trending_issues.invoke({
            "repo_name": "comet-ml/opik",
            "upvote_threshold": 5,
            "days_back": 7
        })
        
        # Handle potential JSON parsing issues from error responses
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            # If JSON parsing fails, check if it's an error string
            if "error" in result.lower():
                pytest.skip("GitHub API error - token may not be configured")
            raise
        
        # May return error if GitHub token not configured in test env
        if "error" in data:
            pytest.skip("GitHub token not configured")
        
        assert "repo" in data
        assert "trending_count" in data
        assert "issues" in data
        assert "alert_needed" in data
    
    def test_send_urgent_alert_structure(self):
        """AC-2.4: Can send urgent alerts."""
        from src.tools.alert_tools import send_urgent_alert
        
        result = send_urgent_alert.invoke({
            "subject": "Test Alert",
            "message": "This is a test alert message.",
            "priority": "high"
        })
        data = json.loads(result)
        
        assert "subject" in data
        assert "priority" in data
        assert "delivery" in data


# ============================================================
# 3. CUSTOMER VOICE REPORT TESTS
# ============================================================

class TestCustomerVoiceReport:
    """
    Acceptance Criteria:
    - AC-3.1: Aggregate feedback from Jira and GitHub
    - AC-3.2: Group requests by theme/component
    - AC-3.3: Identify top requested features
    - AC-3.4: Return structured data for reporting
    """
    
    def test_aggregate_customer_voice_structure(self):
        """AC-3.1, AC-3.4: Returns aggregated customer feedback."""
        from src.tools.pm_tools import aggregate_customer_voice
        
        result = aggregate_customer_voice.invoke({"days_back": 7})
        data = json.loads(result)
        
        assert "period_days" in data
        assert "total_requests" in data
        assert "themes" in data
        assert "top_requests" in data
    
    def test_aggregate_customer_voice_themes(self):
        """AC-3.2: Requests are grouped by theme."""
        from src.tools.pm_tools import aggregate_customer_voice
        
        result = aggregate_customer_voice.invoke({"days_back": 7})
        data = json.loads(result)
        
        for theme in data.get("themes", []):
            assert "theme" in theme
            assert "count" in theme
            assert "sources" in theme
    
    def test_analyze_feature_requests_structure(self):
        """AC-3.3: Feature request analysis available."""
        from src.tools.pm_tools import analyze_feature_requests
        
        result = analyze_feature_requests.invoke({"days_back": 30})
        data = json.loads(result)
        
        assert "total_requests" in data
        assert "top_themes" in data
        assert "recommendations" in data


# ============================================================
# 4. STATUS UPDATE GENERATOR TESTS
# ============================================================

class TestStatusUpdateGenerator:
    """
    Acceptance Criteria:
    - AC-4.1: Include shipped PRs from GitHub
    - AC-4.2: Include in-progress Jira tickets
    - AC-4.3: Include blocked items
    - AC-4.4: Include metrics (PRs merged, tickets closed, bugs fixed)
    - AC-4.5: Include next week priorities
    """
    
    def test_generate_status_update_structure(self):
        """AC-4.1-4.5: Returns complete status update."""
        from src.tools.pm_tools import generate_status_update
        
        result = generate_status_update.invoke({"days_back": 7})
        data = json.loads(result)
        
        assert "period_days" in data
        assert "status" in data
        
        status = data["status"]
        assert "shipped" in status  # AC-4.1
        assert "in_progress" in status  # AC-4.2
        assert "blocked" in status  # AC-4.3
        assert "metrics" in status  # AC-4.4
        assert "next_week" in status  # AC-4.5
    
    def test_generate_status_update_metrics(self):
        """AC-4.4: Metrics include required counts."""
        from src.tools.pm_tools import generate_status_update
        
        result = generate_status_update.invoke({"days_back": 7})
        data = json.loads(result)
        
        metrics = data["status"]["metrics"]
        assert "prs_merged" in metrics
        assert "tickets_closed" in metrics
        assert "bugs_fixed" in metrics
        assert "features_shipped" in metrics


# ============================================================
# 5. INTEGRATED DAILY BRIEFING TESTS
# ============================================================

class TestIntegratedDailyBriefing:
    """
    Acceptance Criteria:
    - AC-5.1: Daily digest scheduled job exists
    - AC-5.2: Weekly customer voice scheduled job exists
    - AC-5.3: Weekly status update scheduled job exists
    - AC-5.4: Critical alert check runs every 15 minutes
    - AC-5.5: All jobs can be manually triggered
    """
    
    def test_scheduled_job_functions_exist(self):
        """AC-5.1-5.4: All scheduled job functions are defined."""
        from api.background import (
            run_daily_digest,
            run_weekly_market_scan,
            run_hourly_check,
            run_critical_alert_check,
            run_weekly_customer_voice,
            run_weekly_status_update,
        )
        
        # Verify all functions exist and are callable
        assert callable(run_daily_digest)
        assert callable(run_weekly_market_scan)
        assert callable(run_hourly_check)
        assert callable(run_critical_alert_check)
        assert callable(run_weekly_customer_voice)
        assert callable(run_weekly_status_update)
    
    def test_scheduler_configuration(self):
        """AC-5.5: Scheduler configuration is correct."""
        from api.background import _parse_cron
        
        # Test cron parsing
        daily_cron = _parse_cron("0 8 * * 1-5")
        assert daily_cron["hour"] == "8"
        assert daily_cron["minute"] == "0"
        assert daily_cron["day_of_week"] == "1-5"
        
        weekly_cron = _parse_cron("0 8 * * 1")
        assert weekly_cron["day_of_week"] == "1"


# ============================================================
# 6. AGENT INTEGRATION TESTS
# ============================================================

class TestAgentIntegration:
    """
    Acceptance Criteria:
    - AC-6.1: Market research agent has competitor tools
    - AC-6.2: Jira agent has alert and PM tools
    - AC-6.3: GitHub agent has trending issues tool
    """
    
    def test_market_research_agent_tools_imported(self):
        """AC-6.1: Market research agent imports competitor tools."""
        from src.tools.competitor_tools import competitor_tools
        from src.tools.research_tools import research_tools
        
        # Verify competitor tools exist
        tool_names = [t.name for t in competitor_tools]
        assert "check_github_releases" in tool_names
        assert "check_competitor_changelogs" in tool_names
        assert "get_competitor_github_activity" in tool_names
        
        # Verify research tools exist
        research_names = [t.name for t in research_tools]
        assert "web_search" in research_names
    
    def test_jira_agent_tools_imported(self):
        """AC-6.2: Jira agent imports alert and PM tools."""
        from src.tools.alert_tools import alert_tools
        from src.tools.pm_tools import pm_tools
        from src.tools.jira_tools import jira_tools
        
        # Verify alert tools exist
        alert_names = [t.name for t in alert_tools]
        assert "check_critical_jira_tickets" in alert_names
        assert "check_blocked_tickets" in alert_names
        
        # Verify PM tools exist
        pm_names = [t.name for t in pm_tools]
        assert "aggregate_customer_voice" in pm_names
        assert "generate_status_update" in pm_names
        
        # Verify jira tools exist
        jira_names = [t.name for t in jira_tools]
        assert "search_jira_issues" in jira_names
    
    def test_github_agent_tools_imported(self):
        """AC-6.3: GitHub agent imports trending issues tool."""
        from src.tools.alert_tools import check_github_trending_issues
        from src.tools.github_tools import github_tools
        
        # Verify trending tool exists
        assert check_github_trending_issues is not None
        assert check_github_trending_issues.name == "check_github_trending_issues"
        
        # Verify github tools exist
        github_names = [t.name for t in github_tools]
        assert "list_github_issues" in github_names


# ============================================================
# 7. API ENDPOINT TESTS
# ============================================================

class TestAPIEndpoints:
    """
    Acceptance Criteria:
    - AC-7.1: /trigger/{job_name} endpoint exists
    - AC-7.2: /scheduler/status endpoint exists
    - AC-7.3: Manual trigger returns success/failure
    """
    
    def test_trigger_endpoint_defined(self):
        """AC-7.1: Trigger endpoint is defined in routes."""
        from api.routes import router
        
        # Check that trigger route exists
        routes = [r.path for r in router.routes]
        assert "/trigger/{job_name}" in routes
    
    def test_scheduler_status_endpoint_defined(self):
        """AC-7.2: Scheduler status endpoint is defined."""
        from api.routes import router
        
        routes = [r.path for r in router.routes]
        assert "/scheduler/status" in routes
    
    def test_trigger_job_names_defined(self):
        """AC-7.3: All job names are defined for triggering."""
        from api.routes import trigger_scheduled_job
        
        # The function should exist and be callable
        assert callable(trigger_scheduled_job)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

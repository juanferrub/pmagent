"""
Tests for capabilities tools.

Tests:
- check_capabilities: Reports integration status
- get_configured_projects: Returns configured projects
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest


class TestCheckCapabilities:
    """Tests for the check_capabilities tool."""
    
    def test_returns_json(self):
        """Should return valid JSON."""
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        
        # Should be valid JSON
        data = json.loads(result)
        assert "integrations" in data
        assert "summary" in data
        assert "checked_at" in data
    
    def test_includes_all_integrations(self):
        """Should check all integration types."""
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        data = json.loads(result)
        
        integrations = data["integrations"]
        assert "jira" in integrations
        assert "github" in integrations
        assert "slack" in integrations
        assert "notion" in integrations
        assert "web_research" in integrations
        assert "email" in integrations
        assert "whatsapp" in integrations
    
    def test_jira_configured_status(self):
        """Should report Jira as configured when credentials are set."""
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        data = json.loads(result)
        
        jira = data["integrations"]["jira"]
        # Test env has JIRA_URL and JIRA_API_TOKEN set
        assert jira["configured"] is True
    
    def test_summary_categorizes_integrations(self):
        """Summary should categorize integrations correctly."""
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        data = json.loads(result)
        
        summary = data["summary"]
        assert "fully_working" in summary
        assert "partial_or_issues" in summary
        assert "not_configured" in summary
        assert isinstance(summary["fully_working"], list)
    
    @patch("src.tools.capabilities_tools._check_jira_status")
    def test_handles_connection_failure(self, mock_jira):
        """Should handle connection failures gracefully."""
        mock_jira.return_value = {
            "name": "Jira",
            "configured": True,
            "can_read": False,
            "issues": ["Connection test failed: timeout"],
        }
        
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        data = json.loads(result)
        
        # Should still return valid response
        assert "integrations" in data
    
    def test_customer_field_status(self):
        """Should report customer field configuration status."""
        from src.tools.capabilities_tools import check_capabilities
        
        result = check_capabilities.invoke({})
        data = json.loads(result)
        
        jira = data["integrations"]["jira"]
        # Should have customer field status
        assert "customer_field_configured" in jira or "issues" in jira


class TestGetConfiguredProjects:
    """Tests for the get_configured_projects tool."""
    
    def test_returns_json(self):
        """Should return valid JSON."""
        from src.tools.capabilities_tools import get_configured_projects
        
        result = get_configured_projects.invoke({})
        
        data = json.loads(result)
        assert isinstance(data, dict)
    
    def test_includes_jira_projects(self):
        """Should include Jira project keys."""
        from src.tools.capabilities_tools import get_configured_projects
        
        result = get_configured_projects.invoke({})
        data = json.loads(result)
        
        assert "jira_projects" in data
        assert isinstance(data["jira_projects"], list)
        # Test env has PROD,SUPPORT
        assert "PROD" in data["jira_projects"]
        assert "SUPPORT" in data["jira_projects"]
    
    def test_includes_github_repos(self):
        """Should include GitHub repositories."""
        from src.tools.capabilities_tools import get_configured_projects
        
        result = get_configured_projects.invoke({})
        data = json.loads(result)
        
        assert "github_repositories" in data
        assert isinstance(data["github_repositories"], list)
    
    def test_includes_slack_channels(self):
        """Should include configured Slack channels."""
        from src.tools.capabilities_tools import get_configured_projects
        
        result = get_configured_projects.invoke({})
        data = json.loads(result)
        
        assert "slack_channels" in data
        assert "alert_channel" in data["slack_channels"]
        assert "summary_channel" in data["slack_channels"]
    
    def test_includes_timezone(self):
        """Should include configured timezone."""
        from src.tools.capabilities_tools import get_configured_projects
        
        result = get_configured_projects.invoke({})
        data = json.loads(result)
        
        assert "timezone" in data
        assert data["timezone"] == "Europe/Madrid"  # From test env

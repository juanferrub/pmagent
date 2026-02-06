"""
Tests for configuration module.

Validates:
- AC-8.1: Environment variable management
- Config loading and LLM factory
"""

from __future__ import annotations

import os

import pytest

from src.config import get_settings, get_llm, Settings


class TestSettings:
    """Test settings loading."""

    def test_settings_load_from_env(self, settings):
        assert settings.anthropic_api_key == "test-key"
        assert settings.slack_bot_token == "xoxb-test"
        assert settings.jira_url == "https://test.atlassian.net"
        assert settings.timezone == "Europe/Madrid"

    def test_jira_projects_property(self, settings):
        projects = settings.jira_projects
        assert projects == ["PROD", "SUPPORT"]

    def test_github_repo_list_property(self, settings):
        repos = settings.github_repo_list
        assert repos == ["org/repo1"]

    def test_settings_defaults(self):
        # Test with minimal env
        s = Settings(
            anthropic_api_key="key",
            jira_project_keys="",
            github_repos="",
        )
        assert s.default_llm_provider == "anthropic"
        assert s.api_port == 8000
        assert s.timezone == "Europe/Madrid"

    def test_auth_token_from_env(self, settings):
        assert settings.api_auth_token == "test-auth-token"


class TestGetLLM:
    """Test LLM factory."""

    def test_get_llm_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(provider="unknown_provider")

    def test_get_llm_anthropic(self):
        # This will try to import langchain_anthropic
        # In test env it should be installed
        try:
            llm = get_llm(provider="anthropic", model="claude-sonnet-4-20250514")
            assert llm is not None
        except ImportError:
            pytest.skip("langchain-anthropic not installed")

    def test_get_llm_openai(self):
        try:
            llm = get_llm(provider="openai", model="gpt-4o")
            assert llm is not None
        except ImportError:
            pytest.skip("langchain-openai not installed")

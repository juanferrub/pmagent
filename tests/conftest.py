"""
Shared test fixtures and configuration.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test env vars before any imports that might read config
os.environ.update({
    "ANTHROPIC_API_KEY": "test-key",
    "OPENAI_API_KEY": "test-key",
    "SLACK_BOT_TOKEN": "xoxb-test",
    "SLACK_SIGNING_SECRET": "test-secret",
    "JIRA_URL": "https://test.atlassian.net",
    "JIRA_USER_EMAIL": "test@test.com",
    "JIRA_API_TOKEN": "test-token",
    "JIRA_PROJECT_KEYS": "PROD,SUPPORT",
    "NOTION_API_KEY": "secret_test",
    "GITHUB_TOKEN": "ghp_test",
    "GITHUB_REPOS": "org/repo1",
    "TAVILY_API_KEY": "tvly-test",
    "REDDIT_CLIENT_ID": "test-id",
    "REDDIT_CLIENT_SECRET": "test-secret",
    "REDDIT_USER_AGENT": "test-agent/1.0",
    "POSTGRES_URI": "",
    "SQLITE_PATH": ":memory:",
    "API_AUTH_TOKEN": "test-auth-token",
    "TIMEZONE": "Europe/Madrid",
    "LANGCHAIN_TRACING_V2": "false",
})


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Reset the cached settings singleton before each test."""
    from src.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings():
    """Get test settings."""
    from src.config import get_settings
    return get_settings()


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value=MagicMock(content="Mock LLM response"))
    mock.ainvoke = AsyncMock(return_value=MagicMock(content="Mock LLM response"))
    mock.bind_tools = MagicMock(return_value=mock)
    return mock

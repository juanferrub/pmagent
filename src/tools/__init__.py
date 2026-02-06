"""PM Agent tools for integrations."""

from src.tools.slack_tools import slack_tools
from src.tools.jira_tools import jira_tools
from src.tools.github_tools import github_tools
from src.tools.notion_tools import notion_tools
from src.tools.research_tools import research_tools
from src.tools.whatsapp_tools import whatsapp_tools
from src.tools.email_tools import email_tools

ALL_TOOLS = (
    slack_tools
    + jira_tools
    + github_tools
    + notion_tools
    + research_tools
    + whatsapp_tools
    + email_tools
)

__all__ = [
    "slack_tools",
    "jira_tools",
    "github_tools",
    "notion_tools",
    "research_tools",
    "whatsapp_tools",
    "email_tools",
    "ALL_TOOLS",
]

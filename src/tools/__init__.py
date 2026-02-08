"""PM Agent tools for integrations."""

from src.tools.slack_tools import slack_tools
from src.tools.jira_tools import jira_tools
from src.tools.github_tools import github_tools
from src.tools.notion_tools import notion_tools
from src.tools.research_tools import research_tools
from src.tools.whatsapp_tools import whatsapp_tools
from src.tools.email_tools import email_tools
from src.tools.competitor_tools import competitor_tools
from src.tools.alert_tools import alert_tools
from src.tools.pm_tools import pm_tools
from src.tools.capabilities_tools import capabilities_tools

ALL_TOOLS = (
    slack_tools
    + jira_tools
    + github_tools
    + notion_tools
    + research_tools
    + whatsapp_tools
    + email_tools
    + competitor_tools
    + alert_tools
    + pm_tools
    + capabilities_tools
)

__all__ = [
    "slack_tools",
    "jira_tools",
    "github_tools",
    "notion_tools",
    "research_tools",
    "whatsapp_tools",
    "email_tools",
    "competitor_tools",
    "alert_tools",
    "pm_tools",
    "capabilities_tools",
    "ALL_TOOLS",
]

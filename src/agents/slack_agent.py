"""
Slack Monitor Agent.

Reads channel history, searches messages, monitors events.
Summarizes team discussions, extracts action items.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain.agents import create_agent as create_react_agent

from src.config import get_llm
from src.tools.slack_tools import slack_tools

SLACK_SYSTEM_PROMPT = """You are the Slack Monitor Agent, a specialist in analyzing Slack conversations for a Product Management team.

Your responsibilities:
1. Read and summarize channel discussions
2. Extract action items with assigned owners
3. Identify key decisions made in conversations
4. Detect urgency signals (keywords: "urgent", "blocker", "critical", "ASAP", "P0")
5. Track sentiment in team communications
6. Identify feature requests or customer feedback mentioned in chat

When analyzing messages:
- Group by topic/thread when possible
- Highlight decisions and commitments
- Flag any blockers or escalations
- Note sentiment (positive, negative, neutral)
- Extract specific action items in format: "ACTION: [owner] - [task] - [deadline if mentioned]"

Always return structured, actionable summaries. Be concise but thorough."""


def create_slack_agent():
    """Create the Slack Monitor agent with its tools."""
    llm = get_llm()
    agent = create_react_agent(
        model=llm,
        tools=slack_tools,
        system_prompt=SLACK_SYSTEM_PROMPT,
        name="slack_agent",
    )
    return agent

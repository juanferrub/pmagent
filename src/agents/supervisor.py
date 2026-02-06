"""
Supervisor Agent.

Central coordinator that receives input, classifies intent,
delegates to specialist agents, and aggregates results.
Uses langgraph-supervisor for the hierarchical multi-agent pattern.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage
from langgraph_supervisor import create_supervisor

from src.agents.slack_agent import create_slack_agent
from src.agents.jira_agent import create_jira_agent
from src.agents.github_agent import create_github_agent
from src.agents.market_research_agent import create_market_research_agent
from src.agents.notion_agent import create_notion_agent
from src.config import get_llm
from src.tools.whatsapp_tools import whatsapp_tools
from src.tools.email_tools import email_tools
from src.utils import logger

SUPERVISOR_SYSTEM_PROMPT = """You are the PM Agent Supervisor, an expert Product Management AI coordinator.

You manage a team of specialist agents:
1. **slack_agent** - Monitors Slack channels, reads messages, posts updates, extracts action items
2. **jira_agent** - Queries/creates/updates Jira issues, tracks priorities and roadmap
3. **github_agent** - Monitors GitHub repos, PRs, issues, analyzes code changes
4. **market_research_agent** - Web search, competitor analysis, Reddit monitoring, market trends
5. **notion_agent** - Creates/updates Notion pages, generates reports and documentation

You also have direct access to communication tools:
- **send_whatsapp_message** / **send_whatsapp_template** - Send WhatsApp messages to users
- **send_email_report** - Send HTML email reports via SMTP

Your job is to:
1. Understand the user's request or the trigger event
2. Decide which agent(s) to delegate to (you can use multiple agents)
3. Aggregate and synthesize results from agents
4. Provide a clear, actionable final response
5. When asked to "email me", "WhatsApp me", or "send me" something, use the appropriate communication tool

Delegation guidelines:
- For "daily digest" or "summary": Use slack_agent + jira_agent + github_agent, then notion_agent
- For "support ticket analysis": Use jira_agent + market_research_agent
- For "PR review" or "code changes": Use github_agent, possibly jira_agent
- For "market feedback" or "competitor": Use market_research_agent
- For "roadmap" or "report": Gather data first, then use notion_agent to create docs
- For "email me" or "send report": Gather data, then use send_email_report tool
- For "WhatsApp me": Gather data, then use send_whatsapp_message tool
- For general questions: Determine which agent(s) have the relevant data

Always synthesize the final answer yourself after receiving agent outputs.
Be concise, structured, and action-oriented in your final responses."""


def create_supervisor_graph(checkpointer=None):
    """
    Build the supervisor multi-agent graph.

    Returns a compiled LangGraph that can be invoked or streamed.
    """
    llm = get_llm()

    # Create specialist agents
    slack_agent = create_slack_agent()
    jira_agent = create_jira_agent()
    github_agent = create_github_agent()
    market_research_agent = create_market_research_agent()
    notion_agent = create_notion_agent()

    logger.info("creating_supervisor_graph", agents=[
        "slack_agent", "jira_agent", "github_agent",
        "market_research_agent", "notion_agent",
    ])

    # Build supervisor using langgraph-supervisor
    # Give the supervisor direct access to communication tools (email + WhatsApp)
    supervisor = create_supervisor(
        model=llm,
        agents=[
            slack_agent,
            jira_agent,
            github_agent,
            market_research_agent,
            notion_agent,
        ],
        tools=whatsapp_tools + email_tools,
        prompt=SUPERVISOR_SYSTEM_PROMPT,
        # Allow supervisor to call multiple agents in parallel
        parallel_tool_calls=True,
    )

    # Compile with checkpointer for persistence
    compiled = supervisor.compile(checkpointer=checkpointer)

    logger.info("supervisor_graph_compiled")
    return compiled

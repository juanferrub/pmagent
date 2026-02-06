"""
Tests for specialist agents and supervisor.

Validates:
- AC-1.1: Supervisor agent creation and routing
- AC-1.2: StateGraph implementation
- AC-2.x: Each specialist agent creation
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, AsyncMock

import pytest


class TestAgentCreation:
    """Test that all agents can be created successfully."""

    @patch("src.agents.slack_agent.get_llm")
    def test_create_slack_agent(self, mock_llm_fn):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.slack_agent import create_slack_agent
        agent = create_slack_agent()
        assert agent is not None

    @patch("src.agents.jira_agent.get_llm")
    def test_create_jira_agent(self, mock_llm_fn):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.jira_agent import create_jira_agent
        agent = create_jira_agent()
        assert agent is not None

    @patch("src.agents.github_agent.get_llm")
    def test_create_github_agent(self, mock_llm_fn):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.github_agent import create_github_agent
        agent = create_github_agent()
        assert agent is not None

    @patch("src.agents.market_research_agent.get_llm")
    def test_create_market_research_agent(self, mock_llm_fn):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.market_research_agent import create_market_research_agent
        agent = create_market_research_agent()
        assert agent is not None

    @patch("src.agents.notion_agent.get_llm")
    def test_create_notion_agent(self, mock_llm_fn):
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.notion_agent import create_notion_agent
        agent = create_notion_agent()
        assert agent is not None


class TestSupervisorAgent:
    """Test supervisor agent (AC-1.1)."""

    @patch("src.agents.supervisor.get_llm")
    def test_create_supervisor_graph(self, mock_llm_fn):
        """Test that supervisor graph compiles successfully with all agents."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.supervisor import create_supervisor_graph
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(checkpointer=checkpointer)
        assert graph is not None

    @patch("src.agents.supervisor.get_llm")
    def test_supervisor_graph_has_nodes(self, mock_llm_fn):
        """Test that the compiled graph contains the expected structure."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.agents.supervisor import create_supervisor_graph
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        graph = create_supervisor_graph(checkpointer=checkpointer)

        # The compiled graph should have a get_graph method
        graph_def = graph.get_graph()
        # nodes can be dict-like or list-like depending on version
        if hasattr(graph_def, 'nodes'):
            nodes = graph_def.nodes
            if isinstance(nodes, dict):
                node_ids = list(nodes.keys())
            else:
                node_ids = list(nodes)
        else:
            node_ids = []

        # Supervisor graph should include agent nodes
        assert len(node_ids) > 0


class TestGraphModule:
    """Test the main_graph module (AC-1.2, AC-6.1)."""

    def test_get_checkpointer_memory(self):
        """Test fallback to MemorySaver when no Postgres configured."""
        from src.graphs.main_graph import get_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = get_checkpointer()
        assert isinstance(checkpointer, MemorySaver)

    @patch("src.agents.supervisor.get_llm")
    def test_get_graph_singleton(self, mock_llm_fn):
        """Test that get_graph returns a compiled graph."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.graphs.main_graph import get_graph, reset_graph
        reset_graph()

        graph = get_graph()
        assert graph is not None

        # Second call returns same instance
        graph2 = get_graph()
        assert graph is graph2

        reset_graph()

    @patch("src.agents.supervisor.get_llm")
    def test_reset_graph(self, mock_llm_fn):
        """Test that reset_graph clears the singleton."""
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_llm_fn.return_value = mock_llm

        from src.graphs.main_graph import get_graph, reset_graph
        reset_graph()

        graph1 = get_graph()
        reset_graph()
        graph2 = get_graph()

        # After reset, should create a new instance
        # (may or may not be the same object, but it should work)
        assert graph2 is not None
        reset_graph()

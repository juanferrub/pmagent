"""
Main LangGraph definition for the PM Agent.

This module provides:
- get_checkpointer(): returns the configured checkpointer (Postgres or SQLite)
- get_graph(): returns the compiled supervisor graph with checkpointing
- invoke_graph(): convenience function to run the graph
- stream_graph(): convenience function to stream graph execution
- Opik observability: all graph invocations are traced automatically
"""

from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from langchain_core.messages import HumanMessage

from src.config import get_settings
from src.utils import logger


def _configure_opik():
    """
    Configure Opik observability if an API key is set.
    Sets the required environment variables so the Opik SDK picks them up.
    """
    settings = get_settings()
    if not settings.opik_api_key:
        logger.info("opik_disabled", reason="OPIK_API_KEY not set")
        return False

    # Set env vars that the Opik SDK reads on import
    os.environ.setdefault("OPIK_API_KEY", settings.opik_api_key)
    os.environ.setdefault("OPIK_WORKSPACE", settings.opik_workspace)
    os.environ.setdefault("OPIK_PROJECT_NAME", settings.opik_project_name)
    os.environ.setdefault("OPIK_URL_OVERRIDE", "https://www.comet.com/opik/api")

    logger.info(
        "opik_configured",
        workspace=settings.opik_workspace,
        project=settings.opik_project_name,
    )
    return True


def get_checkpointer():
    """
    Create the appropriate checkpointer based on config.
    Uses PostgresSaver if POSTGRES_URI is set, otherwise SQLiteSaver.
    """
    settings = get_settings()

    if settings.postgres_uri:
        try:
            from psycopg_pool import ConnectionPool
            from langgraph.checkpoint.postgres import PostgresSaver

            pool = ConnectionPool(
                settings.postgres_uri,
                max_size=10,
                kwargs={"autocommit": True},
            )
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("checkpointer_initialized", type="postgres")
            return checkpointer
        except Exception as e:
            logger.warning("postgres_checkpointer_failed", error=str(e), fallback="sqlite")

    # Fallback to SQLite (MemorySaver for tests)
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("checkpointer_initialized", type="memory")
    return MemorySaver()


_graph = None


def get_graph():
    """
    Get or create the singleton compiled supervisor graph.
    Wraps the graph with Opik's track_langgraph for full observability
    when OPIK_API_KEY is configured.
    """
    global _graph
    if _graph is None:
        from src.agents.supervisor import create_supervisor_graph

        checkpointer = get_checkpointer()
        compiled = create_supervisor_graph(checkpointer=checkpointer)

        # Wrap with Opik tracing if configured
        opik_enabled = _configure_opik()
        if opik_enabled:
            try:
                from opik.integrations.langchain import OpikTracer, track_langgraph

                settings = get_settings()
                opik_tracer = OpikTracer(
                    project_name=settings.opik_project_name,
                    tags=["pm-agent", "langgraph"],
                    metadata={"version": "1.0.0"},
                )
                compiled = track_langgraph(compiled, opik_tracer)
                logger.info("opik_tracing_enabled", project=settings.opik_project_name)
            except Exception as e:
                logger.warning("opik_tracing_failed", error=str(e))

        _graph = compiled
        logger.info("graph_singleton_created")
    return _graph


def reset_graph():
    """Reset the singleton (useful for testing)."""
    global _graph
    _graph = None


async def invoke_graph(
    query: str,
    thread_id: Optional[str] = None,
    trigger_type: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Invoke the PM Agent graph with a query.

    Args:
        query: User query or trigger description.
        thread_id: Thread ID for state persistence. Auto-generated if None.
        trigger_type: Type of trigger (manual, scheduled, webhook, slack_command).
        metadata: Optional metadata dict.

    Returns:
        The final graph state as a dict.
    """
    graph = get_graph()
    thread_id = thread_id or str(uuid.uuid4())

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            "trigger_type": trigger_type,
            **(metadata or {}),
        },
    }

    logger.info("invoke_graph", thread_id=thread_id, trigger_type=trigger_type, query=query[:100])

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )

    logger.info("invoke_graph_complete", thread_id=thread_id)
    return result


async def stream_graph(
    query: str,
    thread_id: Optional[str] = None,
    trigger_type: str = "manual",
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream the PM Agent graph execution.

    Yields events as the graph processes through nodes.
    """
    graph = get_graph()
    thread_id = thread_id or str(uuid.uuid4())

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"trigger_type": trigger_type},
    }

    logger.info("stream_graph", thread_id=thread_id)

    async for event in graph.astream(
        {"messages": [HumanMessage(content=query)]},
        config=config,
        stream_mode="updates",
    ):
        yield event

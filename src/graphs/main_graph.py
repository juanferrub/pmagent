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
    Configure Opik observability.
    
    Supports both:
    - Self-hosted Opik: Set OPIK_URL_OVERRIDE to your local instance URL
    - Opik Cloud: Set OPIK_API_KEY for cloud authentication
    """
    settings = get_settings()
    
    # Check if we have either a URL override (self-hosted) or API key (cloud)
    has_url_override = bool(settings.opik_url_override)
    has_api_key = bool(settings.opik_api_key)
    
    if not has_url_override and not has_api_key:
        logger.info("opik_disabled", reason="Neither OPIK_URL_OVERRIDE nor OPIK_API_KEY set")
        return False
    
    try:
        import opik
        
        if has_url_override:
            # Self-hosted Opik instance
            # Set env vars so the SDK internals also pick them up
            os.environ["OPIK_URL_OVERRIDE"] = settings.opik_url_override
            os.environ["OPIK_WORKSPACE"] = settings.opik_workspace
            os.environ["OPIK_PROJECT_NAME"] = settings.opik_project_name
            
            opik.configure(
                use_local=True,
                url=settings.opik_url_override,
            )
            logger.info(
                "opik_configured_self_hosted",
                url=settings.opik_url_override,
                project=settings.opik_project_name,
            )
        else:
            # Opik Cloud
            os.environ.setdefault("OPIK_API_KEY", settings.opik_api_key)
            os.environ.setdefault("OPIK_WORKSPACE", settings.opik_workspace)
            os.environ.setdefault("OPIK_PROJECT_NAME", settings.opik_project_name)
            os.environ.setdefault("OPIK_URL_OVERRIDE", "https://www.comet.com/opik/api")
            
            opik.configure(use_local=False)
            logger.info(
                "opik_configured_cloud",
                workspace=settings.opik_workspace,
                project=settings.opik_project_name,
            )
        
        return True
    except Exception as e:
        logger.warning("opik_configuration_failed", error=str(e))
        return False


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


def _extract_tool_messages(result: Dict[str, Any]) -> list:
    """Extract tool messages from graph result for grounding validation."""
    tool_messages = []
    
    if not result or "messages" not in result:
        return tool_messages
    
    for msg in result.get("messages", []):
        # Check for tool messages (ToolMessage type or tool-like structure)
        msg_type = type(msg).__name__
        
        if msg_type == "ToolMessage":
            tool_messages.append({
                "name": getattr(msg, "name", ""),
                "content": getattr(msg, "content", ""),
            })
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            # AIMessage with tool calls
            for tc in msg.tool_calls:
                tool_messages.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                })
        elif isinstance(msg, dict) and msg.get("type") == "tool":
            tool_messages.append({
                "name": msg.get("name", ""),
                "content": msg.get("content", ""),
            })
    
    return tool_messages


def _validate_grounding(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the final answer is grounded in tool outputs.
    
    Returns validation result with any violations found.
    """
    from src.grounding import assert_grounded, GroundingResult
    
    # Extract final answer
    output_text = ""
    if result and "messages" in result:
        messages = result["messages"]
        if messages:
            last_msg = messages[-1]
            output_text = getattr(last_msg, "content", str(last_msg))
    
    if not output_text:
        return {"validated": False, "reason": "No output to validate"}
    
    # Extract tool messages
    tool_messages = _extract_tool_messages(result)
    
    # Run grounding check
    grounding_result = assert_grounded(output_text, tool_messages, strict=False)
    
    return {
        "validated": True,
        "is_grounded": grounding_result.is_grounded,
        "violations_count": len(grounding_result.violations),
        "violations": [
            {
                "claim_type": v.claim_type,
                "matched_value": v.matched_value,
                "suggestion": v.suggestion,
            }
            for v in grounding_result.violations[:5]  # Limit to 5
        ],
        "verified_claims_count": len(grounding_result.verified_claims),
        "tool_sources_used": list(grounding_result.tool_sources_used),
    }


async def invoke_graph(
    query: str,
    thread_id: Optional[str] = None,
    trigger_type: str = "manual",
    metadata: Optional[Dict[str, Any]] = None,
    validate_grounding: bool = True,
) -> Dict[str, Any]:
    """
    Invoke the PM Agent graph with a query.

    Args:
        query: User query or trigger description.
        thread_id: Thread ID for state persistence. Auto-generated if None.
        trigger_type: Type of trigger (manual, scheduled, webhook, slack_command).
        metadata: Optional metadata dict.
        validate_grounding: If True, validate final answer is grounded in tool outputs.

    Returns:
        The final graph state as a dict, with grounding validation results.
    """
    # Reset evidence ledger, execution state, and message deduplicator at start of each run
    from src.evidence import reset_ledger
    from src.execution_state import reset_execution_state, get_execution_state
    from src.evidence_callback import get_evidence_callback
    from src.trust_score import calculate_trust_score
    from src.message_dedup import reset_deduplicator
    
    reset_ledger()
    reset_execution_state()
    reset_deduplicator()
    logger.info("trust_critical_run_initialized", thread_id=thread_id)
    
    graph = get_graph()
    thread_id = thread_id or str(uuid.uuid4())

    # Build callbacks list: start with the graph's existing callbacks (e.g. OpikTracer)
    # and append our evidence callback, so we don't clobber Opik tracing.
    existing_callbacks = list(getattr(graph, "config", {}).get("callbacks", []))
    existing_callbacks.append(get_evidence_callback())

    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": existing_callbacks,
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

    # Log evidence and execution state summary at end of run
    from src.evidence import get_ledger
    ledger = get_ledger()
    coverage = ledger.get_coverage_summary()
    exec_state = get_execution_state()
    
    # Calculate trust score
    output_text = ""
    if result and "messages" in result:
        messages = result["messages"]
        if messages:
            last_msg = messages[-1]
            output_text = getattr(last_msg, "content", str(last_msg))
    
    trust_result = calculate_trust_score(output_text=output_text, alert_was_sent=False)
    
    # Validate grounding if enabled
    grounding_validation = None
    if validate_grounding:
        grounding_validation = _validate_grounding(result)
        
        if grounding_validation.get("violations_count", 0) > 0:
            logger.warning(
                "grounding_violations_detected",
                thread_id=thread_id,
                violations_count=grounding_validation["violations_count"],
                violations=grounding_validation.get("violations", []),
            )
    
    logger.info(
        "invoke_graph_complete",
        thread_id=thread_id,
        evidence_entries=coverage["total_entries"],
        evidence_sources=coverage["sources_covered"],
        execution_complete=exec_state.is_complete(),
        execution_all_success=exec_state.is_all_success(),
        trust_score=round(trust_result.overall_score, 3),
        trust_grade=trust_result.get_grade(),
        grounding_validated=grounding_validation.get("is_grounded") if grounding_validation else None,
    )
    
    # Attach validation metadata to result
    if grounding_validation:
        result["_grounding_validation"] = grounding_validation
    
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

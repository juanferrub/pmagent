"""
Tool Wrapper for Evidence Recording.

This module provides decorators and utilities to automatically record
tool calls in the Evidence Ledger for evidence-based reporting.
"""

from __future__ import annotations

import functools
import json
from typing import Any, Callable, Dict, Optional

from langchain_core.tools import BaseTool, StructuredTool, tool

from src.evidence import EvidenceLedger, SourceType, get_ledger
from src.utils import logger


# Mapping of tool names to source types
TOOL_SOURCE_MAPPING = {
    # Jira tools
    "search_jira_issues": SourceType.JIRA,
    "get_jira_issue": SourceType.JIRA,
    "create_jira_issue": SourceType.JIRA,
    "update_jira_issue": SourceType.JIRA,
    "add_jira_comment": SourceType.JIRA,
    "check_critical_jira_tickets": SourceType.JIRA,
    "check_blocked_tickets": SourceType.JIRA,
    "aggregate_customer_voice": SourceType.JIRA,
    "analyze_feature_requests": SourceType.JIRA,
    "generate_status_update": SourceType.JIRA,
    
    # GitHub tools
    "list_github_issues": SourceType.GITHUB,
    "get_github_issue": SourceType.GITHUB,
    "list_github_prs": SourceType.GITHUB,
    "get_github_pr": SourceType.GITHUB,
    "get_github_repo_info": SourceType.GITHUB,
    "check_github_trending_issues": SourceType.GITHUB,
    "check_github_releases": SourceType.GITHUB,
    "get_competitor_github_activity": SourceType.GITHUB,
    
    # Slack tools
    "read_slack_channel": SourceType.SLACK,
    "search_slack_messages": SourceType.SLACK,
    "post_slack_message": SourceType.SLACK,
    "get_slack_thread": SourceType.SLACK,
    
    # Web/Research tools
    "web_search": SourceType.WEB,
    "tavily_search": SourceType.WEB,
    "search_reddit": SourceType.WEB,
    "check_competitor_changelogs": SourceType.COMPETITOR,
    "compare_competitor_features": SourceType.COMPETITOR,
    
    # Notion tools
    "search_notion": SourceType.NOTION,
    "get_notion_page": SourceType.NOTION,
    "create_notion_page": SourceType.NOTION,
    "update_notion_page": SourceType.NOTION,
}


def get_source_type_for_tool(tool_name: str) -> SourceType:
    """Get the source type for a tool name."""
    return TOOL_SOURCE_MAPPING.get(tool_name, SourceType.UNKNOWN)


def record_tool_evidence(
    tool_name: str,
    params: Dict[str, Any],
    result: Any,
    success: bool = True,
    error: Optional[str] = None,
) -> str:
    """
    Record a tool call in the evidence ledger.
    
    Args:
        tool_name: Name of the tool
        params: Parameters passed to the tool
        result: Result from the tool
        success: Whether the call succeeded
        error: Error message if failed
        
    Returns:
        Evidence entry ID
    """
    source_type = get_source_type_for_tool(tool_name)
    ledger = get_ledger()
    
    return ledger.record_tool_call(
        source_type=source_type,
        tool_name=tool_name,
        query_params=params,
        result=result,
        success=success,
        error=error,
    )


def with_evidence_recording(source_type: Optional[SourceType] = None):
    """
    Decorator to automatically record tool calls in the evidence ledger.
    
    Usage:
        @tool
        @with_evidence_recording(SourceType.JIRA)
        def my_jira_tool(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            actual_source = source_type or get_source_type_for_tool(tool_name)
            
            # Capture parameters
            params = kwargs.copy()
            if args:
                # Try to get parameter names from function signature
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        params[param_names[i]] = arg
            
            try:
                result = func(*args, **kwargs)
                
                # Check if result indicates an error
                success = True
                error = None
                if isinstance(result, str):
                    try:
                        result_dict = json.loads(result)
                        if "error" in result_dict:
                            success = False
                            error = result_dict.get("error")
                    except json.JSONDecodeError:
                        pass
                
                # Record in ledger
                record_tool_evidence(
                    tool_name=tool_name,
                    params=params,
                    result=result,
                    success=success,
                    error=error,
                )
                
                return result
                
            except Exception as e:
                # Record failed call
                record_tool_evidence(
                    tool_name=tool_name,
                    params=params,
                    result=None,
                    success=False,
                    error=str(e),
                )
                raise
        
        return wrapper
    return decorator


class EvidenceRecordingToolMixin:
    """
    Mixin class to add evidence recording to LangChain tools.
    
    Add this to tool classes to automatically record calls.
    """
    
    def _run(self, *args, **kwargs):
        """Override _run to add evidence recording."""
        tool_name = getattr(self, 'name', self.__class__.__name__)
        source_type = get_source_type_for_tool(tool_name)
        
        params = kwargs.copy()
        
        try:
            result = super()._run(*args, **kwargs)
            
            # Check for error in result
            success = True
            error = None
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if "error" in result_dict:
                        success = False
                        error = result_dict.get("error")
                except json.JSONDecodeError:
                    pass
            
            record_tool_evidence(
                tool_name=tool_name,
                params=params,
                result=result,
                success=success,
                error=error,
            )
            
            return result
            
        except Exception as e:
            record_tool_evidence(
                tool_name=tool_name,
                params=params,
                result=None,
                success=False,
                error=str(e),
            )
            raise


def wrap_tool_with_evidence(original_tool: BaseTool) -> BaseTool:
    """
    Wrap an existing LangChain tool to add evidence recording.
    
    This is useful for wrapping tools that are already defined.
    """
    original_func = original_tool.func if hasattr(original_tool, 'func') else original_tool._run
    tool_name = original_tool.name
    source_type = get_source_type_for_tool(tool_name)
    
    @functools.wraps(original_func)
    def wrapped_func(*args, **kwargs):
        params = kwargs.copy()
        
        try:
            result = original_func(*args, **kwargs)
            
            # Check for error in result
            success = True
            error = None
            if isinstance(result, str):
                try:
                    result_dict = json.loads(result)
                    if "error" in result_dict:
                        success = False
                        error = result_dict.get("error")
                except json.JSONDecodeError:
                    pass
            
            record_tool_evidence(
                tool_name=tool_name,
                params=params,
                result=result,
                success=success,
                error=error,
            )
            
            return result
            
        except Exception as e:
            record_tool_evidence(
                tool_name=tool_name,
                params=params,
                result=None,
                success=False,
                error=str(e),
            )
            raise
    
    # Create a new tool with the wrapped function
    if isinstance(original_tool, StructuredTool):
        return StructuredTool(
            name=original_tool.name,
            description=original_tool.description,
            func=wrapped_func,
            args_schema=original_tool.args_schema,
        )
    else:
        # For @tool decorated functions
        return tool(wrapped_func)


def wrap_all_tools_with_evidence(tools: list) -> list:
    """
    Wrap a list of tools with evidence recording.
    
    Args:
        tools: List of LangChain tools
        
    Returns:
        List of wrapped tools
    """
    wrapped = []
    for t in tools:
        try:
            wrapped.append(wrap_tool_with_evidence(t))
        except Exception as e:
            logger.warning(f"Could not wrap tool {t.name}: {e}")
            wrapped.append(t)  # Keep original if wrapping fails
    return wrapped

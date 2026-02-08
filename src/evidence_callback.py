"""
Evidence Recording Callback Handler.

This module provides a LangChain callback handler that automatically
records all tool calls in the Evidence Ledger AND updates the
Execution State for trust-critical operations.

This is the cleanest way to integrate evidence recording without
modifying every individual tool.

Version: Trust-Critical / Production
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.outputs import LLMResult

from src.evidence import EvidenceLedger, SourceType, get_ledger
from src.execution_state import (
    CheckType,
    ExecutionState,
    get_execution_state,
    record_check_start,
    record_check_success,
    record_check_failure,
)
from src.tool_wrapper import get_source_type_for_tool
from src.utils import logger


# Map source types to check types
SOURCE_TO_CHECK_TYPE = {
    SourceType.JIRA: CheckType.JIRA,
    SourceType.GITHUB: CheckType.GITHUB,
    SourceType.SLACK: CheckType.SLACK,
    SourceType.WEB: CheckType.WEB,
}


class EvidenceRecordingCallback(BaseCallbackHandler):
    """
    Callback handler that records all tool calls in the Evidence Ledger
    AND updates the Execution State for trust tracking.
    
    Attach this to your LangChain/LangGraph runs to automatically
    track evidence for all tool invocations.
    """
    
    def __init__(
        self,
        ledger: Optional[EvidenceLedger] = None,
        execution_state: Optional[ExecutionState] = None,
    ):
        super().__init__()
        self._ledger = ledger
        self._execution_state = execution_state
        self._pending_calls: Dict[str, Dict[str, Any]] = {}
        self._started_checks: set = set()  # Track which checks we've started
    
    @property
    def ledger(self) -> EvidenceLedger:
        """Get the evidence ledger (lazy init if not provided)."""
        if self._ledger is None:
            self._ledger = get_ledger()
        return self._ledger
    
    @property
    def execution_state(self) -> ExecutionState:
        """Get the execution state (lazy init if not provided)."""
        if self._execution_state is None:
            self._execution_state = get_execution_state()
        return self._execution_state
    
    def _get_check_type(self, source_type: SourceType) -> Optional[CheckType]:
        """Map source type to check type."""
        return SOURCE_TO_CHECK_TYPE.get(source_type)
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        
        # Parse input
        try:
            if isinstance(input_str, str):
                params = json.loads(input_str) if input_str.startswith("{") else {"input": input_str}
            else:
                params = input_str if isinstance(input_str, dict) else {"input": str(input_str)}
        except json.JSONDecodeError:
            params = {"input": input_str}
        
        source_type = get_source_type_for_tool(tool_name)
        
        # Store pending call info
        self._pending_calls[str(run_id)] = {
            "tool_name": tool_name,
            "params": params,
            "source_type": source_type,
        }
        
        # Update execution state - mark check as started
        check_type = self._get_check_type(source_type)
        if check_type and check_type not in self._started_checks:
            record_check_start(check_type)
            self._started_checks.add(check_type)
        
        logger.debug(
            "evidence_tool_start",
            tool_name=tool_name,
            run_id=str(run_id),
            source_type=source_type.value if source_type else "unknown",
        )
    
    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record when a tool completes successfully."""
        run_id_str = str(run_id)
        
        if run_id_str not in self._pending_calls:
            logger.warning("evidence_tool_end_no_start", run_id=run_id_str)
            return
        
        call_info = self._pending_calls.pop(run_id_str)
        
        # Check if output indicates an error
        success = True
        error = None
        if isinstance(output, str):
            try:
                output_dict = json.loads(output)
                if "error" in output_dict:
                    success = False
                    error = str(output_dict.get("error"))
            except json.JSONDecodeError:
                pass
        
        # Record in ledger
        entry_id = self.ledger.record_tool_call(
            source_type=call_info["source_type"],
            tool_name=call_info["tool_name"],
            query_params=call_info["params"],
            result=output,
            success=success,
            error=error,
        )
        
        # Update execution state with result
        check_type = self._get_check_type(call_info["source_type"])
        if check_type:
            if success:
                # Extract findings from output
                findings = self._extract_findings(output, call_info["source_type"])
                record_check_success(
                    check_type,
                    findings,
                    raw_output=str(output)[:500] if output else None,
                )
            else:
                record_check_failure(
                    check_type,
                    error or "Tool returned error in output",
                )
        
        logger.debug(
            "evidence_tool_end",
            tool_name=call_info["tool_name"],
            entry_id=entry_id,
            success=success,
        )
    
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Record when a tool fails with an error."""
        run_id_str = str(run_id)
        
        if run_id_str not in self._pending_calls:
            logger.warning("evidence_tool_error_no_start", run_id=run_id_str)
            return
        
        call_info = self._pending_calls.pop(run_id_str)
        
        # Record failed call
        entry_id = self.ledger.record_tool_call(
            source_type=call_info["source_type"],
            tool_name=call_info["tool_name"],
            query_params=call_info["params"],
            result=None,
            success=False,
            error=str(error),
        )
        
        # Update execution state with failure
        check_type = self._get_check_type(call_info["source_type"])
        if check_type:
            record_check_failure(check_type, str(error))
        
        logger.debug(
            "evidence_tool_error",
            tool_name=call_info["tool_name"],
            entry_id=entry_id,
            error=str(error),
        )
    
    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an agent takes an action (before tool execution)."""
        # This gives us another chance to capture tool calls
        tool_name = action.tool
        tool_input = action.tool_input
        
        logger.debug(
            "evidence_agent_action",
            tool=tool_name,
            run_id=str(run_id),
        )
    
    def _extract_findings(
        self, output: Any, source_type: SourceType
    ) -> List[Dict[str, Any]]:
        """Extract structured findings from tool output."""
        findings = []
        
        if not output:
            return findings
        
        # Try to parse as JSON
        try:
            if isinstance(output, str):
                data = json.loads(output)
            elif isinstance(output, dict):
                data = output
            else:
                return findings
        except json.JSONDecodeError:
            return findings
        
        # Extract findings based on source type
        if source_type == SourceType.JIRA:
            findings = self._extract_jira_findings(data)
        elif source_type == SourceType.GITHUB:
            findings = self._extract_github_findings(data)
        elif source_type == SourceType.SLACK:
            findings = self._extract_slack_findings(data)
        
        return findings
    
    def _extract_jira_findings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract Jira findings from tool output."""
        findings = []
        
        # Handle list of issues
        issues = data.get("issues", [])
        if isinstance(issues, list):
            for issue in issues:
                if isinstance(issue, dict):
                    findings.append({
                        "source": "jira",
                        "issue_id": issue.get("key", ""),
                        "priority": issue.get("priority", {}).get("name", "") if isinstance(issue.get("priority"), dict) else str(issue.get("priority", "")),
                        "status": issue.get("status", {}).get("name", "") if isinstance(issue.get("status"), dict) else str(issue.get("status", "")),
                        "summary": issue.get("summary", ""),
                    })
        
        # Handle single issue
        if "key" in data:
            findings.append({
                "source": "jira",
                "issue_id": data.get("key", ""),
                "priority": data.get("priority", {}).get("name", "") if isinstance(data.get("priority"), dict) else str(data.get("priority", "")),
                "status": data.get("status", {}).get("name", "") if isinstance(data.get("status"), dict) else str(data.get("status", "")),
                "summary": data.get("summary", ""),
            })
        
        return findings
    
    def _extract_github_findings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract GitHub findings from tool output."""
        findings = []
        
        # Handle list of issues/PRs
        items = data.get("items", data.get("issues", data.get("pull_requests", [])))
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    findings.append({
                        "source": "github",
                        "issue_id": f"{item.get('repository', 'unknown')}#{item.get('number', '')}",
                        "labels": [l.get("name", l) if isinstance(l, dict) else l for l in item.get("labels", [])],
                        "state": item.get("state", ""),
                        "repository": item.get("repository", ""),
                        "title": item.get("title", ""),
                        "url": item.get("html_url", ""),
                    })
        
        return findings
    
    def _extract_slack_findings(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract Slack findings from tool output."""
        findings = []
        
        # Handle messages
        messages = data.get("messages", [])
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    findings.append({
                        "source": "slack",
                        "channel": msg.get("channel", ""),
                        "timestamp": msg.get("ts", ""),
                        "message_excerpt": msg.get("text", "")[:200],
                        "permalink": msg.get("permalink", ""),
                    })
        
        return findings


def get_evidence_callback() -> EvidenceRecordingCallback:
    """Get an evidence recording callback with the current ledger."""
    return EvidenceRecordingCallback(get_ledger())


def create_evidence_config() -> Dict[str, Any]:
    """
    Create a config dict with evidence recording callback.
    
    Use this when invoking LangGraph:
        config = create_evidence_config()
        config["configurable"]["thread_id"] = thread_id
        result = await graph.ainvoke(input, config)
    """
    return {
        "callbacks": [get_evidence_callback()],
    }

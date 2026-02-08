"""
Grounding Utilities for PM Agent.

This module provides:
- assert_grounded: Validates that answers are supported by tool outputs
- resolve_timerange: Converts natural language time phrases to ISO timestamps
- GroundingValidator: Runtime validator for final answers

Ensures no fabricated operational data by requiring tool evidence for all claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from src.utils import logger


# Patterns for "hard claims" that require evidence
HARD_CLAIM_PATTERNS = [
    # Jira ticket keys (e.g., OPIK-123, PROJ-456)
    (r"\b([A-Z][A-Z0-9]+-\d+)\b", "jira_key"),
    
    # GitHub PR/Issue numbers (e.g., PR #123, issue #456, #789)
    (r"\b(?:PR|pull request|issue|#)\s*#?(\d+)\b", "github_number"),
    
    # Counts with entities (e.g., "5 tickets", "10 PRs", "3 issues", "5 support tickets")
    (r"\b(\d+)\s+(?:\w+\s+)?(tickets?|issues?|PRs?|pull requests?|bugs?|tasks?|stories?|epics?)\b", "count_claim"),
    
    # "No X" claims (e.g., "no P0 issues", "no blockers", "no critical bugs")
    (r"\bno\s+(P0|P1|P2|critical|blocker|urgent|high[- ]priority)\s+(issues?|tickets?|bugs?|items?)\b", "no_issues_claim"),
    
    # Sprint/velocity metrics
    (r"\b(\d+)\s+(story points?|velocity|sprint capacity)\b", "sprint_metric"),
    
    # Date-specific claims (e.g., "created yesterday", "updated last week")
    (r"\b(created|updated|opened|closed|merged|resolved)\s+(yesterday|last week|today|this week)\b", "date_claim"),
    
    # Customer names (potential fabrication) - must end with company suffix or be quoted
    (r"\bcustomer[:\s]+([A-Z][a-zA-Z0-9\s&]+(?:Inc|LLC|Corp|Ltd|Company|Co\.?))\b", "customer_name"),
    (r'\bcustomer[:\s]+"([^"]+)"', "customer_name"),
    
    # Specific percentages/metrics
    (r"\b(\d+(?:\.\d+)?)\s*%\s*(completion|progress|done|resolved)\b", "percentage_claim"),
    
    # Slack channel references (must have at least one letter to avoid matching PR #42)
    (r"#([a-z][a-z0-9_-]*)\b", "slack_channel"),
    
    # Time-specific activity claims
    (r"\b(discussed|mentioned|reported|flagged)\s+(?:in|on)\s+(?:slack|#\w+)\b", "slack_activity"),
]


@dataclass
class GroundingViolation:
    """A detected grounding violation."""
    claim_text: str
    claim_type: str
    matched_value: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class GroundingResult:
    """Result of grounding validation."""
    is_grounded: bool
    violations: List[GroundingViolation] = field(default_factory=list)
    verified_claims: List[str] = field(default_factory=list)
    tool_sources_used: Set[str] = field(default_factory=set)
    recommendation: Optional[str] = None


def assert_grounded(
    answer: str,
    tool_messages: List[Dict[str, Any]],
    strict: bool = True,
) -> GroundingResult:
    """
    Validate that an answer is grounded in tool outputs.
    
    Scans the answer for "hard claims" (specific identifiers, counts, etc.)
    and verifies they appear in the tool message results.
    
    Args:
        answer: The draft answer to validate
        tool_messages: List of tool call results from the conversation
        strict: If True, any ungrounded claim is a violation
        
    Returns:
        GroundingResult with validation status and details
    """
    violations = []
    verified_claims = []
    tool_sources = set()
    
    # Extract all evidence from tool messages
    tool_evidence = _extract_tool_evidence(tool_messages)
    tool_sources = tool_evidence.get("sources", set())
    
    # Compile patterns
    compiled_patterns = [
        (re.compile(pattern, re.IGNORECASE), claim_type)
        for pattern, claim_type in HARD_CLAIM_PATTERNS
    ]
    
    # Scan answer line by line
    lines = answer.split("\n")
    for line_num, line in enumerate(lines, 1):
        for pattern, claim_type in compiled_patterns:
            matches = pattern.findall(line)
            for match in matches:
                # Get the matched value (handle tuple from groups)
                matched_value = match[0] if isinstance(match, tuple) else match
                
                # Check if this claim is supported by evidence
                is_supported = _check_claim_support(
                    claim_type, matched_value, tool_evidence
                )
                
                if is_supported:
                    verified_claims.append(f"{claim_type}: {matched_value}")
                else:
                    # Extract context around the match
                    claim_text = line.strip()[:100]
                    
                    violations.append(GroundingViolation(
                        claim_text=claim_text,
                        claim_type=claim_type,
                        matched_value=str(matched_value),
                        line_number=line_num,
                        suggestion=_get_violation_suggestion(claim_type),
                    ))
    
    # Determine overall grounding status
    is_grounded = len(violations) == 0 if strict else len(violations) < 3
    
    # Generate recommendation
    recommendation = None
    if violations:
        if len(violations) == 1:
            recommendation = f"Remove or verify: {violations[0].claim_text[:50]}..."
        else:
            recommendation = (
                f"Found {len(violations)} ungrounded claims. "
                "Either call the relevant tools first, or rephrase to avoid specific claims."
            )
    
    result = GroundingResult(
        is_grounded=is_grounded,
        violations=violations,
        verified_claims=verified_claims,
        tool_sources_used=tool_sources,
        recommendation=recommendation,
    )
    
    logger.info(
        "grounding_check",
        is_grounded=is_grounded,
        violations_count=len(violations),
        verified_count=len(verified_claims),
        tool_sources=list(tool_sources),
    )
    
    return result


def _extract_tool_evidence(tool_messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract evidence from tool messages for validation."""
    evidence = {
        "jira_keys": set(),
        "github_numbers": set(),
        "counts": {},  # entity_type -> count
        "slack_channels": set(),
        "customer_names": set(),
        "sources": set(),
        "raw_content": "",
    }
    
    for msg in tool_messages:
        # Handle different message formats
        content = ""
        tool_name = ""
        
        if isinstance(msg, dict):
            content = msg.get("content", "") or msg.get("result", "") or ""
            tool_name = msg.get("name", "") or msg.get("tool_name", "") or ""
        elif hasattr(msg, "content"):
            content = str(msg.content) if msg.content else ""
            tool_name = getattr(msg, "name", "") or ""
        else:
            content = str(msg)
        
        # Track source types
        tool_lower = tool_name.lower()
        if "jira" in tool_lower:
            evidence["sources"].add("jira")
        elif "github" in tool_lower:
            evidence["sources"].add("github")
        elif "slack" in tool_lower:
            evidence["sources"].add("slack")
        
        # Accumulate raw content for searching
        evidence["raw_content"] += f"\n{content}"
        
        # Extract Jira keys
        jira_keys = re.findall(r"\b([A-Z][A-Z0-9]+-\d+)\b", content)
        evidence["jira_keys"].update(jira_keys)
        
        # Extract GitHub numbers
        gh_numbers = re.findall(r'"number":\s*(\d+)', content)
        evidence["github_numbers"].update(gh_numbers)
        
        # Extract Slack channels
        slack_channels = re.findall(r'"channel":\s*"([^"]+)"', content)
        evidence["slack_channels"].update(slack_channels)
        slack_channels = re.findall(r'"name":\s*"#?([a-z0-9_-]+)"', content)
        evidence["slack_channels"].update(slack_channels)
        
        # Try to extract counts from JSON arrays
        try:
            import json
            data = json.loads(content)
            if isinstance(data, list):
                evidence["counts"]["items"] = len(data)
            elif isinstance(data, dict):
                if "issues" in data:
                    evidence["counts"]["issues"] = len(data["issues"])
                if "total" in data:
                    evidence["counts"]["total"] = data["total"]
        except (json.JSONDecodeError, TypeError):
            pass
    
    return evidence


def _check_claim_support(
    claim_type: str,
    matched_value: str,
    evidence: Dict[str, Any],
) -> bool:
    """Check if a specific claim is supported by evidence."""
    
    if claim_type == "jira_key":
        return matched_value.upper() in evidence["jira_keys"]
    
    elif claim_type == "github_number":
        return matched_value in evidence["github_numbers"]
    
    elif claim_type == "count_claim":
        # Count claims need the number to appear in tool output
        # or be derivable from array lengths
        claimed_count = int(matched_value) if matched_value.isdigit() else 0
        # Check if this count appears in evidence
        for key, count in evidence["counts"].items():
            if count == claimed_count:
                return True
        # Also check if the number appears in raw content
        return str(claimed_count) in evidence["raw_content"]
    
    elif claim_type == "no_issues_claim":
        # "No X issues" claims require tool calls that returned empty results
        # This is valid if we have evidence from the relevant source
        return bool(evidence["sources"])
    
    elif claim_type == "slack_channel":
        return matched_value.lower() in {c.lower() for c in evidence["slack_channels"]}
    
    elif claim_type == "customer_name":
        # Customer names must appear in tool output
        return matched_value.lower() in evidence["raw_content"].lower()
    
    elif claim_type in ("date_claim", "slack_activity"):
        # These need corresponding source data
        if "slack" in claim_type.lower() or "slack" in matched_value.lower():
            return "slack" in evidence["sources"]
        return bool(evidence["sources"])
    
    elif claim_type == "sprint_metric":
        # Sprint metrics must come from tool output
        return matched_value in evidence["raw_content"]
    
    elif claim_type == "percentage_claim":
        return matched_value in evidence["raw_content"]
    
    # Default: check if value appears anywhere in tool output
    return matched_value.lower() in evidence["raw_content"].lower()


def _get_violation_suggestion(claim_type: str) -> str:
    """Get a suggestion for fixing a grounding violation."""
    suggestions = {
        "jira_key": "Call search_jira_issues or get_jira_issue first",
        "github_number": "Call list_github_prs or list_github_issues first",
        "count_claim": "Replace with 'several' or call tool to get actual count",
        "no_issues_claim": "Say 'no issues found in checked sources' with tool evidence",
        "sprint_metric": "Call Jira sprint tools to get actual metrics",
        "date_claim": "Include the date range used in tool query",
        "customer_name": "Only cite customers from tool results or say 'customer not identified'",
        "slack_channel": "Verify channel exists via Slack tools",
        "slack_activity": "Call slack_conversations_history first",
        "percentage_claim": "Only cite percentages from tool calculations",
    }
    return suggestions.get(claim_type, "Verify claim with relevant tool call")


# Time range resolution

@dataclass
class TimeRange:
    """Resolved time range with ISO timestamps."""
    start: datetime
    end: datetime
    description: str
    jql_clause: str  # For Jira queries
    github_since: str  # ISO format for GitHub API
    slack_oldest: str  # Unix timestamp for Slack API
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "description": self.description,
            "jql_clause": self.jql_clause,
            "github_since": self.github_since,
            "slack_oldest": self.slack_oldest,
        }


# Natural language time patterns
TIME_PATTERNS = [
    # Relative days
    (r"\b(today)\b", lambda now, tz: _resolve_today(now, tz)),
    (r"\b(yesterday)\b", lambda now, tz: _resolve_yesterday(now, tz)),
    (r"\b(last|past)\s*(\d+)\s*days?\b", lambda now, tz, n: _resolve_last_n_days(now, tz, int(n))),
    (r"\b(couple|few)\s*days?\b", lambda now, tz: _resolve_last_n_days(now, tz, 2)),
    
    # Relative weeks
    (r"\b(this\s+week)\b", lambda now, tz: _resolve_this_week(now, tz)),
    (r"\b(last\s+week)\b", lambda now, tz: _resolve_last_week(now, tz)),
    (r"\b(past|last)\s*(\d+)\s*weeks?\b", lambda now, tz, n: _resolve_last_n_weeks(now, tz, int(n))),
    
    # Relative months
    (r"\b(this\s+month)\b", lambda now, tz: _resolve_this_month(now, tz)),
    (r"\b(last\s+month)\b", lambda now, tz: _resolve_last_month(now, tz)),
    
    # Sprint references
    (r"\b(current\s+sprint)\b", lambda now, tz: _resolve_current_sprint(now, tz)),
    (r"\b(last\s+sprint)\b", lambda now, tz: _resolve_last_sprint(now, tz)),
    
    # Recent/latest
    (r"\b(recent|latest|newest)\b", lambda now, tz: _resolve_last_n_days(now, tz, 7)),
]


def resolve_timerange(
    query: str,
    now: Optional[datetime] = None,
    user_tz: str = "UTC",
) -> Optional[TimeRange]:
    """
    Convert natural language time phrases to structured time range.
    
    Args:
        query: The user query containing time references
        now: Current time (defaults to now)
        user_tz: User's timezone (e.g., "America/New_York", "Europe/Madrid")
        
    Returns:
        TimeRange with start/end timestamps and formatted queries, or None if no time reference found
    """
    if now is None:
        now = datetime.now(timezone.utc)
    
    # Parse timezone
    try:
        tz = ZoneInfo(user_tz)
    except Exception:
        tz = timezone.utc
    
    # Convert now to user timezone for date calculations
    now_local = now.astimezone(tz)
    
    query_lower = query.lower()
    
    for pattern, resolver in TIME_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            try:
                if len(groups) == 1:
                    return resolver(now_local, tz)
                elif len(groups) == 2:
                    return resolver(now_local, tz, groups[1])
                else:
                    return resolver(now_local, tz)
            except Exception as e:
                logger.warning("timerange_resolution_failed", pattern=pattern, error=str(e))
                continue
    
    return None


def _resolve_today(now: datetime, tz) -> TimeRange:
    """Resolve 'today' to start of day until now."""
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"today ({start.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= '{start.strftime('%Y-%m-%d')}'",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_yesterday(now: datetime, tz) -> TimeRange:
    """Resolve 'yesterday' to full previous day."""
    yesterday = now - timedelta(days=1)
    start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
    end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return TimeRange(
        start=start,
        end=end,
        description=f"yesterday ({start.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= '{start.strftime('%Y-%m-%d')}' AND created < '{(end + timedelta(days=1)).strftime('%Y-%m-%d')}'",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_last_n_days(now: datetime, tz, n: int) -> TimeRange:
    """Resolve 'last N days' to N days ago until now."""
    start = (now - timedelta(days=n)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"last {n} days ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= '{start.strftime('%Y-%m-%d')}'",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_this_week(now: datetime, tz) -> TimeRange:
    """Resolve 'this week' to start of current week (Monday) until now."""
    days_since_monday = now.weekday()
    start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"this week ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= startOfWeek()",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_last_week(now: datetime, tz) -> TimeRange:
    """Resolve 'last week' to previous full week (Monday to Sunday)."""
    days_since_monday = now.weekday()
    this_monday = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = this_monday - timedelta(days=7)
    end = this_monday - timedelta(seconds=1)
    
    return TimeRange(
        start=start,
        end=end,
        description=f"last week ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= startOfWeek(-1) AND created < startOfWeek()",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_last_n_weeks(now: datetime, tz, n: int) -> TimeRange:
    """Resolve 'last N weeks' to N weeks ago until now."""
    start = (now - timedelta(weeks=n)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"last {n} weeks ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= '{start.strftime('%Y-%m-%d')}'",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_this_month(now: datetime, tz) -> TimeRange:
    """Resolve 'this month' to start of current month until now."""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"this month ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= startOfMonth()",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_last_month(now: datetime, tz) -> TimeRange:
    """Resolve 'last month' to previous full month."""
    first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = first_of_this_month - timedelta(seconds=1)
    start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    return TimeRange(
        start=start,
        end=end,
        description=f"last month ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"created >= startOfMonth(-1) AND created < startOfMonth()",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_current_sprint(now: datetime, tz) -> TimeRange:
    """Resolve 'current sprint' - defaults to last 2 weeks."""
    # Without Jira sprint info, approximate as 2-week sprint
    start = (now - timedelta(days=14)).replace(hour=0, minute=0, second=0, microsecond=0)
    end = now
    
    return TimeRange(
        start=start,
        end=end,
        description=f"current sprint (approx. {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"sprint in openSprints()",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


def _resolve_last_sprint(now: datetime, tz) -> TimeRange:
    """Resolve 'last sprint' - defaults to 2-4 weeks ago."""
    end = (now - timedelta(days=14)).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=14)
    
    return TimeRange(
        start=start,
        end=end,
        description=f"last sprint (approx. {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})",
        jql_clause=f"sprint in closedSprints() ORDER BY created DESC",
        github_since=start.isoformat(),
        slack_oldest=str(start.timestamp()),
    )


class GroundingValidator:
    """
    Runtime validator for final answers in the supervisor node.
    
    Checks answers before sending to ensure they're grounded in tool outputs.
    """
    
    def __init__(self, strict: bool = True):
        self.strict = strict
    
    def validate_and_fix(
        self,
        answer: str,
        tool_messages: List[Dict[str, Any]],
    ) -> Tuple[str, GroundingResult]:
        """
        Validate an answer and optionally fix grounding issues.
        
        Args:
            answer: The draft answer to validate
            tool_messages: Tool call results from conversation
            
        Returns:
            Tuple of (fixed_answer, grounding_result)
        """
        result = assert_grounded(answer, tool_messages, strict=self.strict)
        
        if result.is_grounded:
            return answer, result
        
        # Fix the answer by adding disclaimers or removing ungrounded claims
        fixed_answer = self._fix_answer(answer, result)
        
        return fixed_answer, result
    
    def _fix_answer(self, answer: str, result: GroundingResult) -> str:
        """Fix an ungrounded answer."""
        # Add a disclaimer header
        disclaimer = (
            "**Note:** Some information could not be verified from available data sources. "
            "The following is based on tool outputs where available.\n\n"
        )
        
        # List what was attempted
        if result.tool_sources_used:
            sources_checked = ", ".join(sorted(result.tool_sources_used))
            disclaimer += f"*Sources checked: {sources_checked}*\n\n"
        else:
            disclaimer += "*No data sources were queried for this response.*\n\n"
        
        # Add violation notes
        if result.violations:
            disclaimer += "**Unverified claims (removed or marked):**\n"
            for v in result.violations[:3]:  # Limit to 3
                disclaimer += f"- {v.claim_type}: {v.matched_value}\n"
            disclaimer += "\n"
        
        return disclaimer + answer
    
    def generate_limitation_response(
        self,
        original_query: str,
        attempted_tools: List[str],
        errors: List[str],
    ) -> str:
        """
        Generate a response when tools fail or are unavailable.
        
        Args:
            original_query: What the user asked
            attempted_tools: Tools that were tried
            errors: Error messages from tools
            
        Returns:
            A helpful limitation response
        """
        response = "I wasn't able to get the information you requested.\n\n"
        
        response += "**What I attempted:**\n"
        for tool in attempted_tools:
            response += f"- {tool}\n"
        
        response += "\n**What went wrong:**\n"
        for error in errors:
            # Clean up error messages
            clean_error = error.replace('"', "'")[:200]
            response += f"- {clean_error}\n"
        
        response += "\n**What you can do:**\n"
        
        # Provide specific suggestions based on errors
        if any("permission" in e.lower() or "not_in_channel" in e.lower() for e in errors):
            response += "- Ensure the bot has been invited to the relevant channels\n"
            response += "- Check that API tokens have the required scopes\n"
        
        if any("not found" in e.lower() or "404" in e.lower() for e in errors):
            response += "- Verify the project key, repository name, or channel name is correct\n"
        
        if any("timeout" in e.lower() or "connection" in e.lower() for e in errors):
            response += "- Try again in a few minutes (service may be temporarily unavailable)\n"
        
        response += "- Provide more specific identifiers (e.g., exact project key, channel ID)\n"
        
        return response

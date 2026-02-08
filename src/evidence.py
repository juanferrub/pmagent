"""
Evidence Ledger and Claim Validation System.

This module provides:
- EvidenceLedger: Tracks all tool calls and their results with citations
- ClaimScanner: Parses reports and validates claims against evidence
- CoverageContract: Enforces required tool calls per report section
- SafetyGate: Blocks email sending without sufficient evidence

Prevents hallucinations by requiring evidence for every factual claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from src.utils import logger


class SourceType(str, Enum):
    """Types of data sources."""
    JIRA = "jira"
    GITHUB = "github"
    SLACK = "slack"
    WEB = "web"
    NOTION = "notion"
    COMPETITOR = "competitor"
    UNKNOWN = "unknown"


@dataclass
class EvidenceEntry:
    """A single piece of evidence from a tool call."""
    id: str
    source_type: SourceType
    tool_name: str
    query_params: Dict[str, Any]
    timestamp: str
    identifiers: List[str]  # Issue keys, PR numbers, URLs, etc.
    snippets: List[str]  # Short extracted content (<= 2 lines each)
    success: bool
    error: Optional[str] = None
    raw_result: Optional[str] = None  # Truncated raw result for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "tool_name": self.tool_name,
            "query_params": self.query_params,
            "timestamp": self.timestamp,
            "identifiers": self.identifiers,
            "snippets": self.snippets,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class Claim:
    """A factual claim extracted from a report."""
    text: str
    claim_type: str  # release, metric, status, activity, etc.
    section: str  # Which report section it belongs to
    required_source: SourceType
    evidence_ids: List[str] = field(default_factory=list)
    is_verified: bool = False
    uncertainty_note: Optional[str] = None


class EvidenceLedger:
    """
    Tracks all tool calls and their results for evidence-based reporting.
    
    Every tool call should register its results here. Reports must cite
    evidence from this ledger to make factual claims.
    """
    
    def __init__(self):
        self._entries: Dict[str, EvidenceEntry] = {}
        self._by_source: Dict[SourceType, List[str]] = defaultdict(list)
        self._entry_counter = 0
    
    def record_tool_call(
        self,
        source_type: SourceType,
        tool_name: str,
        query_params: Dict[str, Any],
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
    ) -> str:
        """
        Record a tool call and its result.
        
        Args:
            source_type: Type of data source (jira, github, etc.)
            tool_name: Name of the tool called
            query_params: Parameters passed to the tool
            result: The result from the tool (will be parsed for identifiers)
            success: Whether the call succeeded
            error: Error message if failed
            
        Returns:
            The evidence entry ID for citation
        """
        self._entry_counter += 1
        entry_id = f"ev-{source_type.value}-{self._entry_counter:04d}"
        
        # Extract identifiers and snippets from result
        identifiers, snippets = self._extract_evidence(source_type, result)
        
        # Truncate raw result for storage
        raw_str = str(result)[:500] if result else None
        
        entry = EvidenceEntry(
            id=entry_id,
            source_type=source_type,
            tool_name=tool_name,
            query_params=query_params,
            timestamp=datetime.now(timezone.utc).isoformat(),
            identifiers=identifiers,
            snippets=snippets[:5],  # Max 5 snippets
            success=success,
            error=error,
            raw_result=raw_str,
        )
        
        self._entries[entry_id] = entry
        self._by_source[source_type].append(entry_id)
        
        logger.info(
            "evidence_recorded",
            entry_id=entry_id,
            source_type=source_type.value,
            tool_name=tool_name,
            identifiers_count=len(identifiers),
            success=success,
        )
        
        return entry_id
    
    def _extract_evidence(
        self, source_type: SourceType, result: Any
    ) -> Tuple[List[str], List[str]]:
        """Extract identifiers and snippets from a tool result."""
        identifiers = []
        snippets = []
        
        if result is None:
            return identifiers, snippets
        
        # Try to parse as JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                # Plain text result
                snippets.append(result[:200])
                return identifiers, snippets
        
        if isinstance(result, dict):
            # Extract based on source type
            if source_type == SourceType.JIRA:
                identifiers.extend(self._extract_jira_identifiers(result))
                snippets.extend(self._extract_jira_snippets(result))
            elif source_type == SourceType.GITHUB:
                identifiers.extend(self._extract_github_identifiers(result))
                snippets.extend(self._extract_github_snippets(result))
            elif source_type == SourceType.SLACK:
                identifiers.extend(self._extract_slack_identifiers(result))
                snippets.extend(self._extract_slack_snippets(result))
            elif source_type in (SourceType.WEB, SourceType.COMPETITOR):
                identifiers.extend(self._extract_web_identifiers(result))
                snippets.extend(self._extract_web_snippets(result))
        elif isinstance(result, list):
            for item in result[:10]:  # Limit to first 10 items
                ids, snips = self._extract_evidence(source_type, item)
                identifiers.extend(ids)
                snippets.extend(snips)
        
        return identifiers[:20], snippets[:5]  # Limit counts
    
    def _extract_jira_identifiers(self, data: Dict) -> List[str]:
        """Extract Jira issue keys and URLs."""
        ids = []
        if "key" in data:
            ids.append(data["key"])
        if "issues" in data:
            for issue in data.get("issues", [])[:10]:
                if isinstance(issue, dict) and "key" in issue:
                    ids.append(issue["key"])
        if "url" in data and "jira" in str(data.get("url", "")).lower():
            ids.append(data["url"])
        return ids
    
    def _extract_jira_snippets(self, data: Dict) -> List[str]:
        """Extract Jira summaries and descriptions."""
        snippets = []
        if "summary" in data:
            snippets.append(f"[{data.get('key', '?')}] {data['summary'][:100]}")
        if "issues" in data:
            for issue in data.get("issues", [])[:3]:
                if isinstance(issue, dict):
                    key = issue.get("key", "?")
                    summary = issue.get("summary", "")[:80]
                    snippets.append(f"[{key}] {summary}")
        return snippets
    
    def _extract_github_identifiers(self, data: Dict) -> List[str]:
        """Extract GitHub PR/issue numbers and URLs."""
        ids = []
        if "number" in data:
            repo = data.get("repository", data.get("repo", ""))
            ids.append(f"{repo}#{data['number']}")
        if "html_url" in data:
            ids.append(data["html_url"])
        if "url" in data and "github.com" in str(data.get("url", "")):
            ids.append(data["url"])
        return ids
    
    def _extract_github_snippets(self, data: Dict) -> List[str]:
        """Extract GitHub titles and bodies."""
        snippets = []
        if "title" in data:
            num = data.get("number", "?")
            snippets.append(f"[#{num}] {data['title'][:100]}")
        return snippets
    
    def _extract_slack_identifiers(self, data: Dict) -> List[str]:
        """Extract Slack permalinks and channel IDs."""
        ids = []
        if "permalink" in data:
            ids.append(data["permalink"])
        if "channel_id" in data:
            ids.append(f"channel:{data['channel_id']}")
        if "ts" in data:
            ids.append(f"ts:{data['ts']}")
        return ids
    
    def _extract_slack_snippets(self, data: Dict) -> List[str]:
        """Extract Slack message text."""
        snippets = []
        if "text" in data:
            snippets.append(data["text"][:150])
        return snippets
    
    def _extract_web_identifiers(self, data: Dict) -> List[str]:
        """Extract URLs from web results."""
        ids = []
        if "url" in data:
            ids.append(data["url"])
        if "html_url" in data:
            ids.append(data["html_url"])
        return ids
    
    def _extract_web_snippets(self, data: Dict) -> List[str]:
        """Extract titles and content from web results."""
        snippets = []
        if "title" in data:
            snippets.append(data["title"][:100])
        if "content" in data:
            snippets.append(data["content"][:150])
        return snippets
    
    def get_entry(self, entry_id: str) -> Optional[EvidenceEntry]:
        """Get an evidence entry by ID."""
        return self._entries.get(entry_id)
    
    def get_entries_by_source(self, source_type: SourceType) -> List[EvidenceEntry]:
        """Get all entries for a source type."""
        return [self._entries[eid] for eid in self._by_source.get(source_type, [])]
    
    def get_successful_sources(self) -> Set[SourceType]:
        """Get set of source types with at least one successful call."""
        return {
            source for source, entry_ids in self._by_source.items()
            if any(self._entries[eid].success for eid in entry_ids)
        }
    
    def get_all_identifiers(self, source_type: Optional[SourceType] = None) -> List[str]:
        """Get all identifiers, optionally filtered by source type."""
        ids = []
        entries = (
            self.get_entries_by_source(source_type) if source_type
            else self._entries.values()
        )
        for entry in entries:
            if entry.success:
                ids.extend(entry.identifiers)
        return ids
    
    def has_evidence_for_source(self, source_type: SourceType) -> bool:
        """Check if we have successful evidence for a source type."""
        entries = self.get_entries_by_source(source_type)
        return any(e.success and e.identifiers for e in entries)
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """Get a summary of evidence coverage."""
        summary = {
            "total_entries": len(self._entries),
            "successful_entries": sum(1 for e in self._entries.values() if e.success),
            "failed_entries": sum(1 for e in self._entries.values() if not e.success),
            "sources_covered": list(self.get_successful_sources()),
            "by_source": {},
        }
        for source in SourceType:
            entries = self.get_entries_by_source(source)
            if entries:
                summary["by_source"][source.value] = {
                    "total": len(entries),
                    "successful": sum(1 for e in entries if e.success),
                    "identifiers": len(self.get_all_identifiers(source)),
                }
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ledger to dict."""
        return {
            "entries": [e.to_dict() for e in self._entries.values()],
            "coverage": self.get_coverage_summary(),
        }
    
    def clear(self):
        """Clear all entries (for testing)."""
        self._entries.clear()
        self._by_source.clear()
        self._entry_counter = 0


# Patterns that require evidence
EVIDENCE_REQUIRED_PATTERNS = [
    # Release/launch claims
    (r"\b(launched|released|announced|shipped|deployed)\b", "release"),
    (r"\b(new (version|release|feature|model))\b", "release"),
    (r"\bpricing (changed|updated|increased|decreased)\b", "pricing"),
    
    # Metric claims
    (r"\b(top \d+|highest|most|least)\s+(issues?|tickets?|bugs?|PRs?)\b", "metric"),
    (r"\bsprint velocity\b", "metric"),
    (r"\b\d+\s*(PRs?|issues?|tickets?|bugs?)\s*(merged|closed|opened|created)\b", "metric"),
    
    # Activity claims
    (r"\bslack highlights?\b", "activity"),
    (r"\bsupport escalations?\b", "activity"),
    (r"\bteam discussions?\b", "activity"),
    (r"\bcustomer feedback\b", "activity"),
    
    # Status claims
    (r"\b(blocked|stale|stuck|aging)\s*(tickets?|issues?|work)\b", "status"),
    (r"\bin progress\b.*\b(items?|tickets?|tasks?)\b", "status"),
]


class ClaimScanner:
    """
    Scans report text for factual claims and validates against evidence.
    
    Identifies claims that require evidence and checks if the EvidenceLedger
    has supporting entries.
    """
    
    def __init__(self, ledger: EvidenceLedger):
        self.ledger = ledger
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), claim_type)
            for pattern, claim_type in EVIDENCE_REQUIRED_PATTERNS
        ]
    
    def scan_for_claims(self, text: str, section: str = "general") -> List[Claim]:
        """
        Scan text for factual claims that require evidence.
        
        Args:
            text: The text to scan
            section: The report section name
            
        Returns:
            List of Claim objects found
        """
        claims = []
        
        # Split into sentences for granular analysis
        sentences = re.split(r'[.!?\n]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            for pattern, claim_type in self._compiled_patterns:
                if pattern.search(sentence):
                    required_source = self._infer_required_source(section, claim_type)
                    claims.append(Claim(
                        text=sentence[:200],
                        claim_type=claim_type,
                        section=section,
                        required_source=required_source,
                    ))
                    break  # One claim per sentence
        
        return claims
    
    def _infer_required_source(self, section: str, claim_type: str) -> SourceType:
        """Infer which source type should support a claim."""
        section_lower = section.lower()
        
        if "jira" in section_lower or "ticket" in section_lower:
            return SourceType.JIRA
        elif "github" in section_lower or "pr" in section_lower or "code" in section_lower:
            return SourceType.GITHUB
        elif "slack" in section_lower or "discussion" in section_lower:
            return SourceType.SLACK
        elif "competitor" in section_lower or "market" in section_lower or "external" in section_lower:
            return SourceType.WEB
        elif claim_type == "release":
            return SourceType.WEB
        elif claim_type == "metric":
            return SourceType.JIRA  # Most metrics come from Jira
        else:
            return SourceType.UNKNOWN
    
    def validate_claims(self, claims: List[Claim]) -> Tuple[List[Claim], List[Claim]]:
        """
        Validate claims against the evidence ledger.
        
        Returns:
            Tuple of (verified_claims, unverified_claims)
        """
        verified = []
        unverified = []
        
        for claim in claims:
            # Check if we have evidence for the required source
            if claim.required_source == SourceType.UNKNOWN:
                # Can't verify unknown source claims
                claim.uncertainty_note = "Unable to determine required evidence source"
                unverified.append(claim)
                continue
            
            entries = self.ledger.get_entries_by_source(claim.required_source)
            successful_entries = [e for e in entries if e.success]
            
            if not successful_entries:
                claim.uncertainty_note = f"No successful {claim.required_source.value} tool calls found"
                unverified.append(claim)
                continue
            
            # Check if any entry has relevant identifiers or snippets
            has_evidence = False
            for entry in successful_entries:
                if entry.identifiers or entry.snippets:
                    claim.evidence_ids.append(entry.id)
                    has_evidence = True
            
            if has_evidence:
                claim.is_verified = True
                verified.append(claim)
            else:
                claim.uncertainty_note = f"Tool calls succeeded but returned no identifiable evidence"
                unverified.append(claim)
        
        return verified, unverified
    
    def rewrite_unverified_claims(self, claims: List[Claim]) -> List[str]:
        """
        Generate uncertainty statements for unverified claims.
        
        Returns list of rewritten statements.
        """
        rewrites = []
        for claim in claims:
            if claim.uncertainty_note:
                rewrites.append(
                    f"**[Unverified]** {claim.text[:100]}... "
                    f"(Reason: {claim.uncertainty_note})"
                )
            else:
                rewrites.append(
                    f"**[Needs verification]** {claim.text[:100]}..."
                )
        return rewrites


# Section to source mapping
SECTION_SOURCE_REQUIREMENTS = {
    "jira": [SourceType.JIRA],
    "jira analysis": [SourceType.JIRA],
    "jira tickets": [SourceType.JIRA],
    "github": [SourceType.GITHUB],
    "github activity": [SourceType.GITHUB],
    "pull requests": [SourceType.GITHUB],
    "prs": [SourceType.GITHUB],
    "slack": [SourceType.SLACK],
    "slack highlights": [SourceType.SLACK],
    "team discussions": [SourceType.SLACK],
    "competitor": [SourceType.WEB, SourceType.COMPETITOR],
    "competitor updates": [SourceType.WEB, SourceType.COMPETITOR],
    "market": [SourceType.WEB],
    "market trends": [SourceType.WEB],
    "external": [SourceType.WEB],
    "llm provider": [SourceType.WEB],
}


class CoverageContract:
    """
    Enforces required tool calls per report section.
    
    Before generating a report section, the contract checks if the
    required data sources have been successfully queried.
    """
    
    def __init__(self, ledger: EvidenceLedger):
        self.ledger = ledger
    
    def check_section_coverage(self, section_name: str) -> Tuple[bool, List[str]]:
        """
        Check if a section has required coverage.
        
        Args:
            section_name: Name of the report section
            
        Returns:
            Tuple of (is_covered, list of missing sources)
        """
        section_lower = section_name.lower()
        
        # Find matching requirements
        required_sources = []
        for key, sources in SECTION_SOURCE_REQUIREMENTS.items():
            if key in section_lower:
                required_sources.extend(sources)
                break
        
        if not required_sources:
            # No specific requirements, allow
            return True, []
        
        # Check coverage
        missing = []
        covered_sources = self.ledger.get_successful_sources()
        
        for source in set(required_sources):
            if source not in covered_sources:
                missing.append(source.value)
        
        return len(missing) == 0, missing
    
    def get_coverage_report(self, requested_sections: List[str]) -> Dict[str, Any]:
        """
        Generate a coverage report for requested sections.
        
        Returns dict with coverage status per section and overall status.
        """
        report = {
            "sections": {},
            "all_covered": True,
            "missing_sources": set(),
            "available_sources": list(self.ledger.get_successful_sources()),
        }
        
        for section in requested_sections:
            is_covered, missing = self.check_section_coverage(section)
            report["sections"][section] = {
                "covered": is_covered,
                "missing": missing,
            }
            if not is_covered:
                report["all_covered"] = False
                report["missing_sources"].update(missing)
        
        report["missing_sources"] = list(report["missing_sources"])
        return report


@dataclass
class SafetyGateResult:
    """Result of the pre-send safety gate check."""
    can_send: bool
    tool_success_rate: float
    evidence_coverage_ratio: float
    sections_covered: Dict[str, bool]
    missing_sources: List[str]
    unverified_claims: List[str]
    needs_human_check: List[str]
    draft_report: Optional[str] = None
    rejection_reason: Optional[str] = None


class SafetyGate:
    """
    Pre-send safety gate that blocks email sending without sufficient evidence.
    
    Computes metrics and blocks send_email_report if thresholds are not met.
    """
    
    # Thresholds
    MIN_TOOL_SUCCESS_RATE = 0.5  # At least 50% of tool calls must succeed
    MIN_EVIDENCE_COVERAGE = 0.7  # At least 70% of claims must have evidence
    
    def __init__(self, ledger: EvidenceLedger):
        self.ledger = ledger
        self.scanner = ClaimScanner(ledger)
        self.contract = CoverageContract(ledger)
    
    def check(
        self,
        report_text: str,
        requested_sections: List[str],
    ) -> SafetyGateResult:
        """
        Check if a report is safe to send.
        
        Args:
            report_text: The report content to validate
            requested_sections: List of section names in the report
            
        Returns:
            SafetyGateResult with decision and details
        """
        # Calculate tool success rate
        coverage = self.ledger.get_coverage_summary()
        total_entries = coverage["total_entries"]
        successful_entries = coverage["successful_entries"]
        
        tool_success_rate = (
            successful_entries / total_entries if total_entries > 0 else 0.0
        )
        
        # Check section coverage
        coverage_report = self.contract.get_coverage_report(requested_sections)
        
        # Scan and validate claims
        all_claims = []
        for section in requested_sections:
            # Extract section content (simplified - in practice parse HTML/markdown)
            section_claims = self.scanner.scan_for_claims(report_text, section)
            all_claims.extend(section_claims)
        
        verified, unverified = self.scanner.validate_claims(all_claims)
        
        total_claims = len(all_claims)
        evidence_coverage_ratio = (
            len(verified) / total_claims if total_claims > 0 else 1.0
        )
        
        # Build needs human check list
        needs_human_check = []
        for claim in unverified:
            needs_human_check.append(
                f"[{claim.section}] {claim.text[:80]}... - {claim.uncertainty_note}"
            )
        
        # Add missing source warnings
        for source in coverage_report["missing_sources"]:
            needs_human_check.append(
                f"[Data Gap] No data from {source} - related sections may be incomplete"
            )
        
        # Determine if we can send
        can_send = True
        rejection_reasons = []
        
        if tool_success_rate < self.MIN_TOOL_SUCCESS_RATE:
            can_send = False
            rejection_reasons.append(
                f"Tool success rate ({tool_success_rate:.0%}) below threshold ({self.MIN_TOOL_SUCCESS_RATE:.0%})"
            )
        
        if evidence_coverage_ratio < self.MIN_EVIDENCE_COVERAGE:
            can_send = False
            rejection_reasons.append(
                f"Evidence coverage ({evidence_coverage_ratio:.0%}) below threshold ({self.MIN_EVIDENCE_COVERAGE:.0%})"
            )
        
        if not coverage_report["all_covered"]:
            can_send = False
            rejection_reasons.append(
                f"Missing required sources: {', '.join(coverage_report['missing_sources'])}"
            )
        
        # If we have zero tool calls, definitely block
        if total_entries == 0:
            can_send = False
            rejection_reasons.append("No tool calls recorded - report would be entirely fabricated")
        
        result = SafetyGateResult(
            can_send=can_send,
            tool_success_rate=tool_success_rate,
            evidence_coverage_ratio=evidence_coverage_ratio,
            sections_covered={
                s: info["covered"]
                for s, info in coverage_report["sections"].items()
            },
            missing_sources=coverage_report["missing_sources"],
            unverified_claims=self.scanner.rewrite_unverified_claims(unverified),
            needs_human_check=needs_human_check,
            rejection_reason="; ".join(rejection_reasons) if rejection_reasons else None,
        )
        
        logger.info(
            "safety_gate_check",
            can_send=can_send,
            tool_success_rate=tool_success_rate,
            evidence_coverage_ratio=evidence_coverage_ratio,
            unverified_claims=len(unverified),
            rejection_reason=result.rejection_reason,
        )
        
        return result
    
    def generate_incomplete_draft(
        self,
        original_report: str,
        gate_result: SafetyGateResult,
    ) -> str:
        """
        Generate an incomplete draft report with explicit gaps.
        
        Called when safety gate blocks sending.
        """
        draft = f"""
## ⚠️ DRAFT REPORT - NOT SENT (Evidence Validation Failed)

**Reason:** {gate_result.rejection_reason}

**Metrics:**
- Tool Success Rate: {gate_result.tool_success_rate:.0%}
- Evidence Coverage: {gate_result.evidence_coverage_ratio:.0%}

### Missing Data Sources
{chr(10).join(f"- {s}" for s in gate_result.missing_sources) or "- None"}

### Needs Human Verification
{chr(10).join(f"- {item}" for item in gate_result.needs_human_check) or "- None"}

### Unverified Claims (Removed or Marked)
{chr(10).join(f"- {claim}" for claim in gate_result.unverified_claims) or "- None"}

---

## Original Report Content (Unvalidated)

{original_report}

---

**To send this report:**
1. Ensure all required data sources are accessible
2. Re-run the data collection
3. Or explicitly approve sending incomplete draft
"""
        return draft


# Global ledger instance (reset per run)
_current_ledger: Optional[EvidenceLedger] = None


def get_ledger() -> EvidenceLedger:
    """Get or create the current evidence ledger."""
    global _current_ledger
    if _current_ledger is None:
        _current_ledger = EvidenceLedger()
    return _current_ledger


def reset_ledger():
    """Reset the ledger (call at start of each run)."""
    global _current_ledger
    _current_ledger = EvidenceLedger()
    return _current_ledger


def get_safety_gate() -> SafetyGate:
    """Get a safety gate instance with the current ledger."""
    return SafetyGate(get_ledger())

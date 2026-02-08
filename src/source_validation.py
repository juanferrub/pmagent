"""
Source Type Validation.

Validates that report sections use correct identifiers for their claimed source.
For example:
- "Jira Analysis" must use Jira identifiers (PROJ-123), not GitHub issue links
- "Slack highlights" must include channel/message permalinks
- "GitHub Activity" must use GitHub URLs or PR/issue numbers

Mismatched sources indicate potential hallucination or data confusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from src.evidence import EvidenceLedger, SourceType
from src.utils import logger


@dataclass
class ValidationResult:
    """Result of source validation for a section."""
    section_name: str
    expected_source: SourceType
    is_valid: bool
    found_identifiers: List[str]
    mismatched_identifiers: List[str]
    error_message: Optional[str] = None


# Regex patterns for identifying source types from identifiers
SOURCE_IDENTIFIER_PATTERNS = {
    SourceType.JIRA: [
        re.compile(r'\b[A-Z]{2,10}-\d+\b'),  # PROJ-123
        re.compile(r'https?://[^/]*jira[^/]*/browse/[A-Z]+-\d+', re.IGNORECASE),
        re.compile(r'https?://[^/]*atlassian[^/]*/browse/[A-Z]+-\d+', re.IGNORECASE),
    ],
    SourceType.GITHUB: [
        re.compile(r'https?://github\.com/[^/]+/[^/]+/(issues?|pull)/\d+'),
        re.compile(r'\b#\d{1,6}\b'),  # #123 (PR/issue number)
        re.compile(r'\b[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+#\d+\b'),  # repo/name#123
    ],
    SourceType.SLACK: [
        re.compile(r'https?://[^/]*slack\.com/archives/[A-Z0-9]+/p\d+'),
        re.compile(r'\bchannel:[A-Z0-9]+\b', re.IGNORECASE),
        re.compile(r'\bts:\d+\.\d+\b'),
    ],
    SourceType.WEB: [
        re.compile(r'https?://(?!github\.com|.*jira|.*slack)[^\s<>"]+'),
    ],
}

# Section name to expected source type
SECTION_EXPECTED_SOURCES = {
    "jira": SourceType.JIRA,
    "jira analysis": SourceType.JIRA,
    "jira tickets": SourceType.JIRA,
    "ticket analysis": SourceType.JIRA,
    "sprint": SourceType.JIRA,
    "backlog": SourceType.JIRA,
    
    "github": SourceType.GITHUB,
    "github activity": SourceType.GITHUB,
    "pull requests": SourceType.GITHUB,
    "prs": SourceType.GITHUB,
    "code changes": SourceType.GITHUB,
    "repository": SourceType.GITHUB,
    
    "slack": SourceType.SLACK,
    "slack highlights": SourceType.SLACK,
    "team discussions": SourceType.SLACK,
    "channel activity": SourceType.SLACK,
    
    "competitor": SourceType.WEB,
    "market": SourceType.WEB,
    "external": SourceType.WEB,
    "industry": SourceType.WEB,
}


class SourceValidator:
    """
    Validates that report sections use identifiers matching their claimed source.
    
    Detects when a section claims to be about one source (e.g., Jira) but
    actually contains identifiers from another source (e.g., GitHub).
    """
    
    def __init__(self, ledger: Optional[EvidenceLedger] = None):
        self.ledger = ledger
    
    def detect_source_from_text(self, text: str) -> Dict[SourceType, List[str]]:
        """
        Detect source types from identifiers found in text.
        
        Returns dict mapping source types to found identifiers.
        """
        found: Dict[SourceType, List[str]] = {}
        
        for source_type, patterns in SOURCE_IDENTIFIER_PATTERNS.items():
            identifiers = []
            for pattern in patterns:
                matches = pattern.findall(text)
                identifiers.extend(matches)
            if identifiers:
                found[source_type] = list(set(identifiers))[:10]  # Dedupe and limit
        
        return found
    
    def get_expected_source(self, section_name: str) -> Optional[SourceType]:
        """Get the expected source type for a section name."""
        section_lower = section_name.lower()
        
        for key, source in SECTION_EXPECTED_SOURCES.items():
            if key in section_lower:
                return source
        
        return None
    
    def validate_section(
        self,
        section_name: str,
        section_content: str,
    ) -> ValidationResult:
        """
        Validate that a section's content matches its expected source.
        
        Args:
            section_name: Name of the report section
            section_content: Content of the section
            
        Returns:
            ValidationResult with validation status and details
        """
        expected_source = self.get_expected_source(section_name)
        
        if expected_source is None:
            # No specific expectation, consider valid
            return ValidationResult(
                section_name=section_name,
                expected_source=SourceType.UNKNOWN,
                is_valid=True,
                found_identifiers=[],
                mismatched_identifiers=[],
            )
        
        # Detect what sources are actually in the content
        found_sources = self.detect_source_from_text(section_content)
        
        # Get identifiers for expected source
        expected_identifiers = found_sources.get(expected_source, [])
        
        # Find mismatched identifiers (from wrong sources)
        mismatched = []
        for source, identifiers in found_sources.items():
            if source != expected_source and source != SourceType.WEB:
                # WEB is often mixed in, so we don't flag it as mismatch
                mismatched.extend(identifiers)
        
        # Determine validity
        is_valid = True
        error_message = None
        
        # If section has content but no expected identifiers, it might be fabricated
        if len(section_content.strip()) > 100 and not expected_identifiers:
            # Check if we have evidence in ledger
            if self.ledger:
                has_ledger_evidence = self.ledger.has_evidence_for_source(expected_source)
                if not has_ledger_evidence:
                    is_valid = False
                    error_message = (
                        f"Section claims {expected_source.value} data but no "
                        f"{expected_source.value} identifiers found and no evidence in ledger"
                    )
        
        # If we have mismatched identifiers, flag as suspicious
        if mismatched and not expected_identifiers:
            is_valid = False
            error_message = (
                f"Section '{section_name}' should contain {expected_source.value} "
                f"identifiers but found identifiers from other sources: {mismatched[:3]}"
            )
        
        result = ValidationResult(
            section_name=section_name,
            expected_source=expected_source,
            is_valid=is_valid,
            found_identifiers=expected_identifiers,
            mismatched_identifiers=mismatched,
            error_message=error_message,
        )
        
        if not is_valid:
            logger.warning(
                "source_validation_failed",
                section=section_name,
                expected=expected_source.value,
                error=error_message,
            )
        
        return result
    
    def validate_report(
        self,
        sections: Dict[str, str],
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate all sections in a report.
        
        Args:
            sections: Dict mapping section names to content
            
        Returns:
            Tuple of (all_valid, list of ValidationResults)
        """
        results = []
        all_valid = True
        
        for section_name, content in sections.items():
            result = self.validate_section(section_name, content)
            results.append(result)
            if not result.is_valid:
                all_valid = False
        
        return all_valid, results
    
    def extract_sections_from_html(self, html_content: str) -> Dict[str, str]:
        """
        Extract sections from HTML report content.
        
        Looks for <h2>, <h3> headers and extracts content until next header.
        """
        sections = {}
        
        # Pattern to match headers
        header_pattern = re.compile(r'<h[23][^>]*>([^<]+)</h[23]>', re.IGNORECASE)
        
        # Find all headers and their positions
        headers = []
        for match in header_pattern.finditer(html_content):
            headers.append((match.start(), match.end(), match.group(1).strip()))
        
        # Extract content between headers
        for i, (start, end, title) in enumerate(headers):
            if i + 1 < len(headers):
                next_start = headers[i + 1][0]
                content = html_content[end:next_start]
            else:
                content = html_content[end:]
            
            # Clean HTML tags for analysis
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            
            sections[title] = content
        
        return sections


def validate_report_sources(
    report_html: str,
    ledger: Optional[EvidenceLedger] = None,
) -> Tuple[bool, List[ValidationResult], List[str]]:
    """
    Convenience function to validate a full report.
    
    Args:
        report_html: The HTML report content
        ledger: Optional evidence ledger for cross-referencing
        
    Returns:
        Tuple of (is_valid, results, escalation_messages)
    """
    validator = SourceValidator(ledger)
    sections = validator.extract_sections_from_html(report_html)
    
    is_valid, results = validator.validate_report(sections)
    
    # Generate escalation messages for invalid sections
    escalations = []
    for result in results:
        if not result.is_valid:
            escalations.append(
                f"⚠️ ESCALATE: {result.section_name} - {result.error_message}"
            )
    
    return is_valid, results, escalations

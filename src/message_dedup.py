"""
Message Deduplication Utilities.

Prevents the supervisor from echoing sub-agent responses verbatim,
reducing redundant output in the conversation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from src.utils import logger


@dataclass
class MessageDeduplicator:
    """
    Tracks seen message content to prevent duplicate responses.
    
    Used by the supervisor to avoid re-emitting sub-agent content verbatim.
    """
    
    # Content hashes we've seen
    _seen_hashes: Set[str] = field(default_factory=set)
    
    # Full content for similarity checking
    _seen_content: List[str] = field(default_factory=list)
    
    # Threshold for considering content "similar" (0-1)
    similarity_threshold: float = 0.8
    
    def _hash_content(self, content: str) -> str:
        """Generate a hash of content for quick lookup."""
        # Normalize: lowercase, strip whitespace, remove common prefixes
        normalized = content.lower().strip()
        
        # Remove common supervisor prefixes
        prefixes_to_strip = [
            "here's what i found:",
            "here is what i found:",
            "based on my analysis:",
            "the results show:",
            "i found the following:",
        ]
        for prefix in prefixes_to_strip:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity ratio between two strings."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def is_duplicate(self, content: str) -> bool:
        """
        Check if content is a duplicate of something we've seen.
        
        Returns True if:
        - Exact hash match
        - Very high similarity to previous content
        """
        if not content or len(content) < 50:
            return False
        
        content_hash = self._hash_content(content)
        
        # Exact match
        if content_hash in self._seen_hashes:
            logger.debug("duplicate_detected", method="exact_hash")
            return True
        
        # Similarity check against recent content
        for seen in self._seen_content[-10:]:  # Check last 10
            similarity = self._calculate_similarity(content, seen)
            if similarity >= self.similarity_threshold:
                logger.debug(
                    "duplicate_detected",
                    method="similarity",
                    similarity=round(similarity, 2),
                )
                return True
        
        return False
    
    def mark_seen(self, content: str) -> None:
        """Mark content as seen."""
        if not content or len(content) < 50:
            return
        
        content_hash = self._hash_content(content)
        self._seen_hashes.add(content_hash)
        self._seen_content.append(content)
        
        # Keep memory bounded
        if len(self._seen_content) > 100:
            self._seen_content = self._seen_content[-50:]
    
    def filter_duplicate_messages(
        self,
        messages: List[Any],
        content_extractor=None,
    ) -> List[Any]:
        """
        Filter out duplicate messages from a list.
        
        Args:
            messages: List of messages to filter
            content_extractor: Optional function to extract content from message
            
        Returns:
            Filtered list with duplicates removed
        """
        if content_extractor is None:
            content_extractor = lambda m: getattr(m, "content", str(m))
        
        filtered = []
        for msg in messages:
            content = content_extractor(msg)
            if not self.is_duplicate(content):
                filtered.append(msg)
                self.mark_seen(content)
        
        return filtered
    
    def clear(self) -> None:
        """Clear all seen content."""
        self._seen_hashes.clear()
        self._seen_content.clear()


# Global deduplicator instance (reset per run)
_deduplicator: Optional[MessageDeduplicator] = None


def get_deduplicator() -> MessageDeduplicator:
    """Get or create the message deduplicator."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = MessageDeduplicator()
    return _deduplicator


def reset_deduplicator() -> MessageDeduplicator:
    """Reset the deduplicator (call at start of each run)."""
    global _deduplicator
    _deduplicator = MessageDeduplicator()
    return _deduplicator


def deduplicate_response(
    supervisor_response: str,
    sub_agent_responses: List[str],
) -> str:
    """
    Remove duplicate content from supervisor response that was already
    provided by sub-agents.
    
    Args:
        supervisor_response: The supervisor's draft response
        sub_agent_responses: List of sub-agent response contents
        
    Returns:
        Cleaned supervisor response with duplicates removed or summarized
    """
    dedup = get_deduplicator()
    
    # Mark all sub-agent responses as seen
    for response in sub_agent_responses:
        dedup.mark_seen(response)
    
    # Check if supervisor response is mostly duplicate
    if dedup.is_duplicate(supervisor_response):
        # Return a brief summary instead
        return "The information has been provided by the specialist agent above."
    
    # Otherwise, return as-is but mark it seen
    dedup.mark_seen(supervisor_response)
    return supervisor_response


def extract_unique_content(
    messages: List[Any],
    agent_name_filter: Optional[str] = None,
) -> List[str]:
    """
    Extract unique content from messages, optionally filtering by agent.
    
    Args:
        messages: List of messages to process
        agent_name_filter: If provided, only include messages from this agent
        
    Returns:
        List of unique content strings
    """
    dedup = MessageDeduplicator()  # Fresh instance for this extraction
    unique_content = []
    
    for msg in messages:
        # Get content
        content = getattr(msg, "content", None)
        if not content:
            continue
        
        # Filter by agent if specified
        if agent_name_filter:
            msg_name = getattr(msg, "name", None)
            if msg_name != agent_name_filter:
                continue
        
        # Check for duplicates
        if not dedup.is_duplicate(content):
            unique_content.append(content)
            dedup.mark_seen(content)
    
    return unique_content

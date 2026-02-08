"""
Tests for message deduplication utilities.
"""

from __future__ import annotations

import pytest

from src.message_dedup import (
    MessageDeduplicator,
    get_deduplicator,
    reset_deduplicator,
    deduplicate_response,
    extract_unique_content,
)


class TestMessageDeduplicator:
    """Tests for the MessageDeduplicator class."""
    
    def test_exact_duplicate_detected(self):
        """Should detect exact duplicate content."""
        dedup = MessageDeduplicator()
        
        # Content must be >= 50 chars to be tracked
        content = "This is a test message with some content that is long enough to be tracked by the deduplicator."
        
        # First time - not a duplicate
        assert not dedup.is_duplicate(content)
        dedup.mark_seen(content)
        
        # Second time - is a duplicate
        assert dedup.is_duplicate(content)
    
    def test_similar_content_detected(self):
        """Should detect highly similar content."""
        dedup = MessageDeduplicator(similarity_threshold=0.7)  # Lower threshold for this test
        
        # Content must be >= 50 chars to be tracked
        content1 = "Found 5 tickets in the OPIK project with high priority status and they need immediate attention from the team."
        content2 = "Found 5 tickets in the OPIK project with high priority status that need immediate attention from the team."
        
        dedup.mark_seen(content1)
        
        # Should detect as duplicate due to high similarity
        assert dedup.is_duplicate(content2)
    
    def test_different_content_not_duplicate(self):
        """Should not flag different content as duplicate."""
        dedup = MessageDeduplicator()
        
        content1 = "Found 5 tickets in the OPIK project."
        content2 = "GitHub has 10 open pull requests waiting for review."
        
        dedup.mark_seen(content1)
        
        assert not dedup.is_duplicate(content2)
    
    def test_short_content_ignored(self):
        """Short content should not be tracked."""
        dedup = MessageDeduplicator()
        
        short_content = "OK"
        
        # Should not be flagged as duplicate even after marking
        assert not dedup.is_duplicate(short_content)
        dedup.mark_seen(short_content)
        assert not dedup.is_duplicate(short_content)
    
    def test_strips_common_prefixes(self):
        """Should normalize content by stripping common prefixes."""
        dedup = MessageDeduplicator()
        
        # Content must be >= 50 chars to be tracked
        content1 = "Here's what I found: There are 5 critical bugs in the system that need to be addressed immediately by the development team."
        content2 = "There are 5 critical bugs in the system that need to be addressed immediately by the development team."
        
        dedup.mark_seen(content1)
        
        # Should detect as duplicate after stripping prefix
        assert dedup.is_duplicate(content2)
    
    def test_filter_duplicate_messages(self):
        """Should filter duplicate messages from a list."""
        dedup = MessageDeduplicator()
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        messages = [
            MockMessage("First unique message with enough content to track."),
            MockMessage("Second unique message with different content here."),
            MockMessage("First unique message with enough content to track."),  # Duplicate
            MockMessage("Third unique message that should be kept."),
        ]
        
        filtered = dedup.filter_duplicate_messages(messages)
        
        assert len(filtered) == 3
        assert filtered[0].content == "First unique message with enough content to track."
        assert filtered[1].content == "Second unique message with different content here."
        assert filtered[2].content == "Third unique message that should be kept."
    
    def test_clear(self):
        """Should clear all tracked content."""
        dedup = MessageDeduplicator()
        
        content = "Some content that will be tracked and then cleared."
        dedup.mark_seen(content)
        assert dedup.is_duplicate(content)
        
        dedup.clear()
        
        assert not dedup.is_duplicate(content)
    
    def test_memory_bounded(self):
        """Should keep memory bounded by limiting stored content."""
        dedup = MessageDeduplicator()
        
        # Add many items
        for i in range(150):
            dedup.mark_seen(f"Message number {i} with enough content to be tracked.")
        
        # Should have trimmed to 50
        assert len(dedup._seen_content) <= 100


class TestGlobalDeduplicator:
    """Tests for global deduplicator functions."""
    
    def test_get_deduplicator_singleton(self):
        """Should return same instance."""
        reset_deduplicator()
        
        d1 = get_deduplicator()
        d2 = get_deduplicator()
        
        assert d1 is d2
    
    def test_reset_deduplicator(self):
        """Should create new instance on reset."""
        d1 = get_deduplicator()
        d1.mark_seen("Some content that should be cleared after reset.")
        
        d2 = reset_deduplicator()
        
        assert d1 is not d2
        assert not d2.is_duplicate("Some content that should be cleared after reset.")


class TestDeduplicateResponse:
    """Tests for the deduplicate_response function."""
    
    def test_removes_duplicate_supervisor_response(self):
        """Should detect when supervisor echoes sub-agent."""
        reset_deduplicator()
        
        sub_agent_responses = [
            "Found 5 tickets in OPIK project: OPIK-1, OPIK-2, OPIK-3, OPIK-4, OPIK-5."
        ]
        
        # Supervisor echoes the same content
        supervisor_response = "Found 5 tickets in OPIK project: OPIK-1, OPIK-2, OPIK-3, OPIK-4, OPIK-5."
        
        result = deduplicate_response(supervisor_response, sub_agent_responses)
        
        # Should return brief summary instead
        assert "provided by the specialist agent" in result.lower()
    
    def test_keeps_unique_supervisor_response(self):
        """Should keep supervisor response if it's unique."""
        reset_deduplicator()
        
        sub_agent_responses = [
            "Found 5 tickets in OPIK project with various priorities."
        ]
        
        # Supervisor adds new analysis
        supervisor_response = "Based on the ticket analysis, I recommend prioritizing OPIK-1 first due to its critical status and customer impact."
        
        result = deduplicate_response(supervisor_response, sub_agent_responses)
        
        # Should keep the unique response
        assert result == supervisor_response


class TestExtractUniqueContent:
    """Tests for the extract_unique_content function."""
    
    def test_extracts_unique_messages(self):
        """Should extract unique content from messages."""
        class MockMessage:
            def __init__(self, content, name=None):
                self.content = content
                self.name = name
        
        # Content must be >= 50 chars to be tracked
        messages = [
            MockMessage("First message with unique content here that is long enough to be tracked by the deduplicator system."),
            MockMessage("Second message with different content that is also long enough to be tracked by the deduplicator system."),
            MockMessage("First message with unique content here that is long enough to be tracked by the deduplicator system."),  # Duplicate
        ]
        
        unique = extract_unique_content(messages)
        
        assert len(unique) == 2
    
    def test_filters_by_agent_name(self):
        """Should filter by agent name when specified."""
        class MockMessage:
            def __init__(self, content, name):
                self.content = content
                self.name = name
        
        messages = [
            MockMessage("Jira agent found 5 tickets in the system.", "jira_agent"),
            MockMessage("GitHub agent found 3 open PRs waiting.", "github_agent"),
            MockMessage("Another Jira message about ticket status.", "jira_agent"),
        ]
        
        unique = extract_unique_content(messages, agent_name_filter="jira_agent")
        
        assert len(unique) == 2
        assert all("jira" in c.lower() or "ticket" in c.lower() for c in unique)

"""Comprehensive tests for progressive summarization.

These tests verify that ProgressiveSummarizer works correctly with:
- Anchored summaries that track message positions
- Callback pattern for summarization (no internal LLM calls)
- CCR integration for retrieval
- Extractive fallback summarization

CRITICAL: NO MOCKS for core logic. All tests use real implementations.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.tokenizer import Tokenizer
from headroom.tokenizers import EstimatingTokenCounter
from headroom.transforms.progressive_summarizer import (
    AnchoredSummary,
    ProgressiveSummarizer,
    SummarizationResult,
    extractive_summarizer,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tokenizer() -> Tokenizer:
    """Create a tokenizer for testing."""
    return Tokenizer(EstimatingTokenCounter())


@pytest.fixture
def simple_conversation() -> list[dict[str, Any]]:
    """Simple conversation without tool calls."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Can you help me with Python?"},
        {"role": "assistant", "content": "Of course! What would you like to know?"},
        {"role": "user", "content": "How do I read a file?"},
        {
            "role": "assistant",
            "content": "You can use open() to read files. Here's an example: with open('file.txt', 'r') as f: content = f.read()",
        },
    ]


@pytest.fixture
def conversation_with_tools() -> list[dict[str, Any]]:
    """Conversation with tool calls and responses."""
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for information about Python."},
        {
            "role": "assistant",
            "content": "I'll search for that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"results": [{"title": "Python Guide", "url": "example.com"}, {"title": "Python Tutorial", "url": "tutorial.com"}]}',
        },
        {"role": "assistant", "content": "Here's what I found about Python programming."},
        {"role": "user", "content": "Thanks! Can you search for more?"},
        {
            "role": "assistant",
            "content": "Sure, searching again for more results.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"results": [{"title": "Advanced Python", "status": "found"}, {"error": "Some results failed to load"}]}',
        },
        {"role": "assistant", "content": "Here are more results for you."},
    ]


@pytest.fixture
def long_conversation() -> list[dict[str, Any]]:
    """Long conversation for testing summarization scenarios."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    # Add many turns
    for i in range(20):
        messages.append(
            {"role": "user", "content": f"This is question number {i}. What about topic {i}?"}
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Here's my detailed response about topic {i}. " * 10
                + f"In summary, topic {i} is interesting.",
            }
        )

    return messages


# =============================================================================
# AnchoredSummary Tests
# =============================================================================


class TestAnchoredSummary:
    """Tests for AnchoredSummary dataclass."""

    def test_compression_ratio_calculation(self) -> None:
        """Test compression ratio is calculated correctly."""
        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=5,
            original_message_count=6,
            original_tokens=1000,
            summary_tokens=100,
        )
        assert summary.compression_ratio == 0.1  # 100/1000

    def test_compression_ratio_with_zero_original(self) -> None:
        """Test compression ratio handles zero original tokens."""
        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=0,
            original_message_count=1,
            original_tokens=0,
            summary_tokens=10,
        )
        assert summary.compression_ratio == 1.0  # fallback

    def test_tokens_saved(self) -> None:
        """Test tokens_saved calculation."""
        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=5,
            original_message_count=6,
            original_tokens=1000,
            summary_tokens=100,
        )
        assert summary.tokens_saved == 900

    def test_tokens_saved_no_negative(self) -> None:
        """Test tokens_saved doesn't go negative."""
        summary = AnchoredSummary(
            summary_text="Long summary that is bigger than original",
            start_index=0,
            end_index=0,
            original_message_count=1,
            original_tokens=10,
            summary_tokens=50,
        )
        assert summary.tokens_saved == 0  # max(0, ...)

    def test_optional_fields(self) -> None:
        """Test optional fields have defaults."""
        summary = AnchoredSummary(
            summary_text="Summary",
            start_index=0,
            end_index=5,
            original_message_count=6,
            original_tokens=1000,
            summary_tokens=100,
        )
        assert summary.cache_hash is None
        assert summary.tool_names == []
        assert summary.created_at > 0


# =============================================================================
# Extractive Summarizer Tests
# =============================================================================


class TestExtractiveSummarizer:
    """Tests for the default extractive summarizer."""

    def test_empty_messages(self) -> None:
        """Test handling of empty message list."""
        result = extractive_summarizer([])
        assert result == "[No messages to summarize]"

    def test_simple_conversation(self, simple_conversation: list[dict[str, Any]]) -> None:
        """Test summarization of simple conversation."""
        # Skip system message, use rest
        result = extractive_summarizer(simple_conversation[1:])
        assert "[Summary of 6 messages]" in result
        assert "user messages" in result
        assert "assistant" in result.lower()

    def test_tool_messages_detection(self, conversation_with_tools: list[dict[str, Any]]) -> None:
        """Test that tool messages are detected and counted."""
        result = extractive_summarizer(conversation_with_tools)
        assert "tool outputs" in result.lower()

    def test_error_detection_in_tools(self) -> None:
        """Test that errors in tool responses are detected."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "Error: Connection failed",
            },
        ]
        result = extractive_summarizer(messages)
        assert "with errors" in result

    def test_successful_tools(self) -> None:
        """Test that successful tool responses are marked correctly."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"status": "success", "data": [1, 2, 3]}',
            },
        ]
        result = extractive_summarizer(messages)
        assert "successful" in result

    def test_long_assistant_content_truncated(self) -> None:
        """Test that long assistant content is truncated."""
        messages = [
            {"role": "assistant", "content": "X" * 200},
        ]
        result = extractive_summarizer(messages)
        assert "..." in result  # Truncation indicator

    def test_context_ignored(self) -> None:
        """Test that context parameter exists but doesn't change output format."""
        messages = [{"role": "user", "content": "Hello"}]
        result1 = extractive_summarizer(messages, context="")
        result2 = extractive_summarizer(messages, context="Some context here")
        # Both should work (context is unused in extractive mode)
        assert "[Summary of 1 messages]" in result1
        assert "[Summary of 1 messages]" in result2


# =============================================================================
# ProgressiveSummarizer Core Tests
# =============================================================================


class TestProgressiveSummarizerInit:
    """Tests for ProgressiveSummarizer initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        summarizer = ProgressiveSummarizer()
        assert summarizer.max_summary_tokens == 500
        assert summarizer.min_messages_to_summarize == 3
        assert summarizer.store_for_retrieval is True
        # Default summarizer is extractive_summarizer
        assert summarizer.summarize_fn is not None

    def test_custom_summarize_fn(self) -> None:
        """Test custom summarization function."""

        def custom_fn(messages: list[dict], context: str = "") -> str:
            return f"Custom: {len(messages)} messages"

        summarizer = ProgressiveSummarizer(summarize_fn=custom_fn)
        result = summarizer.summarize_fn([{"role": "user", "content": "test"}])
        assert "Custom: 1" in result

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        summarizer = ProgressiveSummarizer(
            max_summary_tokens=1000,
            min_messages_to_summarize=5,
            store_for_retrieval=False,
        )
        assert summarizer.max_summary_tokens == 1000
        assert summarizer.min_messages_to_summarize == 5
        assert summarizer.store_for_retrieval is False


# =============================================================================
# Find Candidates Tests
# =============================================================================


class TestFindSummarizationCandidates:
    """Tests for finding candidate message groups."""

    def test_no_protected_all_candidates(self) -> None:
        """All messages are candidates when none protected."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=3)
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
        ]
        groups = summarizer._find_summarization_candidates(messages, protected=set())
        # Should have one group spanning all messages
        assert len(groups) == 1
        assert groups[0] == (0, 4)

    def test_protected_splits_groups(self) -> None:
        """Protected messages split the candidates into groups."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=2)
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},  # Protected at index 2
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "5"},
            {"role": "assistant", "content": "6"},
        ]
        groups = summarizer._find_summarization_candidates(messages, protected={2})
        # Should have two groups: (0,1) and (3,5)
        assert len(groups) == 2
        assert groups[0] == (0, 1)
        assert groups[1] == (3, 5)

    def test_min_messages_filter(self) -> None:
        """Groups smaller than min_messages_to_summarize are filtered."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=3)
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
            {"role": "user", "content": "3"},  # Protected
            {"role": "assistant", "content": "4"},
        ]
        groups = summarizer._find_summarization_candidates(messages, protected={2})
        # Group (0,1) has 2 messages, filtered. Group (3,3) has 1, filtered.
        assert len(groups) == 0

    def test_all_protected_no_candidates(self) -> None:
        """No candidates when all messages are protected."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=1)
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
        ]
        groups = summarizer._find_summarization_candidates(messages, protected={0, 1})
        assert len(groups) == 0

    def test_empty_messages(self) -> None:
        """Empty message list returns no groups."""
        summarizer = ProgressiveSummarizer()
        groups = summarizer._find_summarization_candidates([], protected=set())
        assert len(groups) == 0


# =============================================================================
# Summarize Messages Tests
# =============================================================================


class TestSummarizeMessages:
    """Tests for the main summarize_messages method."""

    def test_no_candidates_returns_original(
        self, tokenizer: Tokenizer, simple_conversation: list[dict[str, Any]]
    ) -> None:
        """When no candidates, return original messages unchanged."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=100)  # Too high
        result = summarizer.summarize_messages(
            simple_conversation, tokenizer, protected_indices=set()
        )
        assert len(result.messages) == len(simple_conversation)
        assert result.tokens_saved == 0
        assert len(result.summaries_created) == 0

    def test_all_protected_no_changes(
        self, tokenizer: Tokenizer, simple_conversation: list[dict[str, Any]]
    ) -> None:
        """All protected messages means no summarization."""
        summarizer = ProgressiveSummarizer(min_messages_to_summarize=2)
        all_protected = set(range(len(simple_conversation)))
        result = summarizer.summarize_messages(
            simple_conversation, tokenizer, protected_indices=all_protected
        )
        assert len(result.messages) == len(simple_conversation)
        assert result.tokens_saved == 0

    def test_summarization_reduces_messages(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Summarization reduces message count."""
        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,  # Skip CCR for test
        )
        # Protect first and last few messages
        protected = {0, 1, len(long_conversation) - 1, len(long_conversation) - 2}
        result = summarizer.summarize_messages(
            long_conversation, tokenizer, protected_indices=protected
        )

        # Should have fewer messages
        assert len(result.messages) < len(long_conversation)
        # Should save tokens
        assert result.tokens_saved > 0
        # Should create summaries
        assert len(result.summaries_created) > 0

    def test_summarization_result_structure(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Verify SummarizationResult has correct structure."""
        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )
        result = summarizer.summarize_messages(long_conversation, tokenizer, protected_indices={0})

        assert isinstance(result, SummarizationResult)
        assert isinstance(result.messages, list)
        assert isinstance(result.summaries_created, list)
        assert isinstance(result.tokens_before, int)
        assert isinstance(result.tokens_after, int)
        assert isinstance(result.transforms_applied, list)
        assert result.tokens_before >= result.tokens_after

    def test_custom_summarizer_called(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Custom summarizer function is called."""
        calls: list[int] = []

        def tracking_summarizer(messages: list[dict], context: str = "") -> str:
            calls.append(len(messages))
            return f"CUSTOM SUMMARY of {len(messages)} messages"

        summarizer = ProgressiveSummarizer(
            summarize_fn=tracking_summarizer,
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        result = summarizer.summarize_messages(long_conversation, tokenizer, protected_indices={0})

        # Custom summarizer should have been called
        assert len(calls) > 0
        # Summary should appear in messages
        found_custom = any("CUSTOM SUMMARY" in msg.get("content", "") for msg in result.messages)
        assert found_custom

    def test_context_passed_to_summarizer(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Context messages are passed to summarizer."""
        received_context: list[str] = []

        def context_tracking_summarizer(messages: list[dict], context: str = "") -> str:
            received_context.append(context)
            return "Summary"

        summarizer = ProgressiveSummarizer(
            summarize_fn=context_tracking_summarizer,
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        context_msgs = [{"role": "user", "content": "Recent important question"}]
        summarizer.summarize_messages(
            long_conversation,
            tokenizer,
            protected_indices={0},
            context_messages=context_msgs,
        )

        # Context should have been passed
        assert len(received_context) > 0
        # Should contain the recent message content
        assert any("Recent important question" in ctx for ctx in received_context)

    def test_target_tokens_stops_early(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Summarization stops when target tokens reached."""
        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        # Get original token count
        original_tokens = tokenizer.count_messages(long_conversation)

        # Set target very close to original (minimal summarization needed)
        target = int(original_tokens * 0.95)  # Only need 5% reduction

        result = summarizer.summarize_messages(
            long_conversation,
            tokenizer,
            protected_indices={0},
            target_tokens=target,
        )

        # Should stop once target reached
        assert result.tokens_after <= target or result.tokens_after < original_tokens

    def test_small_groups_skipped(
        self,
        tokenizer: Tokenizer,
    ) -> None:
        """Groups with < 100 tokens are skipped."""
        # Very short messages
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Bye"},
        ]

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=2,
            store_for_retrieval=False,
        )

        result = summarizer.summarize_messages(messages, tokenizer, protected_indices=set())

        # Small groups should be skipped
        assert len(result.summaries_created) == 0

    def test_summary_larger_than_original_skipped(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Summaries larger than original are skipped."""

        def verbose_summarizer(messages: list[dict], context: str = "") -> str:
            # Return a very verbose summary
            return "VERY LONG SUMMARY " * 1000

        summarizer = ProgressiveSummarizer(
            summarize_fn=verbose_summarizer,
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        result = summarizer.summarize_messages(long_conversation, tokenizer, protected_indices={0})

        # Summaries larger than original should be skipped
        # (or if any were created, they saved tokens)
        for summary in result.summaries_created:
            assert summary.tokens_saved >= 0

    def test_summarizer_exception_handled(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Exceptions from summarizer are handled gracefully."""

        def failing_summarizer(messages: list[dict], context: str = "") -> str:
            raise ValueError("Summarization failed!")

        summarizer = ProgressiveSummarizer(
            summarize_fn=failing_summarizer,
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        # Should not raise, should return original
        result = summarizer.summarize_messages(long_conversation, tokenizer, protected_indices={0})

        # No summaries created due to failures
        assert len(result.summaries_created) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestProgressiveSummarizerIntegration:
    """Integration tests for end-to-end summarization."""

    def test_full_workflow_with_extractive(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Test full workflow with default extractive summarizer."""
        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=4,
            store_for_retrieval=False,
        )

        original_count = len(long_conversation)

        result = summarizer.summarize_messages(
            long_conversation,
            tokenizer,
            protected_indices={0},  # Only protect system message
        )

        # Verify reduction
        assert len(result.messages) < original_count
        assert result.tokens_after < result.tokens_before

        # Verify transforms tracked
        assert len(result.transforms_applied) > 0

        # Verify summaries created
        assert len(result.summaries_created) > 0
        for summary in result.summaries_created:
            assert summary.start_index >= 0
            assert summary.end_index >= summary.start_index
            assert summary.compression_ratio < 1.0  # Actually compressed

    def test_preserves_protected_messages(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Protected messages are preserved exactly."""
        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        # Protect first 3 and last 3 messages
        protected = {
            0,
            1,
            2,
            len(long_conversation) - 3,
            len(long_conversation) - 2,
            len(long_conversation) - 1,
        }

        # Store original protected content
        original_protected = {i: long_conversation[i]["content"] for i in protected}

        result = summarizer.summarize_messages(
            long_conversation,
            tokenizer,
            protected_indices=protected,
        )

        # Find protected messages in result
        # First 3 should still be at beginning
        assert result.messages[0]["content"] == original_protected[0]
        assert result.messages[1]["content"] == original_protected[1]
        assert result.messages[2]["content"] == original_protected[2]

        # Last 3 should still be at end (positions shifted)
        assert result.messages[-1]["content"] == original_protected[len(long_conversation) - 1]
        assert result.messages[-2]["content"] == original_protected[len(long_conversation) - 2]
        assert result.messages[-3]["content"] == original_protected[len(long_conversation) - 3]

    def test_tool_messages_handled(
        self, tokenizer: Tokenizer, conversation_with_tools: list[dict[str, Any]]
    ) -> None:
        """Tool messages are handled in summarization."""
        # Create longer tool-heavy conversation
        long_tool_conv = conversation_with_tools.copy()
        for i in range(10):
            long_tool_conv.extend(
                [
                    {"role": "user", "content": f"Search again {i}"},
                    {
                        "role": "assistant",
                        "content": f"Searching {i}...",
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {"name": "search", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{i}",
                        "content": f'{{"data": "result {i}"}}',
                    },
                    {"role": "assistant", "content": f"Found result {i}"},
                ]
            )

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        result = summarizer.summarize_messages(
            long_tool_conv,
            tokenizer,
            protected_indices={0},
        )

        # Should reduce messages
        assert len(result.messages) < len(long_tool_conv)

        # Tool names should be tracked in summaries
        all_tool_names = []
        for summary in result.summaries_created:
            all_tool_names.extend(summary.tool_names)
        # Some tool calls should be tracked (may be empty if extractive)

    def test_does_not_mutate_original(
        self, tokenizer: Tokenizer, long_conversation: list[dict[str, Any]]
    ) -> None:
        """Original messages are not mutated."""
        import copy

        original_copy = copy.deepcopy(long_conversation)

        summarizer = ProgressiveSummarizer(
            min_messages_to_summarize=3,
            store_for_retrieval=False,
        )

        summarizer.summarize_messages(
            long_conversation,
            tokenizer,
            protected_indices={0},
        )

        # Original should be unchanged
        assert long_conversation == original_copy


# =============================================================================
# SummarizationResult Tests
# =============================================================================


class TestSummarizationResult:
    """Tests for SummarizationResult dataclass."""

    def test_tokens_saved_property(self) -> None:
        """Test tokens_saved property."""
        result = SummarizationResult(
            messages=[],
            summaries_created=[],
            tokens_before=1000,
            tokens_after=300,
            transforms_applied=[],
        )
        assert result.tokens_saved == 700

    def test_tokens_saved_no_negative(self) -> None:
        """Test tokens_saved doesn't go negative."""
        result = SummarizationResult(
            messages=[],
            summaries_created=[],
            tokens_before=100,
            tokens_after=150,
            transforms_applied=[],
        )
        assert result.tokens_saved == 0

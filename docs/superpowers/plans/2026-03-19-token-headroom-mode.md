# Token Headroom Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `HEADROOM_MODE=token_headroom` config that compresses older messages to extend session length, trading prefix cache cost savings for token reduction.

**Architecture:** Content-addressed CompressionCache maps original message content hashes to compressed versions. On each turn, already-compressed messages are swapped from cache and re-frozen; newly aged-out messages are compressed by the existing ContentRouter pipeline; recent messages pass through unchanged. Works for all providers (Anthropic + OpenAI).

**Tech Stack:** Python 3.11+, pytest, existing Headroom transforms (ContentRouter, CodeAwareCompressor, SmartCrusher, Kompress)

**Spec:** `docs/superpowers/specs/2026-03-19-token-headroom-mode-design.md`

---

## Chunk 1: CompressionCache Core + Unit Tests

### Task 1: CompressionCache — Data Structure + Core Methods

**Files:**
- Create: `headroom/cache/compression_cache.py`
- Test: `tests/test_compression_cache.py`

- [ ] **Step 1: Write failing tests for CompressionCache core**

Create `tests/test_compression_cache.py`:

```python
"""Tests for CompressionCache — content-addressed compression result cache."""

import pytest

from headroom.cache.compression_cache import CompressionCache


class TestCompressionCacheStoreAndRetrieve:
    """Basic cache hit/miss behavior."""

    def test_cache_miss_returns_none(self):
        cache = CompressionCache()
        assert cache.get_compressed("nonexistent_hash") is None

    def test_store_and_retrieve(self):
        cache = CompressionCache()
        cache.store_compressed("abc123", "compressed content", tokens_saved=500)
        assert cache.get_compressed("abc123") == "compressed content"

    def test_different_content_different_hash(self):
        cache = CompressionCache()
        cache.store_compressed("hash_v1", "compressed v1", tokens_saved=100)
        cache.store_compressed("hash_v2", "compressed v2", tokens_saved=200)
        assert cache.get_compressed("hash_v1") == "compressed v1"
        assert cache.get_compressed("hash_v2") == "compressed v2"

    def test_overwrite_same_hash(self):
        cache = CompressionCache()
        cache.store_compressed("abc", "old", tokens_saved=100)
        cache.store_compressed("abc", "new", tokens_saved=200)
        assert cache.get_compressed("abc") == "new"

    def test_stats_tracking(self):
        cache = CompressionCache()
        cache.store_compressed("h1", "c1", tokens_saved=100)
        cache.store_compressed("h2", "c2", tokens_saved=200)
        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["total_tokens_saved"] == 300


class TestCompressionCacheLRU:
    """LRU eviction when cache exceeds max_entries."""

    def test_eviction_at_max_entries(self):
        cache = CompressionCache(max_entries=3)
        cache.store_compressed("h1", "c1", tokens_saved=10)
        cache.store_compressed("h2", "c2", tokens_saved=10)
        cache.store_compressed("h3", "c3", tokens_saved=10)
        # h1 is oldest
        cache.store_compressed("h4", "c4", tokens_saved=10)
        assert cache.get_compressed("h1") is None  # evicted
        assert cache.get_compressed("h4") == "c4"  # newest

    def test_access_refreshes_lru(self):
        cache = CompressionCache(max_entries=3)
        cache.store_compressed("h1", "c1", tokens_saved=10)
        cache.store_compressed("h2", "c2", tokens_saved=10)
        cache.store_compressed("h3", "c3", tokens_saved=10)
        # Access h1 to make it recent
        cache.get_compressed("h1")
        # Now add h4 — h2 should be evicted (oldest untouched)
        cache.store_compressed("h4", "c4", tokens_saved=10)
        assert cache.get_compressed("h1") == "c1"  # refreshed, still present
        assert cache.get_compressed("h2") is None  # evicted
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compression_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'headroom.cache.compression_cache'`

- [ ] **Step 3: Write CompressionCache implementation**

Create `headroom/cache/compression_cache.py`:

```python
"""Content-addressed compression cache for token headroom mode.

Maps original content hashes to compressed versions. Used to avoid
re-compressing the same content on every turn. Session-scoped.

The cache is keyed by SHA-256 of original content. This means:
- Same file content across turns = same hash = cache hit
- Edited file = different hash = cache miss = fresh compression
- Claude Code dropping a message = unused cache entry (LRU eviction)
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CompressionCacheEntry:
    """A single cached compression result."""

    compressed: str
    tokens_saved: int
    created_at: float = field(default_factory=time.time)


class CompressionCache:
    """Content-addressed cache of compressed message content.

    Maps content_hash → compressed_content. No position tracking —
    works purely on content identity. Handles Claude Code dropping
    messages gracefully (unused entries stay in cache until LRU eviction).
    """

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._cache: OrderedDict[str, CompressionCacheEntry] = OrderedDict()
        # Stats
        self._hits: int = 0
        self._misses: int = 0

    def get_compressed(self, content_hash: str) -> str | None:
        """Look up cached compression result. O(1).

        Returns compressed content or None if not cached.
        Refreshes LRU position on hit.
        """
        entry = self._cache.get(content_hash)
        if entry is None:
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(content_hash)
        self._hits += 1
        return entry.compressed

    def store_compressed(
        self, content_hash: str, compressed: str, tokens_saved: int
    ) -> None:
        """Cache a compression result.

        Args:
            content_hash: SHA-256 of the original content.
            compressed: The compressed content string.
            tokens_saved: Tokens saved by this compression.
        """
        if content_hash in self._cache:
            # Update existing entry, move to end
            self._cache[content_hash] = CompressionCacheEntry(
                compressed=compressed, tokens_saved=tokens_saved
            )
            self._cache.move_to_end(content_hash)
        else:
            self._cache[content_hash] = CompressionCacheEntry(
                compressed=compressed, tokens_saved=tokens_saved
            )
        # Evict oldest if over limit
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_tokens_saved = sum(e.tokens_saved for e in self._cache.values())
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(
                self._hits / max(1, self._hits + self._misses) * 100, 1
            ),
            "total_tokens_saved": total_tokens_saved,
        }

    @staticmethod
    def content_hash(content: str | list) -> str:
        """Compute SHA-256 hash of message content.

        Handles both string content and Anthropic-format list content.
        """
        if isinstance(content, list):
            # Anthropic format: list of content blocks
            # Hash the text content of each block
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        parts.append(str(block.get("content", "")))
                elif isinstance(block, str):
                    parts.append(block)
            text = "\n".join(parts)
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compression_cache.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add headroom/cache/compression_cache.py tests/test_compression_cache.py
git commit -m "feat: add CompressionCache with LRU eviction for token headroom mode"
```

---

### Task 2: CompressionCache — Frozen Count + Apply/Update Methods

**Files:**
- Modify: `headroom/cache/compression_cache.py`
- Modify: `tests/test_compression_cache.py`

- [ ] **Step 1: Write failing tests for frozen count and apply/update**

Append to `tests/test_compression_cache.py`:

```python
class TestCompressionCacheFrozenCount:
    """Frozen count: consecutive stable messages from start."""

    def _make_msg(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def test_empty_cache_returns_zero(self):
        cache = CompressionCache()
        messages = [
            self._make_msg("user", "hello"),
            self._make_msg("assistant", "hi"),
        ]
        assert cache.compute_frozen_count(messages) == 0

    def test_user_assistant_always_stable(self):
        """User/assistant messages are always stable (never compressed)."""
        cache = CompressionCache()
        messages = [
            self._make_msg("user", "hello"),
            self._make_msg("assistant", "hi"),
            self._make_msg("user", "do something"),
        ]
        # No tool results, all user/assistant → all stable
        assert cache.compute_frozen_count(messages) == 3

    def test_tool_result_with_cache_hit_is_stable(self):
        cache = CompressionCache()
        tool_content = "file contents here..."
        h = CompressionCache.content_hash(tool_content)
        cache.store_compressed(h, "compressed file", tokens_saved=100)

        messages = [
            self._make_msg("user", "read file"),
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": tool_content}]},
        ]
        # All 3 stable: user (always), assistant (always), tool_result (cache hit)
        assert cache.compute_frozen_count(messages) == 3

    def test_tool_result_cache_miss_stops_frozen(self):
        cache = CompressionCache()
        messages = [
            self._make_msg("user", "hello"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "uncached data"}]},
            self._make_msg("user", "continue"),
        ]
        # Message 0: user (stable), Message 1: tool_result (cache miss) → stop
        assert cache.compute_frozen_count(messages) == 1

    def test_frozen_count_with_dropped_messages(self):
        """If Claude Code drops a message, consecutive run may break."""
        cache = CompressionCache()
        content_a = "content A"
        content_b = "content B"
        cache.store_compressed(CompressionCache.content_hash(content_a), "ca", tokens_saved=10)
        # content_b NOT in cache (simulates a dropped/new message)

        messages = [
            self._make_msg("user", "hi"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content_a}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": content_b}]},
        ]
        # user (stable), tool_result_a (cache hit, stable), tool_result_b (cache miss, stop)
        assert cache.compute_frozen_count(messages) == 2


class TestCompressionCacheApplyAndUpdate:
    """apply_cached swaps compressed content; update_from_result caches new compressions."""

    def _make_msg(self, role: str, content: str) -> dict:
        return {"role": role, "content": content}

    def test_apply_cached_swaps_tool_results(self):
        cache = CompressionCache()
        original_content = "long file content " * 100
        compressed = "short version"
        h = CompressionCache.content_hash(original_content)
        cache.store_compressed(h, compressed, tokens_saved=500)

        messages = [
            self._make_msg("user", "read file"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": original_content}]},
        ]
        result = cache.apply_cached(messages)
        # First message unchanged (user)
        assert result[0] == messages[0]
        # Second message: tool_result content swapped
        tool_block = result[1]["content"][0]
        assert tool_block["content"] == compressed

    def test_apply_cached_preserves_uncached_messages(self):
        cache = CompressionCache()
        messages = [
            self._make_msg("user", "hello"),
            self._make_msg("assistant", "world"),
        ]
        result = cache.apply_cached(messages)
        assert result[0]["content"] == "hello"
        assert result[1]["content"] == "world"

    def test_apply_cached_never_adds_messages(self):
        """Critical invariant: output length == input length."""
        cache = CompressionCache()
        cache.store_compressed("orphan_hash", "orphan content", tokens_saved=10)
        messages = [self._make_msg("user", "hello")]
        result = cache.apply_cached(messages)
        assert len(result) == len(messages)

    def test_update_from_result_caches_changes(self):
        cache = CompressionCache()
        originals = [
            self._make_msg("user", "hello"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "original data"}]},
        ]
        compressed = [
            self._make_msg("user", "hello"),  # unchanged
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "compressed data"}]},
        ]
        cache.update_from_result(originals, compressed)

        # Now the original content should be cached
        h = CompressionCache.content_hash("original data")
        assert cache.get_compressed(h) == "compressed data"

    def test_update_from_result_ignores_unchanged(self):
        cache = CompressionCache()
        originals = [self._make_msg("user", "hello")]
        compressed = [self._make_msg("user", "hello")]  # same
        cache.update_from_result(originals, compressed)
        assert cache.get_stats()["entries"] == 0

    def test_apply_does_not_modify_original_messages(self):
        """apply_cached must not mutate the input message list."""
        cache = CompressionCache()
        content = "original file"
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "compressed", tokens_saved=50)

        messages = [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": content}]},
        ]
        import copy
        original_copy = copy.deepcopy(messages)
        cache.apply_cached(messages)
        assert messages == original_copy  # original not mutated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compression_cache.py -v`
Expected: FAIL — `AttributeError: 'CompressionCache' object has no attribute 'compute_frozen_count'`

- [ ] **Step 3: Add compute_frozen_count, apply_cached, update_from_result to CompressionCache**

Add these methods to `headroom/cache/compression_cache.py` in the `CompressionCache` class:

```python
    def _is_tool_result_message(self, message: dict) -> bool:
        """Check if a message contains tool result content."""
        content = message.get("content")
        if isinstance(content, list):
            return any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
        # OpenAI format: role == "tool"
        return message.get("role") == "tool"

    def _extract_tool_result_content(self, message: dict) -> str | None:
        """Extract the text content from a tool result message.

        Returns the content string, or None if not a tool result.
        """
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    inner = block.get("content", "")
                    if isinstance(inner, str):
                        return inner
                    if isinstance(inner, list):
                        # Anthropic nested content blocks
                        parts = []
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(sub.get("text", ""))
                        return "\n".join(parts) if parts else None
            return None
        # OpenAI format
        if message.get("role") == "tool" and isinstance(content, str):
            return content
        return None

    def compute_frozen_count(self, messages: list[dict]) -> int:
        """Count consecutive stable messages from the start.

        A message is stable if:
        - It's a user/assistant/system message (always stable, never compressed)
        - It's a tool_result with content that's in the cache (already compressed)

        First unstable message (tool_result with cache miss) stops the count.
        """
        count = 0
        for msg in messages:
            if self._is_tool_result_message(msg):
                content_str = self._extract_tool_result_content(msg)
                if content_str is not None:
                    h = self.content_hash(content_str)
                    if h not in self._cache:
                        break  # Cache miss — stop frozen prefix
            # User/assistant/system/tool_use OR tool_result with cache hit → stable
            count += 1
        return count

    def apply_cached(self, messages: list[dict]) -> list[dict]:
        """Swap cached compressed content into tool result messages.

        Returns a NEW list of messages. Does NOT mutate the input.
        Non-tool-result messages are passed through unchanged.
        """
        result = []
        for msg in messages:
            content_str = self._extract_tool_result_content(msg)
            if content_str is not None:
                h = self.content_hash(content_str)
                cached = self.get_compressed(h)
                if cached is not None:
                    # Deep copy the message and swap content
                    new_msg = _swap_tool_result_content(msg, cached)
                    result.append(new_msg)
                    continue
            result.append(msg)
        return result

    def update_from_result(
        self, originals: list[dict], compressed: list[dict]
    ) -> None:
        """Cache newly compressed content by comparing originals to compressed.

        Index-aligned comparison: for each position, if the tool result content
        differs, store the mapping original_hash → compressed_content.

        Assumes the pipeline does NOT reorder or merge messages.
        """
        if len(originals) != len(compressed):
            logger.warning(
                "update_from_result: message count mismatch (%d vs %d), skipping",
                len(originals),
                len(compressed),
            )
            return

        for orig_msg, comp_msg in zip(originals, compressed):
            orig_content = self._extract_tool_result_content(orig_msg)
            comp_content = self._extract_tool_result_content(comp_msg)
            if (
                orig_content is not None
                and comp_content is not None
                and orig_content != comp_content
            ):
                h = self.content_hash(orig_content)
                tokens_saved = max(0, len(orig_content) // 4 - len(comp_content) // 4)
                self.store_compressed(h, comp_content, tokens_saved=tokens_saved)


def _swap_tool_result_content(message: dict, new_content: str) -> dict:
    """Create a copy of message with tool result content replaced.

    Handles both Anthropic format (content blocks) and OpenAI format (role=tool).
    Does NOT mutate the original message.
    """
    import copy

    new_msg = copy.deepcopy(message)
    content = new_msg.get("content")

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                block["content"] = new_content
                break
    elif new_msg.get("role") == "tool" and isinstance(content, str):
        new_msg["content"] = new_content

    return new_msg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compression_cache.py -v`
Expected: All 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add headroom/cache/compression_cache.py tests/test_compression_cache.py
git commit -m "feat: add frozen count, apply_cached, update_from_result to CompressionCache"
```

---

### Task 3: Export CompressionCache from cache package

**Files:**
- Modify: `headroom/cache/__init__.py`

- [ ] **Step 1: Add import and export**

In `headroom/cache/__init__.py`, add to imports:
```python
from .compression_cache import CompressionCache
```

Add `"CompressionCache"` to the `__all__` list.

- [ ] **Step 2: Verify import works**

Run: `python -c "from headroom.cache import CompressionCache; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add headroom/cache/__init__.py
git commit -m "feat: export CompressionCache from cache package"
```

---

## Chunk 2: ProxyConfig + Mode-Based Pipeline Construction

### Task 4: Add `mode` field to ProxyConfig

**Files:**
- Modify: `headroom/proxy/server.py`

- [ ] **Step 1: Add `mode` field to ProxyConfig dataclass**

In `headroom/proxy/server.py`, find the `ProxyConfig` class (around line 496, after the `optimize` field). Add:

```python
    # Optimization mode: "cost_savings" (default) or "token_headroom"
    # cost_savings: preserve prefix cache for cost reduction
    # token_headroom: compress older messages for session extension
    mode: str = "cost_savings"
```

- [ ] **Step 2: Add HEADROOM_MODE env var in config construction**

Find the `ProxyConfig(...)` constructor call (around line 7576). Add the `mode` parameter:

```python
    mode=_get_env_str("HEADROOM_MODE", "cost_savings"),
```

- [ ] **Step 3: Add mode validation at startup**

Find the startup logging section (around line 1700 where `logger.info(f"Optimization: ...")` is). Add after it:

```python
        if self.config.mode not in ("cost_savings", "token_headroom"):
            logger.warning(
                f"Unknown HEADROOM_MODE '{self.config.mode}', falling back to 'cost_savings'"
            )
            self.config.mode = "cost_savings"
        logger.info(f"Mode: {self.config.mode}")
```

- [ ] **Step 4: Verify proxy starts with default mode**

Run: `python -c "from headroom.proxy.server import ProxyConfig; c = ProxyConfig(); print(c.mode)"`
Expected: `cost_savings`

- [ ] **Step 5: Commit**

```bash
git add headroom/proxy/server.py
git commit -m "feat: add HEADROOM_MODE config to ProxyConfig"
```

---

### Task 5: Mode-Based Pipeline Configuration at Startup

**Files:**
- Modify: `headroom/proxy/server.py`

- [ ] **Step 1: Set ContentRouterConfig based on mode**

Find where `ContentRouterConfig` is instantiated (around line 1419, inside the `if config.smart_routing:` block). After the `router_config = ContentRouterConfig(...)` line, add:

```python
            # Token headroom mode: allow compression of older excluded-tool results
            if config.mode == "token_headroom":
                router_config.protect_recent_reads_fraction = 0.3
                logger.info("Token headroom mode: protect_recent_reads_fraction=0.3")
```

- [ ] **Step 2: Set CCR TTL based on mode**

In the same pipeline construction area, find where `CodeCompressorConfig` or `ReadLifecycleConfig` is used. After the router_config modification, add:

```python
                # Extend CCR TTL for long sessions
                if hasattr(router_config, 'read_lifecycle'):
                    router_config.read_lifecycle.ccr_ttl = 14400  # 4 hours
```

Also check: if `CodeAwareCompressor` is configured separately, its `ccr_ttl` should also be set. Look for `CodeCompressorConfig` instantiation in the same block and set `ccr_ttl=14400`.

- [ ] **Step 3: Initialize CompressionCache store**

Find where `self.session_tracker_store` is initialized (in `__init__` or `_build_pipelines`). Add nearby:

```python
        # Compression cache for token_headroom mode (session-scoped)
        from headroom.cache.compression_cache import CompressionCache
        self._compression_caches: dict[str, CompressionCache] = {}
```

Add a helper method to the proxy class:

```python
    def _get_compression_cache(self, session_id: str) -> CompressionCache:
        """Get or create a CompressionCache for a session."""
        if session_id not in self._compression_caches:
            self._compression_caches[session_id] = CompressionCache()
        return self._compression_caches[session_id]
```

- [ ] **Step 4: Add startup logging for token_headroom mode**

In the startup logging section (around line 1700), add:

```python
        if self.config.mode == "token_headroom":
            logger.info("Token headroom mode active:")
            logger.info("  Prefix freeze: re-freeze after compression")
            logger.info("  Read protection window: 30%% of excluded-tool messages")
            logger.info("  CCR TTL: 4 hours")
            logger.info("  Compression cache: active")
```

- [ ] **Step 5: Verify startup with HEADROOM_MODE=token_headroom**

Run: `HEADROOM_MODE=token_headroom python -c "from headroom.proxy.server import ProxyConfig; c = ProxyConfig(mode='token_headroom'); print(c.mode)"`
Expected: `token_headroom`

- [ ] **Step 6: Commit**

```bash
git add headroom/proxy/server.py
git commit -m "feat: configure pipeline based on HEADROOM_MODE at startup"
```

---

## Chunk 3: Token Headroom Pipeline Integration (Both Providers)

### Task 6: Token Headroom Branch in Anthropic Handler

**Files:**
- Modify: `headroom/proxy/server.py`

- [ ] **Step 1: Add token_headroom branch in handle_anthropic_messages**

Find the optimization block in `handle_anthropic_messages` (around line 2071). The current code is:

```python
        if self.config.optimize and messages:
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                result = self.anthropic_pipeline.apply(
                    messages=messages,
                    model=model,
                    model_limit=context_limit,
                    frozen_message_count=frozen_message_count,
                    biases=self.config.hooks.compute_biases(messages, _hook_ctx)
                    if self.config.hooks
                    else None,
                )
                ...
```

Replace with:

```python
        if self.config.optimize and messages:
            try:
                context_limit = self.anthropic_provider.get_context_limit(model)
                biases = (
                    self.config.hooks.compute_biases(messages, _hook_ctx)
                    if self.config.hooks
                    else None
                )

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(session_id)

                    # Zone 1: Swap cached compressed versions into working copy
                    working_messages = comp_cache.apply_cached(messages)

                    # Re-freeze boundary: consecutive stable messages from start
                    frozen_message_count = comp_cache.compute_frozen_count(messages)

                    result = self.anthropic_pipeline.apply(
                        messages=working_messages,
                        model=model,
                        model_limit=context_limit,
                        frozen_message_count=frozen_message_count,
                        biases=biases,
                    )

                    # Cache newly compressed messages (index-aligned diff)
                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)
                else:
                    result = self.anthropic_pipeline.apply(
                        messages=messages,
                        model=model,
                        model_limit=context_limit,
                        frozen_message_count=frozen_message_count,
                        biases=biases,
                    )

                if result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
```

**IMPORTANT**: The `if result.messages != messages:` check at the end needs to compare against the right baseline. In token_headroom mode, `result.messages` should be compared against `working_messages` (the pre-swapped version), not `messages` (the originals). Fix:

```python
                baseline = working_messages if self.config.mode == "token_headroom" else messages
                if result.messages != baseline:
                    optimized_messages = result.messages
                    ...
```

Wait — actually in token_headroom mode, we ALWAYS want to use result.messages because apply_cached already swapped Zone 1. The comparison should detect Zone 2 changes. But even if Zone 2 has no changes, Zone 1 swaps still need to be used. So in token_headroom mode, always use result.messages:

```python
                if self.config.mode == "token_headroom":
                    # Always use pipeline result — Zone 1 swaps are already applied
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                elif result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
```

- [ ] **Step 2: Verify no side effects — cost_savings mode unchanged**

Run: `python -m pytest tests/ -k "anthropic" --timeout=30 -x -q`
Expected: Existing tests pass (cost_savings is default, no behavior change)

- [ ] **Step 3: Commit**

```bash
git add headroom/proxy/server.py
git commit -m "feat: add token_headroom branch to Anthropic handler"
```

---

### Task 7: Token Headroom Branch in OpenAI Handler

**Files:**
- Modify: `headroom/proxy/server.py`

- [ ] **Step 1: Apply same pattern to handle_openai_chat**

Find the optimization block in `handle_openai_chat` (around line 4604). Apply the exact same pattern as the Anthropic handler:

```python
        if self.config.optimize and messages:
            try:
                context_limit = self.openai_provider.get_context_limit(model)
                biases = _hook_biases

                if self.config.mode == "token_headroom":
                    comp_cache = self._get_compression_cache(openai_session_id)
                    working_messages = comp_cache.apply_cached(messages)
                    openai_frozen_count = comp_cache.compute_frozen_count(messages)

                    result = self.openai_pipeline.apply(
                        messages=working_messages,
                        model=model,
                        model_limit=context_limit,
                        frozen_message_count=openai_frozen_count,
                        biases=biases,
                    )

                    if result.messages != working_messages:
                        comp_cache.update_from_result(messages, result.messages)
                else:
                    result = self.openai_pipeline.apply(
                        messages=messages,
                        model=model,
                        model_limit=context_limit,
                        frozen_message_count=openai_frozen_count,
                        biases=biases,
                    )

                if self.config.mode == "token_headroom":
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                elif result.messages != messages:
                    optimized_messages = result.messages
                    transforms_applied = result.transforms_applied
                    pipeline_timing = result.timing
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                if result.waste_signals:
                    waste_signals_dict = result.waste_signals.to_dict()
            except Exception as e:
                logger.warning(f"Optimization failed: {e}")
```

- [ ] **Step 2: Verify no side effects**

Run: `python -m pytest tests/ -k "openai" --timeout=30 -x -q`
Expected: Existing tests pass

- [ ] **Step 3: Commit**

```bash
git add headroom/proxy/server.py
git commit -m "feat: add token_headroom branch to OpenAI handler"
```

---

### Task 8: Add CompressionCache Stats to /stats Endpoint

**Files:**
- Modify: `headroom/proxy/server.py`

- [ ] **Step 1: Add compression_cache stats to the stats response**

Find the `/stats` endpoint handler (around line 6243). Find where the response dict is built. Add a new key:

```python
        # Compression cache stats (token_headroom mode)
        compression_cache_stats = {}
        if self.config.mode == "token_headroom":
            all_cache_stats = {}
            for sid, cache in self._compression_caches.items():
                all_cache_stats[sid] = cache.get_stats()
            # Aggregate across sessions
            total_entries = sum(s.get("entries", 0) for s in all_cache_stats.values())
            total_hits = sum(s.get("hits", 0) for s in all_cache_stats.values())
            total_misses = sum(s.get("misses", 0) for s in all_cache_stats.values())
            compression_cache_stats = {
                "active_sessions": len(all_cache_stats),
                "total_entries": total_entries,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": round(total_hits / max(1, total_hits + total_misses) * 100, 1),
            }
```

Add it to the response dict:

```python
        "compression_cache": compression_cache_stats,
```

- [ ] **Step 2: Commit**

```bash
git add headroom/proxy/server.py
git commit -m "feat: add compression_cache stats to /stats endpoint"
```

---

### Task 9: Fix Missing Model Entry (claude-opus-4-6)

**Files:**
- Modify: `headroom/providers/anthropic.py`

- [ ] **Step 1: Add claude-opus-4-6 to ANTHROPIC_CONTEXT_LIMITS**

In `headroom/providers/anthropic.py`, find `ANTHROPIC_CONTEXT_LIMITS` (line 47). Add after the claude-opus-4-5 entry:

```python
    "claude-opus-4-6": 1000000,
```

Also add to `ANTHROPIC_PRICING` (line 72):

```python
    "claude-opus-4-6": {"input": 15.00, "output": 75.00, "cached_input": 1.50},
```

- [ ] **Step 2: Verify**

Run: `python -c "from headroom.providers.anthropic import AnthropicProvider; p = AnthropicProvider.__new__(AnthropicProvider); p._context_limits = {}; from headroom.providers.anthropic import ANTHROPIC_CONTEXT_LIMITS; print(ANTHROPIC_CONTEXT_LIMITS.get('claude-opus-4-6'))"`
Expected: `1000000`

- [ ] **Step 3: Commit**

```bash
git add headroom/providers/anthropic.py
git commit -m "fix: add claude-opus-4-6 (1M context) to ANTHROPIC_CONTEXT_LIMITS"
```

---

## Chunk 4: Integration + E2E Tests

### Task 10: Integration Tests — Token Headroom Pipeline

**Files:**
- Create: `tests/test_token_headroom_mode.py`

- [ ] **Step 1: Write integration tests**

Create `tests/test_token_headroom_mode.py`:

```python
"""Integration tests for token_headroom mode.

Tests the full pipeline flow: CompressionCache + ContentRouter + pipeline
working together across simulated multi-turn conversations.
"""

import copy
import pytest

from headroom.cache.compression_cache import CompressionCache
from headroom.config import HeadroomConfig
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.pipeline import TransformPipeline, PipelineConfig
from headroom.tokenizers import EstimatingTokenCounter


def _make_user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def _make_assistant_msg(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _make_tool_use_msg(tool_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _make_tool_result_msg(tool_id: str, content: str) -> dict:
    """Anthropic-format tool result."""
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
    }


def _make_openai_tool_msg(tool_call_id: str, content: str) -> dict:
    """OpenAI-format tool result."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _large_code_content(lines: int = 200) -> str:
    """Generate realistic Python code content."""
    parts = ["import os\nimport sys\nfrom typing import List, Dict\n\n"]
    for i in range(lines // 10):
        parts.append(
            f"def function_{i}(arg: str) -> str:\n"
            f'    """Docstring for function {i}."""\n'
            f"    result = arg.strip()\n"
            f"    for j in range({i}):\n"
            f"        result += str(j)\n"
            f"    return result\n\n"
        )
    return "".join(parts)


class TestTokenHeadroomMultiTurn:
    """Simulate multi-turn conversations to verify compression cascade."""

    def test_messages_compress_as_they_age_out(self):
        """Turn by turn: messages age out of protection window and get compressed."""
        cache = CompressionCache()
        router_config = ContentRouterConfig(
            protect_recent_reads_fraction=0.3,
            enable_kompress=False,  # Disable ML compressors for test speed
            enable_llmlingua=False,
        )
        router = ContentRouter(router_config)

        # Build a 20-message conversation
        messages = []
        for i in range(10):
            messages.append(_make_user_msg(f"Read file_{i}.py"))
            messages.append(_make_tool_use_msg(f"t{i}", "Read"))
            # Large tool result that should be compressible
            messages.append(_make_tool_result_msg(f"t{i}", _large_code_content(100)))

        assert len(messages) == 30

        # Simulate turn with token_headroom pipeline
        working = cache.apply_cached(messages)
        frozen = cache.compute_frozen_count(messages)

        # First turn: nothing cached yet, frozen=0 for first tool_result
        # (user messages are stable, but first tool_result at index 2 is a miss)
        # Actually: msg 0 = user (stable), msg 1 = assistant/tool_use (stable),
        # msg 2 = tool_result (cache miss) → frozen = 2
        assert frozen <= 2  # First tool result is uncached

    def test_no_message_injection(self):
        """Critical: output message count == input message count."""
        cache = CompressionCache()

        messages = [
            _make_user_msg("hello"),
            _make_tool_use_msg("t1", "Read"),
            _make_tool_result_msg("t1", _large_code_content(50)),
            _make_user_msg("edit it"),
        ]

        # Pre-populate cache with some content
        cache.store_compressed(
            CompressionCache.content_hash(_large_code_content(50)),
            "compressed code",
            tokens_saved=500,
        )

        result = cache.apply_cached(messages)
        assert len(result) == len(messages)

    def test_claude_code_drops_messages(self):
        """When CC drops messages, proxy does not re-add them."""
        cache = CompressionCache()

        # Simulate: first 5 messages were sent on turn 1
        content_a = "content A " * 100
        content_b = "content B " * 100
        cache.store_compressed(
            CompressionCache.content_hash(content_a), "ca", tokens_saved=100
        )
        cache.store_compressed(
            CompressionCache.content_hash(content_b), "cb", tokens_saved=100
        )

        # Turn 2: CC dropped the message with content_b
        messages = [
            _make_user_msg("hello"),
            _make_tool_result_msg("t1", content_a),  # still present
            # content_b message was DROPPED by CC
            _make_user_msg("continue"),
        ]

        result = cache.apply_cached(messages)
        assert len(result) == 3  # NOT 4 — no re-insertion

    def test_user_assistant_never_compressed(self):
        """User and assistant messages are never modified regardless of age."""
        cache = CompressionCache()

        messages = [
            _make_user_msg("important instruction " * 50),
            _make_assistant_msg("detailed response " * 50),
        ]

        result = cache.apply_cached(messages)
        assert result[0]["content"] == messages[0]["content"]
        assert result[1]["content"] == messages[1]["content"]

    def test_openai_format_tool_results(self):
        """OpenAI-format tool messages are handled correctly."""
        cache = CompressionCache()
        content = "large tool output " * 100
        h = CompressionCache.content_hash(content)
        cache.store_compressed(h, "compressed output", tokens_saved=300)

        messages = [
            _make_user_msg("run command"),
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
            ]},
            _make_openai_tool_msg("tc1", content),
        ]

        result = cache.apply_cached(messages)
        assert len(result) == 3
        assert result[2]["content"] == "compressed output"

    def test_re_freeze_consecutive_cached(self):
        """After first compression, Zone 1 forms a stable re-frozen prefix."""
        cache = CompressionCache()

        content_1 = "file one " * 100
        content_2 = "file two " * 100
        h1 = CompressionCache.content_hash(content_1)
        h2 = CompressionCache.content_hash(content_2)
        cache.store_compressed(h1, "c1", tokens_saved=100)
        cache.store_compressed(h2, "c2", tokens_saved=100)

        messages = [
            _make_user_msg("read files"),
            _make_tool_result_msg("t1", content_1),
            _make_tool_result_msg("t2", content_2),
            _make_user_msg("now edit"),
            _make_tool_result_msg("t3", "brand new content not cached"),
        ]

        frozen = cache.compute_frozen_count(messages)
        # msg 0: user (stable), msg 1: tool_result (cached → stable),
        # msg 2: tool_result (cached → stable), msg 3: user (stable),
        # msg 4: tool_result (NOT cached → stop)
        assert frozen == 4
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_token_headroom_mode.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_token_headroom_mode.py
git commit -m "test: add integration tests for token_headroom mode"
```

---

### Task 11: Run Full Test Suite — No Side Effects

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ --timeout=60 -x -q --ignore=tests/test_token_headroom_mode.py`
Expected: All existing tests PASS — no side effects from the new code. The default mode is `cost_savings` so all existing behavior is unchanged.

- [ ] **Step 2: Run new tests**

Run: `python -m pytest tests/test_compression_cache.py tests/test_token_headroom_mode.py -v`
Expected: All new tests PASS

- [ ] **Step 3: Run linting**

Run: `ruff check headroom/cache/compression_cache.py headroom/proxy/server.py headroom/providers/anthropic.py`
Expected: No errors

Run: `ruff format --check headroom/cache/compression_cache.py`
Expected: No formatting issues

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix: address linting issues from token headroom implementation"
```

(Only if linting found issues)

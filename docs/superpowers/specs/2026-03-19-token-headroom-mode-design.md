# Token Headroom Mode — Design Spec

**Date**: 2026-03-19
**Status**: Approved
**Problem**: In long Claude Code sessions, the prefix freeze mechanism prevents compression of cached messages. Users see ~1% compression instead of 20-30%, burning through usage limits faster.

## Background

Headroom's proxy has two competing optimizations:

1. **Prefix freeze**: Preserves Anthropic's prefix cache (90% read discount) by never modifying cached messages. Saves money.
2. **Compression**: Reduces token count by compressing older tool results. Extends sessions.

Currently, prefix freeze always wins. Once Anthropic caches a message prefix (which happens after the first turn), the proxy freezes those messages and never compresses them. As the session grows, the frozen prefix grows to cover nearly all messages. Compression only runs on the last few new messages per turn.

For users who are bottlenecked on **token consumption limits** (not cost), this is the wrong tradeoff. They need fewer tokens, not cheaper tokens.

## Solution: Dual-Mode Optimization

Add a `HEADROOM_MODE` config toggle:

- `cost_savings` (default): Current behavior. Prefix freeze enabled. Optimizes for API cost.
- `token_headroom`: Compresses older messages to reduce token count. Accepts prefix cache busts to gain session length.

## Architecture

### CompressionCache

Content-addressed cache mapping original content hashes to compressed versions. Session-scoped (lives as long as the proxy process).

```python
class CompressionCache:
    """Content-addressed cache of compressed messages.

    Maps original_content_hash → compressed_content.
    No position tracking — works purely on content identity.
    Handles Claude Code dropping messages gracefully (unused entries stay in cache).
    """
    cache: dict[str, str]          # content_hash → compressed_content
    token_savings: dict[str, int]  # content_hash → tokens saved
    max_entries: int = 10000       # LRU eviction beyond this

    def get_compressed(self, content_hash: str) -> str | None
    def store_compressed(self, content_hash: str, compressed: str, tokens_saved: int)
    def compute_frozen_count(self, messages: list[dict]) -> int
    def update_from_result(self, originals: list[dict], compressed: list[dict]) -> None
```

**Cache key**: SHA-256 of original content. Same file content across turns = same hash = cache hit. Edited file = different hash = cache miss = fresh compression.

**Max entries**: 10,000 (not 2,000). Long coding sessions can generate thousands of tool results. Evicting cache entries forces re-compression, defeating the re-freeze optimization. 10K entries at ~1KB average compressed text = ~10MB — acceptable.

**Frozen count calculation**: Count consecutive messages from the start that are "stable" (either have cache hits OR are non-compressible messages like user/assistant/tool-use). The key rule:

- **Tool-result messages**: Stable if content hash is in cache (compressed version available).
- **User, assistant, system, tool-use messages**: Always considered stable (they pass through unchanged, so they won't bust the prefix cache).

First *unstable* message (tool-result with cache miss) = end of frozen prefix. This prevents the frozen count from being 0 just because message 0 is a user message.

This handles Claude Code dropping messages: if a cached tool-result is dropped, the next message may not be in cache, breaking the consecutive run and stopping the frozen prefix there.

**`update_from_result` contract**: Performs index-aligned comparison between `originals` and `compressed`. For each index where content differs, stores the mapping original_hash → compressed_content. Assumes the pipeline does NOT reorder or merge messages (this is a pipeline contract that holds today and must be preserved).

### Pipeline Flow in Token Headroom Mode

Each turn, the proxy receives all messages from Claude Code (originals, uncompressed):

```
Messages: [m1, m2, m3, ..., mP, mP+1, ..., mN]
           |← cache hits →|← cache misses →|← protection window (30%) →|

Zone 1: Cache hits       → Swap in compressed versions, re-freeze as stable prefix
Zone 2: Cache misses     → Run full ContentRouter compression, cache results
Zone 3: Protection window → Pass through unchanged (last 30% of messages)
```

**Zone 1** (already compressed): The proxy swaps in cached compressed content. These form a new stable prefix. Anthropic caches them. Next turn gets 90% read discount on Zone 1.

**Zone 2** (newly eligible): First time outside protection window. ContentRouter runs full compression: CodeAwareCompressor for code, SmartCrusher for JSON, Kompress for text. Results cached. On next turn, these move into Zone 1.

**Zone 3** (protected): Recent messages, untouched. The LLM needs these fresh for current work.

**Critical invariant**: The proxy ONLY operates on messages Claude Code actually sends. It never adds, re-inserts, or reorders messages. If Claude Code drops a message via its own context management, the proxy does not re-add it.

### Zone 3 (Protection Window) Definition

Zone 3 protects the **last 30% of excluded-tool messages** (Read, Glob, Grep, Write, Edit results), not 30% of all messages. This matches the existing `protect_recent_reads_fraction` semantics in ContentRouter. User messages, assistant messages, and tool-use blocks are always protected regardless of position — they don't factor into the 30% calculation.

Example: 100 messages, 40 of which are Read/Glob/Grep results → last 12 excluded-tool results protected (30% of 40). All user/assistant messages protected independently.

### Config Plumbing

`protect_recent_reads_fraction` is a `ContentRouterConfig` instance attribute, not a per-call kwarg. It is set **at proxy startup** based on mode:

```python
# In HeadroomProxy.__init__ or _build_pipelines:
if self.config.mode == "token_headroom":
    router_config.protect_recent_reads_fraction = 0.3
    router_config.ccr_ttl = 14400
else:
    router_config.protect_recent_reads_fraction = 0.0  # protect all (current default)
```

Similarly, `force_compress_threshold` on `PrefixFreezeConfig` is set to `0.0` in token_headroom mode (always allow compression through the cache). This avoids two competing mechanisms for cache-bust decisions.

### Integration Point

In `server.py`, `_handle_anthropic_request` (line ~2071):

```python
if self.config.mode == "token_headroom":
    comp_cache = self.compression_cache_store.get_or_create(session_id)

    # Zone 1: Swap cached compressed versions
    working_messages = comp_cache.apply_cached(messages)

    # Calculate re-freeze boundary (consecutive stable messages from start)
    frozen_count = comp_cache.compute_frozen_count(messages)

    # Zone 2+3: Pipeline compresses Zone 2, skips Zone 3 (protected)
    result = self.anthropic_pipeline.apply(
        working_messages, model,
        frozen_message_count=frozen_count,
    )

    # Cache newly compressed messages (index-aligned diff)
    comp_cache.update_from_result(messages, result.messages)
else:
    # Current cost_savings behavior — unchanged
    frozen_count = prefix_tracker.get_frozen_message_count()
    result = self.anthropic_pipeline.apply(
        messages, model, frozen_message_count=frozen_count
    )
```

## Tool-Specific Compression Strategies

| Tool Result | Compressor | CCR Retrieval | Expected Reduction |
|---|---|---|---|
| Read (code) | CodeAwareCompressor (tree-sitter AST) | Yes — hash marker, original stored | 60-80% |
| Read (non-code) | Kompress / SmartCrusher | Yes | 40-60% |
| Glob | SmartCrusher (JSON array crush) | No (too small) | 50-70% |
| Grep | SmartCrusher (keep matches, drop context) | Yes | 40-60% |
| Bash | ContentRouter auto-routes (log/text/JSON) | Yes for large outputs | 50-80% |
| Stale Read | ReadLifecycle marker (already built) | Yes — original stored | ~95% |
| Superseded Read | ReadLifecycle marker | No (later read has content) | ~95% |

### CodeAwareCompressor (existing, key for Read results)

- Uses tree-sitter for proper AST parsing (multi-language)
- Keeps: imports, function/class signatures, type hints, docstrings, decorators
- Removes: function bodies (replaced with `...` + line count)
- CCR integration: stores original in CCR store, appends `# [N tokens compressed. Retrieve more: hash=abc123.]`
- Syntax validity guaranteed
- If tree-sitter unavailable: falls back to text-level compression via ContentRouter chain

### ReadLifecycle (existing, runs before ContentRouter)

- **Stale reads** (file edited after read): Replaced with `[file.py was modified after this read. Re-read if needed.]`
- **Superseded reads** (same file re-read later): Replaced with marker pointing to later read
- In token_headroom mode: runs without prefix freeze blocking it

## Configuration

### New config on ProxyConfig

```python
mode: str = "cost_savings"  # "cost_savings" | "token_headroom"
```

Activation: `HEADROOM_MODE=token_headroom` environment variable or config file.

### Settings that change per mode

| Setting | cost_savings (default) | token_headroom |
|---|---|---|
| prefix_freeze_enabled | True | True (re-freezes compressed content) |
| protect_recent_reads_fraction | 0.0 (protect all) | 0.3 (protect last 30%) |
| ccr_ttl | 300 (5 min) | 14400 (4 hours) |
| read_lifecycle | True (frozen blocks it) | True (runs freely on aged-out) |
| CompressionCache | Not used | Active per session |
| Excluded tools (Read/Glob/Grep) | Protected forever | Protected in last 30% only |

### What stays the same in both modes

- User messages: always protected
- Assistant messages: always protected
- Tool-use blocks: always protected
- ContentRouter routing logic
- All compressors (CodeAware, SmartCrusher, Kompress)
- MCP server tools

### Startup log

```
Headroom proxy v0.4.6 | Mode: token_headroom
  Prefix freeze: enabled (re-freeze after compression)
  Read protection window: 30% of messages
  CCR TTL: 4 hours
  Compression cache: active
```

## Observability

In token_headroom mode, the `/stats` endpoint and startup logs must expose:

- **Per-turn zone breakdown**: Zone 1/2/3 message counts and token counts
- **Compression cache stats**: entries, hits, misses, hit rate
- **Net token savings**: tokens saved by compression minus tokens lost to cache bust write premium
- **CCR store stats**: entries, memory usage, eviction count

The CCR store uses TTL-based eviction (entries expire after 4 hours). The `CompressionCache` uses LRU eviction at 10K entries. Both report eviction counts in stats so operators can detect if bounds are too tight.

## Also Fix: Missing Model Entry (Separate Task)

`claude-opus-4-6` is not in `ANTHROPIC_CONTEXT_LIMITS`. Falls back to pattern match → 200K. Real context is 1M. Add it with correct limit. (Separate from token_headroom mode but discovered during investigation.)

## Edge Cases

1. **Very short conversations (< 10 messages)**: Protection window covers almost everything. Minimum protection of 4 messages means nothing compressed until message ~6+. Correct — no overhead for short sessions.

2. **All user/assistant messages (no tool results)**: These are always protected. CompressionCache stays empty. No-op. Correct.

3. **CodeAwareCompressor not installed**: Falls back to Kompress/SmartCrusher via ContentRouter's existing chain. Log warning.

4. **CCR store unavailable**: Compression still happens, just no retrieval markers. LLM must re-read if it needs exact content. Acceptable degradation.

5. **Session resume (--resume) with empty cache**: First turn: full compression pass. Turn 2: cache populated, re-freeze, prefix cache rebuilds. Recovery within 1-2 turns.

6. **Compression makes content larger**: ContentRouter already checks compression_ratio >= min_ratio and passes through unchanged. Cache only stores actual reductions.

7. **Claude Code drops messages**: Cache entries go unused (LRU eviction eventually). Proxy never re-inserts dropped messages. Frozen count stops at first unstable message, preventing stale frozen prefix.

8. **Mode switch mid-session**: If proxy restarts with a different mode, the CompressionCache is empty (new process) and the prefix tracker is reset. In token_headroom mode, the first turn does a full compression pass (edge case 5). In cost_savings mode, the prefix tracker rebuilds from the first API response. No stale state carries over.

9. **Large Zone 2 on first turn after --resume**: Many tool results age out simultaneously. ContentRouter processes them sequentially. If compression latency exceeds 10 seconds, log a warning but do not skip — the user explicitly chose token_headroom mode and the one-time cost is justified. Subsequent turns are fast (cache hits).

## Testing Strategy

### Unit Tests — CompressionCache

- `test_store_and_retrieve`: Cache hit/miss behavior
- `test_different_content_different_hash`: Edited file → new hash → fresh compression
- `test_lru_eviction`: Max entries respected
- `test_frozen_count_consecutive_hits`: Stops at first miss
- `test_frozen_count_with_dropped_messages`: Claude Code drops messages → correct frozen boundary
- `test_protection_window_calculation`: 30% of N messages, minimum 4

### Integration Tests — Pipeline in Token Headroom Mode

- `test_mode_toggle`: Same conversation, both modes → token_headroom produces fewer tokens
- `test_multi_turn_waterfall`: 10-turn simulation. Verify messages age out → compress → cache → re-freeze. Cumulative savings > 30%.
- `test_claude_code_drops_messages`: Proxy does NOT re-insert dropped messages
- `test_stale_read_lifecycle`: Read → Edit → verify stale marker with CCR hash
- `test_read_edit_reread`: First read stale (marker), second read fresh (full content)
- `test_code_compressor_with_ccr`: AST compression, CCR marker, TTL = 14400s, valid syntax
- `test_glob_grep_bash_compression`: Each compressor type activates for its content type
- `test_re_freeze_after_compression`: Zone 1 re-frozen, pipeline sees correct frozen_count
- `test_user_assistant_always_protected`: Never compressed regardless of age

### End-to-End Simulation

- `test_e2e_long_session`: 70+ requests simulating bug report. cost_savings ~1%, token_headroom >25%.
- `test_e2e_resumed_session`: Empty cache → full compression → cache recovery in 2 turns.
- `test_e2e_cost_comparison`: Log tokens + cost for both modes. Verify token_headroom extends session >30%.
- `test_e2e_no_message_injection`: Verify output message count <= input message count (never adds messages).

## Files to Modify

1. **New file**: `headroom/cache/compression_cache.py` — CompressionCache class
2. **Modify**: `headroom/proxy/server.py` — ProxyConfig.mode, mode-based config at startup, pipeline flow branch, CompressionCache session store, stats endpoint additions
3. **Modify**: `headroom/proxy/server.py` — Set `ContentRouterConfig.protect_recent_reads_fraction` and `PrefixFreezeConfig.force_compress_threshold` at startup based on mode
4. **Modify**: `headroom/providers/anthropic.py` — Add claude-opus-4-6 to ANTHROPIC_CONTEXT_LIMITS (separate commit)
5. **New file**: `tests/test_compression_cache.py` — Unit tests for CompressionCache
6. **New file**: `tests/test_token_headroom_mode.py` — Integration + E2E tests

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
    max_entries: int = 2000        # LRU eviction beyond this

    def get_compressed(self, content_hash: str) -> str | None
    def store_compressed(self, content_hash: str, compressed: str, tokens_saved: int)
    def compute_frozen_count(self, messages: list[dict]) -> int
```

**Cache key**: SHA-256 of original content. Same file content across turns = same hash = cache hit. Edited file = different hash = cache miss = fresh compression.

**Frozen count calculation**: Count consecutive messages from the start that have cache hits. First cache miss = end of stable compressed prefix. This handles Claude Code dropping messages (breaks the consecutive run, stops the frozen prefix there).

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

### Integration Point

In `server.py`, `_handle_anthropic_request` (line ~2071):

```python
if self.config.mode == "token_headroom":
    comp_cache = self.compression_cache_store.get_or_create(session_id)

    # Zone 1: Swap cached compressed versions
    working_messages = comp_cache.apply_cached(messages)

    # Calculate re-freeze boundary (consecutive cache hits from start)
    frozen_count = comp_cache.compute_frozen_count(messages)

    # Zone 2+3: Pipeline compresses Zone 2, skips Zone 3
    result = self.anthropic_pipeline.apply(
        working_messages, model,
        frozen_message_count=frozen_count,
        protect_recent_fraction=0.3,
    )

    # Cache newly compressed messages
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

## Also Fix: Missing Model Entry

`claude-opus-4-6` is not in `ANTHROPIC_CONTEXT_LIMITS`. Falls back to pattern match → 200K. Real context is 1M. Add it with correct limit. (Separate from token_headroom mode but discovered during investigation.)

## Edge Cases

1. **Very short conversations (< 10 messages)**: Protection window covers almost everything. Minimum protection of 4 messages means nothing compressed until message ~6+. Correct — no overhead for short sessions.

2. **All user/assistant messages (no tool results)**: These are always protected. CompressionCache stays empty. No-op. Correct.

3. **CodeAwareCompressor not installed**: Falls back to Kompress/SmartCrusher via ContentRouter's existing chain. Log warning.

4. **CCR store unavailable**: Compression still happens, just no retrieval markers. LLM must re-read if it needs exact content. Acceptable degradation.

5. **Session resume (--resume) with empty cache**: First turn: full compression pass. Turn 2: cache populated, re-freeze, prefix cache rebuilds. Recovery within 1-2 turns.

6. **Compression makes content larger**: ContentRouter already checks compression_ratio >= min_ratio and passes through unchanged. Cache only stores actual reductions.

7. **Claude Code drops messages**: Cache entries go unused (LRU eviction eventually). Proxy never re-inserts dropped messages. Frozen count stops at first cache miss, preventing stale frozen prefix.

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
2. **Modify**: `headroom/proxy/server.py` — ProxyConfig.mode, pipeline flow branch, session store
3. **Modify**: `headroom/transforms/pipeline.py` — Accept protect_recent_fraction kwarg
4. **Modify**: `headroom/transforms/content_router.py` — Use protect_recent_fraction from kwargs
5. **Modify**: `headroom/providers/anthropic.py` — Add claude-opus-4-6 to ANTHROPIC_CONTEXT_LIMITS
6. **New file**: `tests/test_compression_cache.py` — Unit tests
7. **New file**: `tests/test_token_headroom_mode.py` — Integration + E2E tests

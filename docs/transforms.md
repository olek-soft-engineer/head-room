# Transform Reference

Headroom provides three core transforms that work together to optimize LLM context.

## SmartCrusher

Statistical compression for JSON tool outputs.

### How It Works

SmartCrusher analyzes JSON arrays and selectively keeps important items:

1. **First/Last items** - Context for pagination and recency
2. **Error items** - 100% preservation of error states
3. **Anomalies** - Statistical outliers (> 2 std dev from mean)
4. **Relevant items** - Matches to user's query via BM25/embeddings
5. **Change points** - Significant transitions in data

### Configuration

```python
from headroom import SmartCrusherConfig

config = SmartCrusherConfig(
    min_tokens_to_crush=200,      # Only compress if > 200 tokens
    max_items_after_crush=50,     # Keep at most 50 items
    keep_first=3,                 # Always keep first 3 items
    keep_last=2,                  # Always keep last 2 items
    relevance_threshold=0.3,      # Keep items with relevance > 0.3
    anomaly_std_threshold=2.0,    # Keep items > 2 std dev from mean
    preserve_errors=True,         # Always keep error items
)
```

### Example

```python
from headroom import SmartCrusher

crusher = SmartCrusher(config)

# Before: 1000 search results (45,000 tokens)
tool_output = {"results": [...1000 items...]}

# After: ~50 important items (4,500 tokens) - 90% reduction
compressed = crusher.crush(tool_output, query="user's question")
```

### What Gets Preserved

| Category | Preserved | Why |
|----------|-----------|-----|
| Errors | 100% | Critical for debugging |
| First N | 100% | Context/pagination |
| Last N | 100% | Recency |
| Anomalies | All | Unusual values matter |
| Relevant | Top K | Match user's query |
| Others | Sampled | Statistical representation |

---

## CacheAligner

Prefix stabilization for improved cache hit rates.

### The Problem

LLM providers cache request prefixes. But dynamic content breaks caching:

```
"You are helpful. Today is January 7, 2025."  # Changes daily = no cache
```

### The Solution

CacheAligner extracts dynamic content to stabilize the prefix:

```python
from headroom import CacheAligner

aligner = CacheAligner()
result = aligner.align(messages)

# Static prefix (cacheable):
# "You are helpful."

# Dynamic content moved to end:
# [Current date context]
```

### Configuration

```python
from headroom import CacheAlignerConfig

config = CacheAlignerConfig(
    extract_dates=True,           # Move dates to dynamic section
    normalize_whitespace=True,    # Consistent spacing
    stable_prefix_min_tokens=100, # Min prefix size for alignment
)
```

### Cache Hit Improvement

| Scenario | Before | After |
|----------|--------|-------|
| Daily date in prompt | 0% hits | ~95% hits |
| Dynamic user context | ~10% hits | ~80% hits |
| Consistent prompts | ~90% hits | ~95% hits |

---

## RollingWindow

Context management within token limits.

### The Problem

Long conversations exceed context limits. Naive truncation breaks tool calls:

```
[tool_call: search]  # Kept
[tool_result: ...]   # Dropped = orphaned call!
```

### The Solution

RollingWindow drops complete tool units, preserving pairs:

```python
from headroom import RollingWindow

window = RollingWindow(config)
result = window.apply(messages, max_tokens=100000)

# Guarantees:
# 1. Tool calls paired with results
# 2. System prompt preserved
# 3. Recent turns kept
# 4. Oldest tool outputs dropped first
```

### Configuration

```python
from headroom import RollingWindowConfig

config = RollingWindowConfig(
    max_tokens=100000,            # Target token limit
    preserve_system=True,         # Always keep system prompt
    preserve_recent_turns=5,      # Keep last 5 user/assistant turns
    drop_oldest_first=True,       # Remove oldest tool outputs
)
```

### Drop Priority

1. **Oldest tool outputs** - First to go
2. **Old assistant messages** - Summary preserved
3. **Old user messages** - Only if necessary
4. **Never dropped**: System prompt, recent turns, active tool pairs

---

## TransformPipeline

Combine transforms for optimal results.

```python
from headroom import TransformPipeline, SmartCrusher, CacheAligner, RollingWindow

pipeline = TransformPipeline([
    SmartCrusher(),      # First: compress tool outputs
    CacheAligner(),      # Then: stabilize prefix
    RollingWindow(),     # Finally: fit in context
])

result = pipeline.transform(messages)
print(f"Saved {result.tokens_saved} tokens")
```

### Recommended Order

1. **SmartCrusher** - Reduce individual messages
2. **CacheAligner** - Optimize for caching
3. **RollingWindow** - Final size constraint

---

## Safety Guarantees

All transforms follow strict safety rules:

1. **Never remove human content** - User/assistant text is sacred
2. **Never break tool ordering** - Calls and results stay paired
3. **Parse failures are no-ops** - Malformed content passes through
4. **Preserves recency** - Last N turns always kept
5. **100% error preservation** - Error items never dropped

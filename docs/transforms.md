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

## LLMLinguaCompressor (Optional)

ML-based compression using Microsoft's LLMLingua-2 model.

### When to Use

| Transform | Best For | Speed | Compression |
|-----------|----------|-------|-------------|
| SmartCrusher | JSON arrays | ~1ms | 70-90% |
| Text Utilities | Search/logs | ~1ms | 50-90% |
| **LLMLinguaCompressor** | Any text, max compression | 50-200ms | 80-95% |

### Installation

```bash
pip install "headroom-ai[llmlingua]"  # Adds ~2GB
```

### Configuration

```python
from headroom.transforms import LLMLinguaCompressor, LLMLinguaConfig

config = LLMLinguaConfig(
    device="auto",                    # auto, cuda, cpu, mps
    target_compression_rate=0.3,      # Keep 30% of tokens
    min_tokens_for_compression=100,   # Skip small content
    code_compression_rate=0.4,        # Conservative for code
    json_compression_rate=0.35,       # Moderate for JSON
    text_compression_rate=0.25,       # Aggressive for text
    enable_ccr=True,                  # Store original for retrieval
)

compressor = LLMLinguaCompressor(config)
```

### Content-Aware Rates

LLMLinguaCompressor auto-detects content type:

| Content Type | Default Rate | Behavior |
|--------------|--------------|----------|
| Code | 0.4 | Conservative - preserves syntax |
| JSON | 0.35 | Moderate - keeps structure |
| Text | 0.3 | Aggressive - maximum compression |

### Memory Management

```python
from headroom.transforms import (
    is_llmlingua_model_loaded,
    unload_llmlingua_model,
)

# Check if model is loaded
print(is_llmlingua_model_loaded())  # True/False

# Free ~1GB RAM when done
unload_llmlingua_model()
```

### Proxy Integration

```bash
# Enable in proxy
headroom proxy --llmlingua --llmlingua-device cuda --llmlingua-rate 0.3
```

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

### With LLMLingua (Optional)

```python
from headroom.transforms import (
    TransformPipeline, SmartCrusher, CacheAligner,
    RollingWindow, LLMLinguaCompressor
)

pipeline = TransformPipeline([
    CacheAligner(),         # 1. Stabilize prefix
    SmartCrusher(),         # 2. Compress JSON arrays
    LLMLinguaCompressor(),  # 3. ML compression on remaining text
    RollingWindow(),        # 4. Final size constraint (always last)
])
```

### Recommended Order

| Order | Transform | Purpose |
|-------|-----------|---------|
| 1 | CacheAligner | Stabilize prefix for caching |
| 2 | SmartCrusher | Compress JSON tool outputs |
| 3 | LLMLinguaCompressor | ML compression (optional) |
| 4 | RollingWindow | Enforce token limits (always last) |

**Why this order?**
- CacheAligner first to maximize prefix stability
- SmartCrusher handles JSON arrays efficiently
- LLMLingua compresses remaining long text
- RollingWindow truncates only if still over limit

---

## Safety Guarantees

All transforms follow strict safety rules:

1. **Never remove human content** - User/assistant text is sacred
2. **Never break tool ordering** - Calls and results stay paired
3. **Parse failures are no-ops** - Malformed content passes through
4. **Preserves recency** - Last N turns always kept
5. **100% error preservation** - Error items never dropped

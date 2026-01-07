<p align="center">
  <h1 align="center">Headroom</h1>
  <p align="center">
    <strong>The Context Optimization Layer for LLM Applications</strong>
  </p>
  <p align="center">
    Cut your LLM costs by 50-90% without losing accuracy
  </p>
</p>

<p align="center">
  <a href="https://github.com/headroom-sdk/headroom/actions/workflows/ci.yml">
    <img src="https://github.com/headroom-sdk/headroom/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://pypi.org/project/headroom/">
    <img src="https://img.shields.io/pypi/v/headroom.svg" alt="PyPI">
  </a>
  <a href="https://pypi.org/project/headroom/">
    <img src="https://img.shields.io/pypi/pyversions/headroom.svg" alt="Python">
  </a>
  <a href="https://github.com/headroom-sdk/headroom/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

---

## The Problem

AI coding agents and tool-using applications generate **massive contexts**:

- Tool outputs with 1000s of search results, log entries, API responses
- Long conversation histories that hit token limits
- System prompts with dynamic dates that break provider caching

**Result**: You pay for tokens you don't need, and cache hits are rare.

## The Solution

Headroom is a **smart compression layer** that sits between your app and LLM providers. It applies three transforms:

| Transform | What It Does | Savings |
|-----------|--------------|---------|
| **SmartCrusher** | Compresses tool outputs statistically (keeps errors, anomalies, relevant items) | 70-90% |
| **CacheAligner** | Stabilizes prefixes so provider caching works | Up to 10x |
| **RollingWindow** | Manages context within limits without breaking tool calls | Prevents failures |

**Zero accuracy loss** - we keep what matters: errors, anomalies, relevant items.

## Quick Start

### Option 1: Proxy (Recommended)

Run Headroom as a proxy server - works with any client:

```bash
pip install headroom

# Start the proxy
headroom proxy --port 8787

# Use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8787 claude

# Use with any OpenAI-compatible client
OPENAI_BASE_URL=http://localhost:8787/v1 your-app
```

### Option 2: Python SDK

Wrap your existing client:

```python
from headroom import HeadroomClient
from openai import OpenAI

client = HeadroomClient(
    original_client=OpenAI(),
    default_mode="optimize",
)

# Use exactly like the original client
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
)
```

### Option 3: LangChain Integration

```python
from langchain_openai import ChatOpenAI
from headroom.integrations import HeadroomOptimizer

llm = ChatOpenAI(model="gpt-4o", callbacks=[HeadroomOptimizer()])
```

## Features

### Smart Tool Output Compression

```python
# Before: 50KB tool response with 1000 items
{"results": [{"id": 1, ...}, {"id": 2, ...}, ... 1000 items ...]}

# After: ~2KB with important items preserved
# - First 3 items (context)
# - Last 2 items (recency)
# - All error items
# - Anomalous values (> 2 std dev)
# - Items matching user's query
```

### Cache-Aligned Prefixes

```python
# Before: Cache miss every day due to changing date
"You are helpful. Today is January 7, 2025."

# After: Stable prefix (cache hit!) + dynamic context
"You are helpful."
# [Dynamic context moved to end]
```

### Rolling Window

```python
# Automatically manages context within token limits
# - Drops oldest tool outputs first
# - Never orphans tool call/response pairs
# - Always preserves system prompt and recent turns
```

### Production Proxy Features

- **Semantic Caching**: LRU cache with TTL for repeated queries
- **Rate Limiting**: Token bucket (requests + tokens per minute)
- **Cost Tracking**: Budget enforcement (hourly/daily/monthly)
- **Prometheus Metrics**: `/metrics` endpoint for monitoring
- **Request Logging**: JSONL logs for debugging

## Installation

```bash
# Core (minimal dependencies)
pip install headroom

# With semantic relevance scoring
pip install headroom[relevance]

# With proxy server
pip install headroom[proxy]

# Everything
pip install headroom[all]
```

## Modes

### Audit Mode (Observe Only)

```python
client = HeadroomClient(original_client=base, default_mode="audit")
# Logs metrics but doesn't modify requests
```

### Optimize Mode (Apply Transforms)

```python
client = HeadroomClient(original_client=base, default_mode="optimize")
# Applies safe, deterministic transforms
```

### Simulate Mode (Preview)

```python
plan = client.chat.completions.simulate(model="gpt-4o", messages=[...])
print(f"Would save {plan.tokens_saved} tokens ({plan.savings_percent:.1f}%)")
```

## Configuration

```python
from headroom import HeadroomClient, SmartCrusherConfig

client = HeadroomClient(
    original_client=base,
    default_mode="optimize",
    smart_crusher_config=SmartCrusherConfig(
        min_tokens_to_crush=200,      # Only compress if > 200 tokens
        max_items_after_crush=50,     # Keep at most 50 items
        keep_first=3,                 # Always keep first 3
        keep_last=2,                  # Always keep last 2
        relevance_threshold=0.3,      # Keep items with relevance > 0.3
    ),
)
```

## Supported Providers

| Provider | Token Counting | Status |
|----------|----------------|--------|
| OpenAI | tiktoken | Full support |
| Anthropic | Official API | Full support |
| Google | Official API | Full support |
| Cohere | Official API | Full support |
| Mistral | Official tokenizer | Full support |
| LiteLLM | Via provider | Full support |

## Safety Guarantees

Headroom follows strict safety rules:

1. **Never removes human content** - User/assistant text is sacred
2. **Never breaks tool ordering** - Tool calls and responses stay paired
3. **Parse failures are no-ops** - Malformed content passes through unchanged
4. **Preserves recency** - Last N turns are always kept

## Benchmarks

| Scenario | Before | After | Savings |
|----------|--------|-------|---------|
| Search results (1000 items) | 45,000 tokens | 4,500 tokens | 90% |
| Log analysis (500 entries) | 22,000 tokens | 3,300 tokens | 85% |
| API response (nested JSON) | 15,000 tokens | 2,250 tokens | 85% |
| Long conversation (50 turns) | 80,000 tokens | 32,000 tokens | 60% |

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Proxy Server Documentation](docs/proxy.md)
- [Transform Reference](docs/transforms.md)
- [API Reference](docs/api.md)
- [Examples](examples/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
git clone https://github.com/headroom-sdk/headroom.git
cd headroom
pip install -e ".[dev]"
pytest
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Links

- [GitHub](https://github.com/headroom-sdk/headroom)
- [PyPI](https://pypi.org/project/headroom/)
- [Documentation](https://headroom.dev/docs)
- [Discord](https://discord.gg/headroom)

---

<p align="center">
  <sub>Built with care for the AI developer community</sub>
</p>

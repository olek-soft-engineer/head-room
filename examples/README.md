# Headroom Examples

This directory contains examples demonstrating Headroom's capabilities.

## Quick Start Examples

### basic_usage.py

Basic integration with OpenAI client:

```bash
export OPENAI_API_KEY='your-key'
python examples/basic_usage.py
```

### anthropic_example.py

Integration with Anthropic Claude:

```bash
export ANTHROPIC_API_KEY='your-key'
python examples/anthropic_example.py
```

### streaming_example.py

Streaming responses with optimization:

```bash
export OPENAI_API_KEY='your-key'
python examples/streaming_example.py
```

## Evaluation Examples

### smart_vs_naive_eval.py

Compare SmartCrusher against naive truncation:

```bash
export OPENAI_API_KEY='your-key'
python examples/smart_vs_naive_eval.py
```

### real_world_eval.py

Comprehensive evaluation with Anthropic models:

```bash
export ANTHROPIC_API_KEY='your-key'
python examples/real_world_eval.py
```

### real_world_openai_eval.py

Comprehensive evaluation with OpenAI models:

```bash
export OPENAI_API_KEY='your-key'
python examples/real_world_openai_eval.py
```

## Demo Directories

### langchain_demo/

Full LangChain agent integration demo:

```bash
# No API key needed for compression demo
PYTHONPATH=. python -m examples.langchain_demo.show_compression

# Full comparison (requires API key)
export OPENAI_API_KEY='your-key'
PYTHONPATH=. python -m examples.langchain_demo.run_comparison
```

See [langchain_demo/README.md](langchain_demo/README.md) for details.

### mcp_demo/

MCP (Model Context Protocol) integration demo:

```bash
export OPENAI_API_KEY='your-key'
PYTHONPATH=. python -m examples.mcp_demo.run_agent_eval
```

## Running Examples

All examples can be run from the repository root:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run any example
python examples/<example_name>.py
```

## Expected Results

| Example | Token Savings | Notes |
|---------|---------------|-------|
| basic_usage | 50-70% | Simple tool output compression |
| langchain_demo | 70-85% | Real agent with multiple tools |
| mcp_demo | 60-80% | MCP tool outputs |
| real_world_eval | 50-90% | Varies by scenario |

## Troubleshooting

**ModuleNotFoundError: No module named 'headroom'**

Run from the repository root with PYTHONPATH:

```bash
PYTHONPATH=. python examples/basic_usage.py
```

Or install in development mode:

```bash
pip install -e .
```

**API Key Errors**

Ensure your API keys are set:

```bash
export OPENAI_API_KEY='sk-...'
export ANTHROPIC_API_KEY='sk-ant-...'
```

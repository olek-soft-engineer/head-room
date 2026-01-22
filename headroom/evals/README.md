# Headroom Evaluation Framework

**Prove that compression preserves LLM accuracy through rigorous before/after testing.**

## Installation

```bash
pip install headroom-ai[evals]
```

This installs:
- `datasets` - HuggingFace datasets for benchmark data
- `sentence-transformers` - Semantic similarity computation
- `numpy` & `scikit-learn` - ML metrics
- `anthropic` & `openai` - LLM API clients

## Quick Start

### CLI Usage

```bash
# Quick sanity check (5 samples, uses Anthropic Claude)
python -m headroom.evals quick

# List all available datasets
python -m headroom.evals list

# Run benchmark on specific dataset
python -m headroom.evals benchmark --dataset hotpotqa -n 50

# Run all RAG datasets
python -m headroom.evals benchmark --dataset rag

# Run ALL datasets (comprehensive)
python -m headroom.evals benchmark --dataset all -o results.json

# Generate HTML report
python -m headroom.evals report -i results.json
```

### Python API

```python
from headroom.evals import run_quick_eval, BeforeAfterRunner, LLMConfig
from headroom.evals import load_hotpotqa, load_squad, load_tool_output_samples

# Quick evaluation
results = run_quick_eval(n_samples=5)
print(results.summary())

# Custom evaluation
runner = BeforeAfterRunner(
    llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    use_semantic_similarity=True,
)

# Load a dataset
suite = load_hotpotqa(n=100)

# Run evaluation
results = runner.run(suite, progress_callback=lambda cur, tot, r: print(f"{cur}/{tot}"))
print(results.summary())
```

## The Before/After Evaluation Pattern

This framework proves compression accuracy through a simple but rigorous pattern:

```
┌─────────────────────────────────────────────────────────────┐
│                    BEFORE/AFTER EVAL                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Original Context ────────┐                                │
│   (e.g., 10,000 tokens)    │                                │
│                            ▼                                │
│                      ┌─────────┐                            │
│                      │   LLM   │───► Response A             │
│                      └─────────┘                            │
│                            ▲                                │
│   Compressed Context ──────┘                                │
│   (e.g., 3,000 tokens)     │                                │
│                            ▼                                │
│                      ┌─────────┐                            │
│                      │   LLM   │───► Response B             │
│                      └─────────┘                            │
│                                                             │
│   Compare A vs B:                                           │
│   • F1 Score (token overlap)                               │
│   • Semantic Similarity (embedding cosine)                  │
│   • Ground Truth Match (if available)                       │
│   • Exact Match (normalized)                               │
│                                                             │
│   Result: PASS if responses are functionally equivalent     │
│           FAIL if compression lost critical information     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Available Datasets

### RAG / Retrieval (5 datasets)

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `hotpotqa` | Multi-hop QA requiring reasoning over multiple Wikipedia passages | 100 |
| `natural_questions` | Real Google search questions with Wikipedia answers | 100 |
| `triviaqa` | Large-scale trivia QA with evidence documents | 100 |
| `msmarco` | Real Bing search queries with relevant passages | 100 |
| `squad` | SQuAD v2 reading comprehension with extractive answers | 100 |

### Long Context (2 datasets)

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `longbench` | Long context understanding (4K-128K tokens) | 50 |
| `narrativeqa` | Story comprehension requiring narrative understanding | 100 |

### Tool Use (3 datasets)

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `bfcl` | Berkeley Function Calling Leaderboard - API schemas | 100 |
| `toolbench` | Real-world API tool usage scenarios | 100 |
| `tool_outputs` | Built-in realistic tool outputs (JSON, logs, etc.) | 8 |

### Code (2 datasets)

| Dataset | Description | Default N |
|---------|-------------|-----------|
| `codesearchnet` | Code snippets with natural language descriptions | 100 |
| `humaneval` | Hand-crafted programming problems (OpenAI) | 164 |

## Metrics

### F1 Score
Token-level overlap between original and compressed responses.
```
F1 = 2 * (precision * recall) / (precision + recall)
```
A score of 1.0 means identical tokens, 0.0 means no overlap.

### Semantic Similarity
Cosine similarity between sentence embeddings (using `all-MiniLM-L6-v2`).
Range: 0.0 to 1.0, where 1.0 = semantically identical.

### Exact Match
Boolean indicating if responses are identical after normalization.

### Ground Truth Match
Checks if the compressed response contains the known correct answer.

### Accuracy Preserved
Overall verdict: `True` if ANY of:
- F1 > 0.7
- Semantic Similarity > 0.85
- Contains Ground Truth

## Programmatic Usage

### Loading Datasets

```python
from headroom.evals import (
    load_hotpotqa,
    load_natural_questions,
    load_triviaqa,
    load_msmarco,
    load_squad,
    load_longbench,
    load_narrativeqa,
    load_bfcl,
    load_toolbench,
    load_codesearchnet,
    load_humaneval,
    load_tool_output_samples,
    load_dataset_by_name,
    list_available_datasets,
)

# Load specific dataset
suite = load_hotpotqa(n=50)

# Load by name (useful for dynamic loading)
suite = load_dataset_by_name("msmarco", n=100)

# List all datasets by category
datasets = list_available_datasets()
# {'rag': ['hotpotqa', 'natural_questions', ...], 'tool_use': [...], ...}
```

### Custom Datasets

```python
from headroom.evals import EvalCase, EvalSuite, load_custom_dataset

# Create custom cases
cases = [
    EvalCase(
        id="my_case_001",
        context="Long document content here...",
        query="What is the main point?",
        ground_truth="The main point is X",
        metadata={"source": "my_data"},
    ),
]
suite = EvalSuite(name="MyDataset", cases=cases)

# Or load from JSONL file
suite = load_custom_dataset("my_data.jsonl")
```

JSONL format:
```json
{"id": "case_001", "context": "...", "query": "...", "ground_truth": "..."}
{"id": "case_002", "context": "...", "query": "...", "ground_truth": "..."}
```

### Configuring the Runner

```python
from headroom.evals import BeforeAfterRunner, LLMConfig
from headroom.transforms import SmartCrusherConfig

runner = BeforeAfterRunner(
    llm_config=LLMConfig(
        provider="anthropic",  # or "openai", "ollama"
        model="claude-sonnet-4-20250514",
        temperature=0.0,  # Deterministic for reproducibility
        max_tokens=1024,
    ),
    crusher_config=SmartCrusherConfig(
        target_ratio=0.5,
        preserve_structure=True,
    ),
    use_semantic_similarity=True,  # Requires sentence-transformers
)
```

### Computing Metrics Directly

```python
from headroom.evals import (
    compute_f1,
    compute_exact_match,
    compute_semantic_similarity,
    compute_answer_equivalence,
    compute_rouge_l,
    compute_information_recall,
)

# Compare two responses
f1 = compute_f1("The answer is 42", "The answer is forty-two")
exact = compute_exact_match("yes", "YES")  # True (normalized)
semantic = compute_semantic_similarity("The cat sat", "A feline was sitting")

# Comprehensive equivalence check
result = compute_answer_equivalence(
    response_a="The capital is Paris",
    response_b="Paris is the capital",
    ground_truth="Paris",
)
# {'exact_match': False, 'f1_score': 0.8, 'semantic_similarity': 0.95, 'equivalent': True}

# Information recall (check if facts survived compression)
recall = compute_information_recall(
    original_context="John Smith was born in 1990 in New York",
    compressed_context="J. Smith born 1990 NY",
    probe_facts=["John Smith", "1990", "New York"],
)
# {'recall': 0.67, 'facts_lost': ['John Smith']}
```

## CI/CD Integration

```bash
# Run quick eval and fail if accuracy < 90%
python -m headroom.evals quick -n 10

# Exit code: 0 if accuracy >= 90%, 1 otherwise
```

```yaml
# GitHub Actions example
- name: Run Compression Evals
  run: |
    pip install headroom-ai[evals]
    python -m headroom.evals quick -n 20
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Required for Anthropic Claude models
- `OPENAI_API_KEY` - Required for OpenAI models
- No key needed for Ollama (local models)

## Interpretation Guide

### Ideal Results
```
Accuracy Preservation: 95%+
Average F1: 0.85+
Average Semantic Similarity: 0.90+
Compression Ratio: 40-70%
```

### Warning Signs
- Accuracy < 90% → Compression may be too aggressive
- F1 < 0.7 → Significant token loss
- Semantic Similarity < 0.85 → Meaning drift

### Troubleshooting
1. **Low accuracy on specific dataset**: Try adjusting `target_ratio` in SmartCrusherConfig
2. **Semantic similarity unavailable**: Install `sentence-transformers`
3. **Dataset loading fails**: Ensure `datasets` package is installed

## Architecture

```
headroom/evals/
├── __init__.py          # Public API exports
├── __main__.py          # CLI entry point
├── core.py              # EvalCase, EvalResult, EvalSuite
├── datasets.py          # Dataset loaders (HF + built-in)
├── metrics.py           # F1, semantic similarity, etc.
├── runners/
│   ├── __init__.py
│   └── before_after.py  # BeforeAfterRunner
└── reports/
    └── __init__.py
```

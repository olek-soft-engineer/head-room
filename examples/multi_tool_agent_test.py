#!/usr/bin/env python3
"""
Multi-Tool Agent Demo: Headroom in Action

Shows an AI agent investigating a memory leak using 4 tools.
Headroom compresses the tool outputs while preserving critical information.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/multi_tool_agent_test.py
"""

import json
import os
import time

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY environment variable required")


# =============================================================================
# MOCK TOOL DATA
# Each tool returns many items, but only ONE is critical (the "needle")
# =============================================================================


def generate_github_issues(count: int = 50, needle_at: int = 42) -> list[dict]:
    """Generate GitHub issues with a memory leak bug at needle_at."""
    issues = []
    for i in range(count):
        if i == needle_at:
            issues.append(
                {
                    "number": i,
                    "title": "Memory leak in worker pool",
                    "state": "open",
                    "labels": ["bug", "priority:high"],
                    "body": "Worker threads not released after task completion. Memory grows unboundedly.",
                    "reactions": {"thumbs_up": 47},
                }
            )
        else:
            issues.append(
                {
                    "number": i,
                    "title": f"Feature: {['dark mode', 'pagination', 'webhooks', 'rate limiting'][i % 4]}",
                    "state": "closed" if i % 3 == 0 else "open",
                    "labels": ["enhancement"],
                    "body": "Please add this feature.",
                    "reactions": {"thumbs_up": i % 5},
                }
            )
    return issues


def generate_code_results(count: int = 40, needle_at: int = 23) -> list[dict]:
    """Generate code search results with memory fix at needle_at."""
    results = []
    for i in range(count):
        if i == needle_at:
            results.append(
                {
                    "file": "src/worker.py",
                    "line": 330,
                    "content": """def cleanup_worker(self):
    \"\"\"Release worker resources - MEMORY LEAK FIX\"\"\"
    self.thread_pool.shutdown(wait=True)
    self.connections.clear()
    gc.collect()""",
                    "relevance_score": 0.98,
                }
            )
        else:
            results.append(
                {
                    "file": f"src/{['utils', 'api', 'models', 'handlers'][i % 4]}.py",
                    "line": 100 + i * 10,
                    "content": f"def process_{['data', 'request', 'model', 'event'][i % 4]}(self): pass",
                    "relevance_score": 0.5 - (i * 0.01),
                }
            )
    return results


def generate_db_logs(count: int = 60, needle_at: int = 17) -> list[dict]:
    """Generate database logs with OutOfMemoryError at needle_at."""
    logs = []
    for i in range(count):
        if i == needle_at:
            logs.append(
                {
                    "type": "error",
                    "service": "worker-pool",
                    "message": "OutOfMemoryError: heap space exhausted",
                    "metadata": {"heap_used": "7.8GB", "heap_max": "8GB", "thread_count": 847},
                    "stack_trace": "java.lang.OutOfMemoryError at WorkerPool.execute()",
                }
            )
        else:
            logs.append(
                {
                    "type": "info",
                    "service": ["api", "auth", "db", "cache"][i % 4],
                    "message": "Operation completed successfully",
                    "metadata": {"heap_used": f"{1 + i % 3}GB", "thread_count": 50 + i % 50},
                }
            )
    return logs


def generate_arxiv_papers(count: int = 30, needle_at: int = 15) -> list[dict]:
    """Generate ArXiv papers with relevant research at needle_at."""
    papers = []
    for i in range(count):
        if i == needle_at:
            papers.append(
                {
                    "id": f"2401.{i:05d}",
                    "title": "Memory Management in Worker Pool Architectures",
                    "abstract": "We analyze memory leak patterns in thread pools and propose cleanup strategies.",
                    "citations": 89,
                }
            )
        else:
            papers.append(
                {
                    "id": f"2401.{i:05d}",
                    "title": f"{['Deep Learning', 'Transformers', 'Neural Arch', 'Scaling'][i % 4]} Study",
                    "abstract": "Novel approach with state-of-the-art results.",
                    "citations": i * 3,
                }
            )
    return papers


# Generate the mock data
GITHUB_ISSUES = generate_github_issues()
CODE_RESULTS = generate_code_results()
DB_LOGS = generate_db_logs()
ARXIV_PAPERS = generate_arxiv_papers()


# =============================================================================
# TOOLS
# =============================================================================


@tool(name="search_github_issues")
def search_github_issues(query: str) -> str:
    """Search GitHub issues."""
    return json.dumps(GITHUB_ISSUES, indent=2)


@tool(name="search_code")
def search_code(query: str) -> str:
    """Search codebase."""
    return json.dumps(CODE_RESULTS, indent=2)


@tool(name="query_database")
def query_database(query: str) -> str:
    """Query database logs."""
    return json.dumps(DB_LOGS, indent=2)


@tool(name="search_arxiv")
def search_arxiv(query: str) -> str:
    """Search ArXiv papers."""
    return json.dumps(ARXIV_PAPERS, indent=2)


# =============================================================================
# VERIFICATION
# =============================================================================

NEEDLES = {
    "GitHub Issue #42": lambda r: "memory leak" in r.lower()
    and ("42" in r or "worker" in r.lower()),
    "cleanup_worker() fix": lambda r: "cleanup" in r.lower() or "worker.py" in r.lower(),
    "OutOfMemoryError": lambda r: "outofmemory" in r.lower() or "847" in r or "7.8" in r.lower(),
    "ArXiv paper": lambda r: "paper" in r.lower()
    or "research" in r.lower()
    or "arxiv" in r.lower(),
}


def verify_response(response: str) -> dict[str, bool]:
    """Check if response found all needles."""
    return {name: check(response) for name, check in NEEDLES.items()}


# =============================================================================
# MAIN
# =============================================================================


def main():
    from headroom.integrations.agno import HeadroomAgnoModel
    from headroom.pricing import estimate_cost

    model_id = "claude-sonnet-4-20250514"

    print("\n" + "=" * 70)
    print("  MULTI-TOOL AGENT DEMO")
    print("=" * 70)

    # Show data sizes
    total_chars = sum(
        len(json.dumps(d)) for d in [GITHUB_ISSUES, CODE_RESULTS, DB_LOGS, ARXIV_PAPERS]
    )
    print(f"\n  Tool outputs: {total_chars:,} chars across 4 tools")
    print("  Needles hidden at positions: #42, #23, #17, #15")

    # Create agent with Headroom
    base_model = Claude(id=model_id)
    model = HeadroomAgnoModel(wrapped_model=base_model)
    agent = Agent(
        model=model,
        tools=[search_github_issues, search_code, query_database, search_arxiv],
        markdown=True,
    )

    question = """Investigate a memory leak in our application:
1. Search GitHub for memory-related issues
2. Search code for memory leak fixes
3. Check database logs for OutOfMemory errors
4. Find relevant research papers

Summarize findings and recommend a fix."""

    print("\n  Running agent...")
    start = time.time()

    response = agent.run(question)
    response_text = response.content if hasattr(response, "content") else str(response)

    duration = time.time() - start

    # Get stats from Headroom
    stats = model.get_savings_summary()
    tokens_before = stats["total_tokens_before"]
    tokens_after = stats["total_tokens_after"]
    tokens_saved = stats["total_tokens_saved"]
    pct_saved = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

    # Calculate costs
    cost_before = estimate_cost(model_id, input_tokens=tokens_before)
    cost_after = estimate_cost(model_id, input_tokens=tokens_after)

    # Verify findings
    verification = verify_response(response_text)
    found = sum(verification.values())

    # Results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"""
                              Without         With
                              Headroom        Headroom        Savings
    ─────────────────────────────────────────────────────────────────
    Input Tokens              {tokens_before:>8,}        {tokens_after:>8,}        {tokens_saved:,} ({pct_saved:.0f}%)""")

    if cost_before and cost_after:
        cost_saved = cost_before - cost_after
        print(
            f"    Input Cost                ${cost_before:.4f}         ${cost_after:.4f}         ${cost_saved:.4f}"
        )

    print(f"""
    Duration                                    {duration:.1f}s
    API Requests                                {stats["total_requests"]}
    Needles Found                               {found}/4
    """)

    for name, found_it in verification.items():
        print(f"      {'✓' if found_it else '✗'} {name}")

    print("\n" + "=" * 70)
    print("  RESPONSE (excerpt)")
    print("=" * 70)
    print(response_text[:1500] + "..." if len(response_text) > 1500 else response_text)

    print("\n" + "=" * 70)
    print(f"  {pct_saved:.0f}% token reduction, {found}/4 needles found")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

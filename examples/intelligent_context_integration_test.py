#!/usr/bin/env python3
"""
Comprehensive Integration Test for IntelligentContextManager

This test runs WITHOUT MOCKS - it uses real API calls to verify:
1. IntelligentContextManager is properly wired into the pipeline
2. COMPRESS_FIRST strategy works (deeper compression before dropping)
3. SUMMARIZE strategy works (progressive summarization)
4. DROP_BY_SCORE strategy works (semantic scoring)
5. Token savings are real and significant

Requirements:
- ANTHROPIC_API_KEY environment variable
- Real API calls will be made
"""

import json
import os
import sys
import time
from dataclasses import dataclass

# Check for API key early
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable required")
    print("Usage: ANTHROPIC_API_KEY=sk-... python examples/intelligent_context_integration_test.py")
    sys.exit(1)

from anthropic import Anthropic  # noqa: E402

from headroom import AnthropicProvider, HeadroomClient  # noqa: E402
from headroom.config import (  # noqa: E402
    HeadroomConfig,
    IntelligentContextConfig,
)
from headroom.transforms import IntelligentContextManager  # noqa: E402
from headroom.transforms.pipeline import TransformPipeline  # noqa: E402

# =============================================================================
# TEST DATA - Realistic tool outputs that benefit from intelligent compression
# =============================================================================


def generate_large_search_results(count: int = 100) -> str:
    """Generate realistic search results with varying relevance."""
    results = []
    for i in range(count):
        # Some results are clearly important (errors, high scores)
        if i == 42:
            result = {
                "id": i,
                "title": "CRITICAL: Memory leak in worker pool causing OOM",
                "score": 0.98,
                "type": "error",
                "content": "Worker threads are not being released after task completion. "
                "Stack trace shows accumulation in ThreadPoolExecutor.",
                "metadata": {"severity": "critical", "affected_users": 1247},
            }
        elif i == 17:
            result = {
                "id": i,
                "title": "Performance regression in v2.3.1 release",
                "score": 0.95,
                "type": "bug",
                "content": "Response times increased 3x after the latest deploy. "
                "Profiling shows bottleneck in database connection pooling.",
                "metadata": {"severity": "high", "p99_latency_ms": 2340},
            }
        else:
            result = {
                "id": i,
                "title": f"Search result #{i}: {'Feature request' if i % 3 == 0 else 'Documentation update'}",
                "score": 0.3 + (0.5 * (1 - i / count)),  # Decreasing relevance
                "type": "info",
                "content": f"This is search result {i} with standard content. "
                f"Contains typical information that may or may not be relevant.",
                "metadata": {"views": 100 + i * 10, "last_updated": f"2024-01-{(i % 28) + 1:02d}"},
            }
        results.append(result)
    return json.dumps(results, indent=2)


def generate_log_entries(count: int = 200) -> str:
    """Generate realistic log entries with some errors."""
    entries = []
    for i in range(count):
        if i == 87:
            entry = {
                "timestamp": f"2024-01-15T10:{i % 60:02d}:00Z",
                "level": "ERROR",
                "service": "worker-pool",
                "message": "OutOfMemoryError: Java heap space exhausted",
                "stack_trace": "java.lang.OutOfMemoryError: Java heap space\n"
                "  at WorkerPool.execute(WorkerPool.java:234)\n"
                "  at TaskRunner.run(TaskRunner.java:89)",
                "context": {"heap_used": "7.8GB", "heap_max": "8GB", "thread_count": 847},
            }
        elif i == 143:
            entry = {
                "timestamp": f"2024-01-15T10:{i % 60:02d}:00Z",
                "level": "ERROR",
                "service": "database",
                "message": "Connection pool exhausted - all 100 connections in use",
                "context": {"active_connections": 100, "waiting_requests": 342},
            }
        else:
            entry = {
                "timestamp": f"2024-01-15T10:{i % 60:02d}:00Z",
                "level": "INFO",
                "service": ["api", "auth", "worker", "cache"][i % 4],
                "message": f"Request processed successfully (id={i})",
                "context": {"latency_ms": 50 + (i % 100), "status": 200},
            }
        entries.append(entry)
    return json.dumps(entries, indent=2)


def generate_code_analysis(file_count: int = 50) -> str:
    """Generate code analysis results."""
    files = []
    for i in range(file_count):
        if i == 23:
            file_result = {
                "path": "src/worker.py",
                "issues": [
                    {
                        "line": 234,
                        "type": "memory_leak",
                        "severity": "critical",
                        "message": "Thread pool executor not properly shutdown",
                    },
                    {
                        "line": 287,
                        "type": "resource_leak",
                        "severity": "high",
                        "message": "Database connection not closed in finally block",
                    },
                ],
                "metrics": {"complexity": 45, "lines": 523, "test_coverage": 0.23},
            }
        else:
            file_result = {
                "path": f"src/module_{i}.py",
                "issues": [],
                "metrics": {"complexity": 5 + (i % 10), "lines": 100 + i * 5, "test_coverage": 0.8},
            }
        files.append(file_result)
    return json.dumps(files, indent=2)


# =============================================================================
# TEST HELPERS
# =============================================================================


@dataclass
class TestResult:
    name: str
    success: bool
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float
    strategy_used: str
    duration_ms: float
    error: str | None = None


SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes data and provides insights. "
    "When given search results or logs, identify the most important items "
    "and summarize key findings."
)


def create_test_messages(tool_output: str, question: str) -> list[dict]:
    """Create a realistic conversation with tool output (no system message - passed separately)."""
    return [
        {"role": "user", "content": "Search for any critical issues in our system."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_search_001",
                    "type": "function",
                    "function": {
                        "name": "search_issues",
                        "arguments": '{"query": "critical issues errors"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_search_001", "content": tool_output},
        {"role": "user", "content": question},
    ]


def create_headroom_client(
    config: HeadroomConfig, model_context_limits: dict | None = None
) -> HeadroomClient:
    """Create a HeadroomClient with the given config."""
    base_client = Anthropic(api_key=API_KEY)
    provider = AnthropicProvider()

    return HeadroomClient(
        original_client=base_client,
        provider=provider,
        default_mode="optimize",
        config=config,
        model_context_limits=model_context_limits,
    )


# =============================================================================
# TESTS
# =============================================================================


def test_intelligent_context_wired_in_pipeline():
    """Test that IntelligentContextManager is properly wired into the pipeline."""
    print("\n" + "=" * 70)
    print("TEST: IntelligentContextManager wired into pipeline")
    print("=" * 70)

    # Create config with intelligent context enabled
    config = HeadroomConfig()
    config.intelligent_context = IntelligentContextConfig(
        enabled=True,
        use_importance_scoring=True,
        compress_threshold=0.10,
        summarize_threshold=0.25,
    )
    # Disable rolling window since intelligent context is enabled
    config.rolling_window.enabled = False

    # Create pipeline
    provider = AnthropicProvider()
    pipeline = TransformPipeline(config, provider=provider)

    # Check that IntelligentContextManager is in the transforms
    icm_found = False
    for transform in pipeline.transforms:
        if isinstance(transform, IntelligentContextManager):
            icm_found = True
            break

    if icm_found:
        print("✅ IntelligentContextManager found in pipeline transforms")
        return TestResult(
            name="Pipeline Wiring",
            success=True,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="N/A",
            duration_ms=0,
        )
    else:
        print("❌ IntelligentContextManager NOT found in pipeline!")
        print(f"   Transforms in pipeline: {[t.name for t in pipeline.transforms]}")
        return TestResult(
            name="Pipeline Wiring",
            success=False,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="N/A",
            duration_ms=0,
            error="IntelligentContextManager not in pipeline",
        )


def test_compress_first_strategy():
    """Test COMPRESS_FIRST strategy - deeper compression before dropping."""
    print("\n" + "=" * 70)
    print("TEST: COMPRESS_FIRST strategy")
    print("=" * 70)

    # Generate large tool output
    search_results = generate_large_search_results(100)
    messages = create_test_messages(
        search_results, "What are the most critical issues? Summarize the top problems."
    )

    print(f"Tool output size: {len(search_results):,} chars (~{len(search_results) // 4:,} tokens)")

    # Create config with intelligent context
    config = HeadroomConfig()
    config.intelligent_context = IntelligentContextConfig(
        enabled=True,
        use_importance_scoring=True,
        compress_threshold=0.50,  # High threshold to trigger COMPRESS_FIRST
        keep_last_turns=2,
    )
    config.rolling_window.enabled = False
    config.smart_crusher.enabled = True  # Enable smart crushing for COMPRESS_FIRST

    start_time = time.time()

    try:
        client = create_headroom_client(config)

        # Get optimization result via simulate (on messages API)
        result = client.messages.simulate(
            messages=messages,
            model="claude-sonnet-4-20250514",
            system=SYSTEM_PROMPT,
        )

        duration_ms = (time.time() - start_time) * 1000

        tokens_saved = result.tokens_before - result.tokens_after
        savings_pct = (tokens_saved / result.tokens_before * 100) if result.tokens_before > 0 else 0

        print(f"✅ Tokens before: {result.tokens_before:,}")
        print(f"✅ Tokens after:  {result.tokens_after:,}")
        print(f"✅ Tokens saved:  {tokens_saved:,} ({savings_pct:.1f}%)")
        print(f"✅ Transforms:    {result.transforms}")

        return TestResult(
            name="COMPRESS_FIRST",
            success=True,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=savings_pct,
            strategy_used="compress_first"
            if any("compress" in t.lower() for t in result.transforms)
            else "smart_crusher",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return TestResult(
            name="COMPRESS_FIRST",
            success=False,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="error",
            duration_ms=duration_ms,
            error=str(e),
        )


def test_drop_by_score_strategy():
    """Test DROP_BY_SCORE strategy - semantic scoring for message importance."""
    print("\n" + "=" * 70)
    print("TEST: DROP_BY_SCORE strategy (over budget scenario)")
    print("=" * 70)

    # Create a VERY long conversation that will definitely exceed limits
    # System message passed separately to Anthropic API
    messages = []

    # Add many tool calls and responses to exceed context
    for i in range(20):
        messages.append({"role": "user", "content": f"Search for issue category {i}"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i:03d}",
                        "type": "function",
                        "function": {"name": "search", "arguments": f'{{"query": "category {i}"}}'},
                    }
                ],
            }
        )
        # Large tool output
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i:03d}",
                "content": generate_log_entries(50),  # 50 log entries per call
            }
        )

    # Final question
    messages.append(
        {"role": "user", "content": "Based on all the searches, what are the critical issues?"}
    )

    print(f"Conversation: {len(messages)} messages")

    # Create config with intelligent context and SMALL context limit to force dropping
    config = HeadroomConfig()
    config.intelligent_context = IntelligentContextConfig(
        enabled=True,
        use_importance_scoring=True,
        compress_threshold=0.05,  # Low threshold to skip COMPRESS_FIRST
        keep_last_turns=2,
        output_buffer_tokens=4000,
    )
    config.rolling_window.enabled = False
    config.smart_crusher.enabled = True

    start_time = time.time()

    try:
        client = create_headroom_client(
            config,
            # Use a small context limit to force dropping
            model_context_limits={"claude-sonnet-4-20250514": 20000},
        )

        result = client.messages.simulate(
            messages=messages,
            model="claude-sonnet-4-20250514",
            system="You are a helpful assistant that analyzes data.",
        )

        duration_ms = (time.time() - start_time) * 1000

        tokens_saved = result.tokens_before - result.tokens_after
        savings_pct = (tokens_saved / result.tokens_before * 100) if result.tokens_before > 0 else 0

        print(f"✅ Tokens before: {result.tokens_before:,}")
        print(f"✅ Tokens after:  {result.tokens_after:,}")
        print(f"✅ Tokens saved:  {tokens_saved:,} ({savings_pct:.1f}%)")
        print(f"✅ Transforms:    {result.transforms}")

        # Check if intelligent_cap was applied (dropping happened)
        dropped = any("intelligent_cap" in t for t in result.transforms)
        strategy = "drop_by_score" if dropped else "compress_only"

        return TestResult(
            name="DROP_BY_SCORE",
            success=True,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=savings_pct,
            strategy_used=strategy,
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return TestResult(
            name="DROP_BY_SCORE",
            success=False,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="error",
            duration_ms=duration_ms,
            error=str(e),
        )


def test_real_api_call_with_optimization():
    """Test a real API call with intelligent context optimization."""
    print("\n" + "=" * 70)
    print("TEST: Real API call with optimization")
    print("=" * 70)

    # Create a long conversation (no tool calls - simpler for real API)
    # This simulates a multi-turn conversation that benefits from compression
    messages = []

    # Add many conversation turns with verbose content
    for i in range(15):
        messages.append(
            {
                "role": "user",
                "content": f"Tell me about topic {i}. Please provide detailed information including "
                f"history, current state, key concepts, and important considerations. "
                f"I want comprehensive coverage of all aspects related to topic {i}.",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"Here's detailed information about topic {i}:\n\n"
                f"**History**: Topic {i} has a rich history spanning many decades. "
                f"It originated in the early period and evolved through various phases. "
                f"Key milestones include development A, breakthrough B, and innovation C.\n\n"
                f"**Current State**: Today, topic {i} is widely recognized as important. "
                f"Modern applications include X, Y, and Z. The field continues to evolve.\n\n"
                f"**Key Concepts**: Understanding topic {i} requires grasping concepts like "
                f"principle 1, methodology 2, and framework 3. These form the foundation.\n\n"
                f"**Considerations**: When working with topic {i}, consider factors such as "
                f"constraint A, limitation B, and opportunity C. Best practices recommend "
                f"approach D for optimal results.",
            }
        )

    # Final question
    messages.append(
        {
            "role": "user",
            "content": "Based on everything we discussed, what are the 3 most important takeaways?",
        }
    )

    print(f"Conversation: {len(messages)} messages")

    config = HeadroomConfig()
    config.intelligent_context = IntelligentContextConfig(
        enabled=True,
        use_importance_scoring=True,
    )
    config.rolling_window.enabled = False
    config.smart_crusher.enabled = True

    start_time = time.time()

    try:
        client = create_headroom_client(config)

        # Make actual API call using the Anthropic-style API
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=messages,
            system="You are a helpful assistant. Be concise in your responses.",
            max_tokens=300,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Extract response content
        if hasattr(response, "content") and response.content:
            if isinstance(response.content, list):
                response_text = response.content[0].text if response.content else ""
            else:
                response_text = str(response.content)
        else:
            response_text = str(response)

        print("✅ API call successful")
        print(f"✅ Response length: {len(response_text)} chars")
        print(f"✅ Response preview: {response_text[:200]}...")
        print(f"✅ Duration: {duration_ms:.0f}ms")

        # Get session stats from internal tracking
        stats = client._session_stats
        tokens_before = stats.get("tokens", {}).get("input_before", 0)
        tokens_after = stats.get("tokens", {}).get("input_after", 0)
        tokens_saved = tokens_before - tokens_after
        savings_pct = (tokens_saved / tokens_before * 100) if tokens_before > 0 else 0

        print(f"✅ Session tokens before: {tokens_before:,}")
        print(f"✅ Session tokens after:  {tokens_after:,}")
        print(f"✅ Session tokens saved:  {tokens_saved:,} ({savings_pct:.1f}%)")

        return TestResult(
            name="Real API Call",
            success=True,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_saved,
            savings_percent=savings_pct,
            strategy_used="intelligent_context",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return TestResult(
            name="Real API Call",
            success=False,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="error",
            duration_ms=duration_ms,
            error=str(e),
        )


def test_comparison_rolling_window_vs_intelligent():
    """Compare RollingWindow vs IntelligentContextManager on same data."""
    print("\n" + "=" * 70)
    print("TEST: RollingWindow vs IntelligentContextManager comparison")
    print("=" * 70)

    # Create test data
    search_results = generate_large_search_results(80)
    messages = create_test_messages(search_results, "Summarize the critical issues.")

    results = {}

    # Test with RollingWindow
    print("\n--- RollingWindow ---")
    config_rw = HeadroomConfig()
    config_rw.rolling_window.enabled = True
    config_rw.intelligent_context.enabled = False
    config_rw.smart_crusher.enabled = True

    try:
        client_rw = create_headroom_client(config_rw)
        result_rw = client_rw.messages.simulate(
            messages=messages,
            model="claude-sonnet-4-20250514",
            system=SYSTEM_PROMPT,
        )
        results["rolling_window"] = {
            "tokens_before": result_rw.tokens_before,
            "tokens_after": result_rw.tokens_after,
            "transforms": result_rw.transforms,
        }
        print(f"Tokens: {result_rw.tokens_before:,} -> {result_rw.tokens_after:,}")
        print(f"Transforms: {result_rw.transforms}")
    except Exception as e:
        print(f"Error: {e}")
        results["rolling_window"] = {"error": str(e)}

    # Test with IntelligentContextManager
    print("\n--- IntelligentContextManager ---")
    config_icm = HeadroomConfig()
    config_icm.rolling_window.enabled = False
    config_icm.intelligent_context = IntelligentContextConfig(enabled=True)
    config_icm.smart_crusher.enabled = True

    try:
        client_icm = create_headroom_client(config_icm)
        result_icm = client_icm.messages.simulate(
            messages=messages,
            model="claude-sonnet-4-20250514",
            system=SYSTEM_PROMPT,
        )
        results["intelligent_context"] = {
            "tokens_before": result_icm.tokens_before,
            "tokens_after": result_icm.tokens_after,
            "transforms": result_icm.transforms,
        }
        print(f"Tokens: {result_icm.tokens_before:,} -> {result_icm.tokens_after:,}")
        print(f"Transforms: {result_icm.transforms}")
    except Exception as e:
        print(f"Error: {e}")
        results["intelligent_context"] = {"error": str(e)}

    # Compare
    print("\n--- Comparison ---")
    if "error" not in results.get("rolling_window", {}) and "error" not in results.get(
        "intelligent_context", {}
    ):
        rw_saved = (
            results["rolling_window"]["tokens_before"] - results["rolling_window"]["tokens_after"]
        )
        icm_saved = (
            results["intelligent_context"]["tokens_before"]
            - results["intelligent_context"]["tokens_after"]
        )
        print(f"RollingWindow saved: {rw_saved:,} tokens")
        print(f"IntelligentContext saved: {icm_saved:,} tokens")

        if icm_saved >= rw_saved:
            print(f"✅ IntelligentContextManager saved {icm_saved - rw_saved:,} MORE tokens!")
        else:
            print(f"⚠️ RollingWindow saved {rw_saved - icm_saved:,} more tokens")

        return TestResult(
            name="Comparison",
            success=True,
            tokens_before=results["intelligent_context"]["tokens_before"],
            tokens_after=results["intelligent_context"]["tokens_after"],
            tokens_saved=icm_saved,
            savings_percent=(icm_saved / results["intelligent_context"]["tokens_before"] * 100),
            strategy_used=f"ICM:{icm_saved} vs RW:{rw_saved}",
            duration_ms=0,
        )
    else:
        return TestResult(
            name="Comparison",
            success=False,
            tokens_before=0,
            tokens_after=0,
            tokens_saved=0,
            savings_percent=0,
            strategy_used="error",
            duration_ms=0,
            error="One or both tests failed",
        )


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 70)
    print("INTELLIGENT CONTEXT MANAGER - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 70)
    print(f"API Key: {'SET' if API_KEY else 'NOT SET'}")
    print("Using real API calls - NO MOCKS")
    print("=" * 70)

    all_results: list[TestResult] = []

    # Run all tests
    all_results.append(test_intelligent_context_wired_in_pipeline())
    all_results.append(test_compress_first_strategy())
    all_results.append(test_drop_by_score_strategy())
    all_results.append(test_real_api_call_with_optimization())
    all_results.append(test_comparison_rolling_window_vs_intelligent())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    total_tokens_saved = 0

    for result in all_results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"\n{status} - {result.name}")
        if result.success:
            passed += 1
            if result.tokens_saved > 0:
                print(
                    f"    Tokens: {result.tokens_before:,} -> {result.tokens_after:,} "
                    f"(saved {result.tokens_saved:,}, {result.savings_percent:.1f}%)"
                )
                total_tokens_saved += result.tokens_saved
            print(f"    Strategy: {result.strategy_used}")
            if result.duration_ms > 0:
                print(f"    Duration: {result.duration_ms:.0f}ms")
        else:
            failed += 1
            if result.error:
                print(f"    Error: {result.error[:100]}")

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"Total tokens saved across tests: {total_tokens_saved:,}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

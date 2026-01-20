#!/usr/bin/env python3
"""
Comprehensive Multi-Agent Reasoning Test with Debugging

This test creates:
1. A reasoning agent (reasoning=True)
2. Multiple tool-using agents
3. Tests message flow inter and intra agent

We run WITHOUT Headroom first, then WITH Headroom to find where the issue occurs.
"""

import json
import os
import sys
import traceback
from typing import Any

# Enable maximum Agno debugging
os.environ["AGNO_DEBUG"] = "true"

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools import tool

# Check for API key
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable required")
    sys.exit(1)

# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================

DEBUG_LOG = []


def debug_log(category: str, message: str, data: Any = None):
    """Log debug information."""
    entry = f"[{category}] {message}"
    if data is not None:
        if isinstance(data, list):
            entry += f"\n  Items: {len(data)}"
            for i, item in enumerate(data[:5]):  # First 5 items
                item_type = type(item).__name__
                has_log = hasattr(item, "log")
                has_content = hasattr(item, "content")
                if isinstance(item, dict):
                    keys = list(item.keys())
                    entry += f"\n    [{i}] dict with keys: {keys}"
                else:
                    entry += (
                        f"\n    [{i}] {item_type} (has .log={has_log}, has .content={has_content})"
                    )
        else:
            entry += f"\n  Data: {type(data).__name__}"

    DEBUG_LOG.append(entry)
    print(entry)


def dump_message_details(messages: list, label: str):
    """Dump detailed message information."""
    print(f"\n{'=' * 60}")
    print(f"MESSAGE DUMP: {label}")
    print(f"{'=' * 60}")
    print(f"Total messages: {len(messages)}")

    for i, msg in enumerate(messages):
        print(f"\n--- Message {i} ---")
        print(f"  Type: {type(msg).__name__}")
        print(f"  Is dict: {isinstance(msg, dict)}")
        print(f"  Has .log(): {hasattr(msg, 'log')}")
        print(f"  Has .content: {hasattr(msg, 'content')}")

        if isinstance(msg, dict):
            print(f"  Keys: {list(msg.keys())}")
            print(f"  Role: {msg.get('role', 'N/A')}")
            content = msg.get("content", "")
            print(f"  Content preview: {str(content)[:100]}...")
        elif hasattr(msg, "role"):
            print(f"  Role: {msg.role}")
            content = getattr(msg, "content", "")
            print(f"  Content preview: {str(content)[:100]}...")

        # Try calling .log() to see if it works
        if hasattr(msg, "log"):
            try:
                msg.log(metrics=False)
                print("  .log() call: SUCCESS")
            except Exception as e:
                print(f"  .log() call: FAILED - {e}")

    print(f"{'=' * 60}\n")


# =============================================================================
# MOCK TOOLS
# =============================================================================


@tool(name="search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information.

    Args:
        query: Search query

    Returns:
        Search results as JSON
    """
    debug_log("TOOL", f"search_knowledge_base called with: {query}")
    results = [
        {
            "id": 1,
            "title": "Memory Management Best Practices",
            "content": "Always release resources...",
        },
        {"id": 2, "title": "Worker Pool Optimization", "content": "Use thread pool executors..."},
        {"id": 3, "title": "Garbage Collection Tuning", "content": "Set appropriate heap sizes..."},
    ]
    return json.dumps(results, indent=2)


@tool(name="analyze_code")
def analyze_code(file_path: str) -> str:
    """Analyze code for issues.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Analysis results as JSON
    """
    debug_log("TOOL", f"analyze_code called with: {file_path}")
    return json.dumps(
        {
            "file": file_path,
            "issues": [
                {"line": 42, "type": "memory_leak", "description": "Resource not released"},
                {"line": 87, "type": "performance", "description": "Inefficient loop"},
            ],
            "suggestions": ["Add cleanup in finally block", "Use list comprehension"],
        },
        indent=2,
    )


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_simple_agent(use_headroom: bool, use_reasoning: bool) -> dict:
    """Test a simple agent configuration."""

    label = f"{'WITH' if use_headroom else 'WITHOUT'} Headroom, reasoning={use_reasoning}"
    print(f"\n{'#' * 70}")
    print(f"# TEST: {label}")
    print(f"{'#' * 70}")

    DEBUG_LOG.clear()

    try:
        # Create the model
        if use_headroom:
            from headroom.integrations.agno import HeadroomAgnoModel

            base_model = Claude(id="claude-sonnet-4-20250514")
            model = HeadroomAgnoModel(wrapped_model=base_model)
            debug_log("SETUP", "Created HeadroomAgnoModel wrapping Claude")
        else:
            model = Claude(id="claude-sonnet-4-20250514")
            debug_log("SETUP", "Created Claude model directly")

        # Create the agent
        agent = Agent(
            model=model,
            tools=[search_knowledge_base, analyze_code],
            reasoning=use_reasoning,
            markdown=True,
            debug_mode=True,
        )
        debug_log("SETUP", f"Created Agent with reasoning={use_reasoning}")

        # Simple question that uses tools
        question = "Search the knowledge base for memory management and analyze worker.py for issues. Summarize what you find."
        debug_log("INPUT", f"Question: {question}")

        # Run the agent
        debug_log("RUN", "Starting agent.run()...")
        response = agent.run(question)

        # Extract response
        if hasattr(response, "content") and response.content is not None:
            response_text = response.content
        elif response is not None:
            response_text = str(response)
        else:
            response_text = "(No response content)"

        debug_log("OUTPUT", f"Response length: {len(response_text)} chars")
        debug_log("OUTPUT", f"Response preview: {response_text[:200]}...")

        # Get Headroom stats if available
        headroom_stats = None
        if use_headroom and hasattr(model, "get_savings_summary"):
            headroom_stats = model.get_savings_summary()
            debug_log("HEADROOM", f"Stats: {headroom_stats}")

        return {
            "success": True,
            "label": label,
            "response_length": len(response_text),
            "response_preview": response_text[:500],
            "headroom_stats": headroom_stats,
            "debug_log": DEBUG_LOG.copy(),
        }

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        debug_log("ERROR", f"Exception: {error_msg}")
        debug_log("ERROR", f"Traceback:\n{tb}")

        return {
            "success": False,
            "label": label,
            "error": error_msg,
            "traceback": tb,
            "debug_log": DEBUG_LOG.copy(),
        }


def test_with_message_interception(use_headroom: bool, use_reasoning: bool) -> dict:
    """Test with message interception to see what's being passed around."""

    label = (
        f"INTERCEPTED: {'WITH' if use_headroom else 'WITHOUT'} Headroom, reasoning={use_reasoning}"
    )
    print(f"\n{'#' * 70}")
    print(f"# TEST: {label}")
    print(f"{'#' * 70}")

    DEBUG_LOG.clear()

    # Patch Agno's _log_messages to intercept and debug
    original_log_messages = None

    try:
        from agno.models import base as agno_base

        original_log_messages = agno_base._log_messages

        def intercepted_log_messages(messages):
            debug_log("INTERCEPT", "_log_messages called", messages)
            dump_message_details(messages, "_log_messages input")

            # Check each message
            for i, msg in enumerate(messages):
                if isinstance(msg, dict):
                    debug_log("INTERCEPT", f"Message {i} is a DICT - this will fail!")
                elif not hasattr(msg, "log"):
                    debug_log("INTERCEPT", f"Message {i} has no .log() method!")

            # Call original
            return original_log_messages(messages)

        agno_base._log_messages = intercepted_log_messages
        debug_log("SETUP", "Patched _log_messages for interception")

        # Now run the actual test
        result = test_simple_agent(use_headroom, use_reasoning)
        result["label"] = label
        return result

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        debug_log("ERROR", f"Exception: {error_msg}")
        debug_log("ERROR", f"Traceback:\n{tb}")

        return {
            "success": False,
            "label": label,
            "error": error_msg,
            "traceback": tb,
            "debug_log": DEBUG_LOG.copy(),
        }

    finally:
        # Restore original
        if original_log_messages:
            agno_base._log_messages = original_log_messages
            debug_log("CLEANUP", "Restored original _log_messages")


def run_all_tests():
    """Run all test combinations."""

    print("\n" + "=" * 70)
    print("COMPREHENSIVE MULTI-AGENT REASONING TEST")
    print("=" * 70)
    print(f"API Key: {'SET' if API_KEY else 'NOT SET'}")
    print("=" * 70)

    results = []

    # Test matrix
    test_cases = [
        # (use_headroom, use_reasoning, use_interception)
        (False, False, False),  # Baseline: No Headroom, No Reasoning
        (False, True, False),  # No Headroom, With Reasoning
        (True, False, False),  # With Headroom, No Reasoning
        (True, True, False),  # With Headroom, With Reasoning
        (True, True, True),  # With Headroom, With Reasoning, With Interception
    ]

    for use_headroom, use_reasoning, use_interception in test_cases:
        print(f"\n{'=' * 70}")
        print(
            f"Running: Headroom={use_headroom}, Reasoning={use_reasoning}, Intercept={use_interception}"
        )
        print("=" * 70)

        try:
            if use_interception:
                result = test_with_message_interception(use_headroom, use_reasoning)
            else:
                result = test_simple_agent(use_headroom, use_reasoning)
            results.append(result)
        except Exception as e:
            results.append(
                {
                    "success": False,
                    "label": f"Headroom={use_headroom}, Reasoning={use_reasoning}",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

        print(f"\nResult: {'SUCCESS' if results[-1]['success'] else 'FAILED'}")
        if not results[-1]["success"]:
            print(f"Error: {results[-1].get('error', 'Unknown')}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['label']}")
        if not result["success"]:
            print(f"    Error: {result.get('error', 'Unknown')[:100]}")
        elif result.get("headroom_stats"):
            stats = result["headroom_stats"]
            saved = stats.get("total_tokens_saved", 0)
            before = stats.get("total_tokens_before", 0)
            pct = (saved / before * 100) if before > 0 else 0
            print(f"    Tokens saved: {saved:,} ({pct:.1f}%)")

    print("\n" + "=" * 70)

    # Detailed failure analysis
    failures = [r for r in results if not r["success"]]
    if failures:
        print("\nDETAILED FAILURE ANALYSIS")
        print("=" * 70)
        for failure in failures:
            print(f"\n--- {failure['label']} ---")
            print(f"Error: {failure.get('error', 'Unknown')}")
            if "traceback" in failure:
                print(f"Traceback:\n{failure['traceback']}")
            if "debug_log" in failure:
                print("\nDebug Log:")
                for entry in failure["debug_log"][-20:]:  # Last 20 entries
                    print(f"  {entry}")

    return results


if __name__ == "__main__":
    run_all_tests()

#!/usr/bin/env python3
"""
Groq-Specific Reasoning Test

Tests Groq model with reasoning=True, both with and without Headroom.
This isolates whether the issue is Groq-specific or Headroom-specific.
"""

import json
import os
import sys
import traceback

# Enable Agno debugging
os.environ["AGNO_DEBUG"] = "true"

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import tool

# Check for API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY environment variable required")
    sys.exit(1)

# =============================================================================
# SIMPLE TOOLS
# =============================================================================


@tool(name="get_weather")
def get_weather(city: str) -> str:
    """Get weather for a city.

    Args:
        city: City name

    Returns:
        Weather information as JSON
    """
    print(f"[TOOL] get_weather called with: {city}")
    return json.dumps(
        {"city": city, "temperature": "72°F", "conditions": "Sunny", "humidity": "45%"}
    )


@tool(name="get_time")
def get_time(timezone: str = "UTC") -> str:
    """Get current time in a timezone.

    Args:
        timezone: Timezone name

    Returns:
        Current time
    """
    print(f"[TOOL] get_time called with: {timezone}")
    return json.dumps({"timezone": timezone, "time": "14:30:00", "date": "2025-01-19"})


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


def test_groq(use_headroom: bool, use_reasoning: bool, model_id: str = "llama-3.3-70b-versatile"):
    """Test Groq with specific configuration."""

    label = f"Groq {'+ Headroom' if use_headroom else 'Direct'}, reasoning={use_reasoning}, model={model_id}"
    print(f"\n{'#' * 70}")
    print(f"# TEST: {label}")
    print(f"{'#' * 70}")

    try:
        # Create the model
        if use_headroom:
            from headroom.integrations.agno import HeadroomAgnoModel

            base_model = Groq(id=model_id)
            model = HeadroomAgnoModel(wrapped_model=base_model)
            print("[SETUP] Created HeadroomAgnoModel wrapping Groq")
        else:
            model = Groq(id=model_id)
            print("[SETUP] Created Groq model directly")

        # Create the agent
        agent = Agent(
            model=model,
            tools=[get_weather, get_time],
            reasoning=use_reasoning,
            markdown=True,
            debug_mode=True,
        )
        print(f"[SETUP] Created Agent with reasoning={use_reasoning}")

        # Simple question
        question = "What's the weather in San Francisco and what time is it there?"
        print(f"[INPUT] Question: {question}")

        # Run the agent
        print("[RUN] Starting agent.run()...")
        response = agent.run(question)

        # Extract response
        if hasattr(response, "content") and response.content is not None:
            response_text = response.content
        elif response is not None:
            response_text = str(response)
        else:
            response_text = "(No response content)"

        print(f"[OUTPUT] Response length: {len(response_text)} chars")
        print(f"[OUTPUT] Response preview: {response_text[:300]}...")

        # Get Headroom stats if available
        if use_headroom and hasattr(model, "get_savings_summary"):
            stats = model.get_savings_summary()
            print(f"[HEADROOM] Stats: {stats}")

        return {
            "success": True,
            "label": label,
            "response": response_text[:500],
        }

    except Exception as e:
        error_msg = str(e)
        tb = traceback.format_exc()
        print(f"[ERROR] Exception: {error_msg}")
        print(f"[ERROR] Traceback:\n{tb}")

        return {
            "success": False,
            "label": label,
            "error": error_msg,
            "traceback": tb,
        }


def run_all_tests():
    """Run all Groq test combinations."""

    print("\n" + "=" * 70)
    print("GROQ REASONING TEST")
    print("=" * 70)
    print(f"GROQ_API_KEY: {'SET' if GROQ_API_KEY else 'NOT SET'}")
    print("=" * 70)

    results = []

    # Test with llama-3.3-70b-versatile (most capable)
    model_id = "llama-3.3-70b-versatile"

    test_cases = [
        # (use_headroom, use_reasoning)
        (False, False),  # Baseline: Groq direct, no reasoning
        (False, True),  # Groq direct, with reasoning
        (True, False),  # Groq + Headroom, no reasoning
        (True, True),  # Groq + Headroom, with reasoning <-- This is what fails for user
    ]

    for use_headroom, use_reasoning in test_cases:
        result = test_groq(use_headroom, use_reasoning, model_id)
        results.append(result)
        print(f"\nResult: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        if not result["success"]:
            print(f"Error: {result.get('error', 'Unknown')[:200]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} - {result['label']}")
        if not result["success"]:
            print(f"    Error: {result.get('error', 'Unknown')[:100]}")

    print("\n" + "=" * 70)

    # Analysis
    print("\nANALYSIS:")

    # Check if Groq + reasoning fails without Headroom
    groq_direct_reasoning = next(
        (r for r in results if "Direct" in r["label"] and "reasoning=True" in r["label"]), None
    )
    groq_headroom_reasoning = next(
        (r for r in results if "Headroom" in r["label"] and "reasoning=True" in r["label"]), None
    )

    if groq_direct_reasoning and not groq_direct_reasoning["success"]:
        print("⚠️  Groq + reasoning=True fails WITHOUT Headroom!")
        print("   This is an Agno/Groq bug, NOT a Headroom issue.")

    if (
        groq_direct_reasoning
        and groq_direct_reasoning["success"]
        and groq_headroom_reasoning
        and not groq_headroom_reasoning["success"]
    ):
        print("⚠️  Groq + reasoning=True works without Headroom but FAILS with Headroom!")
        print("   This IS a Headroom issue that needs investigation.")

    if all(r["success"] for r in results):
        print("✅ All tests passed! No issues found.")

    return results


if __name__ == "__main__":
    run_all_tests()

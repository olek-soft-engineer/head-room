"""Record `MessageScorer` parity fixtures.

Captures `MessageScorer.score_messages(messages, protected, tool_unit)`
with `toin=None` and `embedding_provider=None` so all six factors run
through the deterministic-or-neutral code path. The Rust comparator
(`MessageScorerComparator` in `crates/headroom-parity/src/lib.rs`) runs
the same inputs through the Rust port and asserts bit-equal outputs
after a 5-decimal-place rounding step.

Why round: Rust uses `f32::exp` and computes the weighted total in
f32, while Python's `math.exp` and weighted sum are f64. Both are
mathematically identical; they only drift in the low bits. Rounding
both sides to 5 decimals (1e-5 tolerance, 100x looser than f32 ulp
drift on the 0–1 score range) gives byte-equal JSON without masking
real bugs.

Run from repo root:
    python tests/parity/record_message_scorer.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from headroom.config import ScoringWeights
from headroom.transforms.scoring import MessageScorer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES_DIR = _REPO_ROOT / "tests" / "parity" / "fixtures" / "message_scorer"

# 5 decimals: f32 has ~7 decimals of precision, but Rust's port runs
# the weighted-sum in f32 while Python runs it in f64 — six summed
# f32-precision components occasionally drift in the 6th decimal of
# the total. 5 decimals (1e-5 tolerance) is loose enough to absorb
# the drift while still tight enough to catch real bugs.
_FLOAT_ROUND_PLACES = 5


def _round_floats(obj: Any) -> Any:
    """Recursively round every float in a JSON-shaped object."""
    if isinstance(obj, float):
        return round(obj, _FLOAT_ROUND_PLACES)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _record(
    label: str,
    messages: list[dict[str, Any]],
    protected_indices: list[int],
    tool_unit_indices: list[int],
    weights: ScoringWeights | None = None,
    decay_rate: float = 0.1,
) -> Path:
    scorer = MessageScorer(
        weights=weights,
        toin=None,
        embedding_provider=None,
        recency_decay_rate=decay_rate,
    )
    scores = scorer.score_messages(
        messages=messages,
        protected_indices=set(protected_indices),
        tool_unit_indices=set(tool_unit_indices),
    )

    payload_input = {
        "messages": messages,
        "protected_indices": sorted(protected_indices),
        "tool_unit_indices": sorted(tool_unit_indices),
        "decay_rate": decay_rate,
    }
    payload_config = {"weights": asdict(weights) if weights else None}
    payload_output = _round_floats([asdict(s) for s in scores])

    digest_source = {
        "transform": "message_scorer",
        "label": label,
        "input": payload_input,
        "config": payload_config,
    }
    digest = _digest(digest_source)

    fixture = {
        "transform": "message_scorer",
        "label": label,
        "input": payload_input,
        "config": payload_config,
        "output": payload_output,
        "recorded_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "input_sha256": digest,
    }

    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    target = _FIXTURES_DIR / f"{label}_{digest[:12]}.json"
    target.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    return target


def _scenarios() -> list[dict[str, Any]]:
    """Test scenarios covering each deterministic factor + edge cases."""
    out: list[dict[str, Any]] = []

    # 1. Single message — recency=1.0, no refs, no tool unit.
    out.append(
        {
            "label": "single_user_message",
            "messages": [{"role": "user", "content": "hello world"}],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 2. Empty list.
    out.append(
        {
            "label": "empty",
            "messages": [],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 3. Linear conversation (5 messages, no tools) — exercises recency
    # decay over a small range.
    out.append(
        {
            "label": "linear_5_messages",
            "messages": [
                {"role": "user", "content": "what is the capital of france"},
                {"role": "assistant", "content": "the capital of france is paris"},
                {"role": "user", "content": "and germany"},
                {"role": "assistant", "content": "the capital of germany is berlin"},
                {"role": "user", "content": "thanks"},
            ],
            "protected_indices": [0],
            "tool_unit_indices": [],
        }
    )

    # 4. Tool-call pair — exercises forward references.
    out.append(
        {
            "label": "tool_call_pair",
            "messages": [
                {"role": "user", "content": "what's the weather"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temp": 72, "conditions": "sunny"}',
                },
                {"role": "assistant", "content": "it is 72 and sunny"},
            ],
            "protected_indices": [0],
            "tool_unit_indices": [1, 2],
        }
    )

    # 5. Multiple tool-call references to same assistant message.
    out.append(
        {
            "label": "multi_ref_to_assistant",
            "messages": [
                {"role": "user", "content": "do many things"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "f"}},
                        {"id": "c2", "function": {"name": "g"}},
                        {"id": "c3", "function": {"name": "h"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"},
                {"role": "tool", "tool_call_id": "c3", "content": "r3"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [1, 2, 3, 4],
        }
    )

    # 6. High-density message (all-unique tokens).
    out.append(
        {
            "label": "high_density",
            "messages": [
                {"role": "user", "content": "alpha bravo charlie delta echo foxtrot golf hotel"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 7. Low-density (highly repetitive).
    out.append(
        {
            "label": "low_density",
            "messages": [
                {"role": "user", "content": "ok ok ok ok ok ok ok ok ok ok"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 8. Custom weights — exercises ScoringWeights normalization +
    # weighted total.
    out.append(
        {
            "label": "custom_weights_recency_heavy",
            "messages": [
                {"role": "user", "content": "first message in a longer chat"},
                {"role": "assistant", "content": "an assistant reply with substance"},
                {"role": "user", "content": "another follow up question here"},
                {"role": "assistant", "content": "and the closing reply"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
            "weights": ScoringWeights(
                recency=0.6,
                semantic_similarity=0.1,
                toin_importance=0.1,
                error_indicator=0.05,
                forward_reference=0.1,
                token_density=0.05,
            ),
        }
    )

    # 9. Custom decay rate (slower decay).
    out.append(
        {
            "label": "slow_decay",
            "messages": [{"role": "user", "content": f"message number {i}"} for i in range(8)],
            "protected_indices": [],
            "tool_unit_indices": [],
            "decay_rate": 0.02,
        }
    )

    # 10. drop_safe truth table — protected + in_tool_unit.
    out.append(
        {
            "label": "drop_safe_truth_table",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "x", "function": {"name": "f"}}],
                },
                {"role": "tool", "tool_call_id": "x", "content": "result"},
            ],
            "protected_indices": [0, 2],
            "tool_unit_indices": [2, 3],
        }
    )

    # 11. Unicode content — exercises char-count tokens estimate.
    out.append(
        {
            "label": "unicode_content",
            "messages": [
                {"role": "user", "content": "你好世界 это тест 🎉🎉🎉"},
                {"role": "assistant", "content": "received the unicode message"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 12. Tool-call id mismatch — tool message references a non-existent
    # call_id, must NOT contribute to forward refs.
    out.append(
        {
            "label": "orphan_tool_response",
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "nope", "content": "stranded"},
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    # 13. Non-string content (tool_calls list, no text content).
    out.append(
        {
            "label": "non_string_content",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "tc", "function": {"name": "f"}}],
                },
            ],
            "protected_indices": [],
            "tool_unit_indices": [],
        }
    )

    return out


def main() -> int:
    written: list[Path] = []
    for sc in _scenarios():
        path = _record(
            label=sc["label"],
            messages=sc["messages"],
            protected_indices=sc["protected_indices"],
            tool_unit_indices=sc["tool_unit_indices"],
            weights=sc.get("weights"),
            decay_rate=sc.get("decay_rate", 0.1),
        )
        written.append(path)
        print(f"  + {path.relative_to(_REPO_ROOT)}")
    print(f"wrote {len(written)} fixture(s) → {_FIXTURES_DIR.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

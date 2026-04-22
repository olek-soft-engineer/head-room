"""Tests for ``headroom.proxy.helpers.compute_turn_id``."""

from __future__ import annotations

from headroom.proxy.helpers import compute_turn_id

MODEL = "claude-sonnet-4-5"
SYSTEM = "You are helpful."


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tool_use(tool_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": {}}],
    }


def _user_tool_result(tool_id: str, out: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": out}],
    }


def test_returns_none_when_messages_empty():
    assert compute_turn_id(MODEL, SYSTEM, []) is None
    assert compute_turn_id(MODEL, SYSTEM, None) is None


def test_returns_none_when_no_user_text_message():
    messages = [_assistant_tool_use("t1", "bash")]
    assert compute_turn_id(MODEL, SYSTEM, messages) is None


def test_stable_across_agent_loop_iterations():
    iteration_1 = [_user("fix the bug")]
    iteration_2 = iteration_1 + [
        _assistant_tool_use("t1", "read"),
        _user_tool_result("t1", "file contents"),
    ]
    iteration_3 = iteration_2 + [
        _assistant_tool_use("t2", "edit"),
        _user_tool_result("t2", "edit ok"),
    ]

    id1 = compute_turn_id(MODEL, SYSTEM, iteration_1)
    id2 = compute_turn_id(MODEL, SYSTEM, iteration_2)
    id3 = compute_turn_id(MODEL, SYSTEM, iteration_3)

    assert id1 is not None
    assert id1 == id2 == id3


def test_rolls_over_on_new_user_prompt():
    turn_1 = [_user("first prompt")]
    turn_2 = turn_1 + [
        _assistant_tool_use("t1", "bash"),
        _user_tool_result("t1", "ok"),
        _user("second prompt"),
    ]

    id1 = compute_turn_id(MODEL, SYSTEM, turn_1)
    id2 = compute_turn_id(MODEL, SYSTEM, turn_2)

    assert id1 != id2


def test_different_model_yields_different_id():
    messages = [_user("same prompt")]
    id_a = compute_turn_id("claude-sonnet-4-5", SYSTEM, messages)
    id_b = compute_turn_id("claude-opus-4-7", SYSTEM, messages)
    assert id_a != id_b


def test_different_system_yields_different_id():
    messages = [_user("same prompt")]
    id_a = compute_turn_id(MODEL, "system A", messages)
    id_b = compute_turn_id(MODEL, "system B", messages)
    assert id_a != id_b


def test_accepts_list_system_prompt():
    messages = [_user("hi")]
    system_list = [{"type": "text", "text": "You are helpful."}]
    assert compute_turn_id(MODEL, system_list, messages) is not None


def test_text_block_in_list_content_is_a_user_turn():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert compute_turn_id(MODEL, SYSTEM, messages) is not None


def test_tool_result_only_content_is_not_a_turn_boundary():
    # A message whose only content is a tool_result is a continuation, not a
    # new turn — so the function must not latch onto it.
    messages = [_user_tool_result("t1", "result only")]
    assert compute_turn_id(MODEL, SYSTEM, messages) is None


def test_returns_16_hex_chars():
    turn_id = compute_turn_id(MODEL, SYSTEM, [_user("hi")])
    assert turn_id is not None
    assert len(turn_id) == 16
    int(turn_id, 16)  # raises if not hex

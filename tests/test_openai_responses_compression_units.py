from __future__ import annotations

from types import MethodType, SimpleNamespace

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    RouterCompressionResult,
)


class TokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


def _handler_with_router(router: ContentRouter) -> OpenAIHandlerMixin:
    handler = OpenAIHandlerMixin()
    handler.openai_pipeline = SimpleNamespace(transforms=[router])
    handler.openai_provider = SimpleNamespace(
        get_token_counter=lambda _model: TokenCounter(),
    )
    return handler


def test_openai_responses_adapter_compresses_only_live_text_slots():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="kept words",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "reasoning", "encrypted_content": long_text},
            {"type": "function_call", "arguments": long_text},
            {"type": "local_shell_call_output", "call_id": "c1", "output": long_text},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": long_text}],
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["encrypted_content"] == long_text
    assert new_payload["input"][1]["arguments"] == long_text
    assert new_payload["input"][2]["output"] == "kept words"
    assert new_payload["input"][3]["content"][0]["text"] == long_text
    assert any(t.startswith("router:openai:responses:") for t in transforms)
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_compresses_custom_tool_call_output():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="custom output summary",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    long_text = " ".join(f"word{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "custom_tool_call_output",
                "call_id": "c1",
                "output": long_text,
            }
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is True
    assert saved > 0
    assert new_payload["input"][0]["output"] == "custom output summary"
    assert (
        "router:openai:responses:custom_tool_call_output:kompress" in transforms
    )
    assert units_by_category == {"applied": 1}
    assert strategy_chain == []


def test_openai_responses_adapter_accepts_empty_input_list():
    router = ContentRouter()
    handler = _handler_with_router(router)
    payload = {"model": "gpt-5", "input": [], "tools": []}

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert new_payload == payload
    assert modified is False
    assert saved == 0
    assert transforms == []
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_preserves_headroom_retrieve_outputs():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="compressed retrieve output",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    retrieved = " ".join(f"retrieved{i}" for i in range(180))
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_retrieve",
                "name": "mcp__headroom__headroom_retrieve",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve",
                "output": retrieved,
            },
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {}
    assert strategy_chain == []


def test_openai_responses_adapter_keeps_small_and_opaque_items():
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="short",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "local_shell_call_output", "call_id": "c1", "output": "too small"},
            {"type": "compaction", "encrypted_content": " ".join(["secret"] * 200)},
        ],
    }

    new_payload, modified, saved, transforms, units_by_category, strategy_chain, _attempted = (
        handler._compress_openai_responses_live_text_units_with_router(
            payload,
            model="gpt-5",
            request_id="req_test",
        )
    )

    assert modified is False
    assert saved == 0
    assert transforms == []
    assert new_payload == payload
    assert units_by_category == {"size_floor": 1}
    assert strategy_chain == []


def test_openai_responses_payload_routes_through_content_router_without_rust(
    monkeypatch,
):
    router = ContentRouter()

    def compress(self, content: str, **_kwargs):
        return RouterCompressionResult(
            compressed="compressed fallback",
            original=content,
            strategy_used=CompressionStrategy.KOMPRESS,
        )

    router.compress = MethodType(compress, router)
    handler = _handler_with_router(router)

    import headroom._core as core

    def rust_must_not_run(*_args, **_kwargs):
        raise AssertionError("Responses payload compression should route through ContentRouter")

    monkeypatch.setattr(core, "compress_openai_responses_live_zone", rust_must_not_run)

    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "local_shell_call_output",
                "call_id": "c1",
                "output": " ".join(f"word{i}" for i in range(180)),
            }
        ],
    }

    new_payload, modified, saved, transforms, reason, _, _, _ = (
        handler._compress_openai_responses_payload(
            payload,
            model="gpt-5",
            request_id="req_router",
        )
    )

    assert modified is True
    assert saved > 0
    assert reason is None
    assert new_payload["input"][0]["output"] == "compressed fallback"
    assert any(t.startswith("router:openai:responses:") for t in transforms)

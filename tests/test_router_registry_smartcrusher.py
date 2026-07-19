"""Byte-identical differential tests for the SMART_CRUSHER / KOMPRESS / TEXT flip.

PR-C2 flips the PRIMARY compressor invocation of SMART_CRUSHER to the compressor
registry (``_registry_compress("smart_crusher", ...)``), while the shared
post-strategy Kompress -> Log fallback block stays a direct dispatch. KOMPRESS and
TEXT are DEFERRED: they are the ``_try_ml_compressor`` ML boundary, and the
``kompress`` built-in adapter (a) hardcodes ``question=None`` (dropping the real
QA-aware ``question`` argument) and (b) recomputes the token count via
``_estimate_tokens`` instead of returning ``_try_ml_compressor``'s tuple token
count (Kompress's own word-count ``compressed_tokens``, computed pre-CCR-marker),
so a registry round-trip cannot reproduce either the content (when a question is
supplied) or the token metric byte-for-byte. These tests pin both facts.

Every path asserts registry-dispatch output == the historical direct-dispatch
output (content, token metric, ``strategy_chain``, and — via the recorded call
args — the query/bias/question that flow through).

Offline guardrails:
  * No real ML/ONNX/HF inference — the ML boundary is mocked at
    ``_try_ml_compressor`` (SMART_CRUSHER fallback tests) or ``_get_kompress``
    (KOMPRESS/TEXT tests, which exercise the real ``_try_ml_compressor`` with a
    fake in-memory model).
  * SmartCrusher itself is pure-Python (no ML), so the SMART_CRUSHER success path
    runs the real crusher on a small JSON array.
  * ``relevance_split`` / ``lossless_then_lossy`` off and STAGE 0
    (``_lossless_first``) neutralized so the if/elif branch under test is the
    terminal path. All of that is shared, unchanged code.
  * The broad ``content_router``/``compression`` -k selection is NOT exercised
    (it hangs on HF-Hub/ONNX).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _estimate_tokens,
)


def _router() -> ContentRouter:
    """Router whose if/elif branch is the terminal dispatch path.

    ``relevance_split`` off, ``lossless_then_lossy`` off, and
    ``ccr_inject_marker`` off make the compressed output deterministic and
    marker-free; the same config is applied to the direct reference and the
    dispatch router, so the differential holds regardless.
    """
    return ContentRouter(
        ContentRouterConfig(
            relevance_split=False,
            lossless_then_lossy=False,
            ccr_inject_marker=False,
        )
    )


def _isolate_branch(monkeypatch: pytest.MonkeyPatch, router: ContentRouter) -> None:
    """Neutralize STAGE 0 (``_lossless_first``) so the if/elif branch is exercised.

    ``_lossless_first`` is shared, unchanged code (the flip only touches the branch
    body), so forcing it to a no-op isolates what the flip actually changed.
    """
    monkeypatch.setattr(router, "_lossless_first", lambda content, strategy: (content, None))


# A JSON array the real SmartCrusher shrinks (so the fallback chain stays single).
_JSON = json.dumps([{"id": i, "status": "ok", "level": "INFO", "value": i * 2} for i in range(40)])


# ─────────────────────── SMART_CRUSHER: flipped (registry) ────────────────────


def test_smart_crusher_success_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-SUCCEEDS: the flip routes the primary ``.crush`` through the
    # registry "smart_crusher" adapter, which delegates to the SAME getter +
    # ``.crush(query=..., bias=...)``. Differential: registry dispatch == a direct
    # crush on an independent router. A real shrink means no fallback fires.
    router = _router()
    _isolate_branch(monkeypatch, router)
    context, bias = "q", 1.0
    direct = _router()._get_smart_crusher().crush(_JSON, query=context, bias=bias).compressed
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, context, bias=bias
    )
    assert out == direct
    assert tokens == _estimate_tokens(direct)
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]
    assert len(out) < len(_JSON)


def test_smart_crusher_forwards_query_and_bias(monkeypatch: pytest.MonkeyPatch) -> None:
    # The flip must forward content, query (==context), and bias to ``.crush``
    # unchanged. A fake crusher records the exact call args, proving the registry
    # adapter's ``query=inp.query`` / ``bias=budget['bias']`` reproduce the direct
    # ``crush(content, query=context, bias=bias)`` call.
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}

    def _crush(content: str, query: str = "", bias: float = 1.0) -> SimpleNamespace:
        seen.update(content=content, query=query, bias=bias)
        return SimpleNamespace(compressed="CRUSHED " + content[:5])

    monkeypatch.setattr(router, "_get_smart_crusher", lambda: SimpleNamespace(crush=_crush))
    # Sentinel ML: if the flip WRONGLY dropped into the Kompress fallback we'd see
    # KOMPRESS appended to the chain; the chain assertion catches it.
    monkeypatch.setattr(
        router, "_try_ml_compressor", lambda *a, **k: ("KOMPRESS_SENTINEL", 999_999)
    )

    content = "x" * 200
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.SMART_CRUSHER, "myquery", bias=0.7
    )
    assert out == "CRUSHED " + content[:5]
    assert tokens == _estimate_tokens("CRUSHED " + content[:5])
    assert chain == [CompressionStrategy.SMART_CRUSHER.value]
    assert seen == {"content": content, "query": "myquery", "bias": 0.7}


def test_smart_crusher_kompress_fallback_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-NO-SHRINK -> KOMPRESS: the crusher returns the content
    # unchanged (no net saving), so the SHARED, UNCHANGED post-strategy block runs
    # the Kompress fallback. The flip only sets ``compressed``/``compressed_tokens``
    # on entry — identically to the direct crush — so the fallback fires exactly as
    # before: chain [smart_crusher, kompress], Kompress output adopted.
    router = _router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(
        router,
        "_get_smart_crusher",
        lambda: SimpleNamespace(crush=lambda c, query="", bias=1.0: SimpleNamespace(compressed=c)),
    )
    # Kompress shrinks (fallback_tokens < compressed_tokens) so it is adopted.
    monkeypatch.setattr(router, "_try_ml_compressor", lambda c, ctx, q: ("KOMPRESSED::" + c, 3))
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, "ctx", bias=1.0
    )
    assert out == "KOMPRESSED::" + _JSON
    assert tokens == 3
    assert chain == [
        CompressionStrategy.SMART_CRUSHER.value,
        CompressionStrategy.KOMPRESS.value,
    ]


def test_smart_crusher_log_fallback_matches_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    # SMART_CRUSHER-NO-SHRINK -> KOMPRESS-NO-SHRINK -> LOG: crusher passes content
    # through, Kompress fails to shrink (fallback_tokens NOT < compressed_tokens),
    # and — because the content is valid JSON and the log compressor is enabled —
    # the last-ditch Log fallback runs and shrinks. All of that is the shared,
    # unchanged block; the flip only feeds it identical entry state. Chain
    # [smart_crusher, kompress, log].
    router = _router()
    _isolate_branch(monkeypatch, router)
    monkeypatch.setattr(
        router,
        "_get_smart_crusher",
        lambda: SimpleNamespace(crush=lambda c, query="", bias=1.0: SimpleNamespace(compressed=c)),
    )
    # Kompress returns content unchanged with a huge token count -> NOT a shrink,
    # so the else branch (Log fallback) is taken.
    monkeypatch.setattr(router, "_try_ml_compressor", lambda c, ctx, q: (c, 10**9))
    monkeypatch.setattr(
        router,
        "_get_log_compressor",
        lambda: SimpleNamespace(
            compress=lambda c, bias=1.0: SimpleNamespace(compressed="LOG_FOLDED")
        ),
    )
    out, tokens, chain = router._apply_strategy_to_content(
        _JSON, CompressionStrategy.SMART_CRUSHER, "ctx", bias=1.0
    )
    assert out == "LOG_FOLDED"
    assert tokens == _estimate_tokens("LOG_FOLDED")
    assert chain == [
        CompressionStrategy.SMART_CRUSHER.value,
        CompressionStrategy.KOMPRESS.value,
        CompressionStrategy.LOG.value,
    ]


# ───────────────────── KOMPRESS / TEXT: deferred (ML boundary) ─────────────────


def _fake_kompress(seen: dict[str, object]) -> SimpleNamespace:
    """In-memory Kompress model: records compress kwargs, no ONNX/HF."""

    def _compress(text: str, **kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        # ``compressed_tokens`` is the model's OWN word-count (7), deliberately
        # unequal to ``_estimate_tokens`` of the output — this is exactly the
        # value the registry adapter would discard, which is why the branch is
        # deferred.
        return SimpleNamespace(compressed="KOMPRESSED::" + text, compressed_tokens=7)

    return SimpleNamespace(
        is_ready=lambda: True,
        ensure_background_load=lambda: None,
        compress=_compress,
    )


def test_kompress_deferred_ml_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "_get_kompress", lambda: _fake_kompress(seen))

    content = "some plain text that the ML model would compress. " * 4
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.KOMPRESS, "ctx", question="my question", bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.KOMPRESS.value]
    # Token count is the model's tuple value (7), NOT _estimate_tokens(out) — the
    # exact divergence that makes the registry round-trip non-byte-identical.
    assert tokens == 7
    assert tokens != _estimate_tokens(out)
    # The real ``question`` is forwarded (the kompress adapter would pass None).
    assert seen["question"] == "my question"
    # Still the bespoke ML path, byte-identical to a direct _try_ml_compressor call.
    direct, direct_tokens = router._try_ml_compressor(content, "ctx", "my question")
    assert out == direct
    assert tokens == direct_tokens


def test_text_deferred_ml_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    router = _router()
    _isolate_branch(monkeypatch, router)
    seen: dict[str, object] = {}
    monkeypatch.setattr(router, "_get_kompress", lambda: _fake_kompress(seen))

    content = "plain prose the text strategy sends straight to kompress. " * 4
    out, tokens, chain = router._apply_strategy_to_content(
        content, CompressionStrategy.TEXT, "ctx", question="q2", bias=1.0
    )
    assert out == "KOMPRESSED::" + content
    assert chain == [CompressionStrategy.TEXT.value]
    assert tokens == 7
    assert tokens != _estimate_tokens(out)
    assert seen["question"] == "q2"
    direct, direct_tokens = router._try_ml_compressor(content, "ctx", "q2")
    assert out == direct
    assert tokens == direct_tokens

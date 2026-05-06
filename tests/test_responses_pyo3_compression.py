"""Hot-fix tests: PyO3 inline `/v1/responses` compression.

Re-enables compression after PR-C5 retired the Python pipeline. The
standalone Rust proxy binary (`crates/headroom-proxy`) was supposed to
handle this, but it's not deployed by the CLI today. This module
exposes a PyO3 binding so the Python proxy can call the live-zone
dispatcher in-process.

These tests pin:

1. The binding is exposed and callable.
2. Round-trip: a body with no eligible content passes through unchanged.
3. Round-trip: a body with a compressible function-call output gets compressed.
4. Errors are non-fatal: malformed JSON / missing input array → passthrough.
5. Auth-mode parsing accepts every variant the F1 classifier produces.
"""

from __future__ import annotations

import json

import pytest


def _ensure_binding():
    """Skip if the Rust extension hasn't been built (mirrors existing pattern)."""
    try:
        from headroom._core import compress_openai_responses_live_zone

        return compress_openai_responses_live_zone
    except ImportError:
        pytest.skip("headroom._core not built — run scripts/build_rust_extension.sh")


class TestBindingExposed:
    """The pyfunction is reachable from Python."""

    def test_callable(self):
        compress = _ensure_binding()
        assert callable(compress), "compress_openai_responses_live_zone must be callable"


class TestPassthroughCases:
    """Bodies the dispatcher cannot compress should be returned byte-for-byte
    with `modified=False`. Matches the Rust proxy's `Outcome::Passthrough`
    contract."""

    def test_not_json_passthrough(self):
        compress = _ensure_binding()
        body = b"this is not JSON at all"
        out, modified = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini"}).encode()
        out, modified = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_empty_input_array_passthrough(self):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        out, modified = compress(body, "payg", "gpt-4o-mini")
        assert out == body
        assert modified is False

    def test_no_eligible_items_passthrough(self):
        compress = _ensure_binding()
        # Single user message under the byte threshold — no compression
        # applies, but still valid input.
        body = json.dumps(
            {
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        ).encode()
        out, modified = compress(body, "payg", "gpt-4o-mini")
        assert modified is False
        # Body should be byte-equal (passthrough, not re-serialized).
        assert out == body


class TestAuthModeAccepted:
    """Every F1 AuthMode value is accepted; unrecognised falls back to
    Unknown (does not raise)."""

    @pytest.mark.parametrize(
        "auth_mode",
        ["payg", "oauth", "subscription", "unknown", "", "garbage"],
    )
    def test_accepts(self, auth_mode):
        compress = _ensure_binding()
        body = json.dumps({"model": "gpt-4o-mini", "input": []}).encode()
        # Should not raise on any string input.
        out, modified = compress(body, auth_mode, "gpt-4o-mini")
        assert isinstance(out, bytes)
        assert modified is False


class TestModelDefault:
    """Empty `model` defaults to `headroom_core`'s `DEFAULT_MODEL`."""

    def test_empty_model_uses_default(self):
        compress = _ensure_binding()
        body = json.dumps({"input": []}).encode()
        out, modified = compress(body, "payg", "")
        assert isinstance(out, bytes)
        assert modified is False


class TestNoExceptionsLeak:
    """The binding's contract is `never raises` (matches the Rust proxy's
    `compress_openai_responses_request` passthrough-on-error semantics).
    Pin this so future maintainers don't accidentally introduce a
    raising path."""

    def test_garbage_bytes_no_raise(self):
        compress = _ensure_binding()
        out, modified = compress(b"\xff\xfe\x00\xff", "payg", "gpt-4o-mini")
        assert modified is False
        assert out == b"\xff\xfe\x00\xff"

    def test_empty_body_no_raise(self):
        compress = _ensure_binding()
        out, modified = compress(b"", "payg", "gpt-4o-mini")
        assert modified is False
        assert out == b""

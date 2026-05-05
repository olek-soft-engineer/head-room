"""Tests for the Python ``CompressionPolicy`` and its parity with Rust.

The Python module is a hand-mirror of
``headroom_core::compression_policy::CompressionPolicy``. These tests
pin both halves: that the per-mode values are right, and that the
Python and Rust sides agree on the field map. F2.2 will likely retire
the hand-mirror via PyO3 — until then, this file is the canary.
"""

from __future__ import annotations

import pytest

from headroom.proxy.auth_mode import AuthMode
from headroom.transforms.compression_policy import (
    CompressionPolicy,
    policy_default_payg,
    policy_for_mode,
)


class TestCompressionPolicyForMode:
    """Per-mode field assertions. Mirrors the three Rust unit tests in
    `crates/headroom-core/src/compression_policy.rs`.
    """

    def test_payg_is_aggressive(self):
        p = policy_for_mode(AuthMode.PAYG)
        assert p.live_zone_only is False, "PAYG can touch outside live zone"
        assert p.cache_aligner_enabled is True, "PAYG runs cache aligner"

    def test_oauth_matches_payg_today(self):
        # Canary: when F2.2 diverges OAuth from PAYG, this test fails
        # and forces a deliberate update on BOTH sides (Rust + Python).
        oauth = policy_for_mode(AuthMode.OAUTH)
        payg = policy_for_mode(AuthMode.PAYG)
        assert oauth == payg, (
            "F2.1 ships OAuth=PAYG; F2.2 will diverge based on telemetry. "
            "If you are reading this assertion failure: also update "
            "crates/headroom-core/src/compression_policy.rs "
            "::oauth_matches_payg_today, otherwise the Rust + Python "
            "parities silently drift apart."
        )

    def test_subscription_disables_cache_aligner(self):
        p = policy_for_mode(AuthMode.SUBSCRIPTION)
        assert p.live_zone_only is True, "Subscription is live-zone-only"
        assert p.cache_aligner_enabled is False, (
            "Subscription MUST skip cache aligner — load-bearing for issues #327 / #388"
        )


class TestPolicyDefaultPayg:
    """The constant used when the enforcement flag is disabled."""

    def test_default_payg_equals_for_mode_payg(self):
        assert policy_default_payg() == policy_for_mode(AuthMode.PAYG)


class TestImmutability:
    """The struct is `frozen=True`; mutation must raise."""

    def test_policy_is_frozen(self):
        p = policy_for_mode(AuthMode.PAYG)
        with pytest.raises((AttributeError, Exception)):
            # Attempting to mutate a frozen dataclass raises
            # FrozenInstanceError (subclass of AttributeError on
            # CPython 3.10+). Catch both for compatibility.
            p.live_zone_only = True  # type: ignore[misc]


class TestRustParityFieldMap:
    """The Python policy must have the same fields as the Rust struct.

    The canonical Rust struct lives at
    ``crates/headroom-core/src/compression_policy.rs``. When you add a
    field there for F2.2, add it here AND update this test. Otherwise
    the parity silently drifts.
    """

    def test_field_set_matches_rust(self):
        # Hard-coded set — when Rust grows fields, this test fails until
        # Python catches up.
        expected_fields = {"live_zone_only", "cache_aligner_enabled"}
        actual_fields = {f.name for f in CompressionPolicy.__dataclass_fields__.values()}
        assert actual_fields == expected_fields, (
            f"Python CompressionPolicy fields drifted from Rust. "
            f"Expected exactly {expected_fields}, got {actual_fields}. "
            f"Update both `headroom/transforms/compression_policy.py` "
            f"and `crates/headroom-core/src/compression_policy.rs` in "
            f"the same commit."
        )

"""Per-auth-mode compression policy — Phase F PR-F2.1, Python parity.

Hand-mirrored port of `headroom_core::compression_policy::CompressionPolicy`
(Rust). The Rust crate is the source of truth; this module exists so
the Python proxy's `TransformPipeline` (which still runs `CacheAligner`
and other detector-only transforms) can read the same per-mode flags
the Rust dispatcher does.

A parity test (`tests/test_compression_policy.py`) instantiates one of
each variant and asserts the field map matches what the Rust unit
tests assert. F2.2 should consider exposing the Rust struct via PyO3
to retire this hand-mirror — that's deliberately out of scope here so
F2.1 can ship.

See `crates/headroom-core/src/compression_policy.rs` for the canonical
docstring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from headroom.proxy.auth_mode import AuthMode


@dataclass(frozen=True, slots=True)
class CompressionPolicy:
    """Per-auth-mode policy that downstream compression stages consult.

    Two fields in F2.1 — `live_zone_only` and `cache_aligner_enabled`.
    F2.2 will add tuning fields (per-mode volatile threshold,
    max-lossy-ratio cap, per-mode TOIN read-only flag) once telemetry
    from F2.1's bake on `main` shows what each mode actually costs/saves.
    """

    live_zone_only: bool
    """When True, transforms MUST NOT modify bytes outside the post-
    cache-marker live zone. The Rust live-zone dispatcher is already
    live-zone-only by construction, so this flag is effectively a
    no-op on the Rust path; it exists for the Python TransformPipeline
    where transforms like CacheAligner inspect the cached prefix."""

    cache_aligner_enabled: bool
    """When False, the CacheAligner transform's `should_apply` MUST
    return False — that's the load-bearing F2.1 gate for the cache-
    instability complaints in #327 / #388."""


def policy_for_mode(mode: AuthMode) -> CompressionPolicy:
    """Resolve the F2.1 policy for an auth mode.

    PAYG and OAuth are identical in F2.1 (aggressive: live-zone-not-
    only, cache-aligner on). Subscription is the user-visible win:
    live-zone-only with cache aligner disabled.

    F2.2 may diverge OAuth from PAYG once telemetry is collected.
    """
    if mode == AuthMode.PAYG:
        return CompressionPolicy(live_zone_only=False, cache_aligner_enabled=True)
    if mode == AuthMode.OAUTH:
        # Identical to PAYG in F2.1. The parity test in
        # `tests/test_compression_policy.py` is the canary that
        # catches a future divergence and forces a deliberate
        # update there + in the Rust crate.
        return CompressionPolicy(live_zone_only=False, cache_aligner_enabled=True)
    if mode == AuthMode.SUBSCRIPTION:
        return CompressionPolicy(live_zone_only=True, cache_aligner_enabled=False)
    raise ValueError(f"Unhandled AuthMode variant: {mode!r}")


def policy_default_payg() -> CompressionPolicy:
    """The PAYG-equivalent policy used when the
    ``HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT`` flag is disabled
    (default in F2.1 c1-c4; flips to enabled in c5/5).

    Centralised so the proxy handlers do not duplicate the constant,
    and so a future change to PAYG semantics propagates to both the
    enforcement-on and enforcement-off paths.
    """
    return policy_for_mode(AuthMode.PAYG)


_ENFORCEMENT_ENV = "HEADROOM_PROXY_AUTH_MODE_POLICY_ENFORCEMENT"


def is_enforcement_enabled() -> bool:
    """Read the enforcement flag from the environment.

    Same env var the Rust proxy reads (``Config::auth_mode_policy_enforcement``)
    so the two paths stay in lockstep with one operator switch.
    Default (when unset): ``True`` from F2.1 c5/5 onward, matching
    the Rust default after the c5 flip.

    NOT cached — read every call so an operator can flip the env var
    in a hot-reload scenario. The cost is one ``dict.get`` per call,
    well below noise.
    """
    val = os.environ.get(_ENFORCEMENT_ENV, "enabled").strip().lower()
    # Same set of off-values the telemetry beacon honours
    # (`headroom/telemetry/beacon.py::_OFF_VALUES`) so operators don't
    # have to remember a different vocabulary per flag.
    return val not in ("disabled", "off", "false", "0", "no")


def resolve_policy(auth_mode: AuthMode | None) -> CompressionPolicy:
    """Resolve the effective ``CompressionPolicy`` for a request.

    - If the enforcement flag is off, returns PAYG-equivalent
      regardless of the classified auth mode.
    - If the enforcement flag is on and ``auth_mode`` is ``None``,
      returns PAYG-equivalent (defensive default for the unclassified
      / batch-row path).
    - Otherwise returns the per-mode policy.

    This is the single public entry point handlers should call when
    deriving the policy from a request's classification result.
    """
    if auth_mode is None or not is_enforcement_enabled():
        return policy_default_payg()
    return policy_for_mode(auth_mode)

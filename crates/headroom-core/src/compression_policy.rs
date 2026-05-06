//! Per-auth-mode compression policy — Phase F PR-F2.1.
//!
//! F1 (`auth_mode.rs`) classifies each inbound request into one of
//! `{Payg, OAuth, Subscription}`. Phase F2.1 turns that classification
//! into a `CompressionPolicy` that downstream pipeline stages read to
//! decide whether they run.
//!
//! Why a struct instead of `match auth_mode { ... }` everywhere?
//! Two reasons:
//!
//! 1. **Centralisation.** Without a policy struct, the per-mode
//!    decision is duplicated at every gate (E3 cache_control, E4
//!    prompt_cache_key, the new live-zone gate, the new cache_aligner
//!    gate, …). When F2.2 wants to tune (e.g. allow OAuth users a
//!    relaxed live-zone gate but stricter volatile-detector threshold)
//!    we'd need to find every site. The struct is the one place to
//!    edit; call sites just read `policy.field`.
//!
//! 2. **Test surface.** `for_mode(AuthMode) -> CompressionPolicy` is
//!    pure and trivial to property-test. Asserting per-mode values
//!    against the struct catches regressions cheaply, whereas asserting
//!    end-to-end behaviour against the dispatcher requires a full
//!    request fixture.
//!
//! Phase F2.2 will add fields to this struct (per-mode volatile
//! threshold, per-mode max lossy ratio, per-mode TOIN read-only flag).
//! F2.1 keeps the field set minimal — only the two flags load-bearing
//! for closing the cache-instability complaints in issues #327 / #388.
//!
//! ## Field semantics
//!
//! - **`live_zone_only`**: when `true`, downstream stages MUST NOT
//!   modify bytes outside the post-cache-marker live zone. Phase B's
//!   Rust dispatcher is *already* live-zone-only by construction, so
//!   this flag is effectively a no-op on the Rust path and exists for
//!   the Python `TransformPipeline`'s `CacheAligner` / `ContentRouter`
//!   gates. Storing it on the canonical struct keeps the cross-
//!   language parity tests honest — Python and Rust must agree on the
//!   field map even when only one side acts on a value.
//!
//! - **`cache_aligner_enabled`**: when `false`, the Python
//!   `CacheAligner` transform's `should_apply` MUST return `False`.
//!   `CacheAligner` is the load-bearing fix for the cache-instability
//!   complaints — historically it has been mutating cached prefixes
//!   and writing into `_previous_prefix_hash` per pipeline instance,
//!   which is what destabilised Subscription users' prompt caches.
//!   Disabling it for Subscription is the user-visible win of F2.1.
//!
//! ## Per-mode F2.1 values
//!
//! | Mode         | live_zone_only | cache_aligner_enabled |
//! |--------------|----------------|-----------------------|
//! | Payg         | false          | true                  |
//! | OAuth        | false          | true (= PAYG today)   |
//! | Subscription | true           | false                 |
//!
//! OAuth starts identical to PAYG. F2.2 will divide them once
//! telemetry from F2.1's bake on `main` shows what each mode actually
//! costs / saves.
//!
//! ## What this struct does NOT replace
//!
//! Phase E's existing PAYG-only gates (cache_control auto-placement,
//! prompt_cache_key injection) keep matching `auth_mode == Payg`
//! directly. Migrating those to the policy struct is F2.2 cleanup
//! work. Doing it in F2.1 would balloon the diff and the existing
//! gates already produce the correct per-mode behaviour — there's no
//! user-visible reason to refactor them now.

use crate::auth_mode::AuthMode;

/// Per-auth-mode policy that downstream compression stages consult.
///
/// `Copy` because the struct is two `bool`s — passing by value is
/// cheaper than passing a reference and the call sites all want
/// owned copies anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionPolicy {
    /// When `true`, transforms MUST NOT modify bytes outside the
    /// post-cache-marker live zone. See module docs.
    pub live_zone_only: bool,

    /// When `false`, the `CacheAligner` transform MUST be skipped.
    /// See module docs.
    pub cache_aligner_enabled: bool,
}

impl CompressionPolicy {
    /// Resolve the F2.1 policy for an auth mode. See module docs for
    /// per-mode rationale.
    pub fn for_mode(mode: AuthMode) -> Self {
        match mode {
            AuthMode::Payg => Self {
                live_zone_only: false,
                cache_aligner_enabled: true,
            },
            // OAuth identical to PAYG in F2.1. F2.2 may diverge once
            // telemetry shows what OAuth users actually need.
            AuthMode::OAuth => Self {
                live_zone_only: false,
                cache_aligner_enabled: true,
            },
            // The user-visible win of F2.1: subscription users stop
            // seeing cache instability because CacheAligner no longer
            // touches their prefix.
            AuthMode::Subscription => Self {
                live_zone_only: true,
                cache_aligner_enabled: false,
            },
        }
    }

    /// Whether the live-zone dispatcher should run at all for this
    /// policy. Always `true` in F2.1 — every mode still gets live-zone
    /// compression (closing #327/#388 requires Subscription to KEEP
    /// compressing the live zone, just stop destabilising the cache).
    /// F2.2 may flip Subscription to `false` if telemetry shows the
    /// live-zone savings aren't worth the latency.
    pub fn live_zone_compression_enabled(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payg_is_aggressive() {
        let p = CompressionPolicy::for_mode(AuthMode::Payg);
        assert!(!p.live_zone_only, "PAYG can touch outside live zone");
        assert!(p.cache_aligner_enabled, "PAYG runs cache aligner");
        assert!(p.live_zone_compression_enabled());
    }

    #[test]
    fn oauth_matches_payg_today() {
        // Canary: when F2.2 diverges OAuth from PAYG, this test fails
        // and forces a deliberate update — which is the point.
        let oauth = CompressionPolicy::for_mode(AuthMode::OAuth);
        let payg = CompressionPolicy::for_mode(AuthMode::Payg);
        assert_eq!(
            oauth, payg,
            "F2.1 ships OAuth=PAYG; F2.2 will diverge based on telemetry"
        );
    }

    #[test]
    fn subscription_disables_cache_aligner() {
        let p = CompressionPolicy::for_mode(AuthMode::Subscription);
        assert!(p.live_zone_only, "Subscription is live-zone-only");
        assert!(
            !p.cache_aligner_enabled,
            "Subscription MUST skip cache aligner — load-bearing for #327/#388"
        );
        assert!(
            p.live_zone_compression_enabled(),
            "Subscription still gets live-zone compression — closing the cache complaint must NOT mean shipping zero compression"
        );
    }
}

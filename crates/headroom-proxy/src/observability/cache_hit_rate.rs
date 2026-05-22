//! Per-session cache-hit-rate observability — Phase G PR-G3.
//!
//! # The H-blocker metric
//!
//! Phase H ("retire the Python proxy") depends on this metric to
//! confirm parity between the Rust proxy and the soon-to-retire
//! Python proxy during canary. The acceptance gate in
//! `REALIGNMENT/10-phase-H-python-retirement.md:11-12` reads
//! `cache_hit_rate ≥ Python baseline; no 5xx regressions in 24h`.
//! That assertion is meaningless without this histogram.
//!
//! # Definition
//!
//! For every completed request session we observe one sample on the
//! `proxy_cache_hit_rate_per_session` histogram whose value is:
//!
//! ```text
//!   cache_read_input_tokens
//!   ───────────────────────
//!         total_input_tokens
//! ```
//!
//! `total_input_tokens` is the full denominator each provider
//! exposes on the `usage` payload (input + cache_read +
//! cache_creation), NOT the "billable" input that excludes cache
//! hits. Defining the denominator this way means the metric stays
//! meaningful across providers and stays in [0, 1].
//!
//! # When called
//!
//! - Anthropic: at `message_delta` (which carries the final `usage`).
//! - OpenAI Chat: on the final usage chunk (when
//!   `stream_options.include_usage = true`; otherwise no sample is
//!   emitted — the absence is itself a signal).
//! - OpenAI Responses: at `response.completed`.
//!
//! # Cardinality
//!
//! One label only: `provider ∈ {anthropic, openai_chat, openai_responses}`.
//! The metric is *intentionally* low-cardinality — the H1 canary
//! looks at fleet-wide hit rate, not per-model. Per-model breakdown
//! lives in the per-provider info logs we already emit.
//!
//! # Bucket selection
//!
//! Bucket boundaries are picked so the histogram discriminates the
//! "no cache" → "cache landed" transition with high resolution near
//! 0 and the "high-hit-rate" regime (where cache is paying for
//! itself) with high resolution near 1. The middle is coarser because
//! ~50% hit-rate is operationally uninteresting — either the cache
//! is working or it isn't.

use std::sync::OnceLock;

use prometheus::{HistogramOpts, HistogramVec, Registry};

use super::metric_names::{
    LABEL_PROVIDER, METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
    METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION_HELP,
};

/// Histogram buckets in [0, 1]. Tighter near 0 (so a "barely any
/// cache hit" stays distinguishable from a true zero) and near 1
/// (so a "near-perfect cache" stays distinguishable from a real
/// 100% hit). The middle band is intentionally coarser.
pub(super) const CACHE_HIT_RATE_BUCKETS: &[f64] =
    &[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0];

/// Provider label vocabulary. Kept here (rather than scattered
/// across emit sites) so the cardinality budget is reviewable in one
/// place per realignment build-constraint "configurable + scalable".
pub mod provider {
    pub const ANTHROPIC: &str = "anthropic";
    pub const OPENAI_CHAT: &str = "openai_chat";
    pub const OPENAI_RESPONSES: &str = "openai_responses";
}

/// `proxy_cache_hit_rate_per_session{provider}` — initialised on
/// first call from `register_in`.
pub fn histogram(registry: &Registry) -> &'static HistogramVec {
    static HIST: OnceLock<HistogramVec> = OnceLock::new();
    HIST.get_or_init(|| {
        let opts = HistogramOpts::new(
            METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
            METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION_HELP,
        )
        .buckets(CACHE_HIT_RATE_BUCKETS.to_vec());
        let hist = HistogramVec::new(opts, &[LABEL_PROVIDER])
            .expect("proxy_cache_hit_rate_per_session descriptor is well-formed");
        registry
            .register(Box::new(hist.clone()))
            .expect("proxy_cache_hit_rate_per_session registers exactly once");
        hist
    })
}

/// Compute the per-session cache hit rate given the three counters
/// most providers expose on `usage`. Returns `None` when the
/// denominator is zero (no input tokens at all — typically a
/// degenerate request such as an empty messages array), so callers
/// can choose to skip the observation rather than divide-by-zero.
///
/// Per realignment build-constraint "no silent fallbacks", a
/// zero-denominator request is *NOT* coerced to `0.0`; it returns
/// `None` so the emit-site can log + skip instead of polluting the
/// histogram with synthesised samples.
pub fn compute_hit_rate(
    input_tokens: u64,
    cache_read_input_tokens: u64,
    cache_creation_input_tokens: u64,
) -> Option<f64> {
    let denom = input_tokens
        .saturating_add(cache_read_input_tokens)
        .saturating_add(cache_creation_input_tokens);
    if denom == 0 {
        return None;
    }
    Some(cache_read_input_tokens as f64 / denom as f64)
}

/// Observe one per-session sample.
///
/// `provider` MUST be one of the [`provider`] constants — callers
/// from the SSE state machine pass the matching string verbatim. We
/// do not validate the label here because the cardinality is bounded
/// by the static call sites; an invalid label would be a bug, not a
/// runtime mismatch.
pub fn observe(provider: &'static str, request_id: &str, hit_rate: f64) {
    debug_assert!(
        (0.0..=1.0).contains(&hit_rate),
        "cache hit rate must be in [0.0, 1.0]; got {hit_rate}"
    );
    let clamped = hit_rate.clamp(0.0, 1.0);
    histogram(super::prometheus::registry())
        .with_label_values(&[provider])
        .observe(clamped);
    tracing::debug!(
        event = "metric_recorded",
        metric = METRIC_PROXY_CACHE_HIT_RATE_PER_SESSION,
        provider = provider,
        request_id = %request_id,
        hit_rate = clamped,
        "observed proxy_cache_hit_rate_per_session"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_rate_zero_when_no_cache_reads() {
        let r = compute_hit_rate(100, 0, 0).unwrap();
        assert!((r - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_one_when_all_reads_are_cache_hits() {
        // Edge case: input_tokens == 0 means upstream charged us
        // nothing new — every input came from cache.
        let r = compute_hit_rate(0, 1000, 0).unwrap();
        assert!((r - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn hit_rate_split_three_ways() {
        // 50 fresh input, 50 cache_read, 50 cache_creation → 50/150.
        let r = compute_hit_rate(50, 50, 50).unwrap();
        assert!((r - (1.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn hit_rate_none_on_empty_request() {
        // Degenerate: no tokens at all. Caller should skip the
        // observation, not coerce to 0.0.
        assert!(compute_hit_rate(0, 0, 0).is_none());
    }
}

//! Anthropic `/v1/messages` request compression — Phase B PR-B2
//! live-zone dispatcher entry point.
//!
//! # Pipeline
//!
//! 1. Resolve `frozen_message_count` from the request body via
//!    [`crate::compression::resolve_frozen_count`] (PR-A4 helper).
//!    The proxy's `cache_control_auto_frozen` config gates whether
//!    the body is parsed at all — when disabled, floor=0 without
//!    inspection.
//! 2. Hand the buffered body bytes to
//!    [`headroom_core::transforms::compress_live_zone`]. The
//!    dispatcher inspects the live zone (latest user message) and
//!    dispatches per-block compression. PR-B2 wires every per-type
//!    function to a no-op, so the dispatcher always returns
//!    [`headroom_core::transforms::LiveZoneOutcome::NoChange`].
//! 3. Translate the result into [`Outcome`] for the proxy: every
//!    PR-B2 call lands on [`Outcome::NoCompression`].
//!
//! # Cache-safety invariant
//!
//! The dispatcher does not mutate any byte in the request body for
//! PR-B2. The proxy's `proxy.rs` forwards the *original* buffered
//! bytes verbatim. Phase A's SHA-256 fixtures still pass: the
//! introduction of dispatching (vs. unconditional passthrough)
//! changes log lines but no upstream-bound bytes.

use bytes::Bytes;
use headroom_core::transforms::{
    compress_live_zone, AuthMode, BlockAction, ExclusionReason, LiveZoneError, LiveZoneOutcome,
};

use crate::compression::resolve_frozen_count;
use crate::config::{CacheControlAutoFrozen, CompressionMode};

/// What happened. The caller uses the variant to decide whether to
/// forward the original bytes (everything PR-B2 lands on) or a
/// modified body (PR-B3+).
#[derive(Debug)]
pub enum Outcome {
    /// Body was not compressed. Caller forwards the original
    /// buffered bytes byte-equal. Always returned in PR-B2.
    NoCompression,
    /// Reserved for PR-B3+: live-zone compression actually ran and
    /// produced a (smaller) body.
    #[allow(dead_code)]
    Compressed {
        body: Bytes,
        tokens_before: usize,
        tokens_after: usize,
        strategies_applied: Vec<&'static str>,
        markers_inserted: Vec<String>,
    },
    /// Dispatcher opted out for a reason we can name.
    Passthrough { reason: PassthroughReason },
}

/// Reason the live-zone dispatcher fell through. Each variant is
/// logged at warn level by the proxy.
#[derive(Debug, Clone, Copy)]
pub enum PassthroughReason {
    /// Body was not valid JSON — never our job to fix that, but we
    /// log so operators know which requests opted out.
    NotJson,
    /// `messages` was missing or not a JSON array — the upstream
    /// API will reject with a 400 anyway; we're just bystanders.
    NoMessages,
    /// The compression-mode config is `Off`. The dispatcher is not
    /// invoked.
    ModeOff,
}

/// Live-zone compression entry point for Anthropic `/v1/messages`.
///
/// Returns one of:
///
/// - [`Outcome::NoCompression`] — proxy forwards the original
///   buffered body verbatim. PR-B2 always lands here.
/// - [`Outcome::Compressed`] — PR-B3+ produces this when at least
///   one block was rewritten.
/// - [`Outcome::Passthrough`] — invalid body shape; proxy forwards
///   the original bytes anyway.
///
/// # Arguments
///
/// - `body`: the buffered request body. Owned by the caller for the
///   lifetime of the upstream request — we only borrow.
/// - `mode`: configured compression mode. `Off` short-circuits to
///   [`Outcome::Passthrough { reason: ModeOff }`]; `LiveZone` runs
///   the dispatcher.
/// - `cache_control_policy`: gates auto-derivation of
///   `frozen_message_count` from explicit `cache_control` markers
///   in the body. Disabled → floor=0 (everything is in the live
///   zone).
/// - `request_id`: per-request id used for log correlation.
pub fn compress_anthropic_request(
    body: &Bytes,
    mode: CompressionMode,
    cache_control_policy: CacheControlAutoFrozen,
    request_id: &str,
) -> Outcome {
    if matches!(mode, CompressionMode::Off) {
        tracing::info!(
            request_id = %request_id,
            path = "/v1/messages",
            method = "POST",
            compression_mode = mode.as_str(),
            decision = "passthrough",
            reason = "mode_off",
            body_bytes = body.len(),
            "anthropic compression decision"
        );
        return Outcome::Passthrough {
            reason: PassthroughReason::ModeOff,
        };
    }

    // Mode is LiveZone. Resolve the cache-hot floor first; this is
    // the only place the body is parsed at all when the policy is
    // Disabled (resolve_frozen_count short-circuits).
    let parsed: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "not_json",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            return Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            };
        }
    };

    let frozen_count = resolve_frozen_count(&parsed, cache_control_policy, request_id);

    // Run the live-zone dispatcher. PR-B2: every block lands on a
    // no-op compressor, so the result is always NoChange. PR-B3+
    // begin returning Modified.
    match compress_live_zone(body, frozen_count, AuthMode::Payg) {
        Ok(LiveZoneOutcome::NoChange { manifest }) => {
            let block_count = manifest.block_outcomes.len();
            let blocks_excluded = manifest
                .block_outcomes
                .iter()
                .filter(|b| {
                    matches!(
                        b.action,
                        BlockAction::Excluded {
                            reason: ExclusionReason::HotZoneBlockType
                        }
                    )
                })
                .count();
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "no_change",
                reason = "no_op_skeleton_pr_b2",
                body_bytes = body.len(),
                frozen_message_count = frozen_count,
                messages_total = manifest.messages_total,
                latest_user_message_index = ?manifest.latest_user_message_index,
                live_zone_blocks = block_count,
                live_zone_blocks_excluded = blocks_excluded,
                "anthropic live-zone dispatch"
            );
            Outcome::NoCompression
        }
        Ok(LiveZoneOutcome::Modified { .. }) => {
            // PR-B3+ will reach this arm. We keep it in B2 only as
            // a debug_assert so an accidental early Modified emission
            // surfaces immediately rather than producing a body
            // we're not yet prepared to forward.
            debug_assert!(
                false,
                "PR-B2 dispatcher must never return LiveZoneOutcome::Modified"
            );
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                "live-zone dispatcher emitted Modified outcome in PR-B2; \
                 falling back to original bytes (this is a regression)"
            );
            Outcome::NoCompression
        }
        Err(LiveZoneError::BodyNotJson(_)) => {
            // We already parsed successfully above; the dispatcher's
            // independent parse can only fail on a state we missed.
            // Pass through with the same byte-faithful guarantee.
            tracing::warn!(
                request_id = %request_id,
                path = "/v1/messages",
                "live-zone dispatcher rejected JSON body that this layer parsed; \
                 falling back to passthrough"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            }
        }
        Err(LiveZoneError::NoMessagesArray) => {
            tracing::info!(
                request_id = %request_id,
                path = "/v1/messages",
                method = "POST",
                compression_mode = mode.as_str(),
                decision = "passthrough",
                reason = "no_messages",
                body_bytes = body.len(),
                "anthropic compression decision"
            );
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_of(value: serde_json::Value) -> Bytes {
        Bytes::from(serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn mode_off_short_circuits_without_parsing() {
        // Invalid JSON — would fail parse — but mode=Off must not
        // attempt to parse, and instead Passthrough{ModeOff}.
        let body = Bytes::from_static(b"not valid json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::Off,
            CacheControlAutoFrozen::Disabled,
            "req-1",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::ModeOff,
            } => {}
            other => panic!("expected Passthrough{{ModeOff}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_no_messages_field_passthrough() {
        let body = body_of(serde_json::json!({"model": "claude"}));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-2",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NoMessages,
            } => {}
            other => panic!("expected Passthrough{{NoMessages}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_invalid_json_passthrough() {
        let body = Bytes::from_static(b"\x01\x02 not json");
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-3",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn live_zone_mode_with_valid_body_returns_no_compression_pr_b2() {
        // PR-B2 invariant: every well-formed body returns NoCompression.
        let body = body_of(serde_json::json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "hello"}
                ]}
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            "req-4",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }

    #[test]
    fn empty_body_with_live_zone_mode_passthrough_not_json() {
        let body = Bytes::new();
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Enabled,
            "req-5",
        );
        match out {
            Outcome::Passthrough {
                reason: PassthroughReason::NotJson,
            } => {}
            other => panic!("expected Passthrough{{NotJson}}, got {other:?}"),
        }
    }

    #[test]
    fn cache_control_disabled_yields_floor_zero() {
        // With auto-derivation Disabled, frozen floor is 0 even
        // though the body marks every message as cached. The
        // dispatcher will treat the entire array as live zone.
        // (PR-B2: still returns NoCompression — this test pins the
        // policy plumbing rather than compression behaviour.)
        let body = body_of(serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}
                    ]
                }
            ]
        }));
        let out = compress_anthropic_request(
            &body,
            CompressionMode::LiveZone,
            CacheControlAutoFrozen::Disabled,
            "req-6",
        );
        match out {
            Outcome::NoCompression => {}
            other => panic!("expected NoCompression, got {other:?}"),
        }
    }
}

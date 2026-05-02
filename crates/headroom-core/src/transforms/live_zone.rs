//! Live-zone block dispatcher — Phase B PR-B2 skeleton.
//!
//! # The mental model
//!
//! After Phase B PR-B1 retired the message-dropping machinery, all
//! compression happens *within* messages, never *between* them. The
//! live-zone dispatcher walks the request body and identifies the
//! *live zone*: the blocks the model will emit a response *against*,
//! which are the only ones whose bytes can mutate without busting the
//! provider's prompt cache.
//!
//! For Anthropic `/v1/messages`, the live zone is bounded by:
//!
//! - **Floor:** `frozen_message_count` (computed by
//!   [`crate::compute_frozen_count`] from explicit `cache_control`
//!   markers; passed in here). Indices below the floor are in the
//!   prompt cache and MUST be byte-identical.
//! - **Ceiling:** the latest user message. The latest assistant
//!   message (if any) is part of the cache hot zone too — it's what
//!   the next response continues from. We never touch it.
//! - **Inside the latest user message:** every block is a candidate.
//!   The most common compressible block type is `tool_result`
//!   (because tool outputs dominate token budgets); `text` blocks
//!   are also eligible (e.g. user pastes a long log).
//!
//! # What this PR ships
//!
//! The dispatcher *skeleton*: identifies live-zone blocks and routes
//! them to per-type compressor functions. PR-B2 wires every per-type
//! function to a no-op, so [`compress_live_zone`] always returns
//! [`LiveZoneOutcome::NoChange`]. Subsequent PRs replace the no-ops:
//!
//! - **PR-B3** wires SmartCrusher, LogCompressor, SearchCompressor,
//!   DiffCompressor, CodeCompressor.
//! - **PR-B4** adds the tokenizer-validation gate (per-block
//!   `compressed.tokens >= original.tokens` → fall back) and the
//!   per-content-type byte threshold below which compression is
//!   skipped.
//! - **PR-B7** wires CCR retrieval-marker injection.
//!
//! # Cache safety invariant
//!
//! Bytes outside the live zone are NEVER touched. The
//! [`LiveZoneOutcome::Modified`] arm carries a freshly-serialized
//! body when (and only when) at least one block was actually
//! mutated; B2's no-op compressors never trigger this arm, so the
//! current implementation provably round-trips byte-for-byte.
//! Phase A's SHA-256 fixtures pin this in CI.
//!
//! # AuthMode
//!
//! The `AuthMode` parameter is taken in B2 but unused — Phase F
//! PR-F2 wires the gate (PAYG/OAuth/Subscription each demand
//! different policies; see project memory
//! `project_auth_mode_compression_nuances.md`). Keeping the
//! parameter in the signature now means later PRs are pure
//! implementation swaps, not signature redesigns.

use serde_json::value::RawValue;
use serde_json::Value;
use thiserror::Error;

/// Authentication mode of the originating request. Passed through to
/// the dispatcher so PR-F2 can vary policy without re-shaping the
/// public API. PR-B2 ignores the value (always treated as `Payg`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// Pay-as-you-go API key. Most aggressive compression budget —
    /// every saved token is real money for the customer.
    Payg,
    /// OAuth-bearing client (e.g. Anthropic.com OAuth). Compression
    /// must not break the per-account routing the OAuth header pins;
    /// otherwise behaves like PAYG.
    OAuth,
    /// Subscription seat (e.g. Claude.ai usage). The provider
    /// already counts tokens against a fixed quota; aggressive
    /// compression is less compelling and may interact badly with
    /// rate-limit accounting.
    Subscription,
}

/// Per-block decision recorded for observability. Independent of
/// whether the body was actually rewritten.
#[derive(Debug, Clone)]
pub struct BlockOutcome {
    /// Index into the `messages` array.
    pub message_index: usize,
    /// Index into the message's `content` array. `None` when the
    /// content is a plain string (Anthropic accepts both shapes).
    pub block_index: Option<usize>,
    /// Block kind detected on this slot. `text`, `tool_result`,
    /// `tool_use`, `image`, ... or `string_content` for the
    /// string-shaped fallback.
    pub block_type: String,
    /// What the dispatcher decided.
    pub action: BlockAction,
}

/// Disposition of one block.
#[derive(Debug, Clone)]
pub enum BlockAction {
    /// PR-B2: every supported block type currently lands here. The
    /// dispatcher inspected the block but no compressor wrote any
    /// bytes. Replaced in PR-B3 with per-type compressors.
    NoOpSkeleton,
    /// Block type is intentionally outside the live zone (e.g.
    /// `tool_use` → cache hot zone) and is excluded from dispatch.
    Excluded { reason: ExclusionReason },
}

/// Why a block was not eligible for compression.
#[derive(Debug, Clone, Copy)]
pub enum ExclusionReason {
    /// Block is in a message at index `< frozen_message_count`.
    BelowFrozenFloor,
    /// Block belongs to a message above the latest user message
    /// boundary (e.g. an older assistant turn).
    AboveLiveZone,
    /// Block type is on the cache-hot list (e.g. `tool_use`,
    /// `thinking`, `redacted_thinking`).
    HotZoneBlockType,
}

/// Aggregated per-request manifest. Always populated, regardless of
/// whether any bytes were written.
#[derive(Debug, Clone)]
pub struct CompressionManifest {
    /// Total messages in the input array. Matches
    /// `body.messages.len()`.
    pub messages_total: usize,
    /// Messages with index `< frozen_message_count`. Untouched.
    pub messages_below_frozen_floor: usize,
    /// Index of the latest user message in the live zone, if any.
    pub latest_user_message_index: Option<usize>,
    /// Per-block outcomes for the latest user message. Empty when
    /// the live zone has no eligible blocks (or the body has no
    /// messages).
    pub block_outcomes: Vec<BlockOutcome>,
}

impl CompressionManifest {
    fn empty() -> Self {
        Self {
            messages_total: 0,
            messages_below_frozen_floor: 0,
            latest_user_message_index: None,
            block_outcomes: Vec::new(),
        }
    }
}

/// Outcome of dispatching the live zone. Variants:
///
/// - [`LiveZoneOutcome::NoChange`] — caller forwards the original
///   bytes verbatim. PR-B2 always lands here.
/// - [`LiveZoneOutcome::Modified`] — caller forwards `new_body`.
///   PR-B3+ start producing this when per-type compressors mutate
///   blocks.
#[derive(Debug)]
pub enum LiveZoneOutcome {
    /// No bytes were rewritten. The caller must forward the original
    /// buffered request body byte-for-byte.
    NoChange { manifest: CompressionManifest },
    /// The dispatcher rewrote at least one block and emitted a fresh
    /// body. The caller forwards `new_body` upstream.
    Modified {
        new_body: Box<RawValue>,
        manifest: CompressionManifest,
    },
}

/// Compressor errors. Every variant is recoverable by the caller —
/// the proxy turns each into a structured warn-level log and
/// falls back to forwarding the original bytes.
#[derive(Debug, Error)]
pub enum LiveZoneError {
    /// The request body is not valid JSON. The proxy should log
    /// and forward the bytes as-is (the upstream provider will
    /// reject them with a parse error — that's the correct
    /// behaviour, not ours to mask).
    #[error("request body is not valid JSON: {0}")]
    BodyNotJson(serde_json::Error),
    /// `messages` field is missing or not a JSON array. Forward
    /// the bytes — the upstream may accept a body shape we don't
    /// recognize (e.g. a future Anthropic API revision).
    #[error("body has no `messages` array")]
    NoMessagesArray,
}

/// Block types the live-zone dispatcher considers "in the cache hot
/// zone" even when they appear inside a live-zone message. Listed
/// explicitly (no string-prefix matching) so the cache-safety
/// surface is grep-able.
const HOT_ZONE_BLOCK_TYPES: &[&str] = &[
    "tool_use",
    "thinking",
    "redacted_thinking",
    // Anthropic compaction items — once injected they're sticky to
    // the cache as much as `tool_use` is.
    "compaction",
];

/// Entry point: inspect a buffered Anthropic `/v1/messages` body and
/// decide which blocks (if any) to rewrite.
///
/// # Arguments
///
/// - `body_raw`: the buffered request body as bytes. Must be valid
///   UTF-8 JSON; non-JSON returns [`LiveZoneError::BodyNotJson`].
/// - `frozen_message_count`: hot-zone floor. Indices `< floor` are
///   excluded from dispatch.
/// - `_auth_mode`: reserved for PR-F2; B2 ignores it.
///
/// # Returns
///
/// - [`LiveZoneOutcome::NoChange`] (B2 always) when no block was
///   rewritten.
/// - [`LiveZoneOutcome::Modified`] (PR-B3+) when one or more blocks
///   were rewritten — the proxy forwards the new body.
pub fn compress_live_zone(
    body_raw: &[u8],
    frozen_message_count: usize,
    _auth_mode: AuthMode,
) -> Result<LiveZoneOutcome, LiveZoneError> {
    let parsed: Value = serde_json::from_slice(body_raw).map_err(LiveZoneError::BodyNotJson)?;
    let messages = parsed
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(LiveZoneError::NoMessagesArray)?;

    if messages.is_empty() {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest::empty(),
        });
    }

    let messages_total = messages.len();
    let messages_below_frozen_floor = frozen_message_count.min(messages_total);

    // Latest user message index, restricted to the live zone (>= floor).
    let latest_user_message_index = find_latest_user_message_index(messages, frozen_message_count);

    let block_outcomes = match latest_user_message_index {
        Some(idx) => inspect_latest_user_blocks(&messages[idx], idx),
        None => Vec::new(),
    };

    let manifest = CompressionManifest {
        messages_total,
        messages_below_frozen_floor,
        latest_user_message_index,
        block_outcomes,
    };

    // PR-B2: no compressor mutates anything, so the dispatcher
    // always lands on `NoChange`. PR-B3+ replaces this with a
    // per-block accumulator that tracks whether any byte was
    // actually rewritten.
    Ok(LiveZoneOutcome::NoChange { manifest })
}

/// Walk `messages` from the back, returning the index of the latest
/// `role == "user"` message. Restricted to indices `>= floor`; if
/// the latest user message lies in the cache hot zone we return
/// `None` (it's out of bounds for live-zone work).
fn find_latest_user_message_index(messages: &[Value], floor: usize) -> Option<usize> {
    let start = floor.min(messages.len());
    for (offset, msg) in messages.iter().enumerate().rev() {
        if offset < start {
            return None;
        }
        if msg.get("role").and_then(Value::as_str) == Some("user") {
            return Some(offset);
        }
    }
    None
}

/// Identify each block in the latest user message and tag it with
/// the dispatcher action it would receive. PR-B2: every dispatched
/// block lands on [`BlockAction::NoOpSkeleton`].
fn inspect_latest_user_blocks(message: &Value, message_index: usize) -> Vec<BlockOutcome> {
    let content = match message.get("content") {
        Some(c) => c,
        None => return Vec::new(),
    };

    // Anthropic accepts string-shaped content (legacy) and
    // array-of-blocks content (current). String content has no
    // sub-blocks; treat it as one synthetic "string_content" entry
    // so observability still records that the live zone exists.
    if let Some(_text) = content.as_str() {
        return vec![BlockOutcome {
            message_index,
            block_index: None,
            block_type: "string_content".to_string(),
            action: BlockAction::NoOpSkeleton,
        }];
    }

    let Some(blocks) = content.as_array() else {
        return Vec::new();
    };

    let mut outcomes = Vec::with_capacity(blocks.len());
    for (idx, block) in blocks.iter().enumerate() {
        let block_type = block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let action = if HOT_ZONE_BLOCK_TYPES.iter().any(|t| *t == block_type) {
            BlockAction::Excluded {
                reason: ExclusionReason::HotZoneBlockType,
            }
        } else {
            // PR-B2: every other block type routes to the no-op
            // dispatcher. PR-B3 replaces this branch with a real
            // per-type compressor switch (`tool_result` →
            // SmartCrusher / Log / Search / Diff / Code based on
            // content sniffing; `text` → SmartCrusher prose).
            BlockAction::NoOpSkeleton
        };
        outcomes.push(BlockOutcome {
            message_index,
            block_index: Some(idx),
            block_type,
            action,
        });
    }

    outcomes
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body(value: Value) -> Vec<u8> {
        serde_json::to_vec(&value).unwrap()
    }

    fn outcome_block_actions(o: &LiveZoneOutcome) -> Vec<&BlockAction> {
        let manifest = match o {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        manifest.block_outcomes.iter().map(|b| &b.action).collect()
    }

    #[test]
    fn empty_messages_yields_no_change() {
        let b = body(json!({"model": "claude", "messages": []}));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        match out {
            LiveZoneOutcome::NoChange { manifest } => {
                assert_eq!(manifest.messages_total, 0);
                assert_eq!(manifest.latest_user_message_index, None);
                assert!(manifest.block_outcomes.is_empty());
            }
            _ => panic!("expected NoChange"),
        }
    }

    #[test]
    fn no_messages_field_errors() {
        let b = body(json!({"model": "claude"}));
        let err = compress_live_zone(&b, 0, AuthMode::Payg).unwrap_err();
        assert!(matches!(err, LiveZoneError::NoMessagesArray));
    }

    #[test]
    fn invalid_json_errors() {
        let err = compress_live_zone(b"not json", 0, AuthMode::Payg).unwrap_err();
        assert!(matches!(err, LiveZoneError::BodyNotJson(_)));
    }

    #[test]
    fn dispatches_only_to_latest_user_message() {
        // Two user messages; the dispatcher must pick the second (index 2).
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "first user"},
                {"role": "assistant", "content": "first asst"},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                    {"type": "text", "text": "summarize"}
                ]},
            ]
        }));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, Some(2));
        let block_msg_indices: Vec<usize> = manifest
            .block_outcomes
            .iter()
            .map(|b| b.message_index)
            .collect();
        assert!(
            block_msg_indices.iter().all(|i| *i == 2),
            "all block outcomes must reference the latest user message; got {block_msg_indices:?}"
        );
    }

    #[test]
    fn respects_frozen_message_count() {
        // Latest user message is at index 1; floor is 2 → live zone is empty.
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": [{"type": "text", "text": "second"}]},
            ]
        }));
        let out = compress_live_zone(&b, 2, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, None);
        assert!(manifest.block_outcomes.is_empty());
        assert_eq!(manifest.messages_below_frozen_floor, 2);
    }

    #[test]
    fn excludes_hot_zone_block_types() {
        // tool_use inside a user message (uncommon shape but legal in
        // some assistants) must be tagged HotZoneBlockType.
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x"},
                    {"type": "thinking", "thinking": "...", "signature": "sig"},
                    {"type": "text", "text": "ok"},
                ]
            }]
        }));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let actions = outcome_block_actions(&out);
        assert_eq!(actions.len(), 3);
        assert!(matches!(actions[0], BlockAction::NoOpSkeleton));
        assert!(matches!(
            actions[1],
            BlockAction::Excluded {
                reason: ExclusionReason::HotZoneBlockType
            }
        ));
        assert!(matches!(actions[2], BlockAction::NoOpSkeleton));
    }

    #[test]
    fn string_content_message_records_synthetic_block() {
        let b = body(json!({
            "messages": [{"role": "user", "content": "just a string"}]
        }));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.block_outcomes.len(), 1);
        assert_eq!(manifest.block_outcomes[0].block_type, "string_content");
        assert!(matches!(
            manifest.block_outcomes[0].action,
            BlockAction::NoOpSkeleton
        ));
    }

    #[test]
    fn no_user_message_in_live_zone_returns_no_blocks() {
        // Only an assistant message → no live zone candidate.
        let b = body(json!({
            "messages": [{"role": "assistant", "content": "hi"}]
        }));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, None);
        assert!(manifest.block_outcomes.is_empty());
    }

    #[test]
    fn auth_mode_does_not_affect_b2_outcome() {
        // PR-F2 will wire policy; in B2 every mode behaves identically.
        let b = body(json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }));
        let payg = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let oauth = compress_live_zone(&b, 0, AuthMode::OAuth).unwrap();
        let sub = compress_live_zone(&b, 0, AuthMode::Subscription).unwrap();
        for o in [&payg, &oauth, &sub] {
            assert!(matches!(o, LiveZoneOutcome::NoChange { .. }));
        }
    }

    #[test]
    fn no_change_when_no_block_mutated_returns_original_semantics() {
        // PR-B2 invariant: dispatcher always returns NoChange.
        // PR-B3+ will start emitting Modified; this test pins the
        // current contract so an accidental early-Modified emission
        // is caught at compile time of the next phase.
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x".repeat(10_000)},
                ]
            }]
        }));
        let out = compress_live_zone(&b, 0, AuthMode::Payg).unwrap();
        assert!(matches!(out, LiveZoneOutcome::NoChange { .. }));
    }

    #[test]
    fn manifest_records_messages_below_floor() {
        let b = body(json!({
            "messages": [
                {"role": "user", "content": "frozen"},
                {"role": "assistant", "content": "frozen"},
                {"role": "user", "content": "live"},
            ]
        }));
        let out = compress_live_zone(&b, 2, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.messages_total, 3);
        assert_eq!(manifest.messages_below_frozen_floor, 2);
        assert_eq!(manifest.latest_user_message_index, Some(2));
    }

    #[test]
    fn frozen_count_above_messages_clamps() {
        // floor > total: clamped, no live zone.
        let b = body(json!({
            "messages": [{"role": "user", "content": "x"}]
        }));
        let out = compress_live_zone(&b, 99, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.messages_below_frozen_floor, 1);
        assert_eq!(manifest.latest_user_message_index, None);
    }
}

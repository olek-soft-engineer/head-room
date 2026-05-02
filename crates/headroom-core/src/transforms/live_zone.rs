//! Live-zone block dispatcher — Phase B.
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
//! # Provider scope
//!
//! Phase B ships ONE dispatcher entry point —
//! [`compress_anthropic_live_zone`] — that handles the Anthropic
//! Messages API shape (`/v1/messages`). Other providers (OpenAI
//! Chat Completions, OpenAI Responses, Google Gemini, Bedrock with
//! native payloads, …) need their own dispatchers because their
//! request shapes diverge in load-bearing ways:
//!
//! - OpenAI Chat Completions puts tool results in their own
//!   `role: "tool"` messages, not nested in user messages.
//! - OpenAI Responses uses `input` (not `messages`) with item types
//!   like `function_call_output` and `reasoning`.
//! - Gemini uses `contents`/`parts`/`function_response`.
//!
//! Phase C (`REALIGNMENT/05-phase-C-rust-proxy.md`) introduces
//! `compress_openai_chat_live_zone`, `compress_openai_responses_live_zone`,
//! and friends. They share this module's provider-agnostic types
//! ([`LiveZoneOutcome`], [`BlockAction`], [`CompressionManifest`])
//! and the per-content-type compressor backend, but each owns its
//! own walker.
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
//! # Phase B build-up
//!
//! - **PR-B2** shipped the dispatcher *skeleton*: identify live-zone
//!   blocks, route to no-op compressors, always return `NoChange`.
//! - **PR-B3** (this PR) wires per-content-type compressors:
//!   `JsonArray` → SmartCrusher; `BuildOutput` → LogCompressor;
//!   `SearchResults` → SearchCompressor; `GitDiff` → DiffCompressor;
//!   `SourceCode` / `PlainText` / `Html` → no-op (B4 + a Rust
//!   code-compressor port follow-up).
//! - **PR-B4** adds the tokenizer-validation gate (per-block
//!   `compressed.tokens >= original.tokens` → fall back) and the
//!   per-content-type byte threshold below which compression is
//!   skipped.
//! - **PR-B7** wires CCR retrieval-marker injection.
//!
//! # Cache safety invariant
//!
//! Bytes outside the live zone are NEVER touched. PR-B3 writes new
//! bodies via **byte-range surgery**: we locate each rewritten block
//! by pointer arithmetic on `serde_json::value::RawValue` borrowed
//! slices (which retain their offset into the original buffer), then
//! splice the replacement into the output. Concretely:
//!
//! ```text
//!     out = body[..block_start] || replacement || body[block_end..]
//! ```
//!
//! The bytes outside the rewritten ranges are *literally copied*
//! from the input, never re-serialized. This is how we guarantee
//! the SHA-256 of the prefix and suffix are byte-identical to the
//! input — Phase A's fixtures and B3's `byte_fidelity_outside_compressed_block`
//! test pin this in CI.
//!
//! Why byte-range surgery and not "deserialize → mutate → serialize"?
//! Re-serializing a JSON `Value` does not preserve original
//! whitespace, key order subtleties, or numeric formatting that the
//! provider may have already cached against. Byte-faithful copy of
//! everything we don't touch is the only way to guarantee
//! cache stability — see `project_compression_realignment_2026_05`.
//!
//! # AuthMode
//!
//! The `AuthMode` parameter is taken in B3 but unused — Phase F
//! PR-F2 wires the gate (PAYG/OAuth/Subscription each demand
//! different policies; see project memory
//! `project_auth_mode_compression_nuances.md`). Keeping the
//! parameter in the signature now means later PRs are pure
//! implementation swaps, not signature redesigns.

use std::sync::OnceLock;

use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::Value;
use thiserror::Error;

use super::content_detector::{detect_content_type, ContentType};
use super::diff_compressor::{DiffCompressor, DiffCompressorConfig};
use super::log_compressor::{LogCompressor, LogCompressorConfig};
use super::search_compressor::{SearchCompressor, SearchCompressorConfig};
use super::smart_crusher::{SmartCrusher, SmartCrusherConfig};

// ─── Tunable constants (no magic numbers in the dispatch logic) ────────

/// Strategy tag emitted when SmartCrusher rewrote a JSON-array block.
const STRATEGY_SMART_CRUSHER: &str = "smart_crusher";
/// Strategy tag emitted when LogCompressor rewrote a build-output / log block.
const STRATEGY_LOG_COMPRESSOR: &str = "log_compressor";
/// Strategy tag emitted when SearchCompressor rewrote a grep / ripgrep block.
const STRATEGY_SEARCH_COMPRESSOR: &str = "search_compressor";
/// Strategy tag emitted when DiffCompressor rewrote a unified-diff block.
const STRATEGY_DIFF_COMPRESSOR: &str = "diff_compressor";

/// Empty query context passed to compressors that take a relevance
/// query string. PR-B3 dispatcher does not yet plumb the user's last
/// prompt through; PR-F3 will.
const EMPTY_QUERY: &str = "";
/// Default relevance bias passed to scoring-aware compressors. Mirrors
/// the OSS-default behaviour ("no bias").
const DEFAULT_BIAS: f64 = 0.0;

// ─── Public types ──────────────────────────────────────────────────────

/// Authentication mode of the originating request. Passed through to
/// the dispatcher so PR-F2 can vary policy without re-shaping the
/// public API. PR-B3 ignores the value (always treated as `Payg`).
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
    /// Content type was inspected, no compressor was applicable.
    /// Examples: `PlainText` (Kompress wires in PR-B4), `SourceCode`
    /// (Rust code-compressor port pending), `Html` (no compressor),
    /// `Image` (binary), unknown shapes.
    NoCompressionApplied {
        /// String form of the detected content type — `"text"`,
        /// `"source_code"`, `"html"`, `"image"`, `"unknown"`, etc.
        content_type: String,
    },
    /// A compressor ran and produced a smaller output that was
    /// spliced into the body.
    Compressed {
        /// Identifier of the compressor (`"smart_crusher"`,
        /// `"log_compressor"`, ...). Static so the manifest is
        /// allocation-light.
        strategy: &'static str,
        /// Bytes of the original block content (the JSON string
        /// value, after unescaping).
        original_bytes: usize,
        /// Bytes of the replacement block content.
        compressed_bytes: usize,
    },
    /// A compressor was tried but failed loudly. Per project memory
    /// `feedback_no_silent_fallbacks.md`: surface the error in the
    /// manifest; the proxy logs warn-level and forwards the original
    /// bytes for that block (other blocks in the same body still get
    /// compressed normally).
    CompressorError {
        /// Identifier of the compressor that failed.
        strategy: &'static str,
        /// Human-readable error string (from `Display`).
        error: String,
    },
    /// A compressor ran but produced output >= original. Cache
    /// safety + "don't make it worse" → keep the original. PR-B4
    /// swaps this byte-length proxy for a real tokenizer count.
    RejectedNotSmaller {
        /// Identifier of the compressor that was rejected.
        strategy: &'static str,
        /// Original block-content size, bytes.
        original_bytes: usize,
        /// Would-be compressed-block-content size, bytes.
        compressed_bytes: usize,
    },
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

    /// True when at least one block was actually rewritten by a
    /// compressor (used to discriminate the `Modified` arm from
    /// `NoChange`).
    fn has_compressed_block(&self) -> bool {
        self.block_outcomes
            .iter()
            .any(|b| matches!(b.action, BlockAction::Compressed { .. }))
    }
}

/// Outcome of dispatching the live zone.
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

/// Dispatcher errors. Every variant is recoverable by the caller —
/// the proxy turns each into a structured warn-level log and
/// falls back to forwarding the original bytes.
#[derive(Debug, Error)]
pub enum LiveZoneError {
    /// The request body is not valid JSON.
    #[error("request body is not valid JSON: {0}")]
    BodyNotJson(serde_json::Error),
    /// `messages` field is missing or not a JSON array.
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

// ─── Compressor singletons ─────────────────────────────────────────────
//
// Each compressor's struct holds its config + (for SmartCrusher) the
// scoring infrastructure. Allocating one per request would be
// wasteful and (in SmartCrusher's case) defeats the purpose of the
// builder. Hold one instance per process behind `OnceLock`; cheap to
// clone the &reference each call.

fn smart_crusher() -> &'static SmartCrusher {
    static INSTANCE: OnceLock<SmartCrusher> = OnceLock::new();
    INSTANCE.get_or_init(|| SmartCrusher::new(SmartCrusherConfig::default()))
}

fn log_compressor() -> &'static LogCompressor {
    static INSTANCE: OnceLock<LogCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| LogCompressor::new(LogCompressorConfig::default()))
}

fn search_compressor() -> &'static SearchCompressor {
    static INSTANCE: OnceLock<SearchCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| SearchCompressor::new(SearchCompressorConfig::default()))
}

fn diff_compressor() -> &'static DiffCompressor {
    static INSTANCE: OnceLock<DiffCompressor> = OnceLock::new();
    INSTANCE.get_or_init(|| DiffCompressor::new(DiffCompressorConfig::default()))
}

// ─── Public entry point ────────────────────────────────────────────────

/// Inspect a buffered Anthropic `/v1/messages` body and decide which
/// blocks (if any) to rewrite.
///
/// # Provider scope (Phase B)
///
/// This function only handles the Anthropic Messages API shape:
///
/// - `messages: [{role, content}]`, with `content` either a JSON
///   string or an array of typed blocks (`text`, `tool_result`,
///   `tool_use`, `thinking`, `image`, …).
/// - The "live zone" is the latest `role == "user"` message at or
///   above `frozen_message_count`. Earlier messages are in the
///   prompt cache hot zone and are byte-preserved.
///
/// **Other providers need their own dispatchers** because their
/// request shapes diverge:
///
/// - **OpenAI Chat Completions** (`/v1/chat/completions`) — tool
///   results live in their own `role: "tool"` messages, not nested
///   in user messages. The live zone is the trailing run of
///   `tool` messages plus the latest `user` message.
/// - **OpenAI Responses API** (`/v1/responses`) — the request is
///   keyed under `input` (not `messages`) with item types like
///   `function_call_output` and `reasoning`; live zone is the
///   trailing function-call-output items since the last `message`
///   or `reasoning` item.
/// - **Google Gemini** (`/v1beta/.../:generateContent`) — request
///   is keyed under `contents` (not `messages`), with
///   `function_response` parts (not `tool_result`). Function
///   responses can be either string or structured object.
/// - **Bedrock InvokeModel** — the embedded payload follows the
///   model's native format (Anthropic, Llama, Cohere, …); route
///   to the matching dispatcher.
///
/// Phase C (`REALIGNMENT/05-phase-C-rust-proxy.md`) introduces the
/// per-provider dispatchers. Each will live as
/// `compress_<provider>_live_zone` and share the cache-safety
/// invariants and the per-content-type compressor backend
/// (SmartCrusher / LogCompressor / SearchCompressor /
/// DiffCompressor / Code) from this module. The
/// [`LiveZoneOutcome`], [`BlockAction`], and
/// [`CompressionManifest`] types are intentionally
/// provider-agnostic so the per-provider dispatchers can return
/// them unchanged.
///
/// # Arguments
///
/// - `body_raw`: the buffered request body as bytes. Must be valid
///   UTF-8 JSON; non-JSON returns [`LiveZoneError::BodyNotJson`].
/// - `frozen_message_count`: hot-zone floor. Indices `< floor` are
///   excluded from dispatch.
/// - `_auth_mode`: reserved for PR-F2; B3 ignores it.
///
/// # Returns
///
/// - [`LiveZoneOutcome::NoChange`] when no block was rewritten
///   (either nothing was eligible, or every compressor declined /
///   failed / produced larger output).
/// - [`LiveZoneOutcome::Modified`] when at least one block was
///   rewritten — the proxy forwards the new body.
pub fn compress_anthropic_live_zone(
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

    let Some(target_idx) = latest_user_message_index else {
        return Ok(LiveZoneOutcome::NoChange {
            manifest: CompressionManifest {
                messages_total,
                messages_below_frozen_floor,
                latest_user_message_index: None,
                block_outcomes: Vec::new(),
            },
        });
    };

    // Resolve block ranges (byte offsets into `body_raw`) by walking
    // the body via `RawValue` borrowed slices. The Vec<Replacement>
    // produced here is the surgery plan; we do *not* mutate `body_raw`
    // while computing it.
    let plan = match plan_block_replacements(body_raw, target_idx) {
        Ok(p) => p,
        Err(_) => {
            // Body shape doesn't match what we expect (e.g. content
            // is not a string and not an array, or messages is shaped
            // unexpectedly). Treat as no-change; the proxy forwards
            // the original bytes verbatim.
            let block_outcomes =
                inspect_latest_user_blocks_value(&messages[target_idx], target_idx)
                    .unwrap_or_default();
            return Ok(LiveZoneOutcome::NoChange {
                manifest: CompressionManifest {
                    messages_total,
                    messages_below_frozen_floor,
                    latest_user_message_index: Some(target_idx),
                    block_outcomes,
                },
            });
        }
    };

    let mut block_outcomes: Vec<BlockOutcome> = Vec::with_capacity(plan.len());
    let mut replacements: Vec<Replacement> = Vec::new();

    for slot in plan {
        let outcome = match slot.kind {
            SlotKind::HotZone(block_type) => BlockOutcome {
                message_index: target_idx,
                block_index: Some(slot.block_index),
                block_type,
                action: BlockAction::Excluded {
                    reason: ExclusionReason::HotZoneBlockType,
                },
            },
            SlotKind::Compressible {
                block_type,
                content_text,
                content_byte_range,
            } => {
                let detected = detect_content_type(&content_text);
                let _detected_tag = detected.content_type.as_str();
                let outcome: BlockOutcome =
                    match dispatch_compressor(&content_text, detected.content_type) {
                        DispatchResult::NoOp { content_type } => BlockOutcome {
                            message_index: target_idx,
                            block_index: Some(slot.block_index),
                            block_type,
                            action: BlockAction::NoCompressionApplied {
                                content_type: content_type.to_string(),
                            },
                        },
                        DispatchResult::Compressed {
                            strategy,
                            compressed,
                        } => {
                            let original_bytes = content_text.len();
                            let compressed_bytes = compressed.len();
                            if compressed_bytes >= original_bytes {
                                // Byte-length gate (PR-B4 will replace
                                // with a token-count gate). Without this
                                // the dispatcher could ship a "compressed"
                                // body bigger than the input.
                                BlockOutcome {
                                    message_index: target_idx,
                                    block_index: Some(slot.block_index),
                                    block_type,
                                    action: BlockAction::RejectedNotSmaller {
                                        strategy,
                                        original_bytes,
                                        compressed_bytes,
                                    },
                                }
                            } else {
                                // Encode replacement as a JSON string and
                                // record it for the splice.
                                let replacement_bytes = serde_json::to_vec(&compressed)
                                    .expect("string is always JSON-encodable");
                                replacements.push(Replacement {
                                    range: content_byte_range,
                                    replacement: replacement_bytes,
                                });
                                BlockOutcome {
                                    message_index: target_idx,
                                    block_index: Some(slot.block_index),
                                    block_type,
                                    action: BlockAction::Compressed {
                                        strategy,
                                        original_bytes,
                                        compressed_bytes,
                                    },
                                }
                            }
                        }
                        DispatchResult::Error { strategy, error } => BlockOutcome {
                            message_index: target_idx,
                            block_index: Some(slot.block_index),
                            block_type,
                            action: BlockAction::CompressorError { strategy, error },
                        },
                    };
                outcome
            }
            SlotKind::StringContent {
                content_text,
                content_byte_range,
            } => {
                let detected = detect_content_type(&content_text);
                match dispatch_compressor(&content_text, detected.content_type) {
                    DispatchResult::NoOp { content_type } => BlockOutcome {
                        message_index: target_idx,
                        block_index: None,
                        block_type: "string_content".to_string(),
                        action: BlockAction::NoCompressionApplied {
                            content_type: content_type.to_string(),
                        },
                    },
                    DispatchResult::Compressed {
                        strategy,
                        compressed,
                    } => {
                        let original_bytes = content_text.len();
                        let compressed_bytes = compressed.len();
                        if compressed_bytes >= original_bytes {
                            BlockOutcome {
                                message_index: target_idx,
                                block_index: None,
                                block_type: "string_content".to_string(),
                                action: BlockAction::RejectedNotSmaller {
                                    strategy,
                                    original_bytes,
                                    compressed_bytes,
                                },
                            }
                        } else {
                            let replacement_bytes = serde_json::to_vec(&compressed)
                                .expect("string is always JSON-encodable");
                            replacements.push(Replacement {
                                range: content_byte_range,
                                replacement: replacement_bytes,
                            });
                            BlockOutcome {
                                message_index: target_idx,
                                block_index: None,
                                block_type: "string_content".to_string(),
                                action: BlockAction::Compressed {
                                    strategy,
                                    original_bytes,
                                    compressed_bytes,
                                },
                            }
                        }
                    }
                    DispatchResult::Error { strategy, error } => BlockOutcome {
                        message_index: target_idx,
                        block_index: None,
                        block_type: "string_content".to_string(),
                        action: BlockAction::CompressorError { strategy, error },
                    },
                }
            }
        };
        block_outcomes.push(outcome);
    }

    let manifest = CompressionManifest {
        messages_total,
        messages_below_frozen_floor,
        latest_user_message_index: Some(target_idx),
        block_outcomes,
    };

    if !manifest.has_compressed_block() || replacements.is_empty() {
        return Ok(LiveZoneOutcome::NoChange { manifest });
    }

    // Build the new body via byte-range surgery. Replacements are
    // produced in ascending block order; sort defensively.
    let new_bytes = apply_replacements(body_raw, &mut replacements);

    // The output is always still valid JSON: every replacement is a
    // JSON string slot replaced by another JSON string slot. We could
    // round-trip-verify with `serde_json::from_slice` and bail out to
    // NoChange on failure, but that doubles parse cost on the hot
    // path. Rely on type discipline; the byte_fidelity test in
    // `live_zone_dispatch.rs` pins correctness.
    let new_body_str = match std::str::from_utf8(&new_bytes) {
        Ok(s) => s,
        Err(_) => {
            // Should be impossible: input was valid JSON (UTF-8) and
            // every replacement was a JSON-encoded string (also UTF-8).
            // Fall back rather than risk shipping malformed bytes.
            return Ok(LiveZoneOutcome::NoChange { manifest });
        }
    };
    let raw = match RawValue::from_string(new_body_str.to_string()) {
        Ok(r) => r,
        Err(_) => {
            // Same defensive bail-out; should not happen.
            return Ok(LiveZoneOutcome::NoChange { manifest });
        }
    };

    Ok(LiveZoneOutcome::Modified {
        new_body: raw,
        manifest,
    })
}

// ─── Internal helpers ──────────────────────────────────────────────────

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

/// Body-shape view used to find byte ranges.
///
/// `&'a RawValue` borrows are pointer-equal to slices into the input
/// buffer; we use this to compute exact byte offsets via the
/// `bytes_offset_of` helper. The struct intentionally only captures
/// the path we need; everything else is left unparsed.
#[derive(Deserialize)]
struct BodyView<'a> {
    #[serde(borrow)]
    messages: Vec<&'a RawValue>,
}

#[derive(Deserialize)]
struct MessageView<'a> {
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct BlockHeader<'a> {
    #[serde(borrow, default)]
    r#type: Option<&'a str>,
    #[serde(borrow, default)]
    content: Option<&'a RawValue>,
}

/// Per-block dispatch slot the planner emits.
struct PlanSlot {
    block_index: usize,
    kind: SlotKind,
}

enum SlotKind {
    /// Content is a JSON string the dispatcher may compress in place.
    Compressible {
        block_type: String,
        content_text: String,
        content_byte_range: (usize, usize),
    },
    /// String-shaped message content (Anthropic legacy shape: the
    /// whole message's `content` is a JSON string, no per-block
    /// array).
    StringContent {
        content_text: String,
        content_byte_range: (usize, usize),
    },
    /// Block type is on the cache-hot list — record but do not
    /// dispatch.
    HotZone(String),
}

/// Walk the buffered body, return one `PlanSlot` per block in the
/// latest user message. Errors out on shapes the dispatcher does not
/// support (e.g. structured-array `content` inside a tool_result —
/// rare; we degrade to NoChange in that case).
fn plan_block_replacements(
    body_raw: &[u8],
    target_msg_idx: usize,
) -> Result<Vec<PlanSlot>, PlanError> {
    // `serde_json::from_slice` requires UTF-8; we re-validate here
    // explicitly so the pointer-arithmetic helper can take a `&str`
    // without unsafe.
    let body_str = std::str::from_utf8(body_raw).map_err(|_| PlanError::ParseFailed)?;
    let body: BodyView<'_> = serde_json::from_str(body_str).map_err(|_| PlanError::ParseFailed)?;
    let target_msg_raw = body
        .messages
        .get(target_msg_idx)
        .ok_or(PlanError::TargetOutOfBounds)?;

    let msg_view: MessageView<'_> =
        serde_json::from_str(target_msg_raw.get()).map_err(|_| PlanError::ParseFailed)?;

    let Some(content_raw) = msg_view.content else {
        return Ok(Vec::new());
    };

    // Compute the byte offset of the message's `content` value into
    // `body_raw`. The target_msg_raw points into body_raw; content_raw
    // points into target_msg_raw's bytes (which are the same backing
    // memory).
    let content_offset_in_msg =
        bytes_offset_of(target_msg_raw.get(), content_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let msg_offset_in_body =
        bytes_offset_of(body_str, target_msg_raw.get()).ok_or(PlanError::OffsetMissing)?;
    let content_offset_in_body = msg_offset_in_body + content_offset_in_msg;

    let content_str = content_raw.get();

    // Case 1: content is a JSON string (Anthropic legacy shape for
    // user messages).
    if content_str.starts_with('"') {
        let unescaped: String =
            serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;
        return Ok(vec![PlanSlot {
            block_index: 0,
            kind: SlotKind::StringContent {
                content_text: unescaped,
                content_byte_range: (
                    content_offset_in_body,
                    content_offset_in_body + content_str.len(),
                ),
            },
        }]);
    }

    // Case 2: content is an array of blocks. Borrow each block as a
    // &RawValue so we can compute its byte range too.
    let blocks: Vec<&RawValue> =
        serde_json::from_str(content_str).map_err(|_| PlanError::ParseFailed)?;

    let mut slots = Vec::with_capacity(blocks.len());
    for (block_idx, block_raw) in blocks.iter().enumerate() {
        let block_offset_in_content =
            bytes_offset_of(content_str, block_raw.get()).ok_or(PlanError::OffsetMissing)?;
        let block_offset_in_body = content_offset_in_body + block_offset_in_content;

        let header: BlockHeader<'_> =
            serde_json::from_str(block_raw.get()).map_err(|_| PlanError::ParseFailed)?;
        let block_type = header.r#type.unwrap_or("unknown").to_string();

        if HOT_ZONE_BLOCK_TYPES.iter().any(|t| *t == block_type) {
            slots.push(PlanSlot {
                block_index: block_idx,
                kind: SlotKind::HotZone(block_type),
            });
            continue;
        }

        // Find the inner `content` field's byte range. For tool_result
        // blocks this is the field we'd compress. For text blocks
        // it's a `text` field — we read that instead.
        let (inner_field_str, inner_field_offset_in_block) = match block_type.as_str() {
            "tool_result" => {
                let Some(field_raw) = header.content else {
                    // tool_result with no content — skip dispatch.
                    slots.push(PlanSlot {
                        block_index: block_idx,
                        kind: SlotKind::Compressible {
                            block_type,
                            content_text: String::new(),
                            content_byte_range: (block_offset_in_body, block_offset_in_body),
                        },
                    });
                    continue;
                };
                let off = bytes_offset_of(block_raw.get(), field_raw.get())
                    .ok_or(PlanError::OffsetMissing)?;
                (field_raw.get(), off)
            }
            "text" => {
                #[derive(Deserialize)]
                struct TextHeader<'a> {
                    #[serde(borrow, default)]
                    text: Option<&'a RawValue>,
                }
                let h: TextHeader<'_> =
                    serde_json::from_str(block_raw.get()).map_err(|_| PlanError::ParseFailed)?;
                let Some(text_raw) = h.text else {
                    slots.push(PlanSlot {
                        block_index: block_idx,
                        kind: SlotKind::Compressible {
                            block_type,
                            content_text: String::new(),
                            content_byte_range: (block_offset_in_body, block_offset_in_body),
                        },
                    });
                    continue;
                };
                let off = bytes_offset_of(block_raw.get(), text_raw.get())
                    .ok_or(PlanError::OffsetMissing)?;
                (text_raw.get(), off)
            }
            _ => {
                // image, document, etc. — record as compressible
                // block-type but with empty content so no compressor
                // runs.
                slots.push(PlanSlot {
                    block_index: block_idx,
                    kind: SlotKind::Compressible {
                        block_type,
                        content_text: String::new(),
                        content_byte_range: (block_offset_in_body, block_offset_in_body),
                    },
                });
                continue;
            }
        };

        // The compressors expect a plain string, not a JSON-quoted
        // string. `tool_result.content` and `text.text` are
        // either a JSON string or a structured array; we only
        // compress the string shape (B3). Structured-array shape
        // falls through to no-op.
        if !inner_field_str.starts_with('"') {
            slots.push(PlanSlot {
                block_index: block_idx,
                kind: SlotKind::Compressible {
                    block_type,
                    content_text: String::new(),
                    content_byte_range: (block_offset_in_body, block_offset_in_body),
                },
            });
            continue;
        }
        let unescaped: String =
            serde_json::from_str(inner_field_str).map_err(|_| PlanError::ParseFailed)?;

        let inner_field_start_in_body = block_offset_in_body + inner_field_offset_in_block;
        let inner_field_end_in_body = inner_field_start_in_body + inner_field_str.len();

        slots.push(PlanSlot {
            block_index: block_idx,
            kind: SlotKind::Compressible {
                block_type,
                content_text: unescaped,
                content_byte_range: (inner_field_start_in_body, inner_field_end_in_body),
            },
        });
    }

    Ok(slots)
}

#[derive(Debug)]
enum PlanError {
    /// JSON parse failure on a body-shape view we expected to succeed.
    ParseFailed,
    /// Pointer-arithmetic could not locate a sub-slice's offset.
    /// Should not happen for valid JSON; surfacing rather than
    /// silently degrading.
    OffsetMissing,
    /// Latest-user-message index points past the end of `messages`.
    /// The caller already validated this — surfacing for safety.
    TargetOutOfBounds,
}

/// Compute the byte offset of `child` within `parent` when both are
/// `&str` views into the same backing memory. Returns `None` when
/// `child` does not lie strictly inside `parent`.
///
/// We rely on this trick because `serde_json` does not expose the
/// byte offset of a `&RawValue`; the `RawValue::get()` slice points
/// into the input buffer when `from_slice` / `from_str` was used,
/// so pointer arithmetic recovers it.
fn bytes_offset_of(parent: &str, child: &str) -> Option<usize> {
    let parent_start = parent.as_ptr() as usize;
    let parent_end = parent_start + parent.len();
    let child_start = child.as_ptr() as usize;
    if child_start < parent_start || child_start + child.len() > parent_end {
        return None;
    }
    Some(child_start - parent_start)
}

/// One byte-range replacement to apply. Sorted in ascending `range.0`
/// before splicing.
struct Replacement {
    range: (usize, usize),
    replacement: Vec<u8>,
}

/// Apply all `replacements` to `original`, returning the new buffer.
/// `replacements` are sorted in-place by ascending start offset; the
/// caller may inspect them post-call (they remain valid).
fn apply_replacements(original: &[u8], replacements: &mut [Replacement]) -> Vec<u8> {
    replacements.sort_by_key(|r| r.range.0);

    // Pre-size: original_len - sum(removed) + sum(replacement_len).
    let removed: usize = replacements.iter().map(|r| r.range.1 - r.range.0).sum();
    let added: usize = replacements.iter().map(|r| r.replacement.len()).sum();
    let mut out = Vec::with_capacity(original.len().saturating_sub(removed) + added);

    let mut cursor = 0usize;
    for r in replacements.iter() {
        out.extend_from_slice(&original[cursor..r.range.0]);
        out.extend_from_slice(&r.replacement);
        cursor = r.range.1;
    }
    out.extend_from_slice(&original[cursor..]);
    out
}

/// Per-block dispatch result — whether any compressor ran and what
/// it produced.
enum DispatchResult {
    /// No compressor was applicable for this content type.
    NoOp { content_type: &'static str },
    /// A compressor ran and produced a candidate replacement string.
    Compressed {
        strategy: &'static str,
        compressed: String,
    },
    /// A compressor ran and failed loudly. The error string is
    /// surfaced via the manifest; the proxy logs it.
    #[allow(dead_code)]
    Error {
        strategy: &'static str,
        error: String,
    },
}

/// Map `(text, content_type)` to the compressor result.
///
/// Per spec PR-B3:
///
/// - `JsonArray` (with `is_dict_array=true`) → SmartCrusher
/// - `BuildOutput` → LogCompressor
/// - `SearchResults` → SearchCompressor
/// - `GitDiff` → DiffCompressor
/// - `SourceCode` → no-op (Rust port pending; see TODO below)
/// - `PlainText` → no-op (PR-B4 wires Kompress)
/// - `Html` → no-op (no compressor)
fn dispatch_compressor(text: &str, content_type: ContentType) -> DispatchResult {
    if text.is_empty() {
        return DispatchResult::NoOp {
            content_type: content_type.as_str(),
        };
    }

    match content_type {
        ContentType::JsonArray => {
            // The detector classifies arrays-of-scalars as JsonArray
            // too (confidence 0.8). SmartCrusher's `crush` is safe to
            // call on those — it parses, finds no compressible
            // arrays, and returns the input.
            let result = smart_crusher().crush(text, EMPTY_QUERY, DEFAULT_BIAS);
            if !result.was_modified {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_SMART_CRUSHER,
                compressed: result.compressed,
            }
        }
        ContentType::BuildOutput => {
            let (result, _stats) = log_compressor().compress(text, DEFAULT_BIAS);
            if result.compressed == result.original {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_LOG_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        ContentType::SearchResults => {
            let (result, _stats) = search_compressor().compress(text, EMPTY_QUERY, DEFAULT_BIAS);
            if result.compressed == result.original {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_SEARCH_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        ContentType::GitDiff => {
            let result = diff_compressor().compress(text, EMPTY_QUERY);
            if result.compressed == text {
                return DispatchResult::NoOp {
                    content_type: content_type.as_str(),
                };
            }
            DispatchResult::Compressed {
                strategy: STRATEGY_DIFF_COMPRESSOR,
                compressed: result.compressed,
            }
        }
        // TODO(PR-B4 / Rust code-compressor port): Python has a
        // CodeAwareCompressor; the Rust port is not yet shipped. Once
        // that crate lands, `ContentType::SourceCode` routes here
        // exactly as the others above.
        ContentType::SourceCode => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
        // TODO(PR-B4): wire Kompress (lossless prose compressor) for
        // PlainText. For now, leave untouched.
        ContentType::PlainText => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
        // No HTML compressor on the Rust side; pages are handled by
        // upstream extractors, not the proxy.
        ContentType::Html => DispatchResult::NoOp {
            content_type: content_type.as_str(),
        },
    }
}

/// Fallback when byte-range planning fails: still record per-block
/// outcomes so observability covers the request. Mirrors PR-B2's
/// observation-only path.
fn inspect_latest_user_blocks_value(
    message: &Value,
    message_index: usize,
) -> Option<Vec<BlockOutcome>> {
    let content = message.get("content")?;

    if content.as_str().is_some() {
        return Some(vec![BlockOutcome {
            message_index,
            block_index: None,
            block_type: "string_content".to_string(),
            action: BlockAction::NoCompressionApplied {
                content_type: "text".to_string(),
            },
        }]);
    }

    let blocks = content.as_array()?;
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
            BlockAction::NoCompressionApplied {
                content_type: "unknown".to_string(),
            }
        };
        outcomes.push(BlockOutcome {
            message_index,
            block_index: Some(idx),
            block_type,
            action,
        });
    }
    Some(outcomes)
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
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
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
        let err = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap_err();
        assert!(matches!(err, LiveZoneError::NoMessagesArray));
    }

    #[test]
    fn invalid_json_errors() {
        let err = compress_anthropic_live_zone(b"not json", 0, AuthMode::Payg).unwrap_err();
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
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
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
        let out = compress_anthropic_live_zone(&b, 2, AuthMode::Payg).unwrap();
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
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let actions = outcome_block_actions(&out);
        assert_eq!(actions.len(), 3);
        // tool_result with tiny content → NoCompressionApplied (under threshold / detector returns plain text).
        assert!(matches!(
            actions[0],
            BlockAction::NoCompressionApplied { .. }
        ));
        assert!(matches!(
            actions[1],
            BlockAction::Excluded {
                reason: ExclusionReason::HotZoneBlockType
            }
        ));
        assert!(matches!(
            actions[2],
            BlockAction::NoCompressionApplied { .. }
        ));
    }

    #[test]
    fn string_content_message_records_synthetic_block() {
        let b = body(json!({
            "messages": [{"role": "user", "content": "just a string"}]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        assert_eq!(manifest.block_outcomes.len(), 1);
        assert_eq!(manifest.block_outcomes[0].block_type, "string_content");
        assert!(matches!(
            manifest.block_outcomes[0].action,
            BlockAction::NoCompressionApplied { .. }
        ));
    }

    #[test]
    fn no_user_message_in_live_zone_returns_no_blocks() {
        let b = body(json!({
            "messages": [{"role": "assistant", "content": "hi"}]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.latest_user_message_index, None);
        assert!(manifest.block_outcomes.is_empty());
    }

    #[test]
    fn auth_mode_does_not_affect_b3_outcome_for_short_input() {
        // Trivial input → every mode behaves identically.
        let b = body(json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }));
        let payg = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
        let oauth = compress_anthropic_live_zone(&b, 0, AuthMode::OAuth).unwrap();
        let sub = compress_anthropic_live_zone(&b, 0, AuthMode::Subscription).unwrap();
        for o in [&payg, &oauth, &sub] {
            assert!(matches!(o, LiveZoneOutcome::NoChange { .. }));
        }
    }

    #[test]
    fn no_change_when_input_already_minimal_returns_original_semantics() {
        // tiny tool_result → detected as plain text, no-op
        // dispatch → NoChange.
        let b = body(json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": "x"},
                ]
            }]
        }));
        let out = compress_anthropic_live_zone(&b, 0, AuthMode::Payg).unwrap();
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
        let out = compress_anthropic_live_zone(&b, 2, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            LiveZoneOutcome::Modified { manifest, .. } => manifest,
        };
        assert_eq!(manifest.messages_total, 3);
        assert_eq!(manifest.messages_below_frozen_floor, 2);
        assert_eq!(manifest.latest_user_message_index, Some(2));
    }

    #[test]
    fn frozen_count_above_messages_clamps() {
        let b = body(json!({
            "messages": [{"role": "user", "content": "x"}]
        }));
        let out = compress_anthropic_live_zone(&b, 99, AuthMode::Payg).unwrap();
        let manifest = match &out {
            LiveZoneOutcome::NoChange { manifest } => manifest,
            _ => panic!("expected NoChange"),
        };
        assert_eq!(manifest.messages_below_frozen_floor, 1);
        assert_eq!(manifest.latest_user_message_index, None);
    }
}

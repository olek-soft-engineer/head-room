//! Compression pipeline traits + supporting types.
//!
//! Two traits, one shared result type, one per-call context object,
//! one error type. That's the entire surface — every piece earns its
//! place by being needed by either an impl in this PR (JsonMinifier,
//! LineImportanceFilter) or the orchestrator that runs them.

use crate::transforms::ContentType;

/// Errors a transform can return.
///
/// `InvalidInput` is for malformed input the transform can't parse
/// (e.g. JsonMinifier on non-JSON). `Skipped` is for "I ran cleanly
/// but found nothing to do" — used when an early exit is the right
/// call but isn't actually an error. `Internal` is for serializer /
/// allocator / logic-bug failures the orchestrator should record but
/// not crash on.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    /// The transform couldn't parse the input. Orchestrator skips and
    /// tries the next transform.
    #[error("invalid input for {transform}: {message}")]
    InvalidInput {
        transform: &'static str,
        message: String,
    },
    /// The transform decided not to run (e.g. empty input, content
    /// already minimal). Orchestrator skips silently.
    #[error("{transform} skipped: {message}")]
    Skipped {
        transform: &'static str,
        message: String,
    },
    /// Internal failure (serializer, allocator, logic bug). Surfaces
    /// to logs but does not abort the pipeline.
    #[error("{transform} internal error: {message}")]
    Internal {
        transform: &'static str,
        message: String,
    },
}

impl TransformError {
    pub fn invalid_input(transform: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidInput {
            transform,
            message: message.into(),
        }
    }

    pub fn skipped(transform: &'static str, message: impl Into<String>) -> Self {
        Self::Skipped {
            transform,
            message: message.into(),
        }
    }

    pub fn internal(transform: &'static str, message: impl Into<String>) -> Self {
        Self::Internal {
            transform,
            message: message.into(),
        }
    }
}

/// Result of a single transform invocation.
///
/// Every successful run produces an owned `output` string and reports
/// how much was saved. The orchestrator decides whether the savings
/// pass its acceptance threshold; transforms that didn't actually
/// shrink anything return `bytes_saved == 0` and the orchestrator
/// rejects the result.
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// Compressed (or pass-through) output. Always owned because
    /// transforms typically produce new strings; for the rare
    /// pass-through case the allocation is unavoidable but tiny.
    pub output: String,
    /// `original.len() - output.len()`, clamped to 0. Negative
    /// "savings" (output longer than input) are normalized to 0 here
    /// and the orchestrator's [`is_acceptable`] check rejects them.
    ///
    /// [`is_acceptable`]: super::orchestrator::CompressionPipeline::is_acceptable
    pub bytes_saved: usize,
    /// True iff the transform preserves structural invariants. JSON
    /// minification preserves shape → true. A code-comment stripper
    /// preserves the token stream's semantic meaning → true. Filtering
    /// out lines a relevance scorer ranked low → false (lines are
    /// gone, the LLM can't reconstruct them without CCR).
    ///
    /// Used by the orchestrator (and downstream PR3+ CCR-emission
    /// logic) to decide whether to attach a retrieval marker.
    pub structure_preserved: bool,
    /// CCR cache key when the transform deliberately offloaded
    /// content. `None` means the output is self-contained — no
    /// retrieval handle needed. PR1 transforms never set this; the
    /// field exists for PR2/PR3 wrappers around DiffCompressor /
    /// LogCompressor / SmartCrusher which do emit CCR markers.
    pub reversible_via: Option<String>,
}

impl TransformResult {
    /// Construct a result whose `bytes_saved` is computed from the
    /// difference between input and output lengths.
    pub fn from_lengths(input_len: usize, output: String, structure_preserved: bool) -> Self {
        let bytes_saved = input_len.saturating_sub(output.len());
        Self {
            output,
            bytes_saved,
            structure_preserved,
            reversible_via: None,
        }
    }
}

/// Per-call context the orchestrator hands to each transform.
///
/// Holds anything a transform might need that *isn't* the input
/// content — query for relevance scoring, target token budget for
/// aggressiveness tuning. Transforms borrow the context for the
/// duration of their `apply` call.
#[derive(Debug, Default, Clone)]
pub struct CompressionContext {
    /// The user's question for the LLM. Empty string means no query
    /// available; transforms must treat this as a generic compression
    /// pass (no relevance bias).
    pub query: String,
    /// Token budget the orchestrator is targeting, in tokens (not
    /// bytes). `None` means "compress as much as is safe — caller
    /// will accept whatever comes out."
    pub token_budget: Option<usize>,
}

impl CompressionContext {
    pub fn with_query(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            token_budget: None,
        }
    }

    pub fn with_budget(token_budget: usize) -> Self {
        Self {
            query: String::new(),
            token_budget: Some(token_budget),
        }
    }
}

/// Compression that preserves all information (the LLM gets the same
/// bytes back, just packed denser).
///
/// Examples (in this PR or PR2+): JSON minification, code-comment
/// stripping, log RLE, HTML extraction. These run first — if the
/// result fits the budget, no lossy work is needed.
pub trait LosslessTransform: Send + Sync {
    /// Stable telemetry name, e.g. `"json_minifier"`. Used as the
    /// strategy key in the per-strategy stats nest landed in
    /// Phase 3e.0. Must match `^[a-z][a-z0-9_]*$` by convention so
    /// it composes cleanly into JSONB keys.
    fn name(&self) -> &'static str;

    /// Content types this transform accepts. The orchestrator skips
    /// the transform when the input's detected type isn't in this
    /// slice. Returning a `&[ContentType]` borrowed from `&self`
    /// lets impls store either a static literal or an owned config-
    /// driven list.
    fn applies_to(&self) -> &[ContentType];

    /// Run the transform. Returns `Err(TransformError)` for malformed
    /// input or internal failures (orchestrator skips); returns
    /// `Ok(TransformResult)` for successful runs even when nothing
    /// was saved (the orchestrator decides what to do based on
    /// `bytes_saved`).
    fn apply(&self, content: &str) -> Result<TransformResult, TransformError>;
}

/// Compression that deliberately drops content to fit a budget.
///
/// Examples (in this PR or PR2+): line-importance filtering, prose-
/// field compression via a small model, extractive summarization.
/// These run after lossless and gate further passes on byte savings.
pub trait LossyTransform: Send + Sync {
    /// Stable telemetry name (same convention as
    /// [`LosslessTransform::name`]).
    fn name(&self) -> &'static str;

    /// Content types this transform accepts.
    fn applies_to(&self) -> &[ContentType];

    /// Run the transform. Same error semantics as
    /// [`LosslessTransform::apply`]. The `ctx` argument lets the
    /// transform read the query (for relevance) and budget (for
    /// aggressiveness) without taking on tokenizer or proxy state.
    fn apply(
        &self,
        content: &str,
        ctx: &CompressionContext,
    ) -> Result<TransformResult, TransformError>;

    /// Calibrated 0.0–1.0 quality score. Higher = more confident the
    /// output preserves enough information for the LLM to answer the
    /// query. The orchestrator records this in telemetry; selection
    /// between competing lossy transforms (PR4 ProseFieldCompressor
    /// vs LineImportanceFilter on the same content type) becomes a
    /// future extension. For now it's purely observational.
    fn confidence(&self) -> f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One throwaway impl per trait, used by orchestrator-level tests
    /// when they need a deterministic transform. Real impls live in
    /// sibling modules.
    pub struct AlwaysSavesNothing;
    impl LosslessTransform for AlwaysSavesNothing {
        fn name(&self) -> &'static str {
            "test_no_op"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
            Ok(TransformResult {
                output: content.to_string(),
                bytes_saved: 0,
                structure_preserved: true,
                reversible_via: None,
            })
        }
    }

    #[test]
    fn from_lengths_clamps_negative_savings_to_zero() {
        let r = TransformResult::from_lengths(10, "this is longer than 10 bytes".into(), true);
        assert_eq!(r.bytes_saved, 0);
    }

    #[test]
    fn transform_error_messages_round_trip() {
        let e = TransformError::invalid_input("json_minifier", "bad token at line 3");
        let msg = e.to_string();
        assert!(msg.contains("json_minifier"));
        assert!(msg.contains("bad token at line 3"));
    }

    #[test]
    fn compression_context_constructors_are_clean() {
        let q = CompressionContext::with_query("find errors");
        assert_eq!(q.query, "find errors");
        assert_eq!(q.token_budget, None);

        let b = CompressionContext::with_budget(2048);
        assert_eq!(b.query, "");
        assert_eq!(b.token_budget, Some(2048));
    }

    #[test]
    fn no_op_lossless_transform_is_a_real_impl() {
        let t = AlwaysSavesNothing;
        let r = t.apply("hello").expect("no-op never errors");
        assert_eq!(r.bytes_saved, 0);
        assert_eq!(t.applies_to(), &[ContentType::PlainText]);
        assert_eq!(t.name(), "test_no_op");
    }
}

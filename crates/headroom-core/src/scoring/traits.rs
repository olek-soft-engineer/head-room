//! External-dependency trait surfaces for `MessageScorer`.
//!
//! The scorer's six factors split into two groups:
//!
//! - **Pure / deterministic** (recency, forward references, density):
//!   computed in-crate from the message list alone. No traits needed.
//! - **External-dependency** (semantic similarity, TOIN importance,
//!   error indicator): require either an embedding model or learned
//!   TOIN telemetry. These get trait surfaces here so the scorer
//!   doesn't have to know how those subsystems are implemented.
//!
//! In PR-A (this PR), no concrete impls exist. PR-A1 wires
//! `EmbeddingProvider` to fastembed (reusing the `bge-small-en-v1.5`
//! model already loaded for SmartCrusher relevance). PR-A2 wires
//! `ToinProvider` to a PyO3 adapter so Rust can read Python's TOIN
//! state. When both land, the scorer code is unchanged — only the
//! provider construction at startup differs.
//!
//! # Why pass `&serde_json::Value` and not `&str` for tool content
//!
//! Python's `MessageScorer._compute_toin_score` parses the message
//! content as JSON, then computes a `ToolSignature` from the parsed
//! items, then looks up the learned pattern. The trait pushes the
//! parsing into the implementor — partly because the parsing is
//! cheap and message-local, partly because `ToolSignature` itself is
//! TOIN-internal (not yet ported, and not needed outside TOIN).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Embedding provider for semantic-similarity scoring.
///
/// Implementations should return a fixed-dimension `Vec<f32>` for
/// each input text. The dimension must be consistent across calls
/// from a given instance — `MessageScorer` does cosine similarity
/// between vectors and assumes equal dimension.
///
/// `embed` may fail in implementations that wrap external services;
/// the scorer treats failures as "no signal" (returns the neutral
/// `0.5` semantic score), matching Python's try/except behavior.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a string into a fixed-dimension vector. Empty strings
    /// or whitespace-only strings should still return a valid vector
    /// — the scorer pre-filters empty content before calling this.
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
}

/// Error type for embedding providers. Wraps the underlying impl's
/// error as a string for trait-object compatibility — losing typed
/// context is fine here since the scorer just logs + falls back.
#[derive(Debug, Clone)]
pub struct EmbeddingError(pub String);

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "embedding error: {}", self.0)
    }
}

impl std::error::Error for EmbeddingError {}

/// TOIN (Tool Output Intelligence Network) pattern lookup.
///
/// TOIN learns retrieval patterns per tool-output structure. The
/// scorer queries it for two things:
///
/// 1. **Importance** (`toin_score`): high `retrieval_rate` means
///    users repeatedly retrieve this tool's data → keep it.
/// 2. **Error detection** (`error_score`): TOIN classifies fields
///    by inferred type. Fields tagged `error_indicator` boost the
///    error score, in lieu of hardcoded keyword regex.
///
/// The trait takes the parsed JSON content directly (rather than a
/// pre-computed structure hash) because `ToolSignature` derivation
/// is TOIN-internal — pushing it across the trait boundary would
/// leak implementation details.
pub trait ToinProvider: Send + Sync {
    /// Look up the learned pattern for a tool-message content
    /// payload. Returns `None` if the content can't be classified
    /// (not list/dict, empty, etc.) or no pattern has been learned
    /// for this structure yet.
    fn pattern_for_tool_content(&self, content: &serde_json::Value) -> Option<ToinPattern>;
}

/// Snapshot of a TOIN-learned pattern for a single tool-output
/// structure. Mirrors the subset of `headroom.telemetry.toin.ToolPattern`
/// that `MessageScorer` actually reads.
///
/// We don't mirror the full Python class — TOIN updates patterns
/// in-place during learning, but the scorer only reads. Returning
/// a snapshot decouples scorer reads from learning writes (no
/// shared-mutable-state across the FFI boundary in PR-A2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToinPattern {
    /// Overall confidence in this pattern, `[0.0, 1.0]`. Patterns
    /// with `confidence < 0.3` are treated as not-yet-learned and
    /// the scorer falls back to neutral.
    pub confidence: f32,

    /// Fraction of tool invocations whose data was later retrieved
    /// from cache or referenced by a follow-up message. `[0.0, 1.0]`.
    /// High values mean "users keep needing this data" → important.
    pub retrieval_rate: f32,

    /// Field hashes (NOT names — TOIN privacy-hashes them) that
    /// are commonly retrieved from this structure. The scorer just
    /// uses the *count* as a small importance boost; it doesn't
    /// dereference individual hashes.
    pub commonly_retrieved_fields: Vec<String>,

    /// Per-field semantics, keyed by privacy-hashed field name.
    /// `BTreeMap` rather than `HashMap` so serialization order is
    /// deterministic (matters for parity-fixture stability).
    pub field_semantics: BTreeMap<String, ToinFieldSemantic>,
}

/// Inferred semantic type + confidence for a single field, as
/// learned by TOIN. Mirrors a subset of
/// `headroom.telemetry.models.FieldSemantics`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToinFieldSemantic {
    /// Inferred semantic category. The scorer specifically checks
    /// for `"error_indicator"` to compute the error score; other
    /// values (`"identifier"`, `"status"`, `"timestamp"`, etc.) are
    /// not used by scoring but kept for completeness so we can pass
    /// the same struct through to other consumers.
    pub inferred_type: String,

    /// Confidence in the inferred type, `[0.0, 1.0]`. The scorer
    /// applies a `>= 0.7` threshold for a "high-confidence error"
    /// boost; below that it still counts the field but doesn't
    /// boost.
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_round_trips_through_serde() {
        let mut field_semantics = BTreeMap::new();
        field_semantics.insert(
            "abc123".to_string(),
            ToinFieldSemantic {
                inferred_type: "error_indicator".to_string(),
                confidence: 0.85,
            },
        );
        field_semantics.insert(
            "def456".to_string(),
            ToinFieldSemantic {
                inferred_type: "identifier".to_string(),
                confidence: 0.9,
            },
        );

        let p = ToinPattern {
            confidence: 0.75,
            retrieval_rate: 0.6,
            commonly_retrieved_fields: vec!["abc123".to_string()],
            field_semantics,
        };

        let json = serde_json::to_string(&p).unwrap();
        let back: ToinPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn embedding_error_is_displayable() {
        let e = EmbeddingError("model not loaded".to_string());
        assert_eq!(e.to_string(), "embedding error: model not loaded");
    }
}

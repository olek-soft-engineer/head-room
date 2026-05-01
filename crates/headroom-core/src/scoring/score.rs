//! `MessageScore` — scoring output for a single message.
//!
//! Mirrors `headroom.transforms.scoring.MessageScore` byte-for-byte
//! including the per-component breakdown. The breakdown is what
//! IntelligentContextManager logs for debug + what TOIN consumes
//! for learning.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Importance score for a single message.
///
/// All component scores are in the range `[0.0, 1.0]` where higher =
/// more important. `total_score` is a weighted sum of the components
/// using [`crate::scoring::ScoringWeights`].
///
/// # Determinism
///
/// For the deterministic factors (recency, forward_reference,
/// token_density), the score is a pure function of the input
/// messages + index. Two runs with the same input produce
/// byte-identical scores.
///
/// For the external-dep factors (semantic_score, toin_score,
/// error_score), the value depends on whether providers are wired
/// in. Without providers, these return neutral defaults (`0.5` /
/// `0.0`) — same as Python's behavior with `embedding_provider=None`
/// / `toin=None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageScore {
    pub message_index: usize,
    pub total_score: f32,

    pub recency_score: f32,
    pub semantic_score: f32,
    pub toin_score: f32,
    pub error_score: f32,
    pub reference_score: f32,
    pub density_score: f32,

    /// Estimated tokens for this message. Python uses
    /// `len(content) // 4` as a rough heuristic; we mirror exactly.
    /// For non-string content (e.g. tool_calls list), Python returns
    /// `100` as a default — we mirror that too.
    pub tokens: usize,

    pub is_protected: bool,
    /// `not in_tool_unit OR not protected`. Mirrors Python's
    /// confusing-but-faithful definition (it's intentionally the OR
    /// of two negatives — see Python's `MessageScorer._score_message`
    /// line 189).
    pub drop_safe: bool,

    /// Per-factor breakdown for debug logging + TOIN learning.
    /// Keyed `BTreeMap` for deterministic JSON serialization order
    /// (Python's `dict` preserves insertion order; we want stable
    /// alphabetical so parity-fixture diffs are stable across runs).
    pub score_breakdown: BTreeMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_serde() {
        let mut breakdown = BTreeMap::new();
        breakdown.insert("recency".to_string(), 0.9);
        breakdown.insert("semantic".to_string(), 0.5);
        breakdown.insert("toin".to_string(), 0.5);
        breakdown.insert("error".to_string(), 0.0);
        breakdown.insert("reference".to_string(), 0.0);
        breakdown.insert("density".to_string(), 0.7);

        let s = MessageScore {
            message_index: 3,
            total_score: 0.42,
            recency_score: 0.9,
            semantic_score: 0.5,
            toin_score: 0.5,
            error_score: 0.0,
            reference_score: 0.0,
            density_score: 0.7,
            tokens: 25,
            is_protected: false,
            drop_safe: true,
            score_breakdown: breakdown,
        };

        let json = serde_json::to_string(&s).unwrap();
        let back: MessageScore = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn breakdown_serializes_in_alphabetical_order() {
        // The breakdown order matters for parity-fixture stability.
        // BTreeMap iteration is alphabetical → parity diffs stay
        // deterministic across runs.
        let mut breakdown = BTreeMap::new();
        // Insert in non-alphabetical order:
        for (k, v) in [
            ("toin", 1.0),
            ("recency", 2.0),
            ("density", 3.0),
            ("semantic", 4.0),
            ("error", 5.0),
            ("reference", 6.0),
        ] {
            breakdown.insert(k.to_string(), v);
        }
        let json = serde_json::to_string(&breakdown).unwrap();
        // density < error < recency < reference < semantic < toin alphabetically
        assert!(
            json.find("density").unwrap() < json.find("error").unwrap()
                && json.find("error").unwrap() < json.find("recency").unwrap()
                && json.find("recency").unwrap() < json.find("reference").unwrap()
                && json.find("reference").unwrap() < json.find("semantic").unwrap()
                && json.find("semantic").unwrap() < json.find("toin").unwrap()
        );
    }
}

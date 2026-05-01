//! `ScoringWeights` — six-factor weighted importance scoring.
//!
//! Mirrors `headroom.config.ScoringWeights` byte-for-byte. The six
//! factors and their default contributions sum to ~1.0; `normalized()`
//! enforces that explicitly when a caller passes weights that don't
//! sum to 1.0 (e.g. learned weights from TOIN telemetry).
//!
//! Default values match Python's defaults exactly so the parity
//! fixtures byte-equal across implementations.

use serde::{Deserialize, Serialize};

/// Weights for the six importance-scoring factors.
///
/// All weights should sum to ~1.0 for normalized scoring (call
/// [`Self::normalized`] to enforce). Non-normalized weights are
/// permitted — `MessageScorer` does NOT auto-normalize on input —
/// since callers may use raw weights for relative comparison.
///
/// Defaults match Python's `ScoringWeights()`:
///
/// | Factor | Weight |
/// |--------|--------|
/// | recency | 0.20 |
/// | semantic_similarity | 0.20 |
/// | toin_importance | 0.25 |
/// | error_indicator | 0.15 |
/// | forward_reference | 0.15 |
/// | token_density | 0.05 |
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Exponential decay from conversation end. Default 0.20.
    pub recency: f32,
    /// Embedding cosine similarity to recent context. Default 0.20.
    pub semantic_similarity: f32,
    /// TOIN-learned field importance. Default 0.25.
    pub toin_importance: f32,
    /// TOIN-learned error-field detection. Default 0.15.
    pub error_indicator: f32,
    /// Number of later messages referencing this one (tool_call_id). Default 0.15.
    pub forward_reference: f32,
    /// Information density (unique-tokens / total-tokens). Default 0.05.
    pub token_density: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            recency: 0.20,
            semantic_similarity: 0.20,
            toin_importance: 0.25,
            error_indicator: 0.15,
            forward_reference: 0.15,
            token_density: 0.05,
        }
    }
}

impl ScoringWeights {
    /// Return a copy with all weights divided by their sum, so they sum
    /// to 1.0 exactly. If the input sums to 0 (degenerate config),
    /// returns the default weights.
    pub fn normalized(&self) -> Self {
        let total = self.recency
            + self.semantic_similarity
            + self.toin_importance
            + self.error_indicator
            + self.forward_reference
            + self.token_density;
        if total == 0.0 {
            return Self::default();
        }
        Self {
            recency: self.recency / total,
            semantic_similarity: self.semantic_similarity / total,
            toin_importance: self.toin_importance / total,
            error_indicator: self.error_indicator / total,
            forward_reference: self.forward_reference / total,
            token_density: self.token_density / total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let w = ScoringWeights::default();
        assert_eq!(w.recency, 0.20);
        assert_eq!(w.semantic_similarity, 0.20);
        assert_eq!(w.toin_importance, 0.25);
        assert_eq!(w.error_indicator, 0.15);
        assert_eq!(w.forward_reference, 0.15);
        assert_eq!(w.token_density, 0.05);
    }

    #[test]
    fn defaults_sum_to_one() {
        let w = ScoringWeights::default();
        let total = w.recency
            + w.semantic_similarity
            + w.toin_importance
            + w.error_indicator
            + w.forward_reference
            + w.token_density;
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalized_already_normal_is_idempotent() {
        let w = ScoringWeights::default();
        let n = w.normalized();
        // Within float epsilon — divisions can introduce tiny drift.
        assert!((n.recency - w.recency).abs() < 1e-6);
        assert!((n.toin_importance - w.toin_importance).abs() < 1e-6);
    }

    #[test]
    fn normalized_unbalanced_weights_sum_to_one() {
        let w = ScoringWeights {
            recency: 1.0,
            semantic_similarity: 1.0,
            toin_importance: 1.0,
            error_indicator: 1.0,
            forward_reference: 1.0,
            token_density: 1.0,
        };
        let n = w.normalized();
        let total = n.recency
            + n.semantic_similarity
            + n.toin_importance
            + n.error_indicator
            + n.forward_reference
            + n.token_density;
        assert!((total - 1.0).abs() < 1e-6);
        // Each component should now be ~1/6.
        assert!((n.recency - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn normalized_zero_weights_falls_back_to_default() {
        let w = ScoringWeights {
            recency: 0.0,
            semantic_similarity: 0.0,
            toin_importance: 0.0,
            error_indicator: 0.0,
            forward_reference: 0.0,
            token_density: 0.0,
        };
        let n = w.normalized();
        assert_eq!(n, ScoringWeights::default());
    }

    #[test]
    fn round_trips_through_serde() {
        let w = ScoringWeights {
            recency: 0.3,
            semantic_similarity: 0.1,
            toin_importance: 0.2,
            error_indicator: 0.2,
            forward_reference: 0.1,
            token_density: 0.1,
        };
        let json = serde_json::to_string(&w).unwrap();
        let back: ScoringWeights = serde_json::from_str(&json).unwrap();
        assert_eq!(w, back);
    }
}

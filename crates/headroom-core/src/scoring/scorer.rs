//! `MessageScorer` — six-factor importance scoring for messages.
//!
//! Direct port of `headroom.transforms.scoring.MessageScorer`. See
//! the module-level doc in `mod.rs` for what is and isn't wired up
//! in PR-A.
//!
//! # Parity contract
//!
//! For the deterministic factors (recency, forward_reference,
//! token_density), the Rust implementation must produce
//! float-epsilon-equal scores to the Python implementation given
//! the same input. This is what `crates/headroom-parity/` validates.
//!
//! For non-deterministic factors (semantic, toin, error), parity is
//! validated only when both implementations are configured with the
//! same providers. Without providers, both return the same neutral
//! defaults.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{Map, Value};

use crate::scoring::score::MessageScore;
use crate::scoring::traits::{EmbeddingProvider, ToinPattern, ToinProvider};
use crate::scoring::weights::ScoringWeights;

/// Six-factor importance scorer. See module-level docs for the
/// list of factors and their weights.
///
/// # Thread safety
///
/// `MessageScorer` is `Send + Sync`. The internal embedding cache is
/// guarded by a `Mutex` — contention is low because cache writes
/// happen at most once per (message, scorer) pair and scorer
/// instances are typically per-request, not shared.
pub struct MessageScorer {
    weights: ScoringWeights,
    toin: Option<Box<dyn ToinProvider>>,
    embedding_provider: Option<Box<dyn EmbeddingProvider>>,
    recency_decay_rate: f32,
    embedding_cache: Mutex<HashMap<usize, Vec<f32>>>,
}

impl MessageScorer {
    /// Create a new scorer.
    ///
    /// `weights` are normalized on construction (matching Python's
    /// `ScoringWeights().normalized()` in `__init__`). Pass `None`
    /// for `toin` and `embedding_provider` to use neutral defaults
    /// for those factors — same as Python's `toin=None` /
    /// `embedding_provider=None`.
    pub fn new(
        weights: Option<ScoringWeights>,
        toin: Option<Box<dyn ToinProvider>>,
        embedding_provider: Option<Box<dyn EmbeddingProvider>>,
        recency_decay_rate: f32,
    ) -> Self {
        Self {
            weights: weights.unwrap_or_default().normalized(),
            toin,
            embedding_provider,
            recency_decay_rate,
            embedding_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create a scorer with default weights, no providers, and the
    /// Python-default decay rate of 0.1. Useful for tests and the
    /// deterministic-only code path.
    pub fn with_defaults() -> Self {
        Self::new(None, None, None, 0.1)
    }

    /// Score every message in the list.
    ///
    /// `protected_indices` are indices the caller has marked as
    /// system / pinned / never-drop. They get `is_protected=true`.
    /// `tool_unit_indices` are part of an inseparable tool unit
    /// (assistant tool_call + tool response pair) — these get
    /// `drop_safe=false` unless also protected, matching Python's
    /// confusing-but-faithful `not in_tool_unit OR not protected`
    /// rule.
    pub fn score_messages(
        &self,
        messages: &[Value],
        protected_indices: &std::collections::HashSet<usize>,
        tool_unit_indices: &std::collections::HashSet<usize>,
    ) -> Vec<MessageScore> {
        let forward_refs = Self::compute_forward_references(messages);
        let recent_embedding = self.compute_recent_context_embedding(messages, 3);

        messages
            .iter()
            .enumerate()
            .map(|(i, msg)| {
                self.score_message(
                    msg,
                    i,
                    messages.len(),
                    protected_indices.contains(&i),
                    tool_unit_indices.contains(&i),
                    &forward_refs,
                    recent_embedding.as_deref(),
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn score_message(
        &self,
        msg: &Value,
        index: usize,
        total: usize,
        protected: bool,
        in_tool_unit: bool,
        forward_refs: &HashMap<usize, u32>,
        recent_embedding: Option<&[f32]>,
    ) -> MessageScore {
        let recency = self.compute_recency_score(index, total);
        let semantic = self.compute_semantic_score(msg, index, recent_embedding);
        let toin = self.compute_toin_score(msg);
        let error = self.compute_error_score(msg);
        let reference = self.compute_reference_score(index, forward_refs);
        let density = Self::compute_density_score(msg);

        let w = &self.weights;
        let total_score = w.recency * recency
            + w.semantic_similarity * semantic
            + w.toin_importance * toin
            + w.error_indicator * error
            + w.forward_reference * reference
            + w.token_density * density;

        let tokens = estimate_tokens(msg);

        // BTreeMap so JSON serialization order is alphabetical.
        let mut breakdown = std::collections::BTreeMap::new();
        breakdown.insert("recency".to_string(), recency);
        breakdown.insert("semantic".to_string(), semantic);
        breakdown.insert("toin".to_string(), toin);
        breakdown.insert("error".to_string(), error);
        breakdown.insert("reference".to_string(), reference);
        breakdown.insert("density".to_string(), density);

        MessageScore {
            message_index: index,
            total_score,
            recency_score: recency,
            semantic_score: semantic,
            toin_score: toin,
            error_score: error,
            reference_score: reference,
            density_score: density,
            tokens,
            is_protected: protected,
            // Python: `not in_tool_unit or not protected` — see
            // scoring.py:189. This *is* the literal expression.
            drop_safe: !in_tool_unit || !protected,
            score_breakdown: breakdown,
        }
    }

    fn compute_recency_score(&self, index: usize, total: usize) -> f32 {
        if total <= 1 {
            return 1.0;
        }
        let position_from_end = (total - 1 - index) as f32;
        (-self.recency_decay_rate * position_from_end).exp()
    }

    fn compute_semantic_score(
        &self,
        msg: &Value,
        index: usize,
        recent_embedding: Option<&[f32]>,
    ) -> f32 {
        let Some(provider) = self.embedding_provider.as_ref() else {
            return 0.5;
        };
        let Some(recent) = recent_embedding else {
            return 0.5;
        };

        let content = match msg.get("content") {
            Some(Value::String(s)) if !s.trim().is_empty() => s,
            _ => return 0.5,
        };

        // Cache lookup; populate on miss.
        let msg_embedding: Vec<f32> = {
            let mut cache = self.embedding_cache.lock().unwrap();
            if let Some(cached) = cache.get(&index) {
                cached.clone()
            } else {
                match provider.embed(content) {
                    Ok(v) => {
                        cache.insert(index, v.clone());
                        v
                    }
                    Err(_) => return 0.5,
                }
            }
        };

        cosine_similarity(&msg_embedding, recent)
    }

    fn compute_toin_score(&self, msg: &Value) -> f32 {
        let Some(toin) = self.toin.as_ref() else {
            return 0.5;
        };
        if msg.get("role").and_then(Value::as_str) != Some("tool") {
            return 0.5;
        }
        let Some(content_value) = parse_tool_content(msg) else {
            return 0.5;
        };

        let Some(pattern) = toin.pattern_for_tool_content(&content_value) else {
            return 0.5;
        };

        if pattern.confidence < 0.3 {
            return 0.5;
        }

        let mut score = 0.5 + pattern.retrieval_rate * 0.5;

        if !pattern.commonly_retrieved_fields.is_empty() {
            // Python: min(0.1, 0.02 * len(commonly_retrieved_fields))
            let boost = (0.02 * pattern.commonly_retrieved_fields.len() as f32).min(0.1);
            score = (score + boost).min(1.0);
        }

        score
    }

    fn compute_error_score(&self, msg: &Value) -> f32 {
        let Some(toin) = self.toin.as_ref() else {
            return 0.0;
        };
        if msg.get("role").and_then(Value::as_str) != Some("tool") {
            return 0.0;
        }
        let Some(content_value) = parse_tool_content(msg) else {
            return 0.0;
        };
        let Some(pattern) = toin.pattern_for_tool_content(&content_value) else {
            return 0.0;
        };

        let (error_field_count, high_confidence_errors) = count_error_fields(&pattern);

        if error_field_count == 0 {
            return 0.0;
        }

        // Python: base_score = min(1.0, 0.3 * error_field_count)
        //         confidence_boost = min(0.5, 0.2 * high_confidence_errors)
        let base = (0.3 * error_field_count as f32).min(1.0);
        let boost = (0.2 * high_confidence_errors as f32).min(0.5);
        base + boost
    }

    fn compute_reference_score(&self, index: usize, forward_refs: &HashMap<usize, u32>) -> f32 {
        let count = *forward_refs.get(&index).unwrap_or(&0);
        if count == 0 {
            return 0.0;
        }
        // Python: min(1.0, 0.3 + 0.2 * math.log(ref_count + 1))
        // math.log is natural log.
        let v = 0.3 + 0.2 * ((count + 1) as f32).ln();
        v.min(1.0)
    }

    fn compute_density_score(msg: &Value) -> f32 {
        let content = match msg.get("content") {
            Some(Value::String(s)) => s,
            _ => return 0.5,
        };
        // Python: `len(content) < 10` — len counts chars (code points).
        if content.chars().count() < 10 {
            return 0.5;
        }
        let lower = content.to_lowercase();
        let tokens: Vec<&str> = lower.split_whitespace().collect();
        if tokens.len() < 3 {
            return 0.5;
        }
        let unique: std::collections::HashSet<&&str> = tokens.iter().collect();
        let density = unique.len() as f32 / tokens.len() as f32;
        // Python: min(1.0, max(0.0, (density - 0.2) / 0.6))
        ((density - 0.2) / 0.6).clamp(0.0, 1.0)
    }

    fn compute_forward_references(messages: &[Value]) -> HashMap<usize, u32> {
        let mut refs: HashMap<usize, u32> = HashMap::new();
        let mut tool_call_ids: HashMap<String, usize> = HashMap::new();

        for (i, msg) in messages.iter().enumerate() {
            let role = msg.get("role").and_then(Value::as_str);
            match role {
                Some("assistant") => {
                    if let Some(tcs) = msg.get("tool_calls").and_then(Value::as_array) {
                        for tc in tcs {
                            if let Some(id) = tc.get("id").and_then(Value::as_str) {
                                tool_call_ids.insert(id.to_string(), i);
                            }
                        }
                    }
                }
                Some("tool") => {
                    if let Some(tcid) = msg.get("tool_call_id").and_then(Value::as_str) {
                        if let Some(&ref_idx) = tool_call_ids.get(tcid) {
                            *refs.entry(ref_idx).or_insert(0) += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        refs
    }

    fn compute_recent_context_embedding(
        &self,
        messages: &[Value],
        num_recent: usize,
    ) -> Option<Vec<f32>> {
        let provider = self.embedding_provider.as_ref()?;
        let start = messages.len().saturating_sub(num_recent);
        let mut texts: Vec<&str> = Vec::new();
        for msg in &messages[start..] {
            if let Some(Value::String(s)) = msg.get("content") {
                if !s.trim().is_empty() {
                    texts.push(s);
                }
            }
        }
        if texts.is_empty() {
            return None;
        }
        let combined = texts.join(" ");
        provider.embed(&combined).ok()
    }
}

/// Token estimate. Python: `len(content) // 4` for strings, `100`
/// for non-strings. We use `chars().count()` (code points) to match
/// Python's `len(str)`.
fn estimate_tokens(msg: &Value) -> usize {
    match msg.get("content") {
        Some(Value::String(s)) => s.chars().count() / 4,
        _ => 100,
    }
}

/// Parse a tool message's `content` as JSON if it's a string, or
/// return it directly if already an object/array. Matches Python's
/// behavior in `_compute_toin_score` / `_compute_error_score`:
/// returns `None` for anything that isn't a list/dict.
fn parse_tool_content(msg: &Value) -> Option<Value> {
    let content = msg.get("content")?;
    let parsed = match content {
        Value::String(s) => {
            if s.is_empty() {
                return None;
            }
            serde_json::from_str::<Value>(s).ok()?
        }
        v => v.clone(),
    };
    match &parsed {
        Value::Object(_) | Value::Array(_) => {
            // Python: `if not items: return 0.5/0.0` — empty
            // list/object disqualifies.
            if let Value::Array(a) = &parsed {
                if a.is_empty() {
                    return None;
                }
            }
            if let Value::Object(o) = &parsed {
                if o.is_empty() {
                    // Python wraps a dict into a list before checking
                    // truthiness; an empty dict becomes `[{}]` which
                    // is truthy. Mirror exactly:
                    let _ = o;
                    // (still return Some since [{}] is truthy in Py)
                }
            }
            Some(parsed)
        }
        _ => None,
    }
}

/// Count `error_indicator` fields in a TOIN pattern. Returns
/// `(total_error_fields, high_confidence_error_fields)` where
/// high-confidence means `confidence >= 0.7`.
fn count_error_fields(pattern: &ToinPattern) -> (u32, u32) {
    let mut total = 0u32;
    let mut high = 0u32;
    for field_sem in pattern.field_semantics.values() {
        if field_sem.inferred_type == "error_indicator" {
            total += 1;
            if field_sem.confidence >= 0.7 {
                high += 1;
            }
        }
    }
    (total, high)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Helper to build a message `Value` for tests. Public-in-crate so
/// integration tests can use it too.
#[allow(dead_code)]
pub(crate) fn msg(role: &str, content: &str) -> Value {
    let mut m = Map::new();
    m.insert("role".to_string(), Value::String(role.to_string()));
    m.insert("content".to_string(), Value::String(content.to_string()));
    Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn empty_set() -> HashSet<usize> {
        HashSet::new()
    }

    #[test]
    fn recency_single_message_returns_one() {
        let s = MessageScorer::with_defaults();
        assert_eq!(s.compute_recency_score(0, 1), 1.0);
        assert_eq!(s.compute_recency_score(0, 0), 1.0);
    }

    #[test]
    fn recency_last_message_full_score() {
        let s = MessageScorer::with_defaults();
        // Last message: position_from_end = 0, score = e^0 = 1.0
        assert!((s.compute_recency_score(4, 5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recency_decays_exponentially() {
        let s = MessageScorer::with_defaults();
        // total=10, index=0 → position_from_end=9 → e^(-0.1*9) = e^-0.9
        let expected = (-0.9f32).exp();
        let got = s.compute_recency_score(0, 10);
        assert!(
            (got - expected).abs() < 1e-6,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn density_short_content_is_neutral() {
        // Less than 10 chars
        let m = msg("user", "hi");
        assert_eq!(MessageScorer::compute_density_score(&m), 0.5);
    }

    #[test]
    fn density_few_tokens_is_neutral() {
        // >= 10 chars but < 3 tokens after split
        let m = msg("user", "abcdefghij"); // single token, 10 chars
        assert_eq!(MessageScorer::compute_density_score(&m), 0.5);
    }

    #[test]
    fn density_all_unique_clamps_to_one() {
        // 10 unique tokens, 10 total → density=1.0 → (1.0-0.2)/0.6=1.33 → clamped to 1.0
        let m = msg(
            "user",
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
        );
        assert!((MessageScorer::compute_density_score(&m) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn density_repeated_tokens_lowers_score() {
        // 4 unique / 8 total = 0.5 → (0.5-0.2)/0.6 = 0.5
        let m = msg(
            "user",
            "alpha bravo charlie delta alpha bravo charlie delta",
        );
        let got = MessageScorer::compute_density_score(&m);
        assert!((got - 0.5).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn density_non_string_content_is_neutral() {
        let mut m = Map::new();
        m.insert("role".to_string(), Value::String("assistant".to_string()));
        m.insert(
            "tool_calls".to_string(),
            Value::Array(vec![Value::Object(Map::new())]),
        );
        let v = Value::Object(m);
        assert_eq!(MessageScorer::compute_density_score(&v), 0.5);
    }

    #[test]
    fn forward_refs_links_tool_response_to_assistant() {
        let mut tc = Map::new();
        tc.insert("id".to_string(), Value::String("call_1".to_string()));
        let mut assistant = Map::new();
        assistant.insert("role".to_string(), Value::String("assistant".to_string()));
        assistant.insert(
            "tool_calls".to_string(),
            Value::Array(vec![Value::Object(tc)]),
        );

        let mut tool_resp = Map::new();
        tool_resp.insert("role".to_string(), Value::String("tool".to_string()));
        tool_resp.insert(
            "tool_call_id".to_string(),
            Value::String("call_1".to_string()),
        );
        tool_resp.insert("content".to_string(), Value::String("result".to_string()));

        let messages = vec![
            msg("user", "do thing"),
            Value::Object(assistant),
            Value::Object(tool_resp),
        ];

        let refs = MessageScorer::compute_forward_references(&messages);
        assert_eq!(refs.get(&1), Some(&1));
        assert_eq!(refs.get(&0), None);
    }

    #[test]
    fn forward_refs_unmatched_tool_call_id_ignored() {
        let mut tool_resp = Map::new();
        tool_resp.insert("role".to_string(), Value::String("tool".to_string()));
        tool_resp.insert(
            "tool_call_id".to_string(),
            Value::String("nope".to_string()),
        );
        tool_resp.insert("content".to_string(), Value::String("result".to_string()));

        let messages = vec![Value::Object(tool_resp)];
        let refs = MessageScorer::compute_forward_references(&messages);
        assert!(refs.is_empty());
    }

    #[test]
    fn reference_score_zero_refs_returns_zero() {
        let s = MessageScorer::with_defaults();
        let refs = HashMap::new();
        assert_eq!(s.compute_reference_score(0, &refs), 0.0);
    }

    #[test]
    fn reference_score_one_ref_uses_log_formula() {
        let s = MessageScorer::with_defaults();
        let mut refs = HashMap::new();
        refs.insert(0, 1);
        // 0.3 + 0.2 * ln(2) = 0.3 + 0.2 * 0.693... ≈ 0.4386
        let expected = 0.3 + 0.2 * 2f32.ln();
        let got = s.compute_reference_score(0, &refs);
        assert!((got - expected).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn reference_score_clamps_to_one() {
        let s = MessageScorer::with_defaults();
        let mut refs = HashMap::new();
        refs.insert(0, 1_000_000);
        assert!(s.compute_reference_score(0, &refs) <= 1.0);
    }

    #[test]
    fn semantic_score_no_provider_returns_neutral() {
        let s = MessageScorer::with_defaults();
        let m = msg("user", "hello world");
        // recent_embedding is None too without a provider
        assert_eq!(s.compute_semantic_score(&m, 0, None), 0.5);
    }

    #[test]
    fn toin_score_no_provider_returns_neutral() {
        let s = MessageScorer::with_defaults();
        let m = msg("tool", "{}");
        assert_eq!(s.compute_toin_score(&m), 0.5);
    }

    #[test]
    fn error_score_no_provider_returns_zero() {
        let s = MessageScorer::with_defaults();
        let m = msg("tool", "{}");
        assert_eq!(s.compute_error_score(&m), 0.0);
    }

    #[test]
    fn estimate_tokens_string_uses_char_count() {
        // 16 chars / 4 = 4
        let m = msg("user", "abcdefghijklmnop");
        assert_eq!(estimate_tokens(&m), 4);
    }

    #[test]
    fn estimate_tokens_unicode_is_char_aware() {
        // 4 emoji chars (each multi-byte) → len("...") in Python is 4.
        let m = msg("user", "🎉🎉🎉🎉");
        // chars().count() = 4, // 4 = 1
        assert_eq!(estimate_tokens(&m), 1);
    }

    #[test]
    fn estimate_tokens_non_string_returns_default() {
        let mut m = Map::new();
        m.insert("role".to_string(), Value::String("assistant".to_string()));
        m.insert("tool_calls".to_string(), Value::Array(vec![]));
        let v = Value::Object(m);
        assert_eq!(estimate_tokens(&v), 100);
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = [1.0f32, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_is_zero() {
        let a = [0.0f32, 0.0];
        let b = [1.0f32, 1.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_similarity_dim_mismatch_is_zero() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn drop_safe_mirrors_python_or_logic() {
        // Python: drop_safe = not in_tool_unit OR not protected
        // truth table:
        // in_tool_unit=F, protected=F → T
        // in_tool_unit=F, protected=T → T (not F = T)
        // in_tool_unit=T, protected=F → T (not F = T)
        // in_tool_unit=T, protected=T → F
        let s = MessageScorer::with_defaults();
        let m = msg("user", "hi");
        let refs = HashMap::new();

        let cases = [
            (false, false, true),
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ];
        for (in_tu, prot, expected) in cases {
            let score = s.score_message(&m, 0, 1, prot, in_tu, &refs, None);
            assert_eq!(
                score.drop_safe, expected,
                "in_tool_unit={in_tu} protected={prot}"
            );
        }
    }

    #[test]
    fn score_messages_returns_one_score_per_message() {
        let s = MessageScorer::with_defaults();
        let messages = vec![
            msg("user", "hello"),
            msg("assistant", "hi there"),
            msg("user", "thanks"),
        ];
        let scores = s.score_messages(&messages, &empty_set(), &empty_set());
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].message_index, 0);
        assert_eq!(scores[2].message_index, 2);
    }

    #[test]
    fn score_messages_protected_indices_are_marked() {
        let s = MessageScorer::with_defaults();
        let messages = vec![msg("user", "hi"), msg("user", "bye")];
        let mut protected = HashSet::new();
        protected.insert(0);
        let scores = s.score_messages(&messages, &protected, &empty_set());
        assert!(scores[0].is_protected);
        assert!(!scores[1].is_protected);
    }

    #[test]
    fn weights_are_normalized_on_construction() {
        let unbalanced = ScoringWeights {
            recency: 2.0,
            semantic_similarity: 2.0,
            toin_importance: 2.0,
            error_indicator: 2.0,
            forward_reference: 2.0,
            token_density: 2.0,
        };
        let s = MessageScorer::new(Some(unbalanced), None, None, 0.1);
        // After normalization each should be 1/6.
        assert!((s.weights.recency - 1.0 / 6.0).abs() < 1e-6);
    }
}

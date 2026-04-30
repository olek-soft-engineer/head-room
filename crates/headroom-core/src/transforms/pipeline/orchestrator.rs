//! `CompressionPipeline` — content-type-keyed dispatch over lossless
//! and lossy transforms.
//!
//! # Decision flow
//!
//! ```text
//! input + content_type
//!   │
//!   ▼
//! ┌────────────────────────────────────┐
//! │ Lossless transforms for this type  │ ──┐
//! │   for each in registration order:  │   │ stop early if
//! │     try apply                      │   │ current_len/orig_len
//! │     accept iff is_acceptable()     │   │ falls below
//! │                                    │   │ lossless_target_ratio
//! └────────────────────────────────────┘ ──┘
//!   │
//!   ▼
//! ┌────────────────────────────────────┐
//! │ Lossy transforms for this type     │
//! │   for each in registration order:  │
//! │     try apply                      │
//! │     accept iff is_acceptable()     │
//! │     flag structure_preserved=false │
//! └────────────────────────────────────┘
//!   │
//!   ▼ steps_applied[], bytes_saved, structure_preserved, reversible_via
//! ```
//!
//! Acceptance gate (`is_acceptable`):
//! * Reject if output isn't strictly shorter than the input it just
//!   processed (a transform that grew the content gets discarded).
//! * Reject if `bytes_saved / current_len < min_savings_ratio`. The
//!   default is 5% — transforms that move bytes by less than that
//!   aren't worth the runtime overhead they impose.
//!
//! Skip-further-lossless gate (`should_keep_running_lossless`):
//! * If the running output is already at or below
//!   `lossless_target_ratio * original_len`, stop running additional
//!   lossless transforms and move straight to lossy. Default 0.5 —
//!   "we've cut input in half losslessly, no need to keep
//!   structural-pass churning."
//!
//! Per-transform errors are logged at TRACE level and treated as
//! skips. The pipeline never panics on a transform failure; the
//! callers can rely on getting *some* output back, even if it's the
//! original input verbatim (all transforms skipped).

use std::collections::HashMap;
use std::sync::Arc;

use crate::transforms::pipeline::traits::{
    CompressionContext, LosslessTransform, LossyTransform, TransformResult,
};
use crate::transforms::ContentType;

/// Runtime knobs for the orchestrator. Defaults chosen empirically;
/// callers can override per-pipeline.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Minimum savings ratio (saved / current_len) for a transform's
    /// output to be accepted. Default 0.05 — transforms that move
    /// fewer than 5% of bytes don't earn the risk of having moved
    /// the wrong ones.
    pub min_savings_ratio: f64,
    /// Stop the lossless pass once the running output drops below
    /// this fraction of the original input length. Default 0.5 —
    /// once we've cut by half losslessly, downstream lossy passes
    /// can do the rest cheaper than another structural transform.
    pub lossless_target_ratio: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_savings_ratio: 0.05,
            lossless_target_ratio: 0.5,
        }
    }
}

/// Result returned by [`CompressionPipeline::run`].
#[derive(Debug, Clone, Default)]
pub struct PipelineResult {
    /// Final compressed output. Equal to the input if every transform
    /// skipped or was rejected.
    pub output: String,
    /// Total bytes removed across all accepted transforms.
    pub bytes_saved: usize,
    /// True iff every accepted transform preserved structure. A
    /// single accepted lossy transform flips this to false.
    pub structure_preserved: bool,
    /// Names of accepted transforms in execution order. Maps 1:1 to
    /// the per-strategy stats nest from Phase 3e.0.
    pub steps_applied: Vec<String>,
    /// Most recent CCR cache key emitted by an accepted transform.
    /// PR1 transforms never set this; the field exists for PR2/PR3
    /// integrations.
    pub reversible_via: Option<String>,
}

/// Sequential lossless-then-lossy pipeline keyed on `ContentType`.
///
/// Construct via [`builder`](Self::builder), then call
/// [`run`](Self::run) on each input.
pub struct CompressionPipeline {
    lossless_by_type: HashMap<ContentType, Vec<Arc<dyn LosslessTransform>>>,
    lossy_by_type: HashMap<ContentType, Vec<Arc<dyn LossyTransform>>>,
    config: PipelineConfig,
}

impl CompressionPipeline {
    pub fn builder() -> CompressionPipelineBuilder {
        CompressionPipelineBuilder::default()
    }

    /// Run the configured pipeline against `content`. Always returns a
    /// [`PipelineResult`] — failures inside individual transforms are
    /// recorded in tracing and turn into skips, never panics.
    pub fn run(
        &self,
        content: &str,
        content_type: ContentType,
        ctx: &CompressionContext,
    ) -> PipelineResult {
        let original_len = content.len();
        let mut current = content.to_string();
        let mut total_saved: usize = 0;
        let mut structure_preserved = true;
        let mut steps: Vec<String> = Vec::new();
        let mut reversible: Option<String> = None;

        // Phase 1 — lossless. Run in registration order, stop if we've
        // hit the lossless target ratio.
        if let Some(lossless) = self.lossless_by_type.get(&content_type) {
            for transform in lossless {
                if !self.should_keep_running_lossless(current.len(), original_len) {
                    tracing::trace!(
                        target: "headroom::pipeline",
                        transform = transform.name(),
                        current_len = current.len(),
                        original_len,
                        "lossless target reached, stopping lossless phase"
                    );
                    break;
                }
                match transform.apply(&current) {
                    Ok(result) => {
                        if !self.is_acceptable(result.bytes_saved, current.len()) {
                            tracing::trace!(
                                target: "headroom::pipeline",
                                transform = transform.name(),
                                bytes_saved = result.bytes_saved,
                                "lossless transform rejected: insufficient savings"
                            );
                            continue;
                        }
                        Self::accept(
                            &mut current,
                            &mut total_saved,
                            &mut steps,
                            &mut reversible,
                            &mut structure_preserved,
                            transform.name(),
                            result,
                        );
                    }
                    Err(e) => {
                        tracing::trace!(
                            target: "headroom::pipeline",
                            transform = transform.name(),
                            error = %e,
                            "lossless transform errored"
                        );
                    }
                }
            }
        }

        // Phase 2 — lossy. Always runs (PR3 will gate this on token
        // budget once the tokenizer hookup lands).
        if let Some(lossy) = self.lossy_by_type.get(&content_type) {
            for transform in lossy {
                match transform.apply(&current, ctx) {
                    Ok(result) => {
                        if !self.is_acceptable(result.bytes_saved, current.len()) {
                            tracing::trace!(
                                target: "headroom::pipeline",
                                transform = transform.name(),
                                bytes_saved = result.bytes_saved,
                                "lossy transform rejected: insufficient savings"
                            );
                            continue;
                        }
                        Self::accept(
                            &mut current,
                            &mut total_saved,
                            &mut steps,
                            &mut reversible,
                            &mut structure_preserved,
                            transform.name(),
                            result,
                        );
                    }
                    Err(e) => {
                        tracing::trace!(
                            target: "headroom::pipeline",
                            transform = transform.name(),
                            error = %e,
                            "lossy transform errored"
                        );
                    }
                }
            }
        }

        PipelineResult {
            output: current,
            bytes_saved: total_saved,
            structure_preserved,
            steps_applied: steps,
            reversible_via: reversible,
        }
    }

    /// Centralized accept logic. Updates the running output, savings,
    /// step list, reversibility handle, and structure flag as one
    /// step so the lossless and lossy phases share semantics.
    fn accept(
        current: &mut String,
        total_saved: &mut usize,
        steps: &mut Vec<String>,
        reversible: &mut Option<String>,
        structure_preserved: &mut bool,
        name: &'static str,
        result: TransformResult,
    ) {
        *total_saved = total_saved.saturating_add(result.bytes_saved);
        if !result.structure_preserved {
            *structure_preserved = false;
        }
        if let Some(handle) = result.reversible_via {
            *reversible = Some(handle);
        }
        *current = result.output;
        steps.push(name.to_string());
    }

    /// Acceptance gate — whether this transform's savings on the
    /// current state of the buffer are worth keeping.
    pub(crate) fn is_acceptable(&self, bytes_saved: usize, current_len: usize) -> bool {
        if bytes_saved == 0 || current_len == 0 {
            return false;
        }
        let ratio = bytes_saved as f64 / current_len as f64;
        ratio >= self.config.min_savings_ratio
    }

    /// Whether the lossless phase should keep running. False once
    /// `current_len / original_len <= lossless_target_ratio`.
    pub(crate) fn should_keep_running_lossless(
        &self,
        current_len: usize,
        original_len: usize,
    ) -> bool {
        if original_len == 0 {
            return false;
        }
        let ratio = current_len as f64 / original_len as f64;
        ratio > self.config.lossless_target_ratio
    }
}

/// Fluent builder for [`CompressionPipeline`]. The example in
/// `issue #315` is the spec for this surface.
#[derive(Default)]
pub struct CompressionPipelineBuilder {
    lossless_by_type: HashMap<ContentType, Vec<Arc<dyn LosslessTransform>>>,
    lossy_by_type: HashMap<ContentType, Vec<Arc<dyn LossyTransform>>>,
    config: Option<PipelineConfig>,
}

impl CompressionPipelineBuilder {
    /// Register a lossless transform. The transform is added to the
    /// pipeline for every `ContentType` it self-declares via
    /// `applies_to()`. Order of registration is order of execution.
    pub fn with_lossless<T>(mut self, transform: T) -> Self
    where
        T: LosslessTransform + 'static,
    {
        let arc: Arc<dyn LosslessTransform> = Arc::new(transform);
        let types: Vec<ContentType> = arc.applies_to().to_vec();
        for ct in types {
            self.lossless_by_type
                .entry(ct)
                .or_default()
                .push(arc.clone());
        }
        self
    }

    pub fn with_lossy<T>(mut self, transform: T) -> Self
    where
        T: LossyTransform + 'static,
    {
        let arc: Arc<dyn LossyTransform> = Arc::new(transform);
        let types: Vec<ContentType> = arc.applies_to().to_vec();
        for ct in types {
            self.lossy_by_type.entry(ct).or_default().push(arc.clone());
        }
        self
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn build(self) -> CompressionPipeline {
        CompressionPipeline {
            lossless_by_type: self.lossless_by_type,
            lossy_by_type: self.lossy_by_type,
            config: self.config.unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{
        ImportanceCategory, ImportanceContext, ImportanceSignal, LineImportanceDetector,
    };
    use crate::transforms::pipeline::traits::{
        CompressionContext, LosslessTransform, LossyTransform, TransformError, TransformResult,
    };
    use crate::transforms::pipeline::{
        JsonMinifier, LineImportanceFilter, LineImportanceFilterConfig,
    };

    // ── Test helpers ─────────────────────────────────────────────────

    /// Trivial lossless transform that always saves exactly 1 byte
    /// (drops a trailing newline if present, otherwise reports
    /// bytes_saved == 0).
    struct TrivialLossless;
    impl LosslessTransform for TrivialLossless {
        fn name(&self) -> &'static str {
            "trivial_lossless"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
            let trimmed = content.trim_end_matches('\n').to_string();
            Ok(TransformResult::from_lengths(content.len(), trimmed, true))
        }
    }

    /// Lossless transform that always errors. Used to verify the
    /// orchestrator continues past failures.
    struct AlwaysErrors;
    impl LosslessTransform for AlwaysErrors {
        fn name(&self) -> &'static str {
            "always_errors"
        }
        fn applies_to(&self) -> &[ContentType] {
            &[ContentType::PlainText]
        }
        fn apply(&self, _content: &str) -> Result<TransformResult, TransformError> {
            Err(TransformError::invalid_input("always_errors", "by design"))
        }
    }

    /// Stub detector for the LineImportanceFilter integration tests.
    struct StubDetector;
    impl LineImportanceDetector for StubDetector {
        fn score(&self, line: &str, _ctx: ImportanceContext) -> ImportanceSignal {
            if line.contains("KEEP") {
                ImportanceSignal::matched(ImportanceCategory::Importance, 0.9, 0.9)
            } else {
                ImportanceSignal::neutral()
            }
        }
    }

    fn ctx() -> CompressionContext {
        CompressionContext::default()
    }

    // ── Empty-pipeline tests ─────────────────────────────────────────

    #[test]
    fn empty_pipeline_passes_input_through_unchanged() {
        let p = CompressionPipeline::builder().build();
        let input = "hello world";
        let r = p.run(input, ContentType::PlainText, &ctx());
        assert_eq!(r.output, input);
        assert_eq!(r.bytes_saved, 0);
        assert!(r.steps_applied.is_empty());
        assert!(r.structure_preserved);
        assert!(r.reversible_via.is_none());
    }

    #[test]
    fn pipeline_with_no_applicable_transforms_passes_through() {
        // Register a JSON-only transform; run on plain text.
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .build();
        let r = p.run("not json", ContentType::PlainText, &ctx());
        assert_eq!(r.output, "not json");
        assert!(r.steps_applied.is_empty());
    }

    // ── Lossless phase ───────────────────────────────────────────────

    #[test]
    fn lossless_runs_when_applicable() {
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .build();
        let pretty = r#"{
  "a": 1,
  "b": [1, 2, 3]
}"#;
        let r = p.run(pretty, ContentType::JsonArray, &ctx());
        assert!(r.bytes_saved > 0);
        assert_eq!(r.steps_applied, vec!["json_minifier".to_string()]);
        assert!(r.structure_preserved);
        assert!(r.output.len() < pretty.len());
    }

    #[test]
    fn lossless_rejects_below_min_savings_ratio() {
        // Compact JSON minifies to itself → bytes_saved == 0 → rejected.
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .build();
        let r = p.run(r#"{"a":1}"#, ContentType::JsonArray, &ctx());
        assert_eq!(r.bytes_saved, 0);
        assert!(r.steps_applied.is_empty());
    }

    #[test]
    fn lossless_continues_past_error() {
        let p = CompressionPipeline::builder()
            .with_lossless(AlwaysErrors)
            .with_lossless(TrivialLossless)
            .with_config(PipelineConfig {
                min_savings_ratio: 0.0, // accept any savings
                lossless_target_ratio: 0.0,
            })
            .build();
        let r = p.run("hello\n", ContentType::PlainText, &ctx());
        // Both transforms run; first errors and is recorded as skip,
        // second saves 1 byte. Without the per-error-recovery loop the
        // pipeline would short-circuit.
        assert_eq!(r.steps_applied, vec!["trivial_lossless".to_string()]);
        assert_eq!(r.bytes_saved, 1);
    }

    #[test]
    fn lossless_stops_once_target_ratio_reached() {
        struct HalvesContent;
        impl LosslessTransform for HalvesContent {
            fn name(&self) -> &'static str {
                "halver"
            }
            fn applies_to(&self) -> &[ContentType] {
                &[ContentType::PlainText]
            }
            fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
                let half = &content[..content.len() / 2];
                Ok(TransformResult::from_lengths(
                    content.len(),
                    half.to_string(),
                    true,
                ))
            }
        }
        // First halver: 100 → 50 bytes (ratio 0.5 — at the gate).
        // Second halver: should NOT run because ratio is now <= 0.5.
        let p = CompressionPipeline::builder()
            .with_lossless(HalvesContent)
            .with_lossless(HalvesContent)
            .build();
        let input = "x".repeat(100);
        let r = p.run(&input, ContentType::PlainText, &ctx());
        assert_eq!(r.steps_applied.len(), 1);
        assert_eq!(r.output.len(), 50);
    }

    // ── Lossy phase ──────────────────────────────────────────────────

    #[test]
    fn lossy_runs_after_lossless() {
        let lif = LineImportanceFilter::new(Box::new(StubDetector)).with_config(
            LineImportanceFilterConfig {
                min_priority: 0.5,
                keep_first: 1,
                keep_last: 1,
                context: ImportanceContext::Text,
            },
        );
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier) // doesn't apply to plain text
            .with_lossy(lif)
            .build();
        let input = (0..20)
            .map(|i| {
                if i == 10 {
                    "KEEP me".into()
                } else {
                    format!("noise {i}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let r = p.run(&input, ContentType::PlainText, &ctx());
        assert!(r.bytes_saved > 0);
        assert_eq!(r.steps_applied, vec!["line_importance_filter".to_string()]);
        assert!(!r.structure_preserved);
        assert!(r.output.contains("KEEP me"));
    }

    #[test]
    fn lossy_after_lossless_compounds_savings() {
        // Run JsonMinifier on JsonArray (saves whitespace), then a
        // hypothetical lossy on top. We just verify the orchestrator
        // accumulates bytes_saved across phases.
        struct JsonAsTextLossy;
        impl LossyTransform for JsonAsTextLossy {
            fn name(&self) -> &'static str {
                "json_truncator"
            }
            fn applies_to(&self) -> &[ContentType] {
                &[ContentType::JsonArray]
            }
            fn apply(
                &self,
                content: &str,
                _ctx: &CompressionContext,
            ) -> Result<TransformResult, TransformError> {
                let cut = content.len() / 2;
                Ok(TransformResult {
                    output: format!("{}…", &content[..cut]),
                    bytes_saved: content.len().saturating_sub(cut + 3),
                    structure_preserved: false,
                    reversible_via: None,
                })
            }
            fn confidence(&self) -> f32 {
                0.4
            }
        }
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .with_lossy(JsonAsTextLossy)
            .build();
        let pretty = r#"{
  "users": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
}"#;
        let r = p.run(pretty, ContentType::JsonArray, &ctx());
        assert_eq!(
            r.steps_applied,
            vec!["json_minifier".to_string(), "json_truncator".to_string()]
        );
        assert!(!r.structure_preserved);
        assert!(r.bytes_saved > 0);
    }

    // ── Structure-preservation flag ─────────────────────────────────

    #[test]
    fn structure_preserved_stays_true_when_only_lossless_runs() {
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .build();
        let r = p.run(r#"{ "a": 1, "b": 2 }"#, ContentType::JsonArray, &ctx());
        assert!(r.structure_preserved);
    }

    #[test]
    fn structure_preserved_flips_when_lossy_runs() {
        let lif = LineImportanceFilter::new(Box::new(StubDetector));
        let p = CompressionPipeline::builder()
            .with_lossy(lif)
            .with_config(PipelineConfig {
                min_savings_ratio: 0.0,
                lossless_target_ratio: 0.5,
            })
            .build();
        let many = (0..30)
            .map(|i| format!("plain line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let r = p.run(&many, ContentType::PlainText, &ctx());
        if !r.steps_applied.is_empty() {
            assert!(!r.structure_preserved);
        }
    }

    // ── Acceptance gate edge cases ───────────────────────────────────

    #[test]
    fn is_acceptable_handles_zero_lengths() {
        let p = CompressionPipeline::builder().build();
        assert!(!p.is_acceptable(0, 100), "no savings → reject");
        assert!(!p.is_acceptable(50, 0), "current_len 0 → reject");
    }

    #[test]
    fn is_acceptable_uses_min_savings_ratio() {
        let p = CompressionPipeline::builder()
            .with_config(PipelineConfig {
                min_savings_ratio: 0.10,
                lossless_target_ratio: 0.5,
            })
            .build();
        assert!(!p.is_acceptable(5, 100), "5% < 10% threshold → reject");
        assert!(p.is_acceptable(15, 100), "15% > 10% threshold → accept");
    }

    #[test]
    fn should_keep_running_lossless_handles_zero_original() {
        let p = CompressionPipeline::builder().build();
        assert!(!p.should_keep_running_lossless(0, 0));
    }

    // ── Builder-side properties ──────────────────────────────────────

    #[test]
    fn builder_dispatches_by_applies_to() {
        let p = CompressionPipeline::builder()
            .with_lossless(JsonMinifier)
            .with_lossless(TrivialLossless)
            .build();
        // JsonMinifier registered for JsonArray; TrivialLossless for
        // PlainText. Each should be the only one available for its
        // own type.
        assert_eq!(p.lossless_by_type[&ContentType::JsonArray].len(), 1);
        assert_eq!(p.lossless_by_type[&ContentType::PlainText].len(), 1);
    }

    #[test]
    fn builder_preserves_registration_order() {
        struct A;
        impl LosslessTransform for A {
            fn name(&self) -> &'static str {
                "a"
            }
            fn applies_to(&self) -> &[ContentType] {
                &[ContentType::PlainText]
            }
            fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
                Ok(TransformResult::from_lengths(
                    content.len(),
                    content.into(),
                    true,
                ))
            }
        }
        struct B;
        impl LosslessTransform for B {
            fn name(&self) -> &'static str {
                "b"
            }
            fn applies_to(&self) -> &[ContentType] {
                &[ContentType::PlainText]
            }
            fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
                Ok(TransformResult::from_lengths(
                    content.len(),
                    content.into(),
                    true,
                ))
            }
        }
        let p = CompressionPipeline::builder()
            .with_lossless(A)
            .with_lossless(B)
            .build();
        let order: Vec<&str> = p.lossless_by_type[&ContentType::PlainText]
            .iter()
            .map(|t| t.name())
            .collect();
        assert_eq!(order, vec!["a", "b"]);
    }
}

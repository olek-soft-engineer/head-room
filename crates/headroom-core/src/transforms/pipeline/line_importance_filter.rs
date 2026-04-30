//! `LineImportanceFilter` — drop low-priority lines using the signals
//! trait from Phase 3e.1.
//!
//! Smallest useful lossy transform: walk lines, score each via a
//! [`LineImportanceDetector`] (the trait shipped in Phase 3e.1), keep
//! lines whose `priority` is at or above a threshold, plus unconditional
//! anchors at the head and tail. Gaps between kept lines collapse into
//! `[... N lines omitted ...]` markers so the downstream LLM has a
//! visible signal that bytes were dropped.
//!
//! # Why this is lossy
//!
//! Dropped lines are gone — the filter does not emit a CCR retrieval
//! handle in PR1 (CCR offload moves into the pipeline in PR3). The
//! omission markers carry coarse "something was here" signal but not
//! the original bytes. Hence `structure_preserved = false` and
//! `confidence = 0.7` (calibrated from the detector's own confidence
//! aggregate; tuned later in PR4 once we have a labeled corpus).
//!
//! # No regex
//!
//! Walks `str::lines()`. The detector itself is aho-corasick + ASCII
//! word-boundary post-filter, also no regex.

use crate::signals::{ImportanceContext, LineImportanceDetector};
use crate::transforms::pipeline::traits::{
    CompressionContext, LossyTransform, TransformError, TransformResult,
};
use crate::transforms::ContentType;

/// Configuration knobs for [`LineImportanceFilter`].
#[derive(Debug, Clone)]
pub struct LineImportanceFilterConfig {
    /// Minimum priority for a line to survive the filter. Lines with
    /// `priority < min_priority` get dropped (unless they're inside
    /// the head/tail anchor windows). Default `0.4` — picked to keep
    /// warning/importance lines (priority 0.5) and drop neutral lines
    /// (priority 0.0) without aggressively trimming borderline cases.
    pub min_priority: f32,
    /// Always keep the first N lines. Header context, banner, summary
    /// — losing these breaks the LLM's ability to interpret what
    /// follows.
    pub keep_first: usize,
    /// Always keep the last N lines. Trailing summaries, exit codes,
    /// totals — same logic as the head anchor.
    pub keep_last: usize,
    /// Importance context to pass to the detector. Different contexts
    /// fire different keyword sets (markdown structure matters in
    /// `Text`, doesn't in `Diff`).
    pub context: ImportanceContext,
}

impl Default for LineImportanceFilterConfig {
    fn default() -> Self {
        Self {
            min_priority: 0.4,
            keep_first: 5,
            keep_last: 5,
            context: ImportanceContext::Text,
        }
    }
}

/// Drop low-priority lines, keep anchors at head/tail.
pub struct LineImportanceFilter {
    detector: Box<dyn LineImportanceDetector>,
    applies_to: Vec<ContentType>,
    config: LineImportanceFilterConfig,
}

impl LineImportanceFilter {
    pub const NAME: &'static str = "line_importance_filter";

    /// Build a filter with the given detector and the default config.
    /// `applies_to` defaults to the broad set of line-based content
    /// types — caller can narrow via [`with_applies_to`].
    ///
    /// [`with_applies_to`]: Self::with_applies_to
    pub fn new(detector: Box<dyn LineImportanceDetector>) -> Self {
        Self {
            detector,
            applies_to: vec![
                ContentType::PlainText,
                ContentType::BuildOutput,
                ContentType::SearchResults,
            ],
            config: LineImportanceFilterConfig::default(),
        }
    }

    pub fn with_config(mut self, config: LineImportanceFilterConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_applies_to(mut self, types: Vec<ContentType>) -> Self {
        self.applies_to = types;
        self
    }
}

impl LossyTransform for LineImportanceFilter {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        &self.applies_to
    }

    fn apply(
        &self,
        content: &str,
        _ctx: &CompressionContext,
    ) -> Result<TransformResult, TransformError> {
        if content.is_empty() {
            return Err(TransformError::skipped(Self::NAME, "empty input"));
        }

        // `str::lines` iterates without retaining line terminators; we
        // collect once because the keep-decision per line depends on
        // the total count (head/tail anchors).
        let lines: Vec<&str> = content.lines().collect();
        let n = lines.len();
        if n == 0 {
            return Err(TransformError::skipped(Self::NAME, "no lines"));
        }

        // Score each line and decide keep-or-drop.
        let mut keep = vec![false; n];
        let head_end = self.config.keep_first.min(n);
        let tail_start = n.saturating_sub(self.config.keep_last);
        for (i, line) in lines.iter().enumerate() {
            if i < head_end || i >= tail_start {
                keep[i] = true;
                continue;
            }
            let signal = self.detector.score(line, self.config.context);
            if signal.priority >= self.config.min_priority {
                keep[i] = true;
            }
        }

        // Build output, collapsing dropped runs into omission markers.
        let mut out = String::with_capacity(content.len());
        let mut last_kept: Option<usize> = None;
        for (i, line) in lines.iter().enumerate() {
            if !keep[i] {
                continue;
            }
            if let Some(prev) = last_kept {
                let gap = i - prev - 1;
                if gap > 0 {
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(&format!("[... {gap} lines omitted ...]"));
                }
            }
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(line);
            last_kept = Some(i);
        }
        if let Some(prev) = last_kept {
            let gap = n - prev - 1;
            if gap > 0 {
                out.push('\n');
                out.push_str(&format!("[... {gap} lines omitted ...]"));
            }
        }

        // Bytes-saved is computed against the original content, not
        // the line-iterated reconstruction (which can differ by the
        // trailing newline). We account for that by using the
        // input length directly.
        let bytes_saved = content.len().saturating_sub(out.len());
        Ok(TransformResult {
            output: out,
            bytes_saved,
            structure_preserved: false,
            reversible_via: None,
        })
    }

    fn confidence(&self) -> f32 {
        // Calibrated 0.7: the detector is reliable on keyword signals
        // (priority 0.5+) but has no view into semantic relevance to
        // the user's query — that's what PR4 ProseFieldCompressor
        // adds. 0.7 is "decent extractive baseline."
        0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::{ImportanceCategory, ImportanceSignal};

    /// Test detector with deterministic priorities — keyed on a
    /// substring match. Avoids spinning up the real keyword detector
    /// for unit tests.
    struct StubDetector {
        priority_keyword: &'static str,
        priority: f32,
    }
    impl LineImportanceDetector for StubDetector {
        fn score(&self, line: &str, _ctx: ImportanceContext) -> ImportanceSignal {
            if line.contains(self.priority_keyword) {
                ImportanceSignal::matched(ImportanceCategory::Importance, self.priority, 0.9)
            } else {
                ImportanceSignal::neutral()
            }
        }
    }

    fn run(filter: &LineImportanceFilter, content: &str) -> TransformResult {
        filter
            .apply(content, &CompressionContext::default())
            .expect("test inputs are well-formed")
    }

    fn filter_keep_keyword() -> LineImportanceFilter {
        LineImportanceFilter::new(Box::new(StubDetector {
            priority_keyword: "KEEP",
            priority: 0.9,
        }))
        .with_config(LineImportanceFilterConfig {
            min_priority: 0.5,
            keep_first: 1,
            keep_last: 1,
            context: ImportanceContext::Text,
        })
    }

    #[test]
    fn name_matches_telemetry_convention() {
        let f = filter_keep_keyword();
        assert_eq!(f.name(), "line_importance_filter");
    }

    #[test]
    fn applies_to_default_set_includes_plain_text_logs_and_search() {
        let f = filter_keep_keyword();
        let types = f.applies_to();
        assert!(types.contains(&ContentType::PlainText));
        assert!(types.contains(&ContentType::BuildOutput));
        assert!(types.contains(&ContentType::SearchResults));
    }

    #[test]
    fn drops_low_priority_lines_keeps_high_priority() {
        let f = filter_keep_keyword();
        let input = (0..20)
            .map(|i| {
                if i == 10 {
                    "KEEP this line".into()
                } else {
                    format!("noise {i}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let result = run(&f, &input);
        assert!(result.output.contains("KEEP this line"));
        assert!(result.output.contains("[... ") && result.output.contains("lines omitted ...]"));
        assert!(result.bytes_saved > 0);
        assert!(!result.structure_preserved);
    }

    #[test]
    fn always_keeps_first_and_last_anchor_windows() {
        let f = filter_keep_keyword();
        let lines: Vec<String> = (0..20).map(|i| format!("plain line {i}")).collect();
        let input = lines.join("\n");
        let result = run(&f, &input);
        // First line and last line must survive even when nothing
        // matches the keyword.
        assert!(result.output.contains("plain line 0"));
        assert!(result.output.contains("plain line 19"));
    }

    #[test]
    fn empty_input_returns_skipped_error() {
        let f = filter_keep_keyword();
        let err = f
            .apply("", &CompressionContext::default())
            .expect_err("empty input is a skip");
        assert!(matches!(err, TransformError::Skipped { .. }));
    }

    #[test]
    fn single_line_input_passes_through() {
        let f = filter_keep_keyword();
        let result = run(&f, "only one line");
        // Head anchor keeps it; no gap markers possible.
        assert_eq!(result.output, "only one line");
        assert_eq!(result.bytes_saved, 0);
    }

    #[test]
    fn omission_markers_count_dropped_lines_correctly() {
        // 5 lines: keep 1st (head anchor), drop middle 3 (no keyword),
        // keep last (tail anchor). Expect ONE marker reporting 3
        // omitted lines.
        let f = filter_keep_keyword();
        let input = ["first", "drop a", "drop b", "drop c", "last"].join("\n");
        let result = run(&f, &input);
        assert!(result.output.starts_with("first"));
        assert!(result.output.ends_with("last"));
        assert!(result.output.contains("[... 3 lines omitted ...]"));
    }

    #[test]
    fn confidence_is_calibrated_constant_in_pr1() {
        let f = filter_keep_keyword();
        assert!((f.confidence() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn structure_preserved_flag_is_always_false() {
        let f = filter_keep_keyword();
        let result = run(&f, "first\nKEEP me\nlast");
        assert!(!result.structure_preserved);
    }

    #[test]
    fn anchor_windows_overlap_safely_on_short_input() {
        // 3 lines, keep_first=5, keep_last=5 — windows fully overlap,
        // every line should survive without panicking on the index
        // arithmetic.
        let f = LineImportanceFilter::new(Box::new(StubDetector {
            priority_keyword: "_",
            priority: 0.9,
        }))
        .with_config(LineImportanceFilterConfig {
            min_priority: 0.5,
            keep_first: 5,
            keep_last: 5,
            context: ImportanceContext::Text,
        });
        let result = run(&f, "a\nb\nc");
        assert_eq!(result.output, "a\nb\nc");
    }

    #[test]
    fn applies_to_can_be_narrowed() {
        let f = filter_keep_keyword().with_applies_to(vec![ContentType::SearchResults]);
        assert_eq!(f.applies_to(), &[ContentType::SearchResults]);
    }

    #[test]
    fn unicode_lines_handled_safely() {
        let f = filter_keep_keyword();
        let result = run(&f, "first\nKEEP héllo wörld 🌍\nlast");
        assert!(result.output.contains("héllo wörld 🌍"));
    }
}

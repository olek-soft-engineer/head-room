//! `JsonMinifier` — round-trip whitespace removal for JSON content.
//!
//! Smallest useful lossless transform: parse with `serde_json`, write
//! with `serde_json::to_string` (which omits all decorative
//! whitespace). The bytes saved depends on how indented the source is
//! — pretty-printed JSON typically shrinks by ~25–35%, already-
//! compact JSON saves ~0%. The orchestrator's
//! [`min_savings_ratio`] gate rejects no-savings runs automatically.
//!
//! [`min_savings_ratio`]: super::orchestrator::PipelineConfig::min_savings_ratio
//!
//! # Why not strip whitespace by hand
//!
//! Hand-rolled minification has to handle string literals, escape
//! sequences, and Unicode separator characters correctly. `serde_json`
//! already does that — and a parse-then-serialize roundtrip
//! validates the input as a side effect, which means downstream
//! consumers can trust the output is well-formed. The cost is one
//! `Value` allocation per call, which is dwarfed by the I/O the proxy
//! is already doing on the same payload.
//!
//! # No regex
//!
//! Pure `serde_json`. No regex, no hand-rolled scanner.

use crate::transforms::pipeline::traits::{LosslessTransform, TransformError, TransformResult};
use crate::transforms::ContentType;

/// Minify any JSON document by parse-then-serialize round-trip.
#[derive(Debug, Default, Clone, Copy)]
pub struct JsonMinifier;

impl JsonMinifier {
    pub const NAME: &'static str = "json_minifier";

    /// The content types this transform claims. Static slice — the
    /// applicability is fixed by the implementation, not configured
    /// per instance.
    const APPLIES_TO: &'static [ContentType] = &[ContentType::JsonArray];
}

impl LosslessTransform for JsonMinifier {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn applies_to(&self) -> &[ContentType] {
        Self::APPLIES_TO
    }

    fn apply(&self, content: &str) -> Result<TransformResult, TransformError> {
        if content.is_empty() {
            return Err(TransformError::skipped(Self::NAME, "empty input"));
        }
        // `from_str` validates the input as a side effect of building
        // the in-memory tree. Any structural error becomes
        // InvalidInput so the orchestrator can move on.
        let value: serde_json::Value = serde_json::from_str(content).map_err(|e| {
            TransformError::invalid_input(Self::NAME, format!("not valid json: {e}"))
        })?;
        let minified = serde_json::to_string(&value)
            .map_err(|e| TransformError::internal(Self::NAME, format!("serialize failed: {e}")))?;
        Ok(TransformResult::from_lengths(content.len(), minified, true))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(input: &str) -> Result<TransformResult, TransformError> {
        JsonMinifier.apply(input)
    }

    #[test]
    fn name_matches_telemetry_convention() {
        // Lowercase snake_case so the strategy-stats JSONB nest stays
        // clean.
        assert_eq!(JsonMinifier.name(), "json_minifier");
    }

    #[test]
    fn applies_to_json_array() {
        assert_eq!(JsonMinifier.applies_to(), &[ContentType::JsonArray]);
    }

    #[test]
    fn pretty_printed_json_shrinks() {
        let pretty = r#"{
  "name": "Alice",
  "age":   30,
  "tags": [
    "engineer",
    "manager"
  ]
}"#;
        let result = run(pretty).expect("valid json should compress");
        assert!(result.bytes_saved > 0);
        assert!(result.structure_preserved);
        assert!(!result.output.contains('\n'));
        assert!(!result.output.contains("  "));
        // Round-trips back to the same logical value.
        let v_in: serde_json::Value = serde_json::from_str(pretty).unwrap();
        let v_out: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(v_in, v_out);
    }

    #[test]
    fn already_compact_json_returns_zero_savings() {
        let compact = r#"{"a":1,"b":[1,2,3]}"#;
        let result = run(compact).expect("compact json is still valid");
        // serde_json's canonical form may match the input byte-for-byte
        // here; the important contract is bytes_saved == 0 (orchestrator
        // will reject this in is_acceptable).
        assert_eq!(result.bytes_saved, 0);
    }

    #[test]
    fn empty_string_is_skipped_not_invalid() {
        let err = run("").expect_err("empty input is a skip, not an error");
        assert!(matches!(err, TransformError::Skipped { .. }));
    }

    #[test]
    fn malformed_json_is_invalid_input() {
        let err = run("{not json").expect_err("garbage is invalid input");
        assert!(matches!(err, TransformError::InvalidInput { .. }));
    }

    #[test]
    fn nested_arrays_round_trip() {
        let nested = r#"{
  "users": [
    {"id": 1, "name": "a"},
    {"id": 2, "name": "b"}
  ]
}"#;
        let result = run(nested).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(parsed["users"][0]["id"], 1);
        assert_eq!(parsed["users"][1]["name"], "b");
        assert!(result.bytes_saved > 0);
    }

    #[test]
    fn unicode_strings_preserved() {
        let input = r#"{ "greeting": "héllo wörld 🌍" }"#;
        let result = run(input).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert_eq!(parsed["greeting"], "héllo wörld 🌍");
    }

    #[test]
    fn structure_preserved_flag_is_always_true() {
        // Lossless by construction.
        let result = run(r#"{"a": 1}"#).unwrap();
        assert!(result.structure_preserved);
        assert!(result.reversible_via.is_none());
    }

    #[test]
    fn deeply_nested_does_not_blow_up() {
        // 50 levels of nesting — well under any reasonable parse limit
        // but enough to catch a recursion bug if we ever introduce one.
        let mut s = String::new();
        for _ in 0..50 {
            s.push('[');
        }
        s.push('1');
        for _ in 0..50 {
            s.push(']');
        }
        let result = run(&s).unwrap();
        // Parsing+serializing a deeply nested array doesn't add or
        // remove much — important is that we don't panic or fail.
        let parsed: serde_json::Value = serde_json::from_str(&result.output).unwrap();
        assert!(parsed.is_array());
    }
}

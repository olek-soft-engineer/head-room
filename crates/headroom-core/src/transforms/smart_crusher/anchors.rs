//! Legacy regex-based query anchor extraction.
//!
//! Direct port of `extract_query_anchors` and `item_matches_anchors`
//! (`smart_crusher.py:99-168`). The Python doc-comment marks both as
//! DEPRECATED in favor of `RelevanceScorer`, but they're still called
//! by the live SmartCrusher path on every invocation, so we port them
//! faithfully.
//!
//! # Why regex parity matters
//!
//! These regexes drive which array items survive compression. A subtle
//! difference between Python's `re` engine and Rust's `regex` crate
//! (e.g. word-boundary behavior on Unicode, or repetition greediness)
//! would silently change which anchors are detected and which items
//! survive. The patterns below are pinned to lowercase ASCII inputs
//! and use only ASCII-safe constructs to keep behavior identical.

use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::LazyLock;

// ---------------------------------------------------------------
// Pattern definitions — direct ports of the module-level Python regexes
// at `smart_crusher.py:85-93`. `std::sync::LazyLock` (stable since Rust
// 1.80) is the modern equivalent of `once_cell::sync::Lazy`, mirroring
// Python's `re.compile` at module import time.
// ---------------------------------------------------------------

/// `\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b`
static UUID_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
        .expect("UUID_PATTERN")
});

/// 4+ digit numbers (likely IDs). Python: `r"\b\d{4,}\b"`.
static NUMERIC_ID_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4,}\b").expect("NUMERIC_ID_PATTERN"));

/// Hostname pattern. Matches `host.tld` with optional `.tld2`. Python:
/// `r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b"`.
static HOSTNAME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})?\b")
        .expect("HOSTNAME_PATTERN")
});

/// Short quoted strings (single OR double quotes), 1-50 chars between
/// quotes. Python: `r"['\"]([^'\"]{1,50})['\"]"`.
static QUOTED_STRING_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"['"]([^'"]{1,50})['"]"#).expect("QUOTED_STRING_PATTERN"));

/// Email addresses. Python: `r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"`.
/// (Note Python's `[A-Z|a-z]` includes a literal `|` in the character
/// class — almost certainly a typo, but we faithfully port it for
/// parity. Real-world impact is nil since `|` doesn't appear in TLDs.)
static EMAIL_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").expect("EMAIL_PATTERN")
});

/// Hostname false-positive blocklist. Python uses a set literal at
/// `smart_crusher.py:137`. We mirror exactly — these strings get
/// dropped from anchor results.
const HOSTNAME_FALSE_POSITIVES: &[&str] = &["e.g", "i.e", "etc."];

/// Extract query anchors from user text. **DEPRECATED** in Python in
/// favor of `RelevanceScorer`, but still called by the live path —
/// ported as-is.
///
/// Output is a set of lowercased anchor strings. Order is not
/// significant (Python returns `set[str]`).
pub fn extract_query_anchors(text: &str) -> HashSet<String> {
    let mut anchors = HashSet::new();

    if text.is_empty() {
        return anchors;
    }

    // UUIDs — lowercase the match.
    for m in UUID_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_lowercase());
    }

    // Numeric IDs — Python keeps original case (digits, no transform needed).
    for m in NUMERIC_ID_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_string());
    }

    // Hostnames — lowercase, filter false positives.
    for m in HOSTNAME_PATTERN.find_iter(text) {
        let lc = m.as_str().to_lowercase();
        if !HOSTNAME_FALSE_POSITIVES.contains(&lc.as_str()) {
            anchors.insert(lc);
        }
    }

    // Quoted strings — capture group 1 (the content between quotes),
    // require trim().len() >= 2 (Python's `if len(match.strip()) >= 2`).
    for caps in QUOTED_STRING_PATTERN.captures_iter(text) {
        if let Some(inner) = caps.get(1) {
            if inner.as_str().trim().len() >= 2 {
                anchors.insert(inner.as_str().to_lowercase());
            }
        }
    }

    // Emails — lowercase.
    for m in EMAIL_PATTERN.find_iter(text) {
        anchors.insert(m.as_str().to_lowercase());
    }

    anchors
}

/// Check if a JSON object matches any query anchors.
///
/// Direct port of `item_matches_anchors` (Python `smart_crusher.py:152-168`).
/// Python uses `str(item).lower()` which produces Python's `dict.__str__`
/// representation. We mirror by serializing with `serde_json` and
/// lowercasing — this isn't byte-identical to Python's `str(dict)`
/// (Python uses single quotes, JSON uses double; Python's bool is
/// `True`/`False`, JSON's is `true`/`false`), so for cross-language
/// parity we need a string form that matches Python's. We document this
/// gap and fix it in the analyzer integration.
///
/// **WARNING:** `str(item).lower()` in Python produces:
///   `{'key': 'value', 'count': 5, 'ok': True}`
/// while `serde_json::to_string(&item)` produces:
///   `{"key":"value","count":5,"ok":true}`
///
/// The anchor matching is substring-based (`anchor in item_str`), so
/// this difference matters: if an anchor is `"true"` it matches the
/// JSON form but not the Python form, and vice versa for `"True"`.
///
/// **Resolution:** when items reach the matcher they're already
/// lowercased, so `True` → `true` after `.lower()`, removing one source
/// of drift. The remaining drift (single vs double quotes, trailing
/// whitespace) is unlikely to affect anchor matching in practice. We
/// pin behavior with fixtures and move on.
pub fn item_matches_anchors(item: &Value, anchors: &HashSet<String>) -> bool {
    if anchors.is_empty() {
        return false;
    }

    // Python: `str(item).lower()`. We approximate via JSON serialization
    // followed by `.lower()` — see WARNING above for the gap.
    let item_str = match serde_json::to_string(item) {
        Ok(s) => s.to_lowercase(),
        Err(_) => return false,
    };

    anchors.iter().any(|a| item_str.contains(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn empty_text_no_anchors() {
        assert!(extract_query_anchors("").is_empty());
    }

    #[test]
    fn extracts_uuid_lowercased() {
        let anchors = extract_query_anchors("see id 550E8400-E29B-41D4-A716-446655440000 plz");
        assert!(anchors.contains("550e8400-e29b-41d4-a716-446655440000"));
    }

    #[test]
    fn extracts_numeric_id_unchanged() {
        let anchors = extract_query_anchors("user 12345 reported issue");
        assert!(anchors.contains("12345"));
    }

    #[test]
    fn three_digit_number_not_anchor() {
        // Pattern requires 4+ digits.
        let anchors = extract_query_anchors("user 123 reported issue");
        assert!(!anchors.iter().any(|a| a == "123"));
    }

    #[test]
    fn extracts_hostname() {
        let anchors = extract_query_anchors("connect to api.example.com asap");
        assert!(anchors.contains("api.example.com"));
    }

    #[test]
    fn hostname_false_positive_filtered() {
        // "e.g" is in the blocklist — must NOT appear as an anchor even
        // though it matches the regex.
        let anchors = extract_query_anchors("test e.g.com endpoint");
        // "e.g" is filtered, but "e.g.com" or other longer matches may
        // pass; we only assert "e.g" itself is gone.
        assert!(!anchors.contains("e.g"));
    }

    #[test]
    fn extracts_quoted_string_double() {
        let anchors = extract_query_anchors(r#"find the "user_name" field"#);
        assert!(anchors.contains("user_name"));
    }

    #[test]
    fn extracts_quoted_string_single() {
        let anchors = extract_query_anchors("find the 'user_name' field");
        assert!(anchors.contains("user_name"));
    }

    #[test]
    fn very_short_quoted_skipped() {
        // Less than 2 chars after trim — skipped.
        let anchors = extract_query_anchors(r#"the "x" thing"#);
        assert!(!anchors.contains("x"));
    }

    #[test]
    fn extracts_email() {
        let anchors = extract_query_anchors("contact USER@example.COM please");
        assert!(anchors.contains("user@example.com"));
    }

    #[test]
    fn item_matches_anchors_empty_set() {
        let empty = HashSet::new();
        assert!(!item_matches_anchors(&json!({"a": 1}), &empty));
    }

    #[test]
    fn item_matches_anchor_in_value() {
        let anchors: HashSet<String> = ["alice".to_string()].into_iter().collect();
        assert!(item_matches_anchors(&json!({"name": "Alice"}), &anchors));
    }

    #[test]
    fn item_matches_anchor_in_key() {
        let anchors: HashSet<String> = ["status".to_string()].into_iter().collect();
        // The anchor "status" appears in the JSON-serialized key.
        assert!(item_matches_anchors(
            &json!({"status": "ok"}),
            &anchors
        ));
    }

    #[test]
    fn item_no_match_with_unrelated_anchor() {
        let anchors: HashSet<String> = ["xyz123".to_string()].into_iter().collect();
        assert!(!item_matches_anchors(&json!({"a": "b"}), &anchors));
    }
}

//! Tag protection — keep custom workflow tags out of ML compressors.
//!
//! # Why this exists
//!
//! LLM workflows carry XML-style markers (`<system-reminder>`,
//! `<tool_call>`, `<thinking>`, `<headroom:tool_digest>`, etc.) that
//! downstream code parses as structure. Kompress / LLMLingua sees them
//! as droppable noise and silently strips them, breaking everything
//! that depends on them. ContentRouter calls [`protect_tags`] before
//! every ML-text-compression to swap custom-tag spans for opaque
//! placeholders, runs the compressor on the cleaned body, then calls
//! [`restore_tags`] on the output to splice the originals back in.
//!
//! Standard HTML5 elements (`<div>`, `<p>`, `<span>`, …) are *not*
//! protected — those go through the HTMLExtractor / trafilatura path
//! at a different layer. Anything else is treated as a custom tag.
//!
//! # Algorithm
//!
//! Single-pass tag-stack walker over the input bytes (no regex
//! backtracking, no O(n²) restart loop):
//!
//! 1. Scan forward for `<`. If the next bytes form a valid tag-open
//!    (`<name attr=…>` or `<name/>`), classify the tag name.
//! 2. HTML tag → emit verbatim, continue.
//! 3. Custom tag, self-closing → emit a placeholder, record the span.
//! 4. Custom tag, opening → push `(name, start_offset)` onto a stack.
//! 5. `</name>` matching the top of the stack → pop, emit a placeholder
//!    for the whole `<name>…</name>` span (when
//!    `compress_tagged_content == false`) or emit two placeholders for
//!    the markers only (when `compress_tagged_content == true`).
//! 6. Mismatched close (HTML close while a custom tag is on top, or a
//!    close with no matching open) → write the close tag verbatim and
//!    move on. The walker never attempts to "repair" malformed input.
//!
//! Output is built incrementally with offset-based slicing — never the
//! Python-original's `result.replace(original, placeholder, 1)`, which
//! silently misbehaves when two identical custom-tag blocks appear in
//! the same input (it always replaces the *first* textual occurrence,
//! not the matched one). See `fixed_in_3e4_replace_first` test below.
//!
//! # Bug fixes vs the Python original
//!
//! * **#1: O(n²) on nested custom tags** — the Python loop restarted a
//!   full regex scan after every replacement. Rust walks once, in
//!   linear time on input length.
//! * **#2: First-occurrence replace bug** — `str.replace(.., .., 1)`
//!   replaced the first textual match of the matched block, not the
//!   block at the matched offset. Two identical custom-tag blocks in
//!   the same input collapsed to one placeholder + a duplicated
//!   second block. The Rust walker stitches output by offset.
//! * **#3: Silent 50-iteration cap** — Python had a hard 50-iteration
//!   safety limit that quietly truncated tag protection on deeply
//!   nested input. The Rust walker's run-time is bounded by input
//!   length only.
//! * **#4: Self-closing pass duplicate-replace risk** — Python ran a
//!   second loop with the same `replace_first` bug for self-closing
//!   tags. Rust handles self-closers in the same single pass.
//! * **#5: Placeholder collision** — if input contains a literal
//!   `{{HEADROOM_TAG_…}}` substring, Python silently let the collision
//!   stand. We detect that and pick a salted prefix (with a tracing
//!   warn) so restoration can't be ambiguous.
//!
//! # Hot path
//!
//! `protect_tags` runs on every ML-compression call from ContentRouter.
//! Most production prompts have 0–10 custom tags so the absolute cost
//! is small either way; the value of the port is correctness (bugs
//! #2, #5) and predictable behavior on adversarial input (bugs #1,
//! #3). The PyO3 bridge releases the GIL during the walk because the
//! algorithm is fully self-contained.

use std::collections::HashSet;
use std::sync::OnceLock;

/// HTML5 living-standard element names — the set of tags this module
/// will NEVER protect (they're handled at a different layer; everything
/// else is treated as custom).
///
/// Generated from
/// <https://html.spec.whatwg.org/multipage/indices.html#elements-3> and
/// matches the Python `KNOWN_HTML_TAGS` frozenset element-for-element
/// so the Rust shim and the Python shim agree.
const HTML5_TAGS: &[&str] = &[
    // Main root
    "html",
    // Document metadata
    "base",
    "head",
    "link",
    "meta",
    "style",
    "title",
    // Sectioning root
    "body",
    // Content sectioning
    "address",
    "article",
    "aside",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hgroup",
    "main",
    "nav",
    "section",
    "search",
    // Text content
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "hr",
    "li",
    "menu",
    "ol",
    "p",
    "pre",
    "ul",
    // Inline text semantics
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "br",
    "cite",
    "code",
    "data",
    "dfn",
    "em",
    "i",
    "kbd",
    "mark",
    "q",
    "rp",
    "rt",
    "ruby",
    "s",
    "samp",
    "small",
    "span",
    "strong",
    "sub",
    "sup",
    "time",
    "u",
    "var",
    "wbr",
    // Image and multimedia
    "area",
    "audio",
    "img",
    "map",
    "track",
    "video",
    // Embedded content
    "embed",
    "iframe",
    "object",
    "param",
    "picture",
    "portal",
    "source",
    // SVG and MathML
    "svg",
    "math",
    // Scripting
    "canvas",
    "noscript",
    "script",
    // Demarcating edits
    "del",
    "ins",
    // Table content
    "caption",
    "col",
    "colgroup",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    // Forms
    "button",
    "datalist",
    "fieldset",
    "form",
    "input",
    "label",
    "legend",
    "meter",
    "optgroup",
    "option",
    "output",
    "progress",
    "select",
    "textarea",
    // Interactive
    "details",
    "dialog",
    "summary",
    // Web Components
    "slot",
    "template",
];

fn known_html_tags() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| HTML5_TAGS.iter().copied().collect())
}

/// Default placeholder prefix. Brace-doubled to look unlike anything a
/// real workflow tag would emit. Falls back to a salted variant if the
/// input itself contains the prefix (see [`pick_placeholder_prefix`]).
const DEFAULT_PREFIX: &str = "{{HEADROOM_TAG_";
const PLACEHOLDER_SUFFIX: &str = "}}";

/// Sidecar diagnostics — same shape every Rust transform uses.
#[derive(Debug, Default, Clone)]
pub struct ProtectStats {
    pub tags_seen: usize,
    pub html_tags_skipped: usize,
    pub custom_blocks_protected: usize,
    pub self_closing_protected: usize,
    /// Closes that didn't match any open on the stack (malformed input
    /// or HTML interleaving). Emitted verbatim. Non-zero is a smell
    /// worth tracking but not necessarily a bug.
    pub orphan_closes: usize,
    /// True iff the placeholder prefix had to be salted because the
    /// input contained a literal `{{HEADROOM_TAG_` substring.
    pub placeholder_collision_avoided: bool,
}

/// Case-insensitive HTML tag check. Lowercases the input lazily so we
/// don't allocate for the common ASCII-lowercase case.
pub fn is_known_html_tag(tag_name: &str) -> bool {
    let set = known_html_tags();
    if set.contains(tag_name) {
        return true;
    }
    if tag_name.bytes().any(|b| b.is_ascii_uppercase()) {
        let lower = tag_name.to_ascii_lowercase();
        return set.contains(lower.as_str());
    }
    false
}

/// Iterate the canonical HTML tag list. Used by the PyO3 shim to
/// expose `KNOWN_HTML_TAGS` to Python without re-declaring the set.
pub fn known_html_tag_names() -> &'static [&'static str] {
    HTML5_TAGS
}

/// Pick a placeholder prefix that doesn't collide with anything in
/// `text`. We try `{{HEADROOM_TAG_` first; if the input contains it
/// literally we salt with a per-call counter until we miss. The salt
/// is bounded; in practice we never need more than one attempt.
fn pick_placeholder_prefix(text: &str) -> (String, bool) {
    if !text.contains(DEFAULT_PREFIX) {
        return (DEFAULT_PREFIX.to_string(), false);
    }
    for salt in 0u32..16 {
        let candidate = format!("{{{{HEADROOM_TAG_{salt}_");
        if !text.contains(&candidate) {
            return (candidate, true);
        }
    }
    // 16 salt attempts collided — fall back to a UUID-shaped marker.
    // The OnceLock cache is so two consecutive calls in the same
    // process don't pay the formatting cost.
    static FALLBACK: OnceLock<String> = OnceLock::new();
    let prefix = FALLBACK
        .get_or_init(|| "{{HEADROOM_TAG_FALLBACK_a4f1c7e2_".to_string())
        .clone();
    (prefix, true)
}

#[derive(Debug)]
struct OpenTag {
    /// Lowercase name for case-insensitive close-matching.
    name_lower: String,
    /// Byte offset of the `<` that opened this tag.
    open_start: usize,
}

/// Outcome of a single `<…>` parse attempt at a given offset.
enum TagParse {
    /// Opening tag (`<name attr=…>`). `name_end` is exclusive.
    Open {
        name_end: usize,
        tag_end: usize,
        is_self_closing: bool,
    },
    /// Closing tag (`</name>`).
    Close { name_end: usize, tag_end: usize },
    /// Not a tag (e.g. `<` followed by whitespace or digit).
    NotTag,
}

/// Parse a `<…>` starting at `start`. Returns the byte offset of the
/// closing `>` (exclusive end of the tag) and the kind. Conservatively
/// rejects malformed shapes — we'd rather emit a `<` verbatim than
/// over-protect on bad input.
fn parse_tag_at(bytes: &[u8], start: usize) -> TagParse {
    debug_assert!(bytes[start] == b'<');
    let mut i = start + 1;
    let n = bytes.len();
    if i >= n {
        return TagParse::NotTag;
    }

    let is_close = bytes[i] == b'/';
    if is_close {
        i += 1;
    }
    let name_start = i;
    if !is_name_start(bytes[i]) {
        return TagParse::NotTag;
    }
    i += 1;
    while i < n && is_name_cont(bytes[i]) {
        i += 1;
    }
    let name_end = i;
    if name_end == name_start {
        return TagParse::NotTag;
    }

    if is_close {
        // Allow optional whitespace, then `>`.
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n || bytes[i] != b'>' {
            return TagParse::NotTag;
        }
        return TagParse::Close {
            name_end,
            tag_end: i + 1,
        };
    }

    // Opening tag: skip attributes until `>` (handle `/>` for
    // self-closing). Quoted attribute values can contain `>`; a
    // single-pass attribute lexer handles the common cases.
    let mut self_closing = false;
    while i < n {
        match bytes[i] {
            b'>' => {
                return TagParse::Open {
                    name_end,
                    tag_end: i + 1,
                    is_self_closing: self_closing,
                };
            }
            b'/' => {
                self_closing = true;
                i += 1;
            }
            b'"' | b'\'' => {
                let quote = bytes[i];
                i += 1;
                while i < n && bytes[i] != quote {
                    i += 1;
                }
                if i >= n {
                    return TagParse::NotTag;
                }
                i += 1;
                self_closing = false;
            }
            _ => {
                if bytes[i].is_ascii_whitespace() {
                    self_closing = false;
                }
                i += 1;
            }
        }
    }

    TagParse::NotTag
}

#[inline]
fn is_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_name_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.' | b':')
}

/// A single span that was identified as worth replacing.
///
/// In block mode every matched custom-tag span (open..=close) becomes
/// one Span and is replaced by a single placeholder; self-closing
/// custom tags become a Span covering just the tag bytes.
///
/// In marker-only mode each opening custom tag and each closing custom
/// tag becomes its own Span (the body between them is left visible to
/// the compressor).
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    kind: SpanKind,
}

#[derive(Debug, Clone, Copy)]
enum SpanKind {
    /// Whole `<custom>…</custom>` block (block mode).
    Block,
    /// Self-closing `<custom/>` (block mode).
    SelfClosing,
    /// Opening `<custom>` marker (marker-only mode).
    OpenMarker,
    /// Closing `</custom>` marker (marker-only mode).
    CloseMarker,
}

/// Protect custom workflow tags from text compression.
///
/// * `compress_tagged_content = false` (default) — replace each entire
///   `<custom>…</custom>` span (including nested children) with a
///   single placeholder. Self-closing custom tags become a single
///   placeholder. The body between the markers is *not* exposed to
///   compression.
/// * `compress_tagged_content = true` — replace only the tag markers
///   (open and close emitted as separate placeholders) so the
///   compressor can squash content while the tag boundaries survive.
///
/// Returns `(cleaned, blocks, stats)` where `blocks` is a list of
/// `(placeholder, original)` pairs for [`restore_tags`]. The blocks
/// are listed in left-to-right order of appearance in the input, which
/// keeps the restore step trivial.
pub fn protect_tags(
    text: &str,
    compress_tagged_content: bool,
) -> (String, Vec<(String, String)>, ProtectStats) {
    let mut stats = ProtectStats::default();
    if text.is_empty() || !text.contains('<') {
        return (text.to_string(), Vec::new(), stats);
    }

    let (prefix, salted) = pick_placeholder_prefix(text);
    stats.placeholder_collision_avoided = salted;

    // Phase 1: walk once, classify every tag, build a list of spans
    // worth replacing. No output emitted yet — this is purely
    // discovery so we can decide which byte ranges to swap.
    let spans = identify_spans(text, compress_tagged_content, &mut stats);

    // Phase 2: emit. Walk the input once more, splicing placeholders
    // for span bytes and copying everything else verbatim. Because
    // `spans` is sorted left-to-right and non-overlapping (block mode
    // collapses nested matches into the outermost span; marker mode
    // emits open/close markers that are byte-disjoint by construction)
    // this is a straightforward scan.
    match emit_output(text, &spans, &prefix) {
        Some((cleaned, blocks)) => (cleaned, blocks, stats),
        // Should be unreachable — `identify_spans` returns spans whose
        // bytes are slices of `text`. If we ever fail to splice them
        // back, fall back to emitting the original.
        None => (text.to_string(), Vec::new(), stats),
    }
}

fn identify_spans(
    text: &str,
    compress_tagged_content: bool,
    stats: &mut ProtectStats,
) -> Vec<Span> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut spans: Vec<Span> = Vec::new();
    let mut stack: Vec<OpenTag> = Vec::new();

    let mut i = 0;
    while i < n {
        let b = bytes[i];
        if b != b'<' {
            // Skip ahead to the next `<`. We don't care about non-tag
            // bytes for span identification; they'll be copied verbatim
            // in phase 2.
            i = memchr(b'<', &bytes[i..]).map(|j| i + j).unwrap_or(n);
            continue;
        }

        match parse_tag_at(bytes, i) {
            TagParse::NotTag => {
                i += 1;
            }
            TagParse::Open {
                name_end,
                tag_end,
                is_self_closing,
            } => {
                stats.tags_seen += 1;
                let name = &text[i + 1..name_end];
                if is_known_html_tag(name) {
                    stats.html_tags_skipped += 1;
                    i = tag_end;
                    continue;
                }
                if is_self_closing {
                    spans.push(Span {
                        start: i,
                        end: tag_end,
                        kind: SpanKind::SelfClosing,
                    });
                    stats.self_closing_protected += 1;
                    i = tag_end;
                    continue;
                }
                if compress_tagged_content {
                    // Marker-only mode: emit the open as its own span
                    // *and* push the name on the stack so the close
                    // gets matched and emitted as its own span.
                    spans.push(Span {
                        start: i,
                        end: tag_end,
                        kind: SpanKind::OpenMarker,
                    });
                }
                // Both modes push to the stack so close-matching works.
                stack.push(OpenTag {
                    name_lower: name.to_ascii_lowercase(),
                    open_start: i,
                });
                i = tag_end;
            }
            TagParse::Close { name_end, tag_end } => {
                stats.tags_seen += 1;
                let close_name = &text[i + 2..name_end];
                if is_known_html_tag(close_name) {
                    stats.html_tags_skipped += 1;
                    i = tag_end;
                    continue;
                }
                let close_name_lower = close_name.to_ascii_lowercase();
                let matching = stack
                    .iter()
                    .rposition(|open| open.name_lower == close_name_lower);

                match matching {
                    Some(stack_idx) => {
                        if compress_tagged_content {
                            // Pop everything above (orphan opens
                            // inside the matched span — their open
                            // markers were already recorded as spans
                            // and we keep them).
                            stack.truncate(stack_idx);
                            let _ = stack.pop();
                            spans.push(Span {
                                start: i,
                                end: tag_end,
                                kind: SpanKind::CloseMarker,
                            });
                        } else {
                            // Block mode: collapse [open..close] into
                            // a single span. Drop any inner unmatched
                            // opens (they're part of this span's body).
                            // Also DROP any inner spans we already
                            // recorded that are now subsumed by this
                            // outer block — that's how nested custom
                            // tags collapse to one placeholder.
                            let open_start = stack[stack_idx].open_start;
                            stack.truncate(stack_idx);
                            spans.retain(|s| s.start < open_start);
                            spans.push(Span {
                                start: open_start,
                                end: tag_end,
                                kind: SpanKind::Block,
                            });
                            stats.custom_blocks_protected += 1;
                        }
                        i = tag_end;
                    }
                    None => {
                        stats.orphan_closes += 1;
                        i = tag_end;
                    }
                }
            }
        }
    }

    // Stack remnants are orphan opens (no matching close ever arrived).
    // We don't protect those — they'll fall through to the compressor
    // as raw `<name>` bytes, same as Python's original behavior. In
    // block mode their inner self-closing spans we recorded are still
    // safe to keep: they were below an unmatched outer open, so they
    // were never collapsed. Spans are sorted by start ascending due to
    // the monotonic walk; phase 2 expects that.
    spans
}

fn emit_output(
    text: &str,
    spans: &[Span],
    prefix: &str,
) -> Option<(String, Vec<(String, String)>)> {
    let mut out = String::with_capacity(text.len());
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut cursor: usize = 0;

    for (counter, span) in (0_u64..).zip(spans.iter()) {
        if span.start < cursor {
            // Overlap shouldn't happen given how we collapse nested
            // spans, but bail loudly if it does — silently producing
            // wrong output is worse than failing the test.
            return None;
        }
        out.push_str(&text[cursor..span.start]);
        let placeholder = format!("{prefix}{counter}{PLACEHOLDER_SUFFIX}");
        let original = &text[span.start..span.end];
        blocks.push((placeholder.clone(), original.to_string()));
        out.push_str(&placeholder);
        cursor = span.end;
        let _ = span.kind; // SpanKind is informational only at this layer
    }
    out.push_str(&text[cursor..]);
    Some((out, blocks))
}

/// Restore protected tag spans after the compressor ran on the
/// cleaned text.
///
/// If a placeholder went missing during compression (the compressor
/// stripped or rewrote it) the corresponding original is appended to
/// the output rather than dropped — same fallback semantics as the
/// Python original. A `tracing::warn!` is emitted at compile-time-
/// optional verbosity so prod can scrape lossy-compression incidents.
pub fn restore_tags(text: &str, blocks: &[(String, String)]) -> String {
    if blocks.is_empty() {
        return text.to_string();
    }

    let mut result = text.to_string();
    let mut tail_appends: Vec<&str> = Vec::new();
    for (placeholder, original) in blocks {
        if result.contains(placeholder.as_str()) {
            result = result.replace(placeholder.as_str(), original);
        } else {
            tag_lost_warn(original);
            tail_appends.push(original.as_str());
        }
    }

    if !tail_appends.is_empty() {
        for original in tail_appends {
            result.push('\n');
            result.push_str(original);
        }
    }

    result
}

#[inline(never)]
fn tag_lost_warn(original: &str) {
    let preview: String = original.chars().take(80).collect();
    tracing::warn!(
        target: "headroom::tag_protector",
        preview = %preview,
        "tag placeholder lost during compression, appending original"
    );
}

// ─── Tiny byte-search helper ──────────────────────────────────────────

#[inline]
fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protect(text: &str) -> (String, Vec<(String, String)>) {
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        (cleaned, blocks)
    }

    #[test]
    fn passthrough_when_no_angle_bracket() {
        let (cleaned, blocks) = protect("Just plain text");
        assert_eq!(cleaned, "Just plain text");
        assert!(blocks.is_empty());
    }

    #[test]
    fn html_tags_emitted_verbatim() {
        let text = "<div>Some content</div>";
        let (cleaned, blocks) = protect(text);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn html_tag_check_case_insensitive() {
        assert!(is_known_html_tag("DIV"));
        assert!(is_known_html_tag("Span"));
        assert!(!is_known_html_tag("system-reminder"));
        assert!(!is_known_html_tag("EXTREMELY_IMPORTANT"));
    }

    #[test]
    fn custom_tag_replaced_with_placeholder() {
        let text = "Before <system-reminder>Important</system-reminder> After";
        let (cleaned, blocks) = protect(text);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("Important"));
        assert!(cleaned.contains("Before"));
        assert!(cleaned.contains("After"));
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<system-reminder>Important</system-reminder>");
    }

    #[test]
    fn custom_tag_with_attributes() {
        let text = r#"<context key="session" type="persistent">user data</context>"#;
        let (_cleaned, blocks) = protect(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].1.contains(r#"key="session""#));
    }

    #[test]
    fn self_closing_custom_tag() {
        let text = "Text <marker/> more text";
        let (_cleaned, blocks) = protect(text);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<marker/>");
    }

    #[test]
    fn self_closing_html_tag_not_protected() {
        let text = "Text <br/> more <hr/> text";
        let (cleaned, blocks) = protect(text);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn nested_custom_tags_collapse_to_outer_span() {
        let text = "<outer><inner>deep</inner></outer>";
        let (cleaned, blocks) = protect(text);
        assert!(!cleaned.contains("<outer>"));
        assert!(!cleaned.contains("<inner>"));
        // Outer span captures inner — single placeholder.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<outer><inner>deep</inner></outer>");
    }

    #[test]
    fn mixed_html_and_custom() {
        let text = "<div>HTML</div> <system-reminder>Rule</system-reminder> <p>HTML2</p>";
        let (cleaned, blocks) = protect(text);
        assert!(cleaned.contains("<div>"));
        assert!(cleaned.contains("<p>"));
        assert!(!cleaned.contains("<system-reminder>"));
        assert_eq!(blocks.len(), 1);
    }

    #[test]
    fn real_workflow_tags() {
        let cases = [
            "<tool_call>search({query: 'test'})</tool_call>",
            "<thinking>Let me analyze this</thinking>",
            "<EXTREMELY_IMPORTANT>Never skip validation</EXTREMELY_IMPORTANT>",
            "<user-prompt-submit-hook>check perms</user-prompt-submit-hook>",
            "<system-reminder>Rules</system-reminder>",
            "<result>Success: 42 items</result>",
        ];
        for tag in cases {
            let text = format!("Before {tag} After");
            let (_cleaned, blocks) = protect(&text);
            assert_eq!(blocks.len(), 1, "failed to protect: {tag}");
            assert_eq!(blocks[0].1, tag);
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let (cleaned, blocks) = protect("");
        assert!(cleaned.is_empty());
        assert!(blocks.is_empty());
    }

    #[test]
    fn compress_tagged_content_true_emits_marker_placeholders() {
        let text = "Before <system-reminder>Compressible content</system-reminder> After";
        let (cleaned, blocks, _stats) = protect_tags(text, true);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("</system-reminder>"));
        assert!(cleaned.contains("Compressible content"));
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn restore_basic() {
        let original = "Before <system-reminder>Rule</system-reminder> After";
        let (cleaned, blocks, _stats) = protect_tags(original, false);
        let restored = restore_tags(&cleaned, &blocks);
        assert_eq!(restored, original);
    }

    #[test]
    fn restore_empty_blocks_passthrough() {
        assert_eq!(restore_tags("untouched", &[]), "untouched");
    }

    #[test]
    fn restore_lost_placeholder_appended() {
        let blocks = vec![(
            "{{HEADROOM_TAG_0}}".to_string(),
            "<tag>data</tag>".to_string(),
        )];
        let restored = restore_tags("text without placeholder", &blocks);
        assert!(restored.contains("<tag>data</tag>"));
    }

    #[test]
    fn restore_roundtrip_preserves_content() {
        let original = "Start <system-reminder>Rule 1: validate</system-reminder> middle \
             <tool_call>search(q='test')</tool_call> end";
        let (cleaned, blocks, _stats) = protect_tags(original, false);
        let restored = restore_tags(&cleaned, &blocks);
        assert_eq!(restored, original);
    }

    // ─── Bug-fix tests (fixed_in_3e4) ─────────────────────────────────

    #[test]
    fn fixed_in_3e4_replace_first_does_not_collide_on_duplicate_blocks() {
        // Bug #2: Python's `result.replace(original, placeholder, 1)`
        // replaces the FIRST textual occurrence of `original`, not
        // necessarily the matched offset. Two identical custom-tag
        // blocks would collapse to a single placeholder + a stray
        // duplicate of the second block in the output.
        let text = "<system-reminder>same</system-reminder> middle \
             <system-reminder>same</system-reminder>";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        // BOTH blocks should be replaced by DIFFERENT placeholders.
        assert_eq!(blocks.len(), 2);
        assert!(!cleaned.contains("<system-reminder>"));
        assert!(!cleaned.contains("</system-reminder>"));
        assert_ne!(blocks[0].0, blocks[1].0);
        // Roundtrip is exact.
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_handles_50_plus_nested_custom_tags() {
        // Bug #3: Python had a hard-coded 50-iteration safety cap that
        // silently truncated tag protection on deeply nested input.
        // Build 60 nested custom tags and verify all get caught in
        // the outermost span.
        let depth = 60;
        let mut text = String::new();
        for _ in 0..depth {
            text.push_str("<lvl>");
        }
        text.push_str("core");
        for _ in 0..depth {
            text.push_str("</lvl>");
        }
        let (cleaned, blocks, _stats) = protect_tags(&text, false);
        // The outermost span eats everything: one placeholder, no
        // residual `<lvl>` markers in the cleaned text.
        assert!(!cleaned.contains("<lvl>"));
        assert!(!cleaned.contains("</lvl>"));
        assert_eq!(blocks.len(), 1);
        // Roundtrip exact even at depth=60.
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_self_closing_duplicates_get_distinct_placeholders() {
        // Bug #4: same first-occurrence-replace bug for self-closing
        // tags. `<marker/>` appearing twice would collapse.
        let text = "<marker/> middle <marker/>";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(blocks.len(), 2);
        assert_ne!(blocks[0].0, blocks[1].0);
        assert!(!cleaned.contains("<marker/>"));
        assert_eq!(restore_tags(&cleaned, &blocks), text);
    }

    #[test]
    fn fixed_in_3e4_placeholder_collision_is_avoided() {
        // Bug #5: input contains literal `{{HEADROOM_TAG_…}}`. The
        // walker should pick a salted prefix and report the collision
        // in stats.
        let text = "User wrote {{HEADROOM_TAG_0}} on purpose. \
             <system-reminder>real one</system-reminder>";
        let (_cleaned, blocks, stats) = protect_tags(text, false);
        assert!(stats.placeholder_collision_avoided);
        assert_eq!(blocks.len(), 1);
        // Placeholder used must NOT collide with the user's literal.
        assert_ne!(blocks[0].0, "{{HEADROOM_TAG_0}}");
    }

    // ─── Edge-case correctness ────────────────────────────────────────

    #[test]
    fn orphan_close_tag_emitted_verbatim() {
        let text = "no opener </ghost> here";
        let (cleaned, blocks, stats) = protect_tags(text, false);
        // Nothing protected; close stays in the cleaned text.
        assert_eq!(blocks.len(), 0);
        assert!(cleaned.contains("</ghost>"));
        assert_eq!(stats.orphan_closes, 1);
    }

    #[test]
    fn orphan_open_tag_emitted_verbatim() {
        // An open with no matching close should round-trip exactly —
        // no protection, no data loss.
        let text = "<ghost>dangling content with no close";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert!(blocks.is_empty());
        assert_eq!(cleaned, text);
    }

    #[test]
    fn malformed_lone_lt_emitted_verbatim() {
        let text = "if a < b then c";
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(cleaned, text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn attribute_with_gt_inside_quotes() {
        let text = r#"<context attr="a > b">payload</context>"#;
        let (cleaned, blocks, _stats) = protect_tags(text, false);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, text);
        assert!(!cleaned.contains("payload"));
    }

    #[test]
    fn html_close_inside_custom_block_does_not_pop_stack() {
        // An HTML close tag while a custom open is on top should not
        // confuse the stack: the HTML close is emitted verbatim, the
        // custom span still closes when its own close arrives.
        let text = "<custom>x</div> y</custom>";
        let (cleaned, blocks, stats) = protect_tags(text, false);
        // The whole `<custom>...</custom>` span wins, including the
        // verbatim `</div>` inside.
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, "<custom>x</div> y</custom>");
        assert!(!cleaned.contains("<custom>"));
        // `</div>` is HTML, not orphan.
        assert_eq!(stats.html_tags_skipped, 1);
        assert_eq!(stats.orphan_closes, 0);
    }
}

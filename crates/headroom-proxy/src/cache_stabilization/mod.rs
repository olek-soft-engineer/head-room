//! Cache-stabilization observability surface (Phase E).
//!
//! Phase E PRs in this module sit *next to* the request path —
//! never on it. They observe inbound bodies and emit structured
//! warnings so customers can see why their prompt-cache hit rate
//! is degrading. Nothing in here mutates request bytes; the
//! cache-safety invariant from Phase A still holds.
//!
//! Currently shipped:
//!
//! - [`volatile_detector`] (PR-E5): scans inbound bodies for
//!   patterns that bust prompt-cache hits (timestamps, UUIDs,
//!   ID-named fields) and emits one structured WARN log per
//!   finding so customers know what to move out of the cached
//!   prefix.
//!
//! Sibling PRs (PR-E6, ...) will land additional detectors here.
//! Conflict resolution between parallel PRs is intentionally
//! trivial: each detector lives in its own file, the only shared
//! surface is this `mod.rs`'s `pub mod` list.

pub mod volatile_detector;

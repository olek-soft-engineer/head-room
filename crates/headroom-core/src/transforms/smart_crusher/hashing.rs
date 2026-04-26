//! Field-name hashing for cache keys.
//!
//! Direct port of `_hash_field_name` (Python `smart_crusher.py:171-176`).
//! Used to generate stable cache keys for compression hints — must match
//! Python byte-for-byte or cache lookups will miss.

use sha2::{Digest, Sha256};

/// SHA-256 of the UTF-8 bytes, hex-encoded, truncated to 16 chars.
///
/// Python equivalent: `hashlib.sha256(field_name.encode()).hexdigest()[:16]`.
/// We use lowercase hex (the default for both Python and Rust's `sha2`
/// crate) — the test below pins this.
pub fn hash_field_name(field_name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(field_name.as_bytes());
    let digest = hasher.finalize();
    // Convert to lowercase hex, then truncate to first 16 chars (8 bytes).
    let hex = format!("{:x}", digest);
    hex[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_sha256_truncated_to_16() {
        // Verified against Python: hashlib.sha256(b"customer_id").hexdigest()[:16]
        assert_eq!(hash_field_name("customer_id"), "1e38d67dbe8f47d2");
    }

    #[test]
    fn empty_string() {
        // Verified against Python: hashlib.sha256(b"").hexdigest()[:16]
        assert_eq!(hash_field_name(""), "e3b0c44298fc1c14");
    }

    #[test]
    fn unicode_field_name() {
        // Verified against Python: hashlib.sha256("café".encode()).hexdigest()[:16]
        // UTF-8 bytes for "café" are 63 61 66 c3 a9 — must encode same way.
        assert_eq!(hash_field_name("café"), "850f7dc43910ff89");
    }

    #[test]
    fn deterministic() {
        // Same input → same output across calls.
        assert_eq!(hash_field_name("test"), hash_field_name("test"));
    }

    #[test]
    fn output_length_is_16() {
        // Always exactly 16 hex chars regardless of input length.
        assert_eq!(hash_field_name("a").len(), 16);
        assert_eq!(hash_field_name(&"x".repeat(1000)).len(), 16);
    }
}

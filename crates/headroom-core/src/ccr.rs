//! CCR (Compress-Cache-Retrieve) storage layer.
//!
//! When a transform compresses data with row-drop or opaque-string
//! substitution, the *original payload* is stashed here keyed by the
//! hash that ends up in the prompt. The runtime later honors retrieval
//! tool calls by looking up the hash in this store and serving back the
//! original. This is the cornerstone of CCR: lossy on the wire, lossless
//! end-to-end.
//!
//! Mirrors the semantics of Python's [`CompressionStore`] (`headroom/
//! cache/compression_store.py`) but stripped down to the contract that
//! actually matters for retrieval — no BM25 search, no retrieval-event
//! feedback, no per-tool metadata. Those live in the runtime layer; this
//! crate only needs put/get.
//!
//! # Pluggable backend
//!
//! Production deployments swap in their own [`CcrStore`] backed by Redis,
//! MongoDB, or whatever shared cache fits. The default in-memory store
//! ships ready for single-process use.
//!
//! [`CompressionStore`]: https://github.com/chopratejas/headroom/blob/main/headroom/cache/compression_store.py

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Pluggable CCR storage backend. `Send + Sync` so it can sit behind an
/// `Arc` and be shared across threads in the proxy.
pub trait CcrStore: Send + Sync {
    /// Stash `payload` under `hash`. If the hash already exists, the
    /// new payload overwrites — same hash should mean same content, so
    /// re-storing is idempotent.
    fn put(&self, hash: &str, payload: &str);

    /// Look up `hash`. Returns `None` if missing or expired.
    fn get(&self, hash: &str) -> Option<String>;

    /// Number of live entries. Informational; used by tests + telemetry.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Default capacity — matches Python's `CompressionStore` default.
pub const DEFAULT_CAPACITY: usize = 1000;

/// Default TTL — 5 minutes, matching Python.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);

/// Simple in-memory CCR store with TTL + bounded capacity.
///
/// - **TTL**: 5 minutes by default. Entries past their TTL are dropped
///   on the next `get`.
/// - **Capacity**: 1000 entries by default. When full, the oldest
///   insertion is evicted (FIFO).
/// - **Locking**: single `Mutex`. The store sits on the cold path
///   (one call per crushed array, not per token) so coarse locking is
///   fine and keeps the implementation small.
pub struct InMemoryCcrStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    capacity: usize,
}

struct Inner {
    map: HashMap<String, Entry>,
    /// FIFO order of insertion for capacity eviction. Hashes that get
    /// re-stored stay at their original position — same content under
    /// the same hash is idempotent and rare.
    order: VecDeque<String>,
}

struct Entry {
    payload: String,
    inserted: Instant,
}

impl InMemoryCcrStore {
    /// Default: 1000 entries, 5-minute TTL.
    pub fn new() -> Self {
        Self::with_capacity_and_ttl(DEFAULT_CAPACITY, DEFAULT_TTL)
    }

    pub fn with_capacity_and_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
            capacity,
        }
    }
}

impl Default for InMemoryCcrStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CcrStore for InMemoryCcrStore {
    fn put(&self, hash: &str, payload: &str) {
        let mut g = self.inner.lock().expect("ccr store mutex poisoned");

        if g.map.contains_key(hash) {
            // Idempotent re-store. Same hash should mean same content;
            // overwrite the payload (cheap) and keep the original FIFO
            // position so eviction stays predictable.
            g.map.insert(
                hash.to_string(),
                Entry {
                    payload: payload.to_string(),
                    inserted: Instant::now(),
                },
            );
            return;
        }

        // New entry. Evict the oldest if we're at capacity.
        while g.map.len() >= self.capacity {
            let Some(oldest) = g.order.pop_front() else {
                break;
            };
            g.map.remove(&oldest);
        }

        g.map.insert(
            hash.to_string(),
            Entry {
                payload: payload.to_string(),
                inserted: Instant::now(),
            },
        );
        g.order.push_back(hash.to_string());
    }

    fn get(&self, hash: &str) -> Option<String> {
        let mut g = self.inner.lock().expect("ccr store mutex poisoned");
        let expired = match g.map.get(hash) {
            Some(e) => e.inserted.elapsed() > self.ttl,
            None => return None,
        };
        if expired {
            g.map.remove(hash);
            return None;
        }
        g.map.get(hash).map(|e| e.payload.clone())
    }

    fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("ccr store mutex poisoned")
            .map
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_returns_payload() {
        let store = InMemoryCcrStore::new();
        store.put("abc123", r#"[{"id":1}]"#);
        assert_eq!(store.get("abc123"), Some(r#"[{"id":1}]"#.to_string()));
    }

    #[test]
    fn missing_hash_returns_none() {
        let store = InMemoryCcrStore::new();
        assert_eq!(store.get("never_stored"), None);
    }

    #[test]
    fn put_overwrites_under_same_hash() {
        let store = InMemoryCcrStore::new();
        store.put("h", "first");
        store.put("h", "second");
        assert_eq!(store.get("h"), Some("second".to_string()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(2, DEFAULT_TTL);
        store.put("a", "1");
        store.put("b", "2");
        store.put("c", "3");
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("a"), None);
        assert_eq!(store.get("b"), Some("2".to_string()));
        assert_eq!(store.get("c"), Some("3".to_string()));
    }

    #[test]
    fn expired_entries_are_dropped_on_get() {
        let store = InMemoryCcrStore::with_capacity_and_ttl(10, Duration::from_millis(10));
        store.put("a", "1");
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(store.get("a"), None);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn store_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InMemoryCcrStore>();
    }

    #[test]
    fn trait_object_is_usable() {
        let store: Box<dyn CcrStore> = Box::new(InMemoryCcrStore::new());
        store.put("h", "v");
        assert_eq!(store.get("h"), Some("v".to_string()));
        assert!(!store.is_empty());
    }
}

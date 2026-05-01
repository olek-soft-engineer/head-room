//! Message-level importance scoring — used by IntelligentContextManager.
//!
//! # Why this lives at the crate root (parallel to `signals/`)
//!
//! `signals/` scores **lines** for line-level compressors (logs, search,
//! diffs). `scoring/` scores **messages** for conversation-level context
//! management. Different inputs, different consumers — separating them
//! keeps each trait surface clean and lets future ports compose without
//! a giant `signals` god-module.
//!
//! # Port status (Phase 7g PR-A, 2026-04-30)
//!
//! Direct port of `headroom/transforms/scoring.py` (459 LOC). The
//! deterministic factors — recency, forward references, density —
//! are fully implemented and parity-tested against Python. The
//! external-dependency factors are gated behind trait surfaces:
//!
//! - **TOIN** (`ToinProvider` trait): no concrete impl yet. Calls
//!   return `0.5` for `toin_importance` and `0.0` for `error_indicator`
//!   when no provider is wired in. PR-A1 will plug in a `PyO3`
//!   `ToinProvider` so Rust can read Python's TOIN state.
//! - **Embeddings** (`EmbeddingProvider` trait): no concrete impl
//!   here yet (the crate already has `relevance::EmbeddingScorer`
//!   for SmartCrusher; PR-A1 wires the same `bge-small-en-v1.5`
//!   model into a `MessageEmbedder` adapter). Until then,
//!   `semantic_score` returns `0.5` (neutral).
//!
//! The trait surface is FULL — when the providers land, no API
//! changes are needed inside `MessageScorer`. The neutral-value
//! defaults match Python behavior when those subsystems are not
//! configured (`toin=None`, `embedding_provider=None`).
//!
//! # No hardcoded patterns (project convention)
//!
//! Mirrors the Python module's design principle: importance derives
//! from computed metrics (recency/density/refs), TOIN-learned
//! patterns (field semantics, retrieval rates), and embedding
//! similarity. No keyword regex, no hardcoded "error" strings.

pub mod score;
pub mod scorer;
pub mod traits;
pub mod weights;

pub use score::MessageScore;
pub use scorer::MessageScorer;
pub use traits::{EmbeddingProvider, ToinFieldSemantic, ToinPattern, ToinProvider};
pub use weights::ScoringWeights;

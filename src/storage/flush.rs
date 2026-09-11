//! Embedding debt bookkeeping: what is pending, since when, and whether automatic flushes
//! are suspended. Persisted as JSON in the shelf's `meta` table. Unknown fields are ignored,
//! so the record can grow without breaking older readers, and it is never exported.
use serde::{Deserialize, Serialize};

/// Key of the record in the `meta` table.
pub const FLUSH_STATE_KEY: &str = "embedding_flush_state";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FlushState {
    /// Model identity `breaker` and `remote_batch` belong to; under another identity they are stale.
    pub identity: String,
    /// When the current embedding debt started (RFC 3339, UTC); `None` while nothing is pending.
    /// Independent of the identity: entries lack vectors whichever model is configured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_since: Option<String>,
    /// Set while automatic flushes are suspended after a failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub breaker: Option<Breaker>,
    /// Largest remote batch the server accepted after rejecting a larger one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_batch: Option<usize>,
}

impl FlushState {
    pub fn for_identity(identity: &str) -> Self {
        Self {
            identity: identity.into(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breaker {
    /// Permanent failures wait for a configuration change; transient ones retry after `retry_after`.
    pub permanent: bool,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<String>,
    pub failures: u32,
}

/// Outcome of one automatic flush.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlushStats {
    pub installed: usize,
    /// Rows whose content changed, or whose identity was rebound, while they were embedded.
    pub skipped: usize,
    pub failed: usize,
    /// The first failure, if any.
    pub error: Option<String>,
}

/// A shelf's embedding debt, as reported by `hypatia backfill --status` and agent
/// interfaces. A public shape of its own, so the stored `FlushState` can change freely.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingDebt {
    pub pending_knowledge: usize,
    pub pending_statement: usize,
    /// When the current debt started; `None` while nothing is pending.
    pub pending_since: Option<String>,
    /// Why no vector can be written right now; `None` when the debt can be paid.
    pub blocked: Option<Blocked>,
    /// Set while automatic flushes are paused after failures.
    pub paused: Option<Paused>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Blocked {
    pub reason: BlockedReason,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockedReason {
    /// The provider cannot embed: model not installed, API key unset, and so on.
    ProviderUnavailable,
    /// Stored vectors belong to another model; `backfill --reembed` recovers.
    IdentityMismatch,
    /// No model name and no readable model files, so vectors would have no identity.
    UnknownIdentity,
    /// Vectors predate identity tracking; `backfill --reembed` recovers.
    LegacyVectors,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paused {
    /// Permanent pauses wait for a configuration change; others resume after `retry_after`.
    pub permanent: bool,
    pub reason: String,
    pub retry_after: Option<String>,
}

impl From<&Breaker> for Paused {
    fn from(breaker: &Breaker) -> Self {
        Self {
            permanent: breaker.permanent,
            reason: breaker.reason.clone(),
            retry_after: breaker.retry_after.clone(),
        }
    }
}

/// The current time in the format `pending_since` and `retry_after` use.
pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

//! Embedding debt bookkeeping: what is pending, since when, and whether automatic flushes
//! are suspended. Persisted as JSON in the shelf's `meta` table. Unknown fields are ignored,
//! so the record can grow without breaking older readers, and it is never exported.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Key of the record in the `meta` table.
pub const FLUSH_STATE_KEY: &str = "embedding_flush_state";

/// Pending entries at which a write flushes, by provider. A local flush loads the model, so
/// its cost is spread over 64 writes; a remote flush is a single request for a full batch.
pub const LOCAL_WRITE_THRESHOLD: usize = 64;
pub const REMOTE_WRITE_THRESHOLD: usize = 128;
/// Most entries one automatic flush embeds, so no single command pays for a large debt.
pub const FLUSH_BATCH: usize = 128;
/// Most entries remembered as failing on their own; the oldest are forgotten first.
pub const MAX_SKIPPED: usize = 32;
/// How long a remote debt may wait for its batch to fill before the next command pays it.
pub const REMOTE_MAX_DELAY_SECS: i64 = 60;
/// Wait after a first transient failure; it doubles with each further failure.
const FIRST_BACKOFF_SECS: i64 = 60;
/// Longest wait between attempts. Permanent failures are retried this often too: a fixed API
/// key leaves the model identity unchanged, so nothing else would notice the fix.
const MAX_BACKOFF_SECS: i64 = 3600;

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
    /// Batch size automatic flushes send after the server rejected a larger batch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_batch: Option<usize>,
    /// Entries the provider failed on alone. Automatic flushes pass over them until their
    /// content changes; `hypatia backfill` still tries them.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<SkippedEntry>,
}

impl FlushState {
    pub fn for_identity(identity: &str) -> Self {
        Self {
            identity: identity.into(),
            ..Self::default()
        }
    }

    /// Remembers entries as failing on their own, forgetting the oldest beyond `MAX_SKIPPED`.
    pub fn skip(&mut self, entries: impl IntoIterator<Item = SkippedEntry>) {
        for entry in entries {
            if !self.skipped.contains(&entry) {
                self.skipped.push(entry);
            }
        }
        let excess = self.skipped.len().saturating_sub(MAX_SKIPPED);
        self.skipped.drain(..excess);
    }

    pub fn is_skipped(&self, catalog: &str, key: &str, version: i64) -> bool {
        self.skipped
            .iter()
            .any(|entry| entry.version == version && entry.key == key && entry.catalog == catalog)
    }
}

/// One version of a pending entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedEntry {
    pub catalog: String,
    pub key: String,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Breaker {
    /// Permanent failures (a wrong key, model, endpoint or vector size) need a fix: they are
    /// retried hourly, and cleared by a `hypatia backfill` that embeds anything. Transient
    /// ones are retried with backoff.
    pub permanent: bool,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<String>,
    pub failures: u32,
}

impl Breaker {
    /// The breaker after one more failure, following `previous` if it was already open.
    pub fn trip(previous: Option<&Breaker>, permanent: bool, reason: String) -> Self {
        let failures = previous.map_or(0, |b| b.failures).saturating_add(1);
        let wait = if permanent {
            MAX_BACKOFF_SECS
        } else {
            (FIRST_BACKOFF_SECS << (failures - 1).min(16)).min(MAX_BACKOFF_SECS)
        };
        Self {
            permanent,
            reason,
            retry_after: Some(timestamp(Utc::now() + chrono::Duration::seconds(wait))),
            failures,
        }
    }

    /// Whether automatic flushes may try again. An unreadable time never holds them back.
    pub fn is_due(&self) -> bool {
        self.retry_after
            .as_deref()
            .and_then(parse)
            .is_none_or(|at| at <= Utc::now())
    }
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
    /// Pending entries automatic embedding passes over, because the provider failed on them
    /// alone; `hypatia backfill` still tries them.
    #[serde(default)]
    pub passed_over: usize,
    /// When the current debt started; `None` while nothing is pending but passed-over entries.
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
    /// Permanent pauses need a configuration fix; every pause is retried after `retry_after`.
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
    timestamp(Utc::now())
}

/// Whole seconds since a time in that format; `None` if it cannot be read.
pub fn seconds_since(at: &str) -> Option<i64> {
    parse(at).map(|at| (Utc::now() - at).num_seconds())
}

fn timestamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn parse(at: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(at)
        .ok()
        .map(|at| at.with_timezone(&Utc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_waits_double_up_to_an_hour_and_permanent_ones_wait_an_hour() {
        let wait = |b: &Breaker| -seconds_since(b.retry_after.as_deref().unwrap()).unwrap();
        let first = Breaker::trip(None, false, "timeout".into());
        assert_eq!(first.failures, 1);
        assert!((58..=60).contains(&wait(&first)), "{}", wait(&first));
        let second = Breaker::trip(Some(&first), false, "timeout".into());
        assert!((118..=120).contains(&wait(&second)));
        let mut many = second;
        for _ in 0..10 {
            many = Breaker::trip(Some(&many), false, "timeout".into());
        }
        assert!((3598..=3600).contains(&wait(&many)));
        let permanent = Breaker::trip(None, true, "bad key".into());
        assert!(permanent.permanent && (3598..=3600).contains(&wait(&permanent)));
        assert!(!permanent.is_due());
    }

    #[test]
    fn skipped_entries_are_capped_oldest_first() {
        let mut state = FlushState::default();
        let entry = |version| SkippedEntry {
            catalog: "knowledge".into(),
            key: "k".into(),
            version,
        };
        state.skip((0..40).map(entry));
        state.skip([entry(39)]);
        assert_eq!(state.skipped.len(), MAX_SKIPPED);
        assert!(!state.is_skipped("knowledge", "k", 7));
        assert!(state.is_skipped("knowledge", "k", 8));
        assert!(!state.is_skipped("statement", "k", 8));
    }

    #[test]
    fn a_past_or_unreadable_retry_time_is_due() {
        let mut breaker = Breaker::trip(None, false, "timeout".into());
        breaker.retry_after = Some("2000-01-01T00:00:00Z".into());
        assert!(breaker.is_due());
        breaker.retry_after = Some("soon".into());
        assert!(breaker.is_due());
    }
}

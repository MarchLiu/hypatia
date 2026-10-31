//! Shared comparison semantics for the time columns (`created_at`, `tr_start`,
//! `tr_end`). SQLite stores them as TEXT (`YYYY-MM-DD HH:MM:SS.SSS`, produced by
//! `strftime('%Y-%m-%d %H:%M:%f', 'now')`) while the pgvector backend casts the
//! bound value to `::timestamp`. Comparing raw strings against an ISO-8601
//! `T`-separated bound value silently missed every row of the same day
//! (`' ' < 'T'`), and pg accepted date words like `'today'` that SQLite read as
//! text. Both engines now go through the helpers here so the two backends
//! accept the same values and compare them on the same timeline.

use crate::error::{HypatiaError, Result};
use chrono::{DateTime, NaiveDateTime, Utc};

/// The columns whose comparisons are time-normalized.
pub(crate) const TIME_COLUMNS: &[&str] = &["created_at", "tr_start", "tr_end"];

/// Canonical stored/bound format: UTC, space-separated, millisecond precision —
/// the same shape `strftime('%Y-%m-%d %H:%M:%f', ...)` writes, so the
/// normalized forms compare correctly as text (fixed width, zero padded).
const CANONICAL: &str = "%Y-%m-%d %H:%M:%S%.3f";

pub(crate) fn is_time_column(field: &str) -> bool {
    TIME_COLUMNS.contains(&field.trim_start_matches('$'))
}

/// Validate a comparison value for a time column and return the canonical UTC
/// form to bind. Accepts `YYYY-MM-DD HH:MM:SS[.fff]` (the stored shape) and
/// ISO-8601 `T`-separated timestamps with an optional `Z` or offset; a bare
/// date or a date word like `today` is a validation error, on every backend.
/// Non-string values are left to the caller's legacy handling.
pub(crate) fn normalize_time_value(
    field: &str,
    value: &serde_json::Value,
) -> Result<Option<String>> {
    let Some(raw) = value.as_str() else {
        return Ok(None);
    };
    match parse_time(raw) {
        Some(normalized) => Ok(Some(normalized)),
        None => Err(HypatiaError::Validation(format!(
            "invalid timestamp '{raw}' for '{field}': expected 'YYYY-MM-DD HH:MM:SS[.fff]' or an ISO-8601 timestamp"
        ))),
    }
}

/// Parse the accepted timestamp shapes into a UTC naive datetime, then render
/// it in the canonical millisecond format.
fn parse_time(raw: &str) -> Option<String> {
    let naive = NaiveDateTime::parse_from_str(raw, "%Y-%m-%d %H:%M:%S%.f").ok();
    if let Some(dt) = naive {
        return Some(dt.format(CANONICAL).to_string());
    }
    let with_tz = DateTime::parse_from_rfc3339(raw).ok()?;
    Some(with_tz.with_timezone(&Utc).format(CANONICAL).to_string())
}

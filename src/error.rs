use thiserror::Error;

/// Render a `rusqlite::Error` without leaking the statement text.
///
/// `Error::SqlInputError` formats as `"{msg} in {sql} at offset {offset}"`,
/// which echoes the whole generated statement — including any fragment the
/// caller managed to smuggle into it — straight to the user. Everything the
/// user can act on is in `msg`; the SQL is a debugging detail. Set
/// `HYPATIA_DEBUG_SQL=1` to get the full statement back, and note that the
/// `Debug` formatting of these errors always carries it.
pub fn redact_sql(err: &rusqlite::Error) -> String {
    match err {
        rusqlite::Error::SqlInputError { msg, .. } if !sql_debug_enabled() => msg.clone(),
        other => other.to_string(),
    }
}

fn sql_debug_enabled() -> bool {
    std::env::var_os("HYPATIA_DEBUG_SQL").is_some_and(|v| !v.is_empty() && v != "0")
}

#[derive(Debug, Error)]
pub enum HypatiaError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// Direct conversion so `?` works on rusqlite errors without map_err
    /// (From is not transitive through StorageError).
    #[error("SQLite error: {}", redact_sql(.0))]
    Sqlite(#[from] rusqlite::Error),


    #[error("JSE parse error: {0}")]
    Parse(String),

    #[error("JSE evaluation error: {0}")]
    Eval(String),

    #[error("shelf error: {0}")]
    Shelf(String),

    #[error("not found: {kind} '{key}'")]
    NotFound { kind: String, key: String },

    #[error("validation error: {0}")]
    Validation(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("config error: {0}")]
    Config(String),
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[cfg(feature = "legacy-migration")]
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[error("SQLite error: {}", redact_sql(.0))]
    Sqlite(#[from] rusqlite::Error),

    #[error("vector index error: {0}")]
    Vector(String),

    #[error("connection not open for shelf: {0}")]
    NotConnected(String),
}

pub type Result<T> = std::result::Result<T, HypatiaError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn sql_input_error() -> rusqlite::Error {
        rusqlite::Error::SqlInputError {
            error: rusqlite::ffi::Error::new(1),
            msg: "near \"b\": syntax error".to_string(),
            sql: "SELECT * FROM knowledge WHERE json_extract(content, '$.a'b') = ?".to_string(),
            offset: 81,
        }
    }

    #[test]
    fn storage_error_display_omits_sql() {
        let rendered = HypatiaError::from(StorageError::from(sql_input_error())).to_string();
        assert_eq!(rendered, "storage error: SQLite error: near \"b\": syntax error");
        assert!(!rendered.contains("SELECT"));
        assert!(!rendered.contains("json_extract"));
    }

    #[test]
    fn sqlite_error_display_omits_sql() {
        let rendered = HypatiaError::from(sql_input_error()).to_string();
        assert_eq!(rendered, "SQLite error: near \"b\": syntax error");
        assert!(!rendered.contains("SELECT"));
    }

    #[test]
    fn debug_formatting_retains_sql_for_diagnosis() {
        let debug = format!("{:?}", HypatiaError::from(sql_input_error()));
        assert!(debug.contains("json_extract"));
    }

    #[test]
    fn other_variants_render_unchanged() {
        let err = rusqlite::Error::InvalidColumnName("nope".to_string());
        assert_eq!(redact_sql(&err), err.to_string());
    }
}

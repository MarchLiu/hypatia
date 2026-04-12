use thiserror::Error;

#[derive(Debug, Error)]
pub enum HypatiaError {
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

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
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),

    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("connection not open for shelf: {0}")]
    NotConnected(String),
}

pub type Result<T> = std::result::Result<T, HypatiaError>;

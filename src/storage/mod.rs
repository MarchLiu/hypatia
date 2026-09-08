pub mod backend;
pub mod json_index;
#[cfg(feature = "legacy-migration")]
pub mod migrate;
#[cfg(feature = "postgres-backend")]
pub mod postgres_store;
pub mod settings;
pub mod transfer;
#[cfg(not(feature = "legacy-migration"))]
pub mod migrate {
    //! Stub: legacy duckdb→sqlite migration requires the
    //! `legacy-migration` feature (cargo build --features legacy-migration).
    use crate::error::{HypatiaError, Result};
    use crate::model::shelf::ShelfConfig;
    use crate::storage::sqlite_store::SqliteStore;

    pub fn open_or_migrate(config: &ShelfConfig) -> Result<SqliteStore> {
        if config.needs_migration() {
            return Err(HypatiaError::Config(format!(
                "shelf '{}' uses the legacy duckdb layout; rebuild with \
                 --features legacy-migration (or run hypatia 0.2.x) to migrate",
                config.id.name
            )));
        }
        SqliteStore::open(&config.sqlite_path)
    }
}
pub mod shelf_manager;
pub mod shelf_registry;
pub mod sqlite_store;
pub mod vector_index;

pub use json_index::json_contains;
pub use migrate::open_or_migrate;
pub use shelf_manager::{OpenShelf, ShelfManager};
pub use shelf_registry::ShelfRegistry;
pub use sqlite_store::{FtsDoc, SqliteStore, sanitize_fts_query};
pub use vector_index::VectorFileIndex;

#[derive(Debug, Clone)]
pub struct FtsResult {
    pub id: i64,
    pub catalog: String,
    pub key: String,
    pub content: String,
    pub rank: f64,
}

use crate::error::Result;
use crate::model::{QueryResult, QueryTarget, SearchOpts};

/// Abstract storage interface for testability.
/// OpenShelf implements this trait by delegating to the unified SQLite store.
/// Note: No Send+Sync bounds because the connection uses RefCell internally.
pub trait Storage {
    /// SQL dialect accepted by execute_query; existing stores default to SQLite.
    fn sql_dialect(&self) -> crate::engine::SqlDialect {
        crate::engine::SqlDialect::Sqlite
    }

    /// PostgreSQL shelf schema. PostgreSQL queries require an explicit schema.
    fn sql_schema(&self) -> Option<&str> {
        None
    }

    fn execute_query(
        &self,
        target: QueryTarget,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<QueryResult>;

    fn execute_search(&self, query: &str, opts: &SearchOpts) -> Result<QueryResult>;

    /// Execute a semantic similarity search using vector embeddings.
    /// Returns an error if the embedding model is unavailable.
    fn execute_similar(
        &self,
        query_text: &str,
        opts: &SearchOpts,
        target: QueryTarget,
    ) -> Result<QueryResult>;

    /// Execute a k-hop forward graph traversal starting from `head`,
    /// following edges with the given relation (or any relation if None),
    /// up to `depth` hops. Returns matching statement triples.
    fn execute_khop(&self, head: &str, relation: Option<&str>, depth: i64) -> Result<QueryResult>;
}

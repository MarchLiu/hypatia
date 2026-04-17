pub mod duckdb_store;
pub mod shelf_manager;
pub mod shelf_registry;
pub mod sqlite_store;

pub use duckdb_store::DuckDbStore;
pub use shelf_manager::{OpenShelf, ShelfManager};
pub use shelf_registry::ShelfRegistry;
pub use sqlite_store::{sanitize_fts_query, FtsDoc, SqliteStore};

use crate::error::Result;
use crate::model::{QueryResult, QueryTarget, SearchOpts};

/// Abstract storage interface for testability.
/// OpenShelf implements this trait by delegating to DuckDB/SQLite stores.
/// Note: No Send+Sync bounds because DuckDB/SQLite connections use RefCell internally.
pub trait Storage {
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

    /// Execute a k-hop forward graph traversal starting from `subject`,
    /// following edges with the given predicate (or any predicate if None),
    /// up to `depth` hops. Returns matching statement triples.
    fn execute_khop(
        &self,
        subject: &str,
        predicate: Option<&str>,
        depth: i64,
    ) -> Result<QueryResult>;
}

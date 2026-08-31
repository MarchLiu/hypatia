pub mod json_index;
pub mod migrate;
pub mod shelf_manager;
pub mod shelf_registry;
pub mod sqlite_store;
pub mod vector_index;

pub use json_index::json_contains;
pub use migrate::open_or_migrate;
pub use shelf_manager::{OpenShelf, ShelfManager};
pub use shelf_registry::ShelfRegistry;
pub use sqlite_store::{sanitize_fts_query, FtsDoc, SqliteStore};
pub use vector_index::VectorFileIndex;

use crate::error::Result;
use crate::model::{QueryResult, QueryTarget, SearchOpts};

/// Abstract storage interface for testability.
/// OpenShelf implements this trait by delegating to the unified SQLite store.
/// Note: No Send+Sync bounds because the connection uses RefCell internally.
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

    /// Execute a k-hop forward graph traversal starting from `head`,
    /// following edges with the given relation (or any relation if None),
    /// up to `depth` hops. Returns matching statement triples.
    fn execute_khop(
        &self,
        head: &str,
        relation: Option<&str>,
        depth: i64,
    ) -> Result<QueryResult>;
}

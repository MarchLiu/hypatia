//! One-time migration from the legacy duckdb+sqlite layout to the unified
//! SQLite store (SPO → HRT happens here, positionally losslessly).
//!
//! Trigger: `data.duckdb` exists and `hypatia.sqlite` does not.
//! Safety: single transaction on the new DB; on failure the partial file is
//! removed so the next attempt starts clean. Legacy files are renamed `.bak`.

use std::path::Path;

use crate::error::Result;
use crate::model::shelf::ShelfConfig;
use crate::storage::sqlite_store::{fts_doc_for, vector_to_blob, SqliteStore};

/// Open a shelf, migrating first when needed. The single entry point used by
/// ShelfManager.
pub fn open_or_migrate(config: &ShelfConfig) -> Result<SqliteStore> {
    if config.needs_migration() {
        migrate_shelf(config)?;
    }
    SqliteStore::open(&config.sqlite_path)
}

/// Migrate a legacy shelf to the unified layout. Idempotent: skips when no
/// legacy files are present or the new DB already exists.
pub fn migrate_shelf(config: &ShelfConfig) -> Result<()> {
    if !config.needs_migration() {
        return Ok(());
    }

    let result = run_migration(config);
    if let Err(e) = result {
        // Remove partial output so a retry starts clean.
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(suffix_path(&config.sqlite_path, suffix));
        }
        return Err(e);
    }

    // Back up legacy files.
    rename_to_bak(&config.legacy_duckdb_path())?;
    rename_to_bak(&config.legacy_index_path())?;
    Ok(())
}

fn run_migration(config: &ShelfConfig) -> Result<()> {
    let config_duckdb = duckdb::Config::default()
        .access_mode(duckdb::AccessMode::ReadOnly)
        .map_err(crate::error::StorageError::from)?;
    let legacy = duckdb::Connection::open_with_flags(
        &config.legacy_duckdb_path(),
        config_duckdb,
    )
    .map_err(crate::error::StorageError::from)?;

    let store = SqliteStore::open(&config.sqlite_path)?;
    let conn = store.conn();

    let tx = conn.unchecked_transaction().map_err(crate::error::StorageError::from)?;

    // ── knowledge ────────────────────────────────────────────────────
    // Embeddings are read as JSON text (to_json) for duckdb ARRAY portability.
    {
        let mut stmt = legacy
            .prepare(
                "SELECT name, CAST(content AS VARCHAR), \
                        CAST(created_at AS VARCHAR), \
                        CAST(to_json(embedding) AS VARCHAR) \
                 FROM knowledge",
            )
            .map_err(crate::error::StorageError::from)?;
        let mut rows = stmt.query([]).map_err(crate::error::StorageError::from)?;
        while let Some(row) = rows.next().map_err(crate::error::StorageError::from)? {
            let name: String = row.get(0).map_err(crate::error::StorageError::from)?;
            let content_json: String = row.get(1).map_err(crate::error::StorageError::from)?;
            let created_at: String = row.get(2).map_err(crate::error::StorageError::from)?;
            let embedding_json: Option<String> = row.get(3).ok();
            let embedding = parse_embedding(embedding_json)?;

            tx.execute(
                "INSERT INTO knowledge (name, content, embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![
                    name,
                    content_json,
                    embedding.as_ref().map(|v| vector_to_blob(v)),
                    created_at
                ],
            )
            .map_err(crate::error::StorageError::from)?;

            let doc = fts_doc_for_content(&content_json, &name)?;
            insert_docs_row(&tx, "knowledge", &name, &doc)?;
        }
    }

    // ── statement (SPO → HRT is positional, keys unchanged) ─────────
    {
        let mut stmt = legacy
            .prepare(
                "SELECT triple, subject, predicate, object, CAST(content AS VARCHAR), \
                        CAST(created_at AS VARCHAR), \
                        CAST(tr_start AS VARCHAR), CAST(tr_end AS VARCHAR), \
                        CAST(to_json(embedding) AS VARCHAR) \
                 FROM statement",
            )
            .map_err(crate::error::StorageError::from)?;
        let mut rows = stmt.query([]).map_err(crate::error::StorageError::from)?;
        while let Some(row) = rows.next().map_err(crate::error::StorageError::from)? {
            let triple: String = row.get(0).map_err(crate::error::StorageError::from)?;
            let head: String = row.get(1).map_err(crate::error::StorageError::from)?;
            let relation: String = row.get(2).map_err(crate::error::StorageError::from)?;
            let tail: String = row.get(3).map_err(crate::error::StorageError::from)?;
            let content_json: String = row.get(4).map_err(crate::error::StorageError::from)?;
            let created_at: Option<String> = row.get(5).ok();
            let tr_start: Option<String> = row.get(6).ok();
            let tr_end: Option<String> = row.get(7).ok();
            let embedding_json: Option<String> = row.get(8).ok();
            let embedding = parse_embedding(embedding_json)?;

            tx.execute(
                "INSERT INTO statement (triple, head, relation, tail, content, embedding, created_at, tr_start, tr_end)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    triple, head, relation, tail, content_json,
                    embedding.as_ref().map(|v| vector_to_blob(v)),
                    created_at.unwrap_or_default(),
                    tr_start, tr_end
                ],
            )
            .map_err(crate::error::StorageError::from)?;

            let doc = fts_doc_for_content(&content_json, &triple)?;
            insert_docs_row(&tx, "statement", &triple, &doc)?;
        }
    }

    // Safety net: external-content FTS rebuild (triggers already kept it in sync).
    tx.execute_batch("INSERT INTO docs_fts(docs_fts) VALUES('rebuild');")
        .map_err(crate::error::StorageError::from)?;

    tx.commit().map_err(crate::error::StorageError::from)?;
    Ok(())
}

fn parse_embedding(json: Option<String>) -> Result<Option<Vec<f32>>> {
    match json {
        None => Ok(None),
        Some(s) => {
            let v: Vec<f32> = serde_json::from_str(&s)?;
            if v.is_empty() {
                Ok(None)
            } else {
                Ok(Some(v))
            }
        }
    }
}

fn fts_doc_for_content(
    content_json: &str,
    key: &str,
) -> Result<crate::storage::sqlite_store::FtsDoc> {
    let content = crate::model::Content::from_json_str(content_json)
        .unwrap_or_else(|_| crate::model::Content::new(content_json.to_string()));
    Ok(fts_doc_for(&content, key))
}

fn insert_docs_row(
    tx: &rusqlite::Transaction<'_>,
    catalog: &str,
    key: &str,
    doc: &crate::storage::sqlite_store::FtsDoc,
) -> Result<()> {
    tx.execute(
        "INSERT INTO docs(catalog, key, fts_key, fts_data, fts_tags, fts_synonyms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![catalog, key, doc.fts_key, doc.fts_data, doc.fts_tags, doc.fts_synonyms],
    )
    .map_err(crate::error::StorageError::from)?;
    Ok(())
}

fn suffix_path(path: &Path, suffix: &str) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(suffix);
    std::path::PathBuf::from(s)
}

fn rename_to_bak(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut bak = path.as_os_str().to_os_string();
    bak.push(".bak");
    let bak = std::path::PathBuf::from(bak);
    if bak.exists() {
        // Previous backup: drop it, the current legacy file is authoritative.
        std::fs::remove_file(&bak)?;
    }
    std::fs::rename(path, &bak)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::shelf::ShelfId;
    use tempfile::TempDir;

    /// Build a legacy data.duckdb with one knowledge row and one statement row.
    fn seed_legacy(path: &Path) {
        let conn = duckdb::Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE knowledge (
                name TEXT PRIMARY KEY,
                content JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding FLOAT[4]
            );
            CREATE TABLE statement (
                triple TEXT PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                content JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tr_start TIMESTAMP,
                tr_end TIMESTAMP,
                embedding FLOAT[4]
            );",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO knowledge (name, content, embedding)
             VALUES ('rust', '{\"format\":\"markdown\",\"data\":\"Rust language\",\"tags\":[\"lang\"],\"synonyms\":null,\"figures\":null,\"scopes\":null}', [1.0, 0.0, 0.0, 0.0])",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO statement (triple, subject, predicate, object, content)
             VALUES ('Rust,is_a,language', 'Rust', 'is_a', 'language',
                     '{\"format\":\"markdown\",\"data\":\"\",\"tags\":[],\"synonyms\":null,\"figures\":null,\"scopes\":null}')",
            [],
        )
        .unwrap();
    }

    #[test]
    fn migration_roundtrip() {
        let dir = TempDir::new().unwrap();
        let config = ShelfConfig::from_path(dir.path(), Some("mig-test"));

        seed_legacy(&config.legacy_duckdb_path());
        assert!(config.needs_migration());

        migrate_shelf(&config).unwrap();

        // Legacy files renamed to .bak
        assert!(!config.legacy_duckdb_path().exists());
        assert!(config.legacy_duckdb_path().with_extension("duckdb.bak").exists());
        assert!(!config.needs_migration());

        // New store contents
        let store = SqliteStore::open(&config.sqlite_path).unwrap();
        let k = store.get_knowledge("rust").unwrap().unwrap();
        assert_eq!(k.content.data, "Rust language");
        assert_eq!(k.content.tags, vec!["lang".to_string()]);

        let key = crate::model::StatementKey::new("Rust", "is_a", "language");
        let s = store.get_statement(&key).unwrap().unwrap();
        assert_eq!(s.key.head, "Rust");
        assert_eq!(s.key.relation, "is_a");
        assert_eq!(s.key.tail, "language");

        // Embedding survived as BLOB
        assert_eq!(store.knowledge_with_embeddings().unwrap().len(), 1);

        // FTS index works after migration
        let results = store.search("Rust", &crate::model::SearchOpts::default()).unwrap();
        assert!(!results.is_empty());

        // Re-running is a no-op
        migrate_shelf(&config).unwrap();
    }

    #[test]
    fn open_or_migrate_on_fresh_shelf() {
        let dir = TempDir::new().unwrap();
        let config = ShelfConfig::from_path(dir.path(), Some("fresh-test"));
        assert!(!config.needs_migration());
        let store = open_or_migrate(&config).unwrap();
        store
            .insert_knowledge("k", &crate::model::Content::new("v"))
            .unwrap();
        assert!(store.get_knowledge("k").unwrap().is_some());
    }

    #[test]
    fn shelf_id_roundtrip_via_config() {
        let dir = TempDir::new().unwrap();
        let id = ShelfId::new(dir.path().to_path_buf());
        let config = ShelfConfig::from_shelf_id(id);
        assert_eq!(config.sqlite_path, dir.path().join("hypatia.sqlite"));
    }
}

use std::path::Path;

use rusqlite::{params, OptionalExtension, Connection};

use crate::error::{Result, StorageError};
use crate::model::SearchOpts;

const DOCS_META_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS docs_meta (
    id INTEGER PRIMARY KEY,
    catalog TEXT,
    key TEXT,
    content TEXT,
    fts_key TEXT NOT NULL DEFAULT '',
    fts_data TEXT NOT NULL DEFAULT '',
    fts_tags TEXT NOT NULL DEFAULT '',
    fts_synonyms TEXT NOT NULL DEFAULT ''
)";

const DOCS_META_INDEX: &str = "\
CREATE INDEX IF NOT EXISTS idx_docs_catalog ON docs_meta(catalog)";

const DOCS_FTS_SCHEMA: &str = "\
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    fts_key,
    fts_data,
    fts_tags,
    fts_synonyms,
    content='docs_meta',
    content_rowid='id',
    tokenize='porter unicode61'
)";

const TRIGGER_INSERT: &str = "\
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs_meta BEGIN
    INSERT INTO docs_fts(rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES (new.id, new.fts_key, new.fts_data, new.fts_tags, new.fts_synonyms);
END";

const TRIGGER_DELETE: &str = "\
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs_meta BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES('delete', old.id, old.fts_key, old.fts_data, old.fts_tags, old.fts_synonyms);
END";

const TRIGGER_UPDATE: &str = "\
CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs_meta BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES('delete', old.id, old.fts_key, old.fts_data, old.fts_tags, old.fts_synonyms);
    INSERT INTO docs_fts(rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES (new.id, new.fts_key, new.fts_data, new.fts_tags, new.fts_synonyms);
END";

/// BM25 column weights: fts_key=10, fts_data=1, fts_tags=5, fts_synonyms=3
const BM25_WEIGHTS: &str = "bm25(docs_fts, 10.0, 1.0, 5.0, 3.0)";

#[derive(Debug, Clone)]
pub struct FtsResult {
    pub id: i64,
    pub catalog: String,
    pub key: String,
    pub content: String,
    pub rank: f64,
}

/// Structured document for FTS indexing with multi-column support.
pub struct FtsDoc {
    /// Full serialized content JSON (stored in content column for retrieval).
    pub content: String,
    /// Key/name text for FTS (knowledge name or CSV triple key).
    pub fts_key: String,
    /// Data text for FTS.
    pub fts_data: String,
    /// Tags text for FTS (space-joined).
    pub fts_tags: String,
    /// Synonyms text for FTS (flattened).
    pub fts_synonyms: String,
}

fn row_to_fts_result(row: &rusqlite::Row) -> rusqlite::Result<FtsResult> {
    Ok(FtsResult {
        id: row.get(0)?,
        catalog: row.get(1)?,
        key: row.get(2)?,
        content: row.get(3)?,
        rank: row.get(4)?,
    })
}

pub struct SqliteStore {
    conn: Connection,
}

impl SqliteStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(StorageError::from)?;
        let store = Self { conn };
        store.init_schema()?;
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        // Create base schema (IF NOT EXISTS for fresh DBs)
        self.conn
            .execute_batch(&format!(
                "{DOCS_META_SCHEMA}; {DOCS_META_INDEX};"
            ))
            .map_err(StorageError::from)?;

        // Migrate: add columns if they don't exist (existing DBs)
        for col in ["fts_key", "fts_data", "fts_tags", "fts_synonyms"] {
            let sql = format!("ALTER TABLE docs_meta ADD COLUMN {col} TEXT NOT NULL DEFAULT ''");
            match self.conn.execute_batch(&sql) {
                Ok(_) => {}
                Err(e) if is_duplicate_column_error(&e) => {}
                Err(e) => return Err(StorageError::from(e).into()),
            }
        }

        // Drop and recreate FTS virtual table (FTS5 schema cannot be ALTERed)
        self.conn
            .execute_batch("DROP TABLE IF EXISTS docs_fts")
            .ok();
        self.conn
            .execute_batch(DOCS_FTS_SCHEMA)
            .map_err(StorageError::from)?;

        // Recreate triggers
        self.conn
            .execute_batch(&format!(
                "DROP TRIGGER IF EXISTS docs_ai; \
                 DROP TRIGGER IF EXISTS docs_ad; \
                 DROP TRIGGER IF EXISTS docs_au; \
                 {TRIGGER_INSERT}; {TRIGGER_DELETE}; {TRIGGER_UPDATE};"
            ))
            .map_err(StorageError::from)?;

        // Rebuild FTS index from existing docs_meta rows
        self.conn
            .execute_batch("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
            .map_err(StorageError::from)?;

        Ok(())
    }

    pub fn upsert_doc(&self, catalog: &str, key: &str, doc: &FtsDoc) -> Result<()> {
        let existing_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM docs_meta WHERE catalog = ?1 AND key = ?2",
                params![catalog, key],
                |row| row.get(0),
            )
            .optional()
            .map_err(StorageError::from)?;

        match existing_id {
            Some(id) => {
                self.conn
                    .execute(
                        "UPDATE docs_meta SET content = ?1, fts_key = ?2, fts_data = ?3, fts_tags = ?4, fts_synonyms = ?5 WHERE id = ?6",
                        params![doc.content, doc.fts_key, doc.fts_data, doc.fts_tags, doc.fts_synonyms, id],
                    )
                    .map_err(StorageError::from)?;
            }
            None => {
                self.conn
                    .execute(
                        "INSERT INTO docs_meta (catalog, key, content, fts_key, fts_data, fts_tags, fts_synonyms) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![catalog, key, doc.content, doc.fts_key, doc.fts_data, doc.fts_tags, doc.fts_synonyms],
                    )
                    .map_err(StorageError::from)?;
            }
        }
        Ok(())
    }

    pub fn delete_doc(&self, catalog: &str, key: &str) -> Result<()> {
        self.conn
            .execute(
                "DELETE FROM docs_meta WHERE catalog = ?1 AND key = ?2",
                params![catalog, key],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn search(&self, query: &str, opts: &SearchOpts) -> Result<Vec<FtsResult>> {
        let query = crate::text::segment_for_fts(query);
        let query = sanitize_fts_query(&query);
        let catalog_filter = opts.catalog.as_deref();
        let sql = match catalog_filter {
            Some(_) =>
                &format!("SELECT m.id, m.catalog, m.key, m.content, {BM25_WEIGHTS} as rank \
                 FROM docs_meta m \
                 JOIN docs_fts f ON m.id = f.rowid \
                 WHERE docs_fts MATCH ?1 AND m.catalog = ?2 \
                 ORDER BY rank LIMIT ?3 OFFSET ?4"),
            None =>
                &format!("SELECT m.id, m.catalog, m.key, m.content, {BM25_WEIGHTS} as rank \
                 FROM docs_meta m \
                 JOIN docs_fts f ON m.id = f.rowid \
                 WHERE docs_fts MATCH ?1 \
                 ORDER BY rank LIMIT ?2 OFFSET ?3"),
        };

        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;

        let rows: Vec<rusqlite::Result<FtsResult>> = match catalog_filter {
            Some(cat) => stmt
                .query_map(params![query, cat, opts.limit, opts.offset], row_to_fts_result)
                .map_err(StorageError::from)?
                .collect(),
            None => stmt
                .query_map(params![query, opts.limit, opts.offset], row_to_fts_result)
                .map_err(StorageError::from)?
                .collect(),
        };

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }
}

fn is_duplicate_column_error(e: &rusqlite::Error) -> bool {
    match e {
        rusqlite::Error::ExecuteReturnedResults => false,
        _ => e.to_string().contains("duplicate column name"),
    }
}

/// Sanitize a query string for SQLite FTS5 by removing special characters
/// that cause parse errors. Replaces them with spaces and collapses whitespace.
pub fn sanitize_fts_query(query: &str) -> String {
    let sanitized: String = query
        .chars()
        .map(|c| {
            matches!(
                c,
                ':' | '"' | '\'' | '*' | '^' | '+' | '-' | '(' | ')' | '.' | '?'
                | '!' | ',' | '/' | '`' | '{' | '}' | '[' | ']' | '~' | '@'
                | '#' | '%' | ';' | '&' | '|' | '<' | '>'
            )
            .then_some(' ')
            .unwrap_or(c)
        })
        .collect();
    let mut result = String::with_capacity(sanitized.len());
    let mut prev_space = false;
    for c in sanitized.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(c);
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, SqliteStore) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.sqlite");
        let store = SqliteStore::open(&db_path).unwrap();
        (dir, store)
    }

    fn test_doc(key: &str, data: &str, tags: &str) -> FtsDoc {
        FtsDoc {
            content: format!("{{\"data\":\"{data}\"}}"),
            fts_key: key.to_string(),
            fts_data: data.to_string(),
            fts_tags: tags.to_string(),
            fts_synonyms: String::new(),
        }
    }

    #[test]
    fn schema_init() {
        let (_dir, _store) = setup();
    }

    #[test]
    fn upsert_and_search() {
        let (_dir, store) = setup();
        store
            .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust is a systems programming language", ""))
            .unwrap();
        store
            .upsert_doc("knowledge", "python", &test_doc("python", "Python is a scripting language", ""))
            .unwrap();

        let results = store
            .search("programming", &SearchOpts::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "rust");
    }

    #[test]
    fn upsert_updates_existing() {
        let (_dir, store) = setup();
        store
            .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust is fast", ""))
            .unwrap();
        store
            .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust is fast and safe", ""))
            .unwrap();

        let results = store.search("safe", &SearchOpts::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "rust");
    }

    #[test]
    fn search_with_catalog_filter() {
        let (_dir, store) = setup();
        store
            .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust programming", ""))
            .unwrap();
        store
            .upsert_doc("statement", "rust,is,language", &test_doc("rust,is,language", "Rust is a programming language", ""))
            .unwrap();

        let opts = SearchOpts {
            catalog: Some("knowledge".to_string()),
            ..Default::default()
        };
        let results = store.search("programming", &opts).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].catalog, "knowledge");
    }

    #[test]
    fn fts_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.sqlite");

        // Insert data in first connection
        {
            let store = SqliteStore::open(&db_path).unwrap();
            store
                .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust is a systems programming language", ""))
                .unwrap();
        }

        // Reopen (triggers DROP + CREATE + rebuild) and verify search still works
        {
            let store = SqliteStore::open(&db_path).unwrap();
            let results = store.search("programming", &SearchOpts::default()).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].key, "rust");
        }
    }

    #[test]
    fn delete_doc() {
        let (_dir, store) = setup();
        store
            .upsert_doc("knowledge", "rust", &test_doc("rust", "Rust programming language", ""))
            .unwrap();
        store.delete_doc("knowledge", "rust").unwrap();

        let results = store.search("Rust", &SearchOpts::default()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn porter_stemmer() {
        let (_dir, store) = setup();
        store
            .upsert_doc("knowledge", "auth", &test_doc("auth", "user authenticating via OAuth2", ""))
            .unwrap();

        // "authentication" should stem-match "authenticating"
        let results = store
            .search("authentication", &SearchOpts::default())
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "auth");
    }

    #[test]
    fn bm25_weights_key_over_data() {
        let (_dir, store) = setup();

        // Entry where "rust" only appears in data, not key
        store
            .upsert_doc("knowledge", "lang1", &FtsDoc {
                content: "{}".to_string(),
                fts_key: "lang1".to_string(),
                fts_data: "Rust programming language".to_string(),
                fts_tags: String::new(),
                fts_synonyms: String::new(),
            })
            .unwrap();

        // Entry where "rust" appears in key (should rank higher)
        store
            .upsert_doc("knowledge", "rust", &FtsDoc {
                content: "{}".to_string(),
                fts_key: "rust".to_string(),
                fts_data: "systems programming".to_string(),
                fts_tags: String::new(),
                fts_synonyms: String::new(),
            })
            .unwrap();

        let results = store
            .search("rust", &SearchOpts::default())
            .unwrap();
        assert!(results.len() >= 2);
        // Key match should rank first (more negative = better)
        assert_eq!(results[0].key, "rust");
    }
}

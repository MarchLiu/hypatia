//! Single-file SQLite store — the only source of truth.
//!
//! Layout (see docs/sqlite-refactor-plan.md):
//! - `meta`        schema versioning
//! - `knowledge` / `statement`  source tables (HRT triples, content TEXT, embedding BLOB)
//! - `docs`        unified document anchor (doc_id = usearch key = FTS content_rowid)
//! - `json_index`  path-tree inverted index (populated from P2 on)
//! - `docs_fts`    FTS5 external-content table anchored on `docs`

use std::path::Path;

use chrono::NaiveDateTime;
use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{HypatiaError, Result, StorageError};
use crate::storage::json_index::{json_contains_str, rebuild_all, replace_postings_in};
use crate::model::{Content, Knowledge, SearchOpts, Statement, StatementKey};

const SCHEMA_VERSION: &str = "2";

const META_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS meta(
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
)";

const KNOWLEDGE_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS knowledge(
    name       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    embedding  BLOB,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))
)";

const STATEMENT_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS statement(
    triple     TEXT PRIMARY KEY,
    head       TEXT NOT NULL,
    relation   TEXT NOT NULL,
    tail       TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  BLOB,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now')),
    tr_start   TEXT,
    tr_end     TEXT
);
CREATE INDEX IF NOT EXISTS idx_stmt_head     ON statement(head);
CREATE INDEX IF NOT EXISTS idx_stmt_relation ON statement(relation);
CREATE INDEX IF NOT EXISTS idx_stmt_tail     ON statement(tail);
";

const DOCS_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS docs(
    id           INTEGER PRIMARY KEY,
    catalog      TEXT NOT NULL,
    key          TEXT NOT NULL,
    fts_key      TEXT NOT NULL DEFAULT '',
    fts_data     TEXT NOT NULL DEFAULT '',
    fts_tags     TEXT NOT NULL DEFAULT '',
    fts_synonyms TEXT NOT NULL DEFAULT '',
    UNIQUE(catalog, key)
);
CREATE INDEX IF NOT EXISTS idx_docs_catalog ON docs(catalog);
";

const JSON_INDEX_SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS json_index(
    doc_id      INTEGER NOT NULL,
    path        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    value       TEXT,
    array_index INTEGER,
    PRIMARY KEY(doc_id, path, array_index, value)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_json_path_value ON json_index(path, value, doc_id);
";

const DOCS_FTS_SCHEMA: &str = "\
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    fts_key, fts_data, fts_tags, fts_synonyms,
    content='docs', content_rowid='id',
    tokenize='porter unicode61'
)";

const TRIGGERS_SCHEMA: &str = "\
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
    INSERT INTO docs_fts(rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES (new.id, new.fts_key, new.fts_data, new.fts_tags, new.fts_synonyms);
END;
CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES('delete', old.id, old.fts_key, old.fts_data, old.fts_tags, old.fts_synonyms);
END;
CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE ON docs BEGIN
    INSERT INTO docs_fts(docs_fts, rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES('delete', old.id, old.fts_key, old.fts_data, old.fts_tags, old.fts_synonyms);
    INSERT INTO docs_fts(rowid, fts_key, fts_data, fts_tags, fts_synonyms)
    VALUES (new.id, new.fts_key, new.fts_data, new.fts_tags, new.fts_synonyms);
END;
";

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
    pub fts_key: String,
    pub fts_data: String,
    pub fts_tags: String,
    pub fts_synonyms: String,
}

pub struct SqliteStore {
    conn: Connection,
}

// ── vector/blob helpers ──────────────────────────────────────────────

/// Encode an f32 vector as little-endian BLOB (source-of-truth format).
pub fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|f| {
            let f = if f.is_nan() || f.is_infinite() { 0.0f32 } else { *f };
            f.to_le_bytes()
        })
        .collect()
}

/// Decode a little-endian f32 BLOB back into a vector.
pub fn blob_to_vector(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine distance in [0, 2]; 0 = identical direction.
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

/// Convert a serde_json query parameter into a typed SQLite value.
fn json_to_sql(v: &serde_json::Value) -> rusqlite::types::Value {
    use rusqlite::types::Value;
    match v {
        serde_json::Value::String(s) => Value::Text(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else {
                Value::Real(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::Bool(b) => Value::Integer(*b as i64),
        serde_json::Value::Null => Value::Null,
        other => Value::Text(other.to_string()),
    }
}

fn parse_timestamp(s: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f").map_err(|e| {
        HypatiaError::Storage(StorageError::Sqlite(rusqlite::Error::FromSqlConversionFailure(
            0,
            rusqlite::types::Type::Text,
            Box::new(e),
        )))
    })
}

fn format_timestamp(dt: &NaiveDateTime) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.f").to_string()
}

fn row_to_knowledge(row: &rusqlite::Row) -> rusqlite::Result<Knowledge> {
    let name: String = row.get(0)?;
    let json: String = row.get(1)?;
    let created_at: String = row.get(2)?;
    let content = Content::from_json_str(&json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let created_at = parse_timestamp(&created_at).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(Knowledge { name, content, created_at })
}

fn row_to_statement(row: &rusqlite::Row) -> rusqlite::Result<Statement> {
    let triple: String = row.get(0)?;
    let head: String = row.get(1)?;
    let relation: String = row.get(2)?;
    let tail: String = row.get(3)?;
    let json: String = row.get(4)?;
    let created_at: String = row.get(5)?;
    let tr_start: Option<String> = row.get(6)?;
    let tr_end: Option<String> = row.get(7)?;
    let parse = |s: String| -> rusqlite::Result<NaiveDateTime> {
        parse_timestamp(&s).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
        })
    };
    let content = Content::from_json_str(&json).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e))
    })?;
    let _ = triple; // PK; key derived from columns
    Ok(Statement {
        key: StatementKey { head, relation, tail },
        content,
        created_at: parse(created_at)?,
        tr_start: tr_start.map(parse).transpose()?,
        tr_end: tr_end.map(parse).transpose()?,
    })
}

impl SqliteStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(StorageError::from)?;
        let store = Self { conn };
        store.init_schema()?;
        store.register_udfs()?;
        store.ensure_json_index_populated()?;
        Ok(store)
    }

    /// Register SQL UDFs (per-connection).
    fn register_udfs(&self) -> Result<()> {
        use rusqlite::functions::FunctionFlags;
        self.conn
            .create_scalar_function(
                "json_contains",
                2,
                FunctionFlags::SQLITE_UTF8,
                |ctx| {
                    let lhs: Option<String> = ctx.get(0)?;
                    let rhs: Option<String> = ctx.get(1)?;
                    match (lhs, rhs) {
                        (Some(l), Some(r)) => Ok(json_contains_str(&l, &r)),
                        _ => Ok(false),
                    }
                },
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    /// Populates json_index for shelves migrated before P2 (one-time, idempotent).
    fn ensure_json_index_populated(&self) -> Result<()> {
        let needs: bool = self
            .conn
            .query_row(
                "SELECT NOT EXISTS(SELECT 1 FROM json_index)                  AND (EXISTS(SELECT 1 FROM knowledge) OR EXISTS(SELECT 1 FROM statement))",
                [],
                |r| r.get(0),
            )
            .map_err(StorageError::from)?;
        if needs {
            self.rebuild_json_index()?;
        }
        Ok(())
    }

    /// Rebuild the whole json_index from source-table content.
    pub fn rebuild_json_index(&self) -> Result<usize> {
        rebuild_all(&self.conn)
    }

    /// Access the underlying connection (used by the migration tool).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             PRAGMA busy_timeout = 5000;",
        )?;

        self.conn.execute_batch(&format!(
            "{META_SCHEMA}; {KNOWLEDGE_SCHEMA}; {STATEMENT_SCHEMA}; \
             {DOCS_SCHEMA}; {JSON_INDEX_SCHEMA}; {DOCS_FTS_SCHEMA}; {TRIGGERS_SCHEMA};"
        ))?;

        let version: Option<String> = self
            .conn
            .query_row("SELECT v FROM meta WHERE k = 'schema_version'", [], |r| {
                r.get(0)
            })
            .optional()?;
        if version.is_none() {
            self.conn.execute(
                "INSERT INTO meta(k, v) VALUES('schema_version', ?1)",
                params![SCHEMA_VERSION],
            )?;
        }
        Ok(())
    }

    // ── docs (FTS anchor) helpers ────────────────────────────────────

    fn docs_upsert_in(
        &self,
        tx: &rusqlite::Transaction<'_>,
        catalog: &str,
        key: &str,
        doc: &FtsDoc,
    ) -> Result<i64> {
        let existing: Option<i64> = tx
            .query_row(
                "SELECT id FROM docs WHERE catalog = ?1 AND key = ?2",
                params![catalog, key],
                |r| r.get(0),
            )
            .optional()?;
        if existing.is_some() {
            tx.execute(
                "UPDATE docs SET fts_key=?1, fts_data=?2, fts_tags=?3, fts_synonyms=?4
                 WHERE catalog=?5 AND key=?6",
                params![doc.fts_key, doc.fts_data, doc.fts_tags, doc.fts_synonyms, catalog, key],
            )?;
        } else {
            tx.execute(
                "INSERT INTO docs(catalog, key, fts_key, fts_data, fts_tags, fts_synonyms)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
                params![catalog, key, doc.fts_key, doc.fts_data, doc.fts_tags, doc.fts_synonyms],
            )?;
        }
        let id: i64 = tx
            .query_row(
                "SELECT id FROM docs WHERE catalog = ?1 AND key = ?2",
                params![catalog, key],
                |r| r.get(0),
            )
            .optional()?
            .ok_or_else(|| {
                HypatiaError::Storage(StorageError::NotConnected(format!(
                    "docs row missing after upsert: {catalog}/{key}"
                )))
            })?;
        Ok(id)
    }

    // ── Knowledge CRUD ───────────────────────────────────────────────

    pub fn insert_knowledge(&self, name: &str, content: &Content) -> Result<()> {
        let json = content.to_json_string();
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO knowledge (name, content) VALUES (?1, ?2)",
            params![name, json],
        )
        .map_err(StorageError::from)?;
        let doc_id = self.docs_upsert_in(&tx, "knowledge", name, &fts_doc_for(content, name))?;
        replace_postings_in(&tx, doc_id, &json)?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_knowledge(&self, name: &str) -> Result<Option<Knowledge>> {
        let result = self
            .conn
            .query_row(
                "SELECT name, content, created_at FROM knowledge WHERE name = ?1",
                params![name],
                row_to_knowledge,
            )
            .optional()?;
        Ok(result)
    }

    pub fn update_knowledge(&self, name: &str, content: &Content) -> Result<()> {
        let json = content.to_json_string();
        let tx = self.conn.unchecked_transaction()?;
        let rows = tx
            .execute(
                "UPDATE knowledge SET content = ?1 WHERE name = ?2",
                params![json, name],
            )
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            });
        }
        let doc_id = self.docs_upsert_in(&tx, "knowledge", name, &fts_doc_for(content, name))?;
        replace_postings_in(&tx, doc_id, &json)?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_knowledge(&self, name: &str) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        let rows = tx
            .execute("DELETE FROM knowledge WHERE name = ?1", params![name])
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            });
        }
        tx.execute(
            "DELETE FROM docs WHERE catalog = 'knowledge' AND key = ?1",
            params![name],
        )
        .map_err(StorageError::from)?;
        tx.commit()?;
        Ok(())
    }

    pub fn query_knowledge(
        &self,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<Vec<Knowledge>> {
        let sql_params: Vec<rusqlite::types::Value> = params.iter().map(json_to_sql).collect();
        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(sql_params.clone()), row_to_knowledge)
            .map_err(StorageError::from)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    // ── Statement CRUD ───────────────────────────────────────────────

    pub fn insert_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<()> {
        let json = content.to_json_string();
        let triple = key.to_csv_key();
        let tr_start_str = tr_start.as_ref().map(format_timestamp);
        let tr_end_str = tr_end.as_ref().map(format_timestamp);
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO statement (triple, head, relation, tail, content, tr_start, tr_end)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![triple, key.head, key.relation, key.tail, json, tr_start_str, tr_end_str],
        )
        .map_err(StorageError::from)?;
        let doc_id = self.docs_upsert_in(&tx, "statement", &triple, &fts_doc_for(content, &triple))?;
        replace_postings_in(&tx, doc_id, &json)?;
        tx.commit()?;
        Ok(())
    }

    pub fn get_statement(&self, key: &StatementKey) -> Result<Option<Statement>> {
        let triple = key.to_csv_key();
        let result = self
            .conn
            .query_row(
                "SELECT triple, head, relation, tail, content, created_at, tr_start, tr_end
                 FROM statement WHERE triple = ?1",
                params![triple],
                row_to_statement,
            )
            .optional()?;
        Ok(result)
    }

    pub fn update_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<()> {
        let json = content.to_json_string();
        let triple = key.to_csv_key();
        let tr_start_str = tr_start.as_ref().map(format_timestamp);
        let tr_end_str = tr_end.as_ref().map(format_timestamp);
        let tx = self.conn.unchecked_transaction()?;
        let rows = tx
            .execute(
                "UPDATE statement SET content = ?1, tr_start = ?2, tr_end = ?3 WHERE triple = ?4",
                params![json, tr_start_str, tr_end_str, triple],
            )
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: triple,
            });
        }
        let doc_id = self.docs_upsert_in(&tx, "statement", &triple, &fts_doc_for(content, &triple))?;
        replace_postings_in(&tx, doc_id, &json)?;
        tx.commit()?;
        Ok(())
    }

    pub fn delete_statement(&self, key: &StatementKey) -> Result<()> {
        let triple = key.to_csv_key();
        let tx = self.conn.unchecked_transaction()?;
        let rows = tx
            .execute("DELETE FROM statement WHERE triple = ?1", params![triple])
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: triple,
            });
        }
        let doc_id: Option<i64> = tx
            .query_row(
                "SELECT id FROM docs WHERE catalog = 'statement' AND key = ?1",
                params![triple],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = doc_id {
            tx.execute("DELETE FROM json_index WHERE doc_id = ?1", params![id])
                .map_err(StorageError::from)?;
        }
        tx.execute(
            "DELETE FROM docs WHERE catalog = 'statement' AND key = ?1",
            params![triple],
        )
        .map_err(StorageError::from)?;
        tx.commit()?;
        Ok(())
    }

    pub fn query_statements(
        &self,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<Vec<Statement>> {
        let sql_params: Vec<rusqlite::types::Value> = params.iter().map(json_to_sql).collect();
        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(sql_params.clone()), row_to_statement)
            .map_err(StorageError::from)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    // ── K-hop graph traversal ────────────────────────────────────────

    /// Forward traversal from `head` following `head → tail` edges up to `depth`.
    pub fn query_khop(
        &self,
        head: &str,
        relation: Option<&str>,
        depth: i64,
    ) -> Result<Vec<Statement>> {
        let (anchor_pred, recursive_pred, mut sql_params) = match relation {
            Some(p) => (
                "AND relation = ?".to_string(),
                "AND s.relation = ?".to_string(),
                vec![
                    serde_json::Value::String(head.to_string()),
                    serde_json::Value::String(p.to_string()),
                ],
            ),
            None => (
                String::new(),
                String::new(),
                vec![serde_json::Value::String(head.to_string())],
            ),
        };
        // Depth MUST bind as INTEGER: in SQLite `int < 'text'` is always true
        // (type ordering), which would disable the depth bound entirely and
        // make cyclic graphs recurse forever.
        sql_params.push(serde_json::Value::Number(depth.into()));
        if let Some(p) = relation {
            sql_params.push(serde_json::Value::String(p.to_string()));
        }

        // GROUP BY triple + MIN(depth): SQLite returns bare columns from the
        // min-depth row, replicating the old DISTINCT ON behaviour.
        let sql = format!(
            "WITH RECURSIVE hop AS (\
               SELECT triple, head, relation, tail, content, \
                      created_at, tr_start, tr_end, 1 AS depth \
               FROM statement WHERE head = ? {anchor_pred} \
               UNION ALL \
               SELECT s.triple, s.head, s.relation, s.tail, s.content, \
                      s.created_at, s.tr_start, s.tr_end, h.depth + 1 \
               FROM hop h JOIN statement s ON h.tail = s.head \
               WHERE h.depth < ? {recursive_pred}\
             ) \
             SELECT triple, head, relation, tail, content, created_at, tr_start, tr_end \
             FROM hop GROUP BY triple ORDER BY MIN(depth), created_at DESC"
        );

        let sql_params: Vec<rusqlite::types::Value> =
            sql_params.iter().map(json_to_sql).collect();
        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(sql_params.clone()), row_to_statement)
            .map_err(StorageError::from)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    // ── Embeddings (BLOB = source of truth) ──────────────────────────

    pub fn upsert_knowledge_embedding(&self, name: &str, vector: &[f32]) -> Result<()> {
        self.conn
            .execute(
                "UPDATE knowledge SET embedding = ?1 WHERE name = ?2",
                params![vector_to_blob(vector), name],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn upsert_statement_embedding(&self, triple: &str, vector: &[f32]) -> Result<()> {
        self.conn
            .execute(
                "UPDATE statement SET embedding = ?1 WHERE triple = ?2",
                params![vector_to_blob(vector), triple],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn clear_knowledge_embedding(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE knowledge SET embedding = NULL WHERE name = ?1",
                params![name],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn clear_statement_embedding(&self, triple: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE statement SET embedding = NULL WHERE triple = ?1",
                params![triple],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    fn entries_with_embeddings(&self, sql: &str) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |row| {
                let key: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((key, content))
            })
            .map_err(StorageError::from)?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    pub fn knowledge_with_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries_with_embeddings(
            "SELECT name, content FROM knowledge WHERE embedding IS NOT NULL",
        )
    }

    pub fn knowledge_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries_with_embeddings(
            "SELECT name, content FROM knowledge WHERE embedding IS NULL",
        )
    }

    pub fn statements_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        self.entries_with_embeddings(
            "SELECT triple, content FROM statement WHERE embedding IS NULL",
        )
    }

    // ── Vector index plumbing (doc_id ↔ embedding pairs) ─────────────

    /// Number of rows carrying an embedding for one catalog.
    pub fn embedding_row_count(&self, catalog: &str) -> Result<usize> {
        let sql = match catalog {
            "statement" => "SELECT COUNT(*) FROM statement WHERE embedding IS NOT NULL",
            _ => "SELECT COUNT(*) FROM knowledge WHERE embedding IS NOT NULL",
        };
        let n: i64 = self.conn.query_row(sql, [], |r| r.get(0)).map_err(StorageError::from)?;
        Ok(n as usize)
    }

    /// (doc_id, embedding BLOB) pairs for one catalog — authoritative data
    /// for rebuilding an external vector index.
    pub fn embedding_pairs(&self, catalog: &str) -> Result<Vec<(i64, Vec<u8>)>> {
        let sql = match catalog {
            "statement" => {
                "SELECT d.id, s.embedding FROM docs d                  JOIN statement s ON s.triple = d.key                  WHERE d.catalog = 'statement' AND s.embedding IS NOT NULL"
            }
            _ => {
                "SELECT d.id, k.embedding FROM docs d                  JOIN knowledge k ON k.name = d.key                  WHERE d.catalog = 'knowledge' AND k.embedding IS NOT NULL"
            }
        };
        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(StorageError::from)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(StorageError::from)?);
        }
        Ok(out)
    }

    /// doc_id for a catalog key (name / triple CSV).
    pub fn doc_id_by_key(&self, catalog: &str, key: &str) -> Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM docs WHERE catalog = ?1 AND key = ?2",
                params![catalog, key],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// Fetch (doc_id, key, content) rows by doc_id (for $similar results).
    pub fn rows_by_doc_ids(
        &self,
        catalog: &str,
        ids: &[i64],
    ) -> Result<Vec<(i64, String, String)>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let (sql, params): (String, Vec<rusqlite::types::Value>) = match catalog {
            "statement" => (
                format!(
                    "SELECT d.id, s.triple, s.content FROM statement s \
                     JOIN docs d ON d.catalog='statement' AND d.key = s.triple \
                     WHERE d.id IN ({placeholders})"
                ),
                ids.iter().map(|i| json_to_sql(&serde_json::Value::from(*i))).collect(),
            ),
            _ => (
                format!(
                    "SELECT d.id, k.name, k.content FROM knowledge k \
                     JOIN docs d ON d.catalog='knowledge' AND d.key = k.name \
                     WHERE d.id IN ({placeholders})"
                ),
                ids.iter().map(|i| json_to_sql(&serde_json::Value::from(*i))).collect(),
            ),
        };
        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.iter()), |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(StorageError::from)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(StorageError::from)?);
        }
        Ok(out)
    }

    // ── Vector search: Rust brute-force kNN over embedding BLOBs ─────

    pub fn vector_search_knowledge(
        &self,
        query_vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.brute_force_search(
            "SELECT name, content, embedding FROM knowledge WHERE embedding IS NOT NULL",
            query_vector,
            limit,
        )
    }

    pub fn vector_search_statements(
        &self,
        query_vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.brute_force_search(
            "SELECT triple, content, embedding FROM statement WHERE embedding IS NOT NULL",
            query_vector,
            limit,
        )
    }

    fn brute_force_search(
        &self,
        sql: &str,
        query_vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let mut rows = stmt.query([]).map_err(StorageError::from)?;
        let mut scored: Vec<(String, String, f64)> = Vec::new();
        while let Some(row) = rows.next().map_err(StorageError::from)? {
            let key: String = row.get(0)?;
            let content: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let v = blob_to_vector(&blob);
            if v.len() != query_vector.len() {
                continue; // dimension mismatch (e.g. model changed): skip
            }
            scored.push((key, content, cosine_distance(query_vector, &v)));
        }
        scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit.max(0) as usize);
        Ok(scored)
    }

    // ── FTS search ───────────────────────────────────────────────────

    pub fn search(&self, query: &str, opts: &SearchOpts) -> Result<Vec<FtsResult>> {
        let query = crate::text::segment_for_fts(query);
        let query = sanitize_fts_query(&query);
        let catalog_filter = opts.catalog.as_deref();
        // docs no longer stores a content copy: JOIN the source tables.
        let sql = format!(
            "SELECT d.id, d.catalog, d.key, {BM25_WEIGHTS} AS rank, \
                    COALESCE(k.content, s.content, '') AS content \
             FROM docs d \
             JOIN docs_fts f ON d.id = f.rowid \
             LEFT JOIN knowledge k ON d.catalog = 'knowledge' AND k.name = d.key \
             LEFT JOIN statement s ON d.catalog = 'statement' AND s.triple = d.key \
             WHERE docs_fts MATCH ?1 {} \
             ORDER BY rank LIMIT ?2 OFFSET ?3",
            if catalog_filter.is_some() { "AND d.catalog = ?4" } else { "" }
        );
        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let map_row = |row: &rusqlite::Row| -> rusqlite::Result<FtsResult> {
            Ok(FtsResult {
                id: row.get(0)?,
                catalog: row.get(1)?,
                key: row.get(2)?,
                rank: row.get(3)?,
                content: row.get(4)?,
            })
        };
        let rows: Vec<rusqlite::Result<FtsResult>> = if let Some(cat) = catalog_filter {
            stmt.query_map(rusqlite::params![query, opts.limit, opts.offset, cat], map_row)
                .map_err(StorageError::from)?
                .collect()
        } else {
            stmt.query_map(rusqlite::params![query, opts.limit, opts.offset], map_row)
                .map_err(StorageError::from)?
                .collect()
        };
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }
}

/// Build FTS columns from Content (segmented for CJK), mirroring the legacy pipeline.
pub(crate) fn fts_doc_for(content: &Content, key: &str) -> FtsDoc {
    let fields = content.fts_fields(key);
    FtsDoc {
        fts_key: fields.key,
        fts_data: fields.data,
        fts_tags: fields.tags,
        fts_synonyms: fields.synonyms,
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
    use crate::model::Format;
    use tempfile::TempDir;

    fn setup() -> (TempDir, SqliteStore) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("hypatia.sqlite");
        let store = SqliteStore::open(&db_path).unwrap();
        (dir, store)
    }

    #[test]
    fn schema_init() {
        let (_dir, _store) = setup();
    }

    #[test]
    fn knowledge_roundtrip() {
        let (_dir, store) = setup();
        let content = Content::new("hello world").with_tags(vec!["test".to_string()]);
        store.insert_knowledge("test-entry", &content).unwrap();

        let loaded = store.get_knowledge("test-entry").unwrap().unwrap();
        assert_eq!(loaded.name, "test-entry");
        assert_eq!(loaded.content.data, "hello world");
        assert_eq!(loaded.content.tags, vec!["test"]);
    }

    #[test]
    fn knowledge_not_found() {
        let (_dir, store) = setup();
        assert!(store.get_knowledge("nonexistent").unwrap().is_none());
    }

    #[test]
    fn knowledge_update() {
        let (_dir, store) = setup();
        store.insert_knowledge("k1", &Content::new("v1")).unwrap();

        let updated = Content::new("v2").with_format(Format::Json);
        store.update_knowledge("k1", &updated).unwrap();

        let loaded = store.get_knowledge("k1").unwrap().unwrap();
        assert_eq!(loaded.content.data, "v2");
        assert_eq!(loaded.content.format, Format::Json);
    }

    #[test]
    fn knowledge_delete() {
        let (_dir, store) = setup();
        store.insert_knowledge("k1", &Content::default()).unwrap();
        store.delete_knowledge("k1").unwrap();
        assert!(store.get_knowledge("k1").unwrap().is_none());
    }

    #[test]
    fn statement_roundtrip() {
        let (_dir, store) = setup();
        let key = StatementKey::new("Alice", "knows", "Bob");
        let content = Content::new("they are friends");
        store.insert_statement(&key, &content, None, None).unwrap();

        let loaded = store.get_statement(&key).unwrap().unwrap();
        assert_eq!(loaded.key.to_csv_key(), "Alice,knows,Bob");
        assert_eq!(loaded.key.head, "Alice");
        assert_eq!(loaded.key.relation, "knows");
        assert_eq!(loaded.key.tail, "Bob");
        assert_eq!(loaded.content.data, "they are friends");
    }

    #[test]
    fn statement_with_temporal_range() {
        use chrono::NaiveDate;
        let (_dir, store) = setup();
        let key = StatementKey::new("Alice", "worked_at", "Company");
        let content = Content::default();
        let start = NaiveDate::from_ymd_opt(2020, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        let end = NaiveDate::from_ymd_opt(2023, 12, 31)
            .unwrap()
            .and_hms_opt(23, 59, 59)
            .unwrap();
        store
            .insert_statement(&key, &content, Some(start), Some(end))
            .unwrap();

        let loaded = store.get_statement(&key).unwrap().unwrap();
        assert_eq!(loaded.tr_start, Some(start));
        assert_eq!(loaded.tr_end, Some(end));
    }

    #[test]
    fn statement_delete() {
        let (_dir, store) = setup();
        let key = StatementKey::new("A", "rel", "B");
        store
            .insert_statement(&key, &Content::default(), None, None)
            .unwrap();
        store.delete_statement(&key).unwrap();
        assert!(store.get_statement(&key).unwrap().is_none());
    }

    #[test]
    fn fts_upsert_and_search() {
        let (_dir, store) = setup();
        let content = Content::new("Rust is a systems programming language");
        store.insert_knowledge("rust", &content).unwrap();
        let content2 = Content::new("Python is a scripting language");
        store.insert_knowledge("python", &content2).unwrap();

        let results = store.search("programming", &SearchOpts::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "rust");
        // content is JOINed from the source table, not a copy
        assert!(results[0].content.contains("systems programming"));
    }

    #[test]
    fn fts_delete_removes_from_index() {
        let (_dir, store) = setup();
        store
            .insert_knowledge("rust", &Content::new("Rust programming language"))
            .unwrap();
        store.delete_knowledge("rust").unwrap();
        let results = store.search("programming", &SearchOpts::default()).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn fts_survives_reopen() {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("hypatia.sqlite");
        {
            let store = SqliteStore::open(&db_path).unwrap();
            store
                .insert_knowledge("rust", &Content::new("Rust is a systems programming language"))
                .unwrap();
        }
        {
            let store = SqliteStore::open(&db_path).unwrap();
            let results = store.search("programming", &SearchOpts::default()).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].key, "rust");
        }
    }

    #[test]
    fn search_with_catalog_filter() {
        let (_dir, store) = setup();
        store
            .insert_knowledge("rust", &Content::new("Rust programming"))
            .unwrap();
        store
            .insert_statement(
                &StatementKey::new("rust", "is_a", "language"),
                &Content::new("Rust is a programming language"),
                None,
                None,
            )
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
    fn porter_stemmer() {
        let (_dir, store) = setup();
        store
            .insert_knowledge("auth", &Content::new("user authenticating via OAuth2"))
            .unwrap();
        let results = store.search("authentication", &SearchOpts::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "auth");
    }

    #[test]
    fn bm25_weights_key_over_data() {
        let (_dir, store) = setup();
        store
            .insert_knowledge("lang1", &Content::new("Rust programming language"))
            .unwrap();
        store
            .insert_knowledge("rust", &Content::new("systems programming"))
            .unwrap();

        let results = store.search("rust", &SearchOpts::default()).unwrap();
        assert!(results.len() >= 2);
        // Key match should rank first (more negative = better)
        assert_eq!(results[0].key, "rust");
    }

    #[test]
    fn query_knowledge_with_condition() {
        let (_dir, store) = setup();
        store.insert_knowledge("k1", &Content::new("v1")).unwrap();
        store.insert_knowledge("k2", &Content::new("v2")).unwrap();

        let rows = store
            .query_knowledge(
                "SELECT name, content, created_at FROM knowledge WHERE name = ? LIMIT ? OFFSET ?",
                vec![serde_json::json!("k1"), serde_json::json!(100), serde_json::json!(0)],
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].name, "k1");
    }

    #[test]
    fn knowledge_vector_upsert_and_search() {
        let (_dir, store) = setup();
        store
            .insert_knowledge("rust", &Content::new("Rust programming language"))
            .unwrap();
        store
            .insert_knowledge("python", &Content::new("Python scripting language"))
            .unwrap();

        let vector_a = vec![1.0f32; 64];
        let vector_b = vec![0.0f32; 64];
        store.upsert_knowledge_embedding("rust", &vector_a).unwrap();
        store.upsert_knowledge_embedding("python", &vector_b).unwrap();

        let results = store.vector_search_knowledge(&vector_a, 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "rust"); // distance 0 = identical
    }

    #[test]
    fn vector_search_excludes_null_embeddings() {
        let (_dir, store) = setup();
        store.insert_knowledge("with_vec", &Content::new("data")).unwrap();
        store.insert_knowledge("no_vec", &Content::new("data")).unwrap();

        let vector = vec![0.5f32; 32];
        store.upsert_knowledge_embedding("with_vec", &vector).unwrap();

        let results = store.vector_search_knowledge(&vector, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "with_vec");
    }

    #[test]
    fn without_embeddings_lists_only_missing() {
        let (_dir, store) = setup();
        store.insert_knowledge("has_vec", &Content::new("data")).unwrap();
        store.insert_knowledge("no_vec", &Content::new("data")).unwrap();

        let vector = vec![0.5f32; 32];
        store.upsert_knowledge_embedding("has_vec", &vector).unwrap();

        let missing = store.knowledge_without_embeddings().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].0, "no_vec");
    }

    #[test]
    fn clear_embedding() {
        let (_dir, store) = setup();
        store.insert_knowledge("k1", &Content::new("data")).unwrap();
        let vector = vec![0.5f32; 32];
        store.upsert_knowledge_embedding("k1", &vector).unwrap();
        assert_eq!(store.knowledge_with_embeddings().unwrap().len(), 1);

        store.clear_knowledge_embedding("k1").unwrap();
        assert_eq!(store.knowledge_with_embeddings().unwrap().len(), 0);
    }

    #[test]
    fn khop_1hop_specific_relation() {
        let (_dir, store) = setup();
        store
            .insert_statement(&StatementKey::new("Alice", "knows", "Bob"), &Content::new("a->b"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("Bob", "knows", "Carol"), &Content::new("b->c"), None, None)
            .unwrap();

        let results = store.query_khop("Alice", Some("knows"), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key.tail, "Bob");
    }

    #[test]
    fn khop_2hop_specific_relation() {
        let (_dir, store) = setup();
        store
            .insert_statement(&StatementKey::new("Alice", "knows", "Bob"), &Content::new("a->b"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("Bob", "knows", "Carol"), &Content::new("b->c"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("Bob", "works_with", "Dave"), &Content::new("b->d"), None, None)
            .unwrap();

        let results = store.query_khop("Alice", Some("knows"), 2).unwrap();
        assert_eq!(results.len(), 2);
        let tails: Vec<&str> = results.iter().map(|s| s.key.tail.as_str()).collect();
        assert!(tails.contains(&"Bob"));
        assert!(tails.contains(&"Carol"));
    }

    #[test]
    fn khop_wildcard_relation() {
        let (_dir, store) = setup();
        store
            .insert_statement(&StatementKey::new("Alice", "knows", "Bob"), &Content::new("a->b"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("Bob", "knows", "Carol"), &Content::new("b->c"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("Bob", "works_with", "Dave"), &Content::new("b->d"), None, None)
            .unwrap();

        let results = store.query_khop("Alice", None, 2).unwrap();
        assert_eq!(results.len(), 3);
        let tails: Vec<&str> = results.iter().map(|s| s.key.tail.as_str()).collect();
        assert!(tails.contains(&"Bob"));
        assert!(tails.contains(&"Carol"));
        assert!(tails.contains(&"Dave"));
    }

    #[test]
    fn khop_cycle() {
        let (_dir, store) = setup();
        store
            .insert_statement(&StatementKey::new("A", "knows", "B"), &Content::new("a->b"), None, None)
            .unwrap();
        store
            .insert_statement(&StatementKey::new("B", "knows", "A"), &Content::new("b->a"), None, None)
            .unwrap();

        let results = store.query_khop("A", Some("knows"), 5).unwrap();
        assert_eq!(results.len(), 2); // GROUP BY triple dedupes
    }

    #[test]
    fn khop_no_results() {
        let (_dir, store) = setup();
        let results = store.query_khop("NonExistent", Some("knows"), 3).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn json_contains_query_e2e() {
        let dir = TempDir::new().unwrap();
        let store = SqliteStore::open(dir.path().join("db.sqlite").as_path()).unwrap();
        store
            .insert_knowledge(
                "plan",
                &Content::new("data").with_tags(vec!["hypatia".into(), "refactor".into()]),
            )
            .unwrap();
        store
            .insert_knowledge("other", &Content::new("x").with_tags(vec!["misc".into()]))
            .unwrap();

        use crate::engine::ast::AstNode;
        use crate::model::QueryTarget;
        use crate::engine::operators::{evaluate_operator, OpContext};
        let ctx = OpContext::for_target(QueryTarget::Knowledge);
        let mut map = serde_json::Map::new();
        map.insert("tags".to_string(), serde_json::json!(["refactor"]));
        let result = evaluate_operator(
            "$json-contains",
            &[AstNode::Object(map)],
            &serde_json::Map::new(),
            &ctx,
            &|_| unreachable!(),
        )
        .unwrap();
        let (fragment, params) = match result {
            crate::engine::operators::OperatorResult::SqlCondition { fragment, params } => {
                (fragment, params)
            }
            other => panic!("expected SqlCondition, got {other:?}"),
        };
        let sql = format!(
            "SELECT name, content, created_at FROM knowledge WHERE {fragment}"
        );
        let rows = store.query_knowledge(&sql, params).unwrap();
        let names: Vec<String> = rows.into_iter().map(|k| k.name).collect();
        assert_eq!(names, vec!["plan".to_string()], "fragment: {fragment}");

        // Split: recall only
        let recall_only = "SELECT name, content, created_at FROM knowledge WHERE EXISTS (SELECT 1 FROM docs d              JOIN json_index j ON j.doc_id = d.id WHERE d.catalog = 'knowledge'              AND d.key = knowledge.name AND j.path = 'tags' AND j.value = 'refactor')";
        let rows = store.query_knowledge(recall_only, vec![]).unwrap();
        assert_eq!(rows.len(), 1, "recall failed");

        // Split: recheck only
        let recheck_only =
            "SELECT name, content, created_at FROM knowledge WHERE name = 'plan' AND json_contains(content, ?)";
        let rows = store
            .query_knowledge(recheck_only, vec![serde_json::json!({"tags": ["refactor"]})])
            .unwrap();
        let recheck_names: Vec<String> = rows.into_iter().map(|k| k.name).collect();
        assert_eq!(recheck_names, vec!["plan".to_string()], "recheck failed");
    }

    #[test]
    fn vector_blob_roundtrip() {
        let v = vec![1.5f32, -2.25, 0.0, f32::NAN];
        let blob = vector_to_blob(&v);
        let decoded = blob_to_vector(&blob);
        assert_eq!(decoded.len(), 4);
        assert_eq!(decoded[0], 1.5);
        assert_eq!(decoded[1], -2.25);
        assert_eq!(decoded[2], 0.0);
        assert!(decoded[3] == 0.0); // NaN sanitized
    }
}

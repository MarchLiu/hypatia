use std::path::Path;

use chrono::NaiveDateTime;
use duckdb::{params, Connection, OptionalExt};

use crate::error::{HypatiaError, Result, StorageError};
use crate::model::{Content, Knowledge, Statement, StatementKey};

fn knowledge_schema(dimensions: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS knowledge (
    name TEXT PRIMARY KEY,
    content JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding FLOAT[{dimensions}]
)"
    )
}

fn statement_schema(dimensions: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS statement (
    triple TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    content JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tr_start TIMESTAMP,
    tr_end TIMESTAMP,
    embedding FLOAT[{dimensions}]
);
CREATE INDEX IF NOT EXISTS idx_stmt_subject ON statement(subject);
CREATE INDEX IF NOT EXISTS idx_stmt_predicate ON statement(predicate);
CREATE INDEX IF NOT EXISTS idx_stmt_object ON statement(object)"
    )
}

/// SQL fragment for selecting all knowledge columns with timestamps as strings.
const KNOWLEDGE_SELECT: &str = "\
SELECT name, content, CAST(created_at AS VARCHAR) AS created_at FROM knowledge";

/// SQL fragment for selecting all statement columns with timestamps as strings.
const STATEMENT_SELECT: &str = "\
SELECT triple, subject, predicate, object, content, \
       CAST(created_at AS VARCHAR) AS created_at, \
       CAST(tr_start AS VARCHAR) AS tr_start, \
       CAST(tr_end AS VARCHAR) AS tr_end FROM statement";

pub struct DuckDbStore {
    conn: Connection,
}

/// Convert a serde_json::Value to a DuckDB-compatible string parameter.
fn json_value_to_sql_param(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Parse a timestamp string from DuckDB into NaiveDateTime.
fn parse_timestamp(s: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S"))
        .map_err(|e| HypatiaError::Storage(StorageError::DuckDb(
            duckdb::Error::FromSqlConversionFailure(0, duckdb::types::Type::Text, Box::new(e)),
        )))
}

/// Format a NaiveDateTime for DuckDB parameter binding.
fn format_timestamp(dt: &NaiveDateTime) -> String {
    dt.format("%Y-%m-%d %H:%M:%S%.f").to_string()
}

impl DuckDbStore {
    pub fn open(path: &Path, dimensions: usize) -> Result<Self> {
        let conn = Connection::open(path).map_err(StorageError::from)?;
        let store = Self { conn };
        store.init_schema(dimensions)?;
        Ok(store)
    }

    fn init_schema(&self, dimensions: usize) -> Result<()> {
        self.conn
            .execute_batch(&format!("{}; {}", knowledge_schema(dimensions), statement_schema(dimensions)))
            .map_err(StorageError::from)?;
        Ok(())
    }

    // --- Knowledge CRUD ---

    pub fn insert_knowledge(&self, name: &str, content: &Content) -> Result<()> {
        let json = content.to_json_string();
        self.conn
            .execute(
                "INSERT INTO knowledge (name, content) VALUES (?, ?)",
                params![name, json],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn get_knowledge(&self, name: &str) -> Result<Option<Knowledge>> {
        let result = self
            .conn
            .query_row(
                &format!("{KNOWLEDGE_SELECT} WHERE name = ?"),
                params![name],
                |row| {
                    let name: String = row.get(0)?;
                    let json: String = row.get(1)?;
                    let created_at_str: String = row.get(2)?;
                    Ok((name, json, created_at_str))
                },
            )
            .optional()
            .map_err(StorageError::from)?;

        result
            .map(|(name, json, created_at_str)| {
                let content = Content::from_json_str(&json)?;
                let created_at = parse_timestamp(&created_at_str)?;
                Ok(Knowledge { name, content, created_at })
            })
            .transpose()
    }

    pub fn update_knowledge(&self, name: &str, content: &Content) -> Result<()> {
        let json = content.to_json_string();
        let rows = self
            .conn
            .execute(
                "UPDATE knowledge SET content = ? WHERE name = ?",
                params![json, name],
            )
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            });
        }
        Ok(())
    }

    pub fn delete_knowledge(&self, name: &str) -> Result<()> {
        let rows = self
            .conn
            .execute("DELETE FROM knowledge WHERE name = ?", params![name])
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "knowledge".to_string(),
                key: name.to_string(),
            });
        }
        Ok(())
    }

    pub fn query_knowledge(
        &self,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<Vec<Knowledge>> {
        let sql_params: Vec<String> = params.iter().map(json_value_to_sql_param).collect();
        let param_refs: Vec<&dyn duckdb::ToSql> =
            sql_params.iter().map(|s| s as &dyn duckdb::ToSql).collect();

        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let name: String = row.get(0)?;
                let json: String = row.get(1)?;
                let created_at_str: String = row.get(2)?;
                Ok((name, json, created_at_str))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            let (name, json, created_at_str) = row.map_err(StorageError::from)?;
            let content = Content::from_json_str(&json)?;
            let created_at = parse_timestamp(&created_at_str)?;
            result.push(Knowledge { name, content, created_at });
        }
        Ok(result)
    }

    // --- Statement CRUD ---

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
        self.conn
            .execute(
                "INSERT INTO statement (triple, subject, predicate, object, content, tr_start, tr_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![triple, key.subject, key.predicate, key.object, json, tr_start_str, tr_end_str],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    pub fn get_statement(&self, key: &StatementKey) -> Result<Option<Statement>> {
        let triple = key.to_csv_key();
        let result = self
            .conn
            .query_row(
                &format!("{STATEMENT_SELECT} WHERE triple = ?"),
                params![triple],
                |row| {
                    let triple: String = row.get(0)?;
                    let subject: String = row.get(1)?;
                    let predicate: String = row.get(2)?;
                    let object: String = row.get(3)?;
                    let json: String = row.get(4)?;
                    let created_at_str: String = row.get(5)?;
                    let tr_start_str: Option<String> = row.get(6)?;
                    let tr_end_str: Option<String> = row.get(7)?;
                    Ok((triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str))
                },
            )
            .optional()
            .map_err(StorageError::from)?;

        result
            .map(|(triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str)| {
                let content = Content::from_json_str(&json)?;
                let created_at = parse_timestamp(&created_at_str)?;
                let tr_start = tr_start_str.as_deref().map(parse_timestamp).transpose()?;
                let tr_end = tr_end_str.as_deref().map(parse_timestamp).transpose()?;
                let key = StatementKey { subject, predicate, object };
                let _ = triple; // triple is the PK, key is derived from columns
                Ok(Statement { key, content, created_at, tr_start, tr_end })
            })
            .transpose()
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
        let rows = self.conn.execute(
            "UPDATE statement SET content = ?, tr_start = ?, tr_end = ? WHERE triple = ?",
            params![json, tr_start_str, tr_end_str, triple],
        ).map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: triple,
            });
        }
        Ok(())
    }

    pub fn delete_statement(&self, key: &StatementKey) -> Result<()> {
        let triple = key.to_csv_key();
        let rows = self
            .conn
            .execute(
                "DELETE FROM statement WHERE triple = ?",
                params![triple],
            )
            .map_err(StorageError::from)?;
        if rows == 0 {
            return Err(HypatiaError::NotFound {
                kind: "statement".to_string(),
                key: triple,
            });
        }
        Ok(())
    }

    pub fn query_statements(
        &self,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<Vec<Statement>> {
        let sql_params: Vec<String> = params.iter().map(json_value_to_sql_param).collect();
        let param_refs: Vec<&dyn duckdb::ToSql> =
            sql_params.iter().map(|s| s as &dyn duckdb::ToSql).collect();

        let mut stmt = self.conn.prepare(sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                let triple: String = row.get(0)?;
                let subject: String = row.get(1)?;
                let predicate: String = row.get(2)?;
                let object: String = row.get(3)?;
                let json: String = row.get(4)?;
                let created_at_str: String = row.get(5)?;
                let tr_start_str: Option<String> = row.get(6)?;
                let tr_end_str: Option<String> = row.get(7)?;
                Ok((triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            let (triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str) =
                row.map_err(StorageError::from)?;
            let content = Content::from_json_str(&json)?;
            let created_at = parse_timestamp(&created_at_str)?;
            let tr_start = tr_start_str.as_deref().map(parse_timestamp).transpose()?;
            let tr_end = tr_end_str.as_deref().map(parse_timestamp).transpose()?;
            let key = StatementKey { subject, predicate, object };
            let _ = triple;
            result.push(Statement { key, content, created_at, tr_start, tr_end });
        }
        Ok(result)
    }

    // --- Vector operations ---

    /// Store an embedding vector for a knowledge entry.
    pub fn upsert_knowledge_embedding(&self, name: &str, vector: &[f32]) -> Result<()> {
        let vec_literal = vector_to_sql_literal(vector);
        let sql = format!(
            "UPDATE knowledge SET embedding = {vec_literal}::FLOAT[{}] WHERE name = ?",
            vector.len()
        );
        self.conn
            .execute(&sql, params![name])
            .map_err(StorageError::from)?;
        Ok(())
    }

    /// Store an embedding vector for a statement entry.
    pub fn upsert_statement_embedding(&self, triple: &str, vector: &[f32]) -> Result<()> {
        let vec_literal = vector_to_sql_literal(vector);
        let sql = format!(
            "UPDATE statement SET embedding = {vec_literal}::FLOAT[{}] WHERE triple = ?",
            vector.len()
        );
        self.conn
            .execute(&sql, params![triple])
            .map_err(StorageError::from)?;
        Ok(())
    }

    /// Clear the embedding vector for a knowledge entry.
    pub fn clear_knowledge_embedding(&self, name: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE knowledge SET embedding = NULL WHERE name = ?",
                params![name],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    /// Clear the embedding vector for a statement entry.
    pub fn clear_statement_embedding(&self, triple: &str) -> Result<()> {
        self.conn
            .execute(
                "UPDATE statement SET embedding = NULL WHERE triple = ?",
                params![triple],
            )
            .map_err(StorageError::from)?;
        Ok(())
    }

    /// Search knowledge entries by vector similarity (cosine distance).
    /// Returns (name, content_json, distance) tuples sorted by similarity.
    pub fn vector_search_knowledge(
        &self,
        query_vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        let vec_literal = vector_to_sql_literal(query_vector);
        let dims = query_vector.len();
        let sql = format!(
            "SELECT name, CAST(content AS VARCHAR), \
             array_cosine_distance(embedding, {vec_literal}::FLOAT[{dims}]) AS distance \
             FROM knowledge \
             WHERE embedding IS NOT NULL \
             ORDER BY distance ASC \
             LIMIT ?"
        );
        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(params![limit], |row| {
                let name: String = row.get(0)?;
                let content: String = row.get(1)?;
                let distance: f64 = row.get(2)?;
                Ok((name, content, distance))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    /// Search statement entries by vector similarity (cosine distance).
    /// Returns (triple, content_json, distance) tuples sorted by similarity.
    pub fn vector_search_statements(
        &self,
        query_vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        let vec_literal = vector_to_sql_literal(query_vector);
        let dims = query_vector.len();
        let sql = format!(
            "SELECT triple, CAST(content AS VARCHAR), \
             array_cosine_distance(embedding, {vec_literal}::FLOAT[{dims}]) AS distance \
             FROM statement \
             WHERE embedding IS NOT NULL \
             ORDER BY distance ASC \
             LIMIT ?"
        );
        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(params![limit], |row| {
                let triple: String = row.get(0)?;
                let content: String = row.get(1)?;
                let distance: f64 = row.get(2)?;
                Ok((triple, content, distance))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    /// Get all knowledge entries that have embeddings.
    /// Returns (name, content_json) pairs for backfill skip detection.
    pub fn knowledge_with_embeddings(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, CAST(content AS VARCHAR) FROM knowledge WHERE embedding IS NOT NULL")
            .map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |row| {
                let name: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((name, content))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    /// Get all knowledge entries that do NOT have embeddings.
    /// Returns (name, content_json) pairs for backfill.
    pub fn knowledge_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name, CAST(content AS VARCHAR) FROM knowledge WHERE embedding IS NULL")
            .map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |row| {
                let name: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((name, content))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    /// Get all statement entries that do NOT have embeddings.
    /// Returns (triple, content_json) pairs for backfill.
    pub fn statements_without_embeddings(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT triple, CAST(content AS VARCHAR) FROM statement WHERE embedding IS NULL")
            .map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |row| {
                let triple: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((triple, content))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(StorageError::from)?);
        }
        Ok(result)
    }

    // --- K-hop graph traversal ---

    /// Execute a k-hop forward graph traversal using a recursive CTE.
    /// Starting from `subject`, follows `subject → object` edges up to `depth` hops.
    /// If `predicate` is Some, only edges matching that predicate are followed.
    /// Returns matching statements ordered by hop distance.
    pub fn query_khop(
        &self,
        subject: &str,
        predicate: Option<&str>,
        depth: i64,
    ) -> Result<Vec<Statement>> {
        let (anchor_pred, recursive_pred, mut sql_params) = match predicate {
            Some(p) => (
                "AND predicate = ?".to_string(),
                "AND s.predicate = ?".to_string(),
                {
                    vec![subject.to_string(), p.to_string()]
                },
            ),
            None => (String::new(), String::new(), vec![subject.to_string()]),
        };
        sql_params.push(depth.to_string());
        if let Some(p) = predicate {
            sql_params.push(p.to_string());
        }

        let sql = format!(
            "WITH RECURSIVE hop AS (\
               SELECT triple, subject, predicate, object, content, \
                      CAST(created_at AS VARCHAR) AS created_at, \
                      CAST(tr_start AS VARCHAR) AS tr_start, \
                      CAST(tr_end AS VARCHAR) AS tr_end, \
                      1 AS depth \
               FROM statement WHERE subject = ? {anchor_pred} \
               UNION ALL \
               SELECT s.triple, s.subject, s.predicate, s.object, s.content, \
                      CAST(s.created_at AS VARCHAR), CAST(s.tr_start AS VARCHAR), CAST(s.tr_end AS VARCHAR), \
                      h.depth + 1 \
               FROM hop h JOIN statement s ON h.object = s.subject \
               WHERE h.depth < ? {recursive_pred}\
             ) \
             SELECT DISTINCT ON (triple) \
               triple, subject, predicate, object, content, created_at, tr_start, tr_end \
             FROM hop ORDER BY depth, created_at DESC"
        );

        let param_refs: Vec<&dyn duckdb::ToSql> =
            sql_params.iter().map(|s| s as &dyn duckdb::ToSql).collect();

        let mut stmt = self.conn.prepare(&sql).map_err(StorageError::from)?;
        let rows = stmt
            .query_map(param_refs.as_slice(), |row: &duckdb::Row| {
                let triple: String = row.get(0)?;
                let subject: String = row.get(1)?;
                let predicate: String = row.get(2)?;
                let object: String = row.get(3)?;
                let json: String = row.get(4)?;
                let created_at_str: String = row.get(5)?;
                let tr_start_str: Option<String> = row.get(6)?;
                let tr_end_str: Option<String> = row.get(7)?;
                Ok((triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str))
            })
            .map_err(StorageError::from)?;

        let mut result = Vec::new();
        for row in rows {
            let (triple, subject, predicate, object, json, created_at_str, tr_start_str, tr_end_str) =
                row.map_err(StorageError::from)?;
            let content = Content::from_json_str(&json)?;
            let created_at = parse_timestamp(&created_at_str)?;
            let tr_start = tr_start_str.as_deref().map(parse_timestamp).transpose()?;
            let tr_end = tr_end_str.as_deref().map(parse_timestamp).transpose()?;
            let key = StatementKey { subject, predicate, object };
            let _ = triple;
            result.push(Statement { key, content, created_at, tr_start, tr_end });
        }
        Ok(result)
    }
}

fn vector_to_sql_literal(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|f| {
        if f.is_nan() || f.is_infinite() {
            "0.0".to_string()
        } else {
            format!("{f:.8}")
        }
    }).collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Format;
    use tempfile::TempDir;

    fn setup() -> (TempDir, DuckDbStore) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.duckdb");
        let store = DuckDbStore::open(&db_path, 1024).unwrap();
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
    fn knowledge_vector_upsert_and_search() {
        let (_dir, store) = setup();
        store.insert_knowledge("rust", &Content::new("Rust programming language")).unwrap();
        store.insert_knowledge("python", &Content::new("Python scripting language")).unwrap();

        // Rust and "memory safety" are semantically closer
        let vector_a = vec![1.0f32; 1024];
        let vector_b = vec![0.0f32; 1024];
        store.upsert_knowledge_embedding("rust", &vector_a).unwrap();
        store.upsert_knowledge_embedding("python", &vector_b).unwrap();

        // Search with vector similar to rust's embedding
        let results = store.vector_search_knowledge(&vector_a, 10).unwrap();
        assert_eq!(results.len(), 2);
        // First result should be "rust" (distance 0 = identical)
        assert_eq!(results[0].0, "rust");
    }

    #[test]
    fn statement_vector_upsert_and_search() {
        let (_dir, store) = setup();
        let key1 = StatementKey::new("Alice", "knows", "Bob");
        let key2 = StatementKey::new("Carol", "manages", "Dave");
        store.insert_statement(&key1, &Content::new("friends"), None, None).unwrap();
        store.insert_statement(&key2, &Content::new("coworkers"), None, None).unwrap();

        let vector_a = vec![1.0f32; 1024];
        let vector_b = vec![0.0f32; 1024];
        store.upsert_statement_embedding(&key1.to_csv_key(), &vector_a).unwrap();
        store.upsert_statement_embedding(&key2.to_csv_key(), &vector_b).unwrap();

        let results = store.vector_search_statements(&vector_a, 10).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].0.contains("Alice"));
    }

    #[test]
    fn vector_search_excludes_null_embeddings() {
        let (_dir, store) = setup();
        store.insert_knowledge("with_vec", &Content::new("data")).unwrap();
        store.insert_knowledge("no_vec", &Content::new("data")).unwrap();

        let vector = vec![0.5f32; 1024];
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

        let vector = vec![0.5f32; 1024];
        store.upsert_knowledge_embedding("has_vec", &vector).unwrap();

        let missing = store.knowledge_without_embeddings().unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].0, "no_vec");
    }

    #[test]
    fn clear_embedding() {
        let (_dir, store) = setup();
        store.insert_knowledge("k1", &Content::new("data")).unwrap();
        let vector = vec![0.5f32; 1024];
        store.upsert_knowledge_embedding("k1", &vector).unwrap();
        assert_eq!(store.knowledge_with_embeddings().unwrap().len(), 1);

        store.clear_knowledge_embedding("k1").unwrap();
        assert_eq!(store.knowledge_with_embeddings().unwrap().len(), 0);
    }

    // --- K-hop graph traversal tests ---

    #[test]
    fn khop_1hop_specific_predicate() {
        let (_dir, store) = setup();
        // Alice --knows--> Bob --knows--> Carol
        store.insert_statement(
            &StatementKey::new("Alice", "knows", "Bob"),
            &Content::new("a->b"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("Bob", "knows", "Carol"),
            &Content::new("b->c"), None, None,
        ).unwrap();

        let results = store.query_khop("Alice", Some("knows"), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key.object, "Bob");
    }

    #[test]
    fn khop_2hop_specific_predicate() {
        let (_dir, store) = setup();
        // Alice --knows--> Bob --knows--> Carol
        //               Bob --works_with--> Dave (different predicate)
        store.insert_statement(
            &StatementKey::new("Alice", "knows", "Bob"),
            &Content::new("a->b"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("Bob", "knows", "Carol"),
            &Content::new("b->c"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("Bob", "works_with", "Dave"),
            &Content::new("b->d"), None, None,
        ).unwrap();

        // 2-hop with "knows" only follows the knows chain
        let results = store.query_khop("Alice", Some("knows"), 2).unwrap();
        assert_eq!(results.len(), 2);
        let objects: Vec<&str> = results.iter().map(|s| s.key.object.as_str()).collect();
        assert!(objects.contains(&"Bob"));
        assert!(objects.contains(&"Carol"));
    }

    #[test]
    fn khop_wildcard_predicate() {
        let (_dir, store) = setup();
        store.insert_statement(
            &StatementKey::new("Alice", "knows", "Bob"),
            &Content::new("a->b"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("Bob", "knows", "Carol"),
            &Content::new("b->c"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("Bob", "works_with", "Dave"),
            &Content::new("b->d"), None, None,
        ).unwrap();

        // Wildcard follows all predicates
        let results = store.query_khop("Alice", None, 2).unwrap();
        assert_eq!(results.len(), 3);
        let objects: Vec<&str> = results.iter().map(|s| s.key.object.as_str()).collect();
        assert!(objects.contains(&"Bob"));
        assert!(objects.contains(&"Carol"));
        assert!(objects.contains(&"Dave"));
    }

    #[test]
    fn khop_cycle() {
        let (_dir, store) = setup();
        // A --knows--> B --knows--> A (cycle)
        store.insert_statement(
            &StatementKey::new("A", "knows", "B"),
            &Content::new("a->b"), None, None,
        ).unwrap();
        store.insert_statement(
            &StatementKey::new("B", "knows", "A"),
            &Content::new("b->a"), None, None,
        ).unwrap();

        let results = store.query_khop("A", Some("knows"), 5).unwrap();
        // DISTINCT ON (triple) ensures exactly 2 unique triples
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn khop_no_results() {
        let (_dir, store) = setup();
        let results = store.query_khop("NonExistent", Some("knows"), 3).unwrap();
        assert!(results.is_empty());
    }
}

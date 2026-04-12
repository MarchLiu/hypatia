use std::collections::HashMap;
use std::path::Path;

use crate::embedding::{Embedder, EmbeddingConfig};
use crate::error::{HypatiaError, Result};
use crate::model::{QueryResult, QueryTarget, SearchOpts, ShelfConfig, ShelfId};
use crate::storage::{DuckDbStore, SqliteStore, Storage};

pub struct OpenShelf {
    pub id: ShelfId,
    pub config: ShelfConfig,
    pub duckdb: DuckDbStore,
    pub sqlite: SqliteStore,
    pub embedder: Embedder,
}

impl Storage for OpenShelf {
    fn execute_query(
        &self,
        target: QueryTarget,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<QueryResult> {
        let rows = match target {
            QueryTarget::Knowledge => {
                let knowledge = self.duckdb.query_knowledge(sql, params)?;
                knowledge.into_iter().map(|k| knowledge_to_row(&k)).collect()
            }
            QueryTarget::Statement => {
                let statements = self.duckdb.query_statements(sql, params)?;
                statements.into_iter().map(|s| statement_to_row(&s)).collect()
            }
        };
        Ok(QueryResult::new(rows))
    }

    fn execute_search(&self, query: &str, opts: &SearchOpts) -> Result<QueryResult> {
        let results = self.sqlite.search(query, opts)?;
        let rows = results
            .into_iter()
            .map(|r| {
                let mut map = serde_json::Map::new();
                map.insert("id".to_string(), serde_json::Value::Number(r.id.into()));
                map.insert("catalog".to_string(), serde_json::Value::String(r.catalog));
                map.insert("key".to_string(), serde_json::Value::String(r.key));
                map.insert("content".to_string(), serde_json::Value::String(r.content));
                map.insert(
                    "rank".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(r.rank)
                            .unwrap_or(serde_json::Number::from(0)),
                    ),
                );
                map
            })
            .collect();
        Ok(QueryResult::new(rows))
    }

    fn execute_similar(
        &self,
        query_text: &str,
        opts: &SearchOpts,
        target: QueryTarget,
    ) -> Result<QueryResult> {
        let query_vector = self.embedder.embed(query_text)?;

        let rows = match target {
            QueryTarget::Knowledge => {
                let results = self.duckdb.vector_search_knowledge(&query_vector, opts.limit)?;
                results.into_iter().map(|(name, content, distance)| {
                    let mut map = serde_json::Map::new();
                    map.insert("name".to_string(), serde_json::Value::String(name));
                    map.insert("content".to_string(), serde_json::Value::String(content));
                    map.insert(
                        "distance".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(distance)
                                .unwrap_or(serde_json::Number::from(0)),
                        ),
                    );
                    map
                }).collect()
            }
            QueryTarget::Statement => {
                let results = self.duckdb.vector_search_statements(&query_vector, opts.limit)?;
                results.into_iter().map(|(triple, content, distance)| {
                    let mut map = serde_json::Map::new();
                    map.insert("triple".to_string(), serde_json::Value::String(triple));
                    map.insert("content".to_string(), serde_json::Value::String(content));
                    map.insert(
                        "distance".to_string(),
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(distance)
                                .unwrap_or(serde_json::Number::from(0)),
                        ),
                    );
                    map
                }).collect()
            }
        };
        Ok(QueryResult::new(rows))
    }

    fn execute_khop(
        &self,
        subject: &str,
        predicate: Option<&str>,
        depth: i64,
    ) -> Result<QueryResult> {
        let statements = self.duckdb.query_khop(subject, predicate, depth)?;
        let rows = statements.into_iter().map(|s| statement_to_row(&s)).collect();
        Ok(QueryResult::new(rows))
    }
}

fn knowledge_to_row(k: &crate::model::Knowledge) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("name".to_string(), serde_json::Value::String(k.name.clone()));
    map.insert(
        "content".to_string(),
        serde_json::to_value(&k.content).unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "created_at".to_string(),
        serde_json::Value::String(k.created_at.to_string()),
    );
    map
}

fn statement_to_row(s: &crate::model::Statement) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert("triple".to_string(), serde_json::Value::String(s.key.to_csv_key()));
    map.insert("subject".to_string(), serde_json::Value::String(s.key.subject.clone()));
    map.insert("predicate".to_string(), serde_json::Value::String(s.key.predicate.clone()));
    map.insert("object".to_string(), serde_json::Value::String(s.key.object.clone()));
    map.insert(
        "content".to_string(),
        serde_json::to_value(&s.content).unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "created_at".to_string(),
        serde_json::Value::String(s.created_at.to_string()),
    );
    if let Some(ts) = s.tr_start {
        map.insert("tr_start".to_string(), serde_json::Value::String(ts.to_string()));
    }
    if let Some(te) = s.tr_end {
        map.insert("tr_end".to_string(), serde_json::Value::String(te.to_string()));
    }
    map
}

pub struct ShelfManager {
    shelves: HashMap<String, OpenShelf>,
}

impl ShelfManager {
    pub fn new() -> Self {
        Self {
            shelves: HashMap::new(),
        }
    }

    pub fn connect(&mut self, path: &Path, name: Option<&str>) -> Result<String> {
        // Ensure directory exists
        std::fs::create_dir_all(path)?;

        let config = ShelfConfig::from_path(path, name);
        let shelf_name = config.id.name.clone();

        // Check if already connected
        if self.shelves.contains_key(&shelf_name) {
            return Err(HypatiaError::Shelf(format!(
                "shelf '{}' is already connected",
                shelf_name
            )));
        }

        let duckdb = DuckDbStore::open(&config.duckdb_path)?;
        let sqlite = SqliteStore::open(&config.sqlite_path)?;

        let embedder_config = EmbeddingConfig {
            model_path: config.model_path.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            dimensions: 768,
            max_seq_length: 8192,
        };
        let embedder = Embedder::new(embedder_config);

        let shelf = OpenShelf {
            id: config.id.clone(),
            config,
            duckdb,
            sqlite,
            embedder,
        };

        self.shelves.insert(shelf_name.clone(), shelf);
        Ok(shelf_name)
    }

    pub fn disconnect(&mut self, name: &str) -> Result<()> {
        if self.shelves.remove(name).is_none() {
            return Err(HypatiaError::Shelf(format!(
                "shelf '{name}' is not connected"
            )));
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&OpenShelf> {
        self.shelves.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut OpenShelf> {
        self.shelves.get_mut(name)
    }

    pub fn list(&self) -> Vec<&ShelfId> {
        self.shelves.values().map(|s| &s.id).collect()
    }

    pub fn export(&self, name: &str, dest: &Path) -> Result<()> {
        let shelf = self
            .shelves
            .get(name)
            .ok_or_else(|| HypatiaError::Shelf(format!("shelf '{name}' is not connected")))?;

        std::fs::create_dir_all(dest)?;

        // Copy DuckDB file
        std::fs::copy(&shelf.config.duckdb_path, dest.join("data.duckdb"))?;
        // Copy SQLite file
        std::fs::copy(&shelf.config.sqlite_path, dest.join("index.sqlite"))?;

        Ok(())
    }

    pub fn ensure_default(&mut self) -> Result<String> {
        let default_path = dirs_home().join(".hypatia").join("default");
        if self.shelves.contains_key("default") {
            return Ok("default".to_string());
        }
        self.connect(&default_path, Some("default"))
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Content;
    use tempfile::TempDir;

    #[test]
    fn connect_and_list() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new();
        let name = mgr.connect(dir.path(), Some("test-shelf")).unwrap();
        assert_eq!(name, "test-shelf");

        let shelves = mgr.list();
        assert_eq!(shelves.len(), 1);
        assert_eq!(shelves[0].name, "test-shelf");
    }

    #[test]
    fn connect_duplicate_fails() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new();
        mgr.connect(dir.path(), Some("dup")).unwrap();
        assert!(mgr.connect(dir.path(), Some("dup")).is_err());
    }

    #[test]
    fn disconnect() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new();
        mgr.connect(dir.path(), Some("tmp")).unwrap();
        mgr.disconnect("tmp").unwrap();
        assert!(mgr.list().is_empty());
    }

    #[test]
    fn disconnect_nonexistent_fails() {
        let mut mgr = ShelfManager::new();
        assert!(mgr.disconnect("nonexistent").is_err());
    }

    #[test]
    fn get_shelf() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new();
        mgr.connect(dir.path(), Some("my-shelf")).unwrap();
        assert!(mgr.get("my-shelf").is_some());
        assert!(mgr.get("other").is_none());
    }

    #[test]
    fn export_shelf() {
        let dir = TempDir::new().unwrap();
        let dest = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new();
        mgr.connect(dir.path(), Some("export-test")).unwrap();

        // Add some data
        let shelf = mgr.get("export-test").unwrap();
        shelf
            .duckdb
            .insert_knowledge("test", &Content::new("data"))
            .unwrap();

        mgr.export("export-test", dest.path()).unwrap();
        assert!(dest.path().join("data.duckdb").exists());
        assert!(dest.path().join("index.sqlite").exists());
    }

    #[test]
    fn ensure_default_creates_directory() {
        let tmp = TempDir::new().unwrap();
        let default_path = tmp.path().join(".hypatia").join("default");

        let mut mgr = ShelfManager::new();
        // Override HOME for test
        // Since ensure_default uses dirs_home(), we can't easily test this
        // without env var manipulation, so just verify connect works
        mgr.connect(&default_path, Some("default")).unwrap();
        assert!(mgr.get("default").is_some());
    }
}

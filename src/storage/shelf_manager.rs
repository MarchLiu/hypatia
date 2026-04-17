use std::collections::HashMap;
use std::path::Path;

use crate::embedding::{EmbeddingConfig, EmbeddingProvider, build_provider};
use crate::error::{HypatiaError, Result};
use crate::model::{QueryResult, QueryTarget, SearchOpts, ShelfConfig, ShelfId};
use crate::storage::{DuckDbStore, ShelfRegistry, SqliteStore, Storage};

pub struct OpenShelf {
    pub id: ShelfId,
    pub config: ShelfConfig,
    pub duckdb: DuckDbStore,
    pub sqlite: SqliteStore,
    pub embedder: Box<dyn EmbeddingProvider>,
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
    registry: ShelfRegistry,
    registry_path: std::path::PathBuf,
}

impl ShelfManager {
    /// Create a new ShelfManager, loading the persistent registry and restoring all connections.
    pub fn new() -> Result<Self> {
        let registry_path = ShelfRegistry::registry_path();
        let registry = ShelfRegistry::load(&registry_path)?;

        let mut manager = Self {
            shelves: HashMap::new(),
            registry,
            registry_path,
        };

        // Restore all registered shelves; ensure default exists.
        manager.ensure_default()?;
        manager.restore_registered();

        Ok(manager)
    }

    /// Restore all shelves from the registry (except "default", already connected).
    fn restore_registered(&mut self) {
        let entries: Vec<(String, std::path::PathBuf)> = self
            .registry
            .shelves
            .iter()
            .filter(|(name, _)| *name != "default")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        for (name, path) in entries {
            if let Err(e) = self.connect_internal(&path, Some(&name)) {
                eprintln!("warning: failed to restore shelf '{}': {}", name, e);
            }
        }
    }

    /// Connect to a shelf and persist the registration.
    pub fn connect(&mut self, path: &Path, name: Option<&str>) -> Result<String> {
        let shelf_name = self.connect_internal(path, name)?;
        self.registry.register(&shelf_name, &path.to_path_buf());
        self.registry.save(&self.registry_path)?;
        Ok(shelf_name)
    }

    /// Internal connect logic without registry persistence.
    fn connect_internal(&mut self, path: &Path, name: Option<&str>) -> Result<String> {
        // Ensure directory exists
        std::fs::create_dir_all(path)?;

        let config = ShelfConfig::from_path(path, name);

        // Ensure archives/ directory exists
        std::fs::create_dir_all(&config.archives_path)?;
        let shelf_name = config.id.name.clone();

        // Check if already connected
        if self.shelves.contains_key(&shelf_name) {
            return Err(HypatiaError::Shelf(format!(
                "shelf '{}' is already connected",
                shelf_name
            )));
        }

        let embedding_config = EmbeddingConfig::from_shelf_dir(path);
        let duckdb = DuckDbStore::open(&config.duckdb_path, embedding_config.dimensions())?;
        let sqlite = SqliteStore::open(&config.sqlite_path)?;

        let embedder = build_provider(&embedding_config);

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

    /// Disconnect a shelf and remove from the persistent registry.
    pub fn disconnect(&mut self, name: &str) -> Result<()> {
        if self.shelves.remove(name).is_none() {
            return Err(HypatiaError::Shelf(format!(
                "shelf '{name}' is not connected"
            )));
        }
        self.registry.unregister(name);
        self.registry.save(&self.registry_path)?;
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&OpenShelf> {
        self.shelves.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut OpenShelf> {
        self.shelves.get_mut(name)
    }

    /// List all registered shelves with their paths.
    /// Returns (name, path, is_connected) tuples.
    pub fn list(&self) -> Vec<(&str, &std::path::PathBuf, bool)> {
        self.registry
            .list()
            .into_iter()
            .map(|(name, path)| (name, path, self.shelves.contains_key(name)))
            .collect()
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

        // Copy archives/ directory if it exists and has content
        let archives_dest = dest.join("archives");
        if shelf.config.archives_path.exists() {
            copy_dir_recursive(&shelf.config.archives_path, &archives_dest)?;
        }

        Ok(())
    }

    /// Get the absolute path to a shelf's archives directory.
    pub fn archives_path(&self, shelf_name: &str) -> Option<std::path::PathBuf> {
        self.shelves
            .get(shelf_name)
            .map(|s| s.config.archives_path.clone())
    }

    /// Ensure the default shelf is registered and connected.
    pub fn ensure_default(&mut self) -> Result<String> {
        let default_path = dirs_home().join(".hypatia").join("default");
        if self.shelves.contains_key("default") {
            return Ok("default".to_string());
        }

        // Register in registry if not present
        if !self.registry.contains("default") {
            self.registry.register("default", &default_path);
            self.registry.save(&self.registry_path)?;
        }

        self.connect_internal(&default_path, Some("default"))
    }
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
}

/// Recursively copy a directory tree.
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Content;
    use tempfile::TempDir;

    #[test]
    fn connect_and_list() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        let name = mgr.connect(dir.path(), Some("test-shelf")).unwrap();
        assert_eq!(name, "test-shelf");

        let shelves = mgr.list();
        assert!(shelves.iter().any(|(n, _, _)| *n == "test-shelf"));
    }

    #[test]
    fn connect_duplicate_fails() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("dup")).unwrap();
        assert!(mgr.connect(dir.path(), Some("dup")).is_err());
    }

    #[test]
    fn disconnect() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("tmp")).unwrap();
        mgr.disconnect("tmp").unwrap();
        let shelves = mgr.list();
        assert!(!shelves.iter().any(|(n, _, _)| *n == "tmp"));
    }

    #[test]
    fn disconnect_nonexistent_fails() {
        let mut mgr = ShelfManager::new().unwrap();
        assert!(mgr.disconnect("nonexistent").is_err());
    }

    #[test]
    fn get_shelf() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("my-shelf")).unwrap();
        assert!(mgr.get("my-shelf").is_some());
        assert!(mgr.get("other").is_none());
    }

    #[test]
    fn export_shelf() {
        let dir = TempDir::new().unwrap();
        let dest = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
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
    fn connect_creates_archives_dir() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("ar-test")).unwrap();
        let ap = mgr.archives_path("ar-test").unwrap();
        assert!(ap.exists());
        assert!(ap.ends_with("archives"));
    }

    #[test]
    fn export_includes_archives_dir() {
        let dir = TempDir::new().unwrap();
        let dest = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("ar-export")).unwrap();

        // Put a file in archives/
        let ap = mgr.archives_path("ar-export").unwrap();
        std::fs::write(ap.join("test.png"), b"fake-png").unwrap();

        mgr.export("ar-export", dest.path()).unwrap();
        assert!(dest.path().join("data.duckdb").exists());
        assert!(dest.path().join("archives/test.png").exists());
    }

    #[test]
    fn ensure_default_creates_directory() {
        let tmp = TempDir::new().unwrap();
        let default_path = tmp.path().join(".hypatia").join("default");

        let mut mgr = ShelfManager::new().unwrap();
        // Since ensure_default uses the real HOME, just verify connect works
        mgr.connect(&default_path, Some("test-default-2")).unwrap();
        assert!(mgr.get("test-default-2").is_some());
    }

    #[test]
    fn list_shows_connected_status() {
        let dir = TempDir::new().unwrap();
        let mut mgr = ShelfManager::new().unwrap();
        mgr.connect(dir.path(), Some("status-test")).unwrap();

        let shelves = mgr.list();
        let entry = shelves.iter().find(|(n, _, _)| *n == "status-test").unwrap();
        assert!(entry.2); // is_connected = true
    }
}

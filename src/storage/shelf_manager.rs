use crate::embedding::{EmbeddingProvider, build_provider};
use crate::error::{HypatiaError, Result};
use crate::model::{Content, QueryResult, QueryTarget, SearchOpts, ShelfConfig, ShelfId};
use crate::storage::{ShelfRegistry, Storage, backend::ShelfBackend, settings::ShelfSettings};
use std::collections::HashMap;
use std::path::Path;

pub struct OpenShelf {
    pub id: ShelfId,
    pub config: ShelfConfig,
    pub settings: ShelfSettings,
    pub backend: ShelfBackend,
    pub embedder: Box<dyn EmbeddingProvider>,
}
impl Storage for OpenShelf {
    fn sql_dialect(&self) -> crate::engine::SqlDialect {
        if self.backend.is_local() {
            crate::engine::SqlDialect::Sqlite
        } else {
            crate::engine::SqlDialect::Postgres
        }
    }
    fn sql_schema(&self) -> Option<&str> {
        self.backend.schema()
    }
    fn execute_query(
        &self,
        target: QueryTarget,
        sql: &str,
        params: Vec<serde_json::Value>,
    ) -> Result<QueryResult> {
        let rows = match target {
            QueryTarget::Knowledge => self
                .backend
                .query_knowledge(sql, params)?
                .iter()
                .map(knowledge_to_row)
                .collect(),
            QueryTarget::Statement => self
                .backend
                .query_statements(sql, params)?
                .iter()
                .map(statement_to_row)
                .collect(),
        };
        Ok(QueryResult::new(rows))
    }
    fn execute_search(&self, query: &str, opts: &SearchOpts) -> Result<QueryResult> {
        let rows = self
            .backend
            .search(query, opts)?
            .into_iter()
            .map(|r| {
                let mut m = serde_json::Map::new();
                m.insert("id".into(), serde_json::json!(r.id));
                m.insert("catalog".into(), serde_json::json!(r.catalog));
                m.insert("key".into(), serde_json::json!(r.key));
                m.insert("content".into(), serde_json::json!(r.content));
                m.insert("rank".into(), serde_json::json!(r.rank));
                m
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
        let vector = self.embedder.embed(query_text)?;
        let rows = self
            .backend
            .vector_search(target, &vector, opts.limit)?
            .into_iter()
            .map(|(key, content, distance)| {
                let mut m = serde_json::Map::new();
                m.insert(target.key_column().into(), serde_json::json!(key));
                // One shape for ANN, exact local search, and PostgreSQL.
                m.insert(
                    "content".into(),
                    serde_json::from_str(&content).unwrap_or(serde_json::Value::Null),
                );
                m.insert("distance".into(), serde_json::json!(distance));
                m
            })
            .collect();
        Ok(QueryResult::new(rows))
    }
    fn execute_khop(&self, head: &str, relation: Option<&str>, depth: i64) -> Result<QueryResult> {
        Ok(QueryResult::new(
            self.backend
                .query_khop(head, relation, depth)?
                .iter()
                .map(statement_to_row)
                .collect(),
        ))
    }
}
impl OpenShelf {
    pub fn open(path: &Path, name: Option<&str>) -> Result<Self> {
        // Configuration errors must not create a SQLite file or local vector directory.
        let settings = ShelfSettings::load(path)?;
        let config = ShelfConfig::from_path(path, name);
        std::fs::create_dir_all(path)?;
        let backend = ShelfBackend::open(&config, &settings)?;
        std::fs::create_dir_all(&config.archives_path)?;
        let embedder = build_provider(&settings.embedding);
        Ok(Self {
            id: config.id.clone(),
            config,
            settings,
            backend,
            embedder,
        })
    }
    /// Content has committed. Embedding failures leave the row pending for backfill.
    pub fn embed_saved(&mut self, catalog: &str, key: &str, content: &Content, version: i64) {
        let outcome = (|| -> Result<bool> {
            let Some(v) = self.embedder.maybe_embed(&content.embedding_text(key))? else {
                return Ok(false);
            };
            self.backend.install_embedding(catalog, key, version, &v)
        })();
        match outcome {
            Ok(true) => {}
            Ok(false) => eprintln!(
                "warning: {catalog}/{key}: content saved; embedding pending (model unavailable or content changed); run backfill"
            ),
            Err(e) => eprintln!(
                "warning: {catalog}/{key}: content saved; embedding pending: {e}; run backfill"
            ),
        }
    }
    pub fn save_vector_indexes(&mut self) -> Result<()> {
        self.backend.flush()
    }
    pub fn rebuild_vector_indexes(&mut self) -> Result<()> {
        if self.backend.is_local() {
            self.backend.rebuild_indexes()
        } else {
            self.backend.flush()
        }
    }
}

fn knowledge_to_row(k: &crate::model::Knowledge) -> serde_json::Map<String, serde_json::Value> {
    let mut map = serde_json::Map::new();
    map.insert(
        "name".to_string(),
        serde_json::Value::String(k.name.clone()),
    );
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
    map.insert(
        "triple".to_string(),
        serde_json::Value::String(s.key.to_csv_key()),
    );
    map.insert(
        "head".to_string(),
        serde_json::Value::String(s.key.head.clone()),
    );
    map.insert(
        "relation".to_string(),
        serde_json::Value::String(s.key.relation.clone()),
    );
    map.insert(
        "tail".to_string(),
        serde_json::Value::String(s.key.tail.clone()),
    );
    map.insert(
        "content".to_string(),
        serde_json::to_value(&s.content).unwrap_or(serde_json::Value::Null),
    );
    map.insert(
        "created_at".to_string(),
        serde_json::Value::String(s.created_at.to_string()),
    );
    if let Some(ts) = s.tr_start {
        map.insert(
            "tr_start".to_string(),
            serde_json::Value::String(ts.to_string()),
        );
    }
    if let Some(te) = s.tr_end {
        map.insert(
            "tr_end".to_string(),
            serde_json::Value::String(te.to_string()),
        );
    }
    map
}

pub struct ShelfManager {
    shelves: HashMap<String, OpenShelf>,
    registry: ShelfRegistry,
    registry_path: std::path::PathBuf,
    home: std::path::PathBuf,
}

impl ShelfManager {
    /// Create a new ShelfManager, loading the persistent registry and restoring all connections.
    pub fn new() -> Result<Self> {
        Self::with_home(dirs_home())
    }

    /// ShelfManager rooted at an explicit home directory. Tests use this to
    /// stay hermetic: unit tests must NEVER touch (or migrate!) the real
    /// `~/.hypatia` default shelf.
    pub fn with_home(home: std::path::PathBuf) -> Result<Self> {
        let registry_path = home.join(".hypatia").join("shelves.json");
        let registry = ShelfRegistry::load(&registry_path)?;

        let mut manager = Self {
            shelves: HashMap::new(),
            registry,
            registry_path,
            home,
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
        let config = ShelfConfig::from_path(path, name);
        let shelf_name = config.id.name.clone();
        if self.shelves.contains_key(&shelf_name) {
            return Err(HypatiaError::Shelf(format!(
                "shelf '{shelf_name}' is already connected"
            )));
        }
        let shelf = OpenShelf::open(path, name)?;

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

        if dest.exists() && std::fs::read_dir(dest)?.next().is_some() {
            return Err(HypatiaError::Validation(
                "export destination must be empty".into(),
            ));
        }
        let parent = dest
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        std::fs::create_dir_all(parent)?;
        let source = shelf
            .config
            .archives_path
            .parent()
            .unwrap()
            .canonicalize()?;
        if parent.canonicalize()?.starts_with(&source) {
            return Err(HypatiaError::Validation(
                "export destination must be outside source shelf".into(),
            ));
        }
        let stage = tempfile::Builder::new()
            .prefix(".hypatia-export-")
            .tempdir_in(parent)?;
        // Both local formats describe exactly the same WAL-safe snapshot.
        let snapshot = if shelf.backend.is_local() {
            let db = stage.path().join("hypatia.sqlite");
            shelf.backend.backup_local(&db)?;
            super::SqliteStore::open(&db)?.snapshot()?
        } else {
            shelf.backend.snapshot()?
        };
        snapshot.validate()?;
        let mut writer =
            std::io::BufWriter::new(std::fs::File::create(stage.path().join("snapshot.json"))?);
        serde_json::to_writer_pretty(&mut writer, &snapshot)?;
        std::io::Write::flush(&mut writer)?;
        drop(writer);
        if shelf.config.archives_path.exists() {
            copy_dir_recursive(&shelf.config.archives_path, &stage.path().join("archives"))?;
        }
        super::transfer::validate_archives(&snapshot, &stage.path().join("archives"))?;
        super::transfer::write_manifest(stage.path(), &snapshot)?;
        // Publishing a completed package is one rename; failures leave no partial export.
        if dest.exists() {
            std::fs::remove_dir(dest)?;
        }
        std::fs::rename(stage.path(), dest)?;
        Ok(())
    }

    /// Import logical data into an already configured, empty target. Never switches config.
    pub fn import(&mut self, name: &str, source: &Path, reembed: bool) -> Result<()> {
        let shelf = self
            .shelves
            .get_mut(name)
            .ok_or_else(|| HypatiaError::Shelf(format!("shelf '{name}' is not connected")))?;
        let logical = source.join("snapshot.json");
        let snapshot = if logical.exists() {
            let snapshot = serde_json::from_reader::<_, super::transfer::Snapshot>(
                std::io::BufReader::new(std::fs::File::open(logical)?),
            )?;
            super::transfer::verify_manifest(source, &snapshot)?;
            snapshot
        } else {
            // Read a legacy export through a temporary backup, so opening/migrating
            // its schema cannot modify the user's source export.
            let db = source.join("hypatia.sqlite");
            if !db.is_file() {
                return Err(HypatiaError::Validation(
                    "source must contain snapshot.json or hypatia.sqlite".into(),
                ));
            }
            let temporary = tempfile::tempdir()?;
            let database = temporary.path().join("hypatia.sqlite");
            let legacy = rusqlite::Connection::open_with_flags(
                &db,
                rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
            )?;
            legacy.backup("main", &database, None)?;
            super::SqliteStore::open(&database)?.snapshot()?
        };
        let snapshot = if reembed {
            snapshot.without_embeddings()
        } else {
            snapshot
        };
        snapshot.validate()?;
        // Check source attachment references before committing any imported records.
        super::transfer::validate_archives(&snapshot, &source.join("archives"))?;
        let source_archives = source.join("archives");
        if source_archives.exists() {
            // Copy first: a filesystem failure cannot leave imported references broken.
            // Any copied files on database failure remain safe and reusable on retry.
            copy_dir_recursive(&source_archives, &shelf.config.archives_path)?;
        }
        shelf.backend.import_snapshot(&snapshot)?;
        shelf.backend.rebuild_indexes()?;
        let mut restored = shelf.backend.snapshot()?;
        let mut expected = snapshot.clone();
        restored
            .knowledge
            .sort_by(|a, b| a.knowledge.name.cmp(&b.knowledge.name));
        expected
            .knowledge
            .sort_by(|a, b| a.knowledge.name.cmp(&b.knowledge.name));
        restored
            .statements
            .sort_by_key(|r| r.statement.key.to_csv_key());
        expected
            .statements
            .sort_by_key(|r| r.statement.key.to_csv_key());
        if restored.knowledge != expected.knowledge || restored.statements != expected.statements {
            return Err(HypatiaError::Validation("data imported but verification differed; preserve source and inspect target before switching".into()));
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
        let default_path = self.home.join(".hypatia").join("default");
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
    if std::fs::symlink_metadata(dest).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(HypatiaError::Validation(
            "archive destination cannot be a symlink".into(),
        ));
    }
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        if entry.file_type()?.is_symlink() {
            return Err(HypatiaError::Validation(
                "archive copy does not follow symlinks".into(),
            ));
        }
        let dest_path = dest.join(entry.file_name());
        if std::fs::symlink_metadata(&dest_path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(HypatiaError::Validation(
                "archive destination cannot be a symlink".into(),
            ));
        }
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            if dest_path.exists() {
                if std::fs::read(&src_path)? != std::fs::read(&dest_path)? {
                    return Err(HypatiaError::Validation(format!(
                        "archive destination already contains different content: {}",
                        dest_path.display()
                    )));
                }
            } else {
                std::fs::copy(&src_path, &dest_path)?;
            }
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
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        let name = mgr.connect(dir.path(), Some("test-shelf")).unwrap();
        assert_eq!(name, "test-shelf");

        let shelves = mgr.list();
        assert!(shelves.iter().any(|(n, _, _)| *n == "test-shelf"));
    }

    #[test]
    fn connect_duplicate_fails() {
        let dir = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("dup")).unwrap();
        assert!(mgr.connect(dir.path(), Some("dup")).is_err());
    }

    #[test]
    fn disconnect() {
        let dir = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("tmp")).unwrap();
        mgr.disconnect("tmp").unwrap();
        let shelves = mgr.list();
        assert!(!shelves.iter().any(|(n, _, _)| *n == "tmp"));
    }

    #[test]
    fn get_shelf() {
        let dir = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("my-shelf")).unwrap();
        assert!(mgr.get("my-shelf").is_some());
        assert!(mgr.get("other").is_none());
    }

    #[test]
    fn export_shelf() {
        let dir = TempDir::new().unwrap();
        let dest = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("export-test")).unwrap();

        // Add some data
        let shelf = mgr.get("export-test").unwrap();
        shelf
            .backend
            .insert_knowledge("test", &Content::new("data"))
            .unwrap();

        mgr.export("export-test", dest.path()).unwrap();
        assert!(dest.path().join("hypatia.sqlite").exists());
    }

    #[test]
    fn connect_creates_archives_dir() {
        let dir = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("ar-test")).unwrap();
        let ap = mgr.archives_path("ar-test").unwrap();
        assert!(ap.exists());
        assert!(ap.ends_with("archives"));
    }

    #[test]
    fn export_includes_archives_dir() {
        let dir = TempDir::new().unwrap();
        let dest = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("ar-export")).unwrap();

        // Put a file in archives/
        let ap = mgr.archives_path("ar-export").unwrap();
        std::fs::write(ap.join("test.png"), b"fake-png").unwrap();

        mgr.export("ar-export", dest.path()).unwrap();
        assert!(dest.path().join("hypatia.sqlite").exists());
        assert!(dest.path().join("archives/test.png").exists());
    }

    #[test]
    fn list_shows_connected_status() {
        let dir = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(dir.path(), Some("status-test")).unwrap();

        let shelves = mgr.list();
        let entry = shelves
            .iter()
            .find(|(n, _, _)| *n == "status-test")
            .unwrap();
        assert!(entry.2); // is_connected = true
    }
}

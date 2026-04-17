use chrono::NaiveDateTime;
use std::path::Path;

use crate::engine::Evaluator;
use crate::error::Result;
use crate::model::*;
use crate::service::{KnowledgeService, StatementService};
use crate::storage::{ShelfManager, Storage};

/// Statistics returned by backfill operation.
pub struct BackfillStats {
    pub created: usize,
    pub skipped: usize,
    pub errors: usize,
}

pub struct Lab {
    shelf_manager: ShelfManager,
}

impl Lab {
    pub fn new() -> Result<Self> {
        let shelf_manager = ShelfManager::new()?;
        Ok(Self { shelf_manager })
    }

    // --- Shelf operations ---

    pub fn connect_shelf(&mut self, path: &Path, name: Option<&str>) -> Result<String> {
        self.shelf_manager.connect(path, name)
    }

    pub fn disconnect_shelf(&mut self, name: &str) -> Result<()> {
        self.shelf_manager.disconnect(name)
    }

    pub fn list_shelves(&self) -> Vec<(&str, &std::path::PathBuf, bool)> {
        self.shelf_manager.list()
    }

    pub fn export_shelf(&self, name: &str, dest: &Path) -> Result<()> {
        self.shelf_manager.export(name, dest)
    }

    // --- JSE Query ---

    pub fn query(&mut self, shelf_name: &str, jse: &serde_json::Value) -> Result<QueryResult> {
        let shelf = self.shelf_manager.get(shelf_name).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf_name}' is not connected"))
        })?;
        Evaluator::execute(jse, shelf)
    }

    // --- Knowledge CRUD ---

    pub fn create_knowledge(&mut self, shelf: &str, name: &str, content: Content) -> Result<Knowledge> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = KnowledgeService::new(shelf_ref);
        svc.create(name, content)
    }

    pub fn get_knowledge(&self, shelf: &str, name: &str) -> Result<Option<Knowledge>> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.duckdb.get_knowledge(name)
    }

    pub fn update_knowledge(&mut self, shelf: &str, name: &str, content: Content) -> Result<Knowledge> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = KnowledgeService::new(shelf_ref);
        svc.update(name, content)
    }

    pub fn delete_knowledge(&mut self, shelf: &str, name: &str) -> Result<()> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = KnowledgeService::new(shelf_ref);
        svc.delete(name)
    }

    // --- Statement CRUD ---

    pub fn create_statement(
        &mut self,
        shelf: &str,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Statement> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = StatementService::new(shelf_ref);
        svc.create(key, content, tr_start, tr_end)
    }

    pub fn get_statement(&self, shelf: &str, key: &StatementKey) -> Result<Option<Statement>> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.duckdb.get_statement(key)
    }

    pub fn delete_statement(&mut self, shelf: &str, key: &StatementKey) -> Result<()> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = StatementService::new(shelf_ref);
        svc.delete(key)
    }

    // --- Search ---

    pub fn search(&self, shelf: &str, query: &str, opts: SearchOpts) -> Result<QueryResult> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.execute_search(query, &opts)
    }

    // --- Archive files ---

    /// Store a file in the shelf's archives/ directory.
    /// `dest_relative` is the target path relative to archives/ (e.g., "euclid/fig1.png").
    /// Returns the absolute path of the stored file.
    pub fn store_archive(&self, shelf: &str, src: &Path, dest_relative: &str) -> Result<std::path::PathBuf> {
        let archives_dir = self.shelf_manager.archives_path(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let dest = archives_dir.join(dest_relative);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(src, &dest)?;
        Ok(dest)
    }

    /// Get the absolute path for an archive file by its relative path.
    pub fn get_archive_path(&self, shelf: &str, relative_path: &str) -> Option<std::path::PathBuf> {
        let archives_dir = self.shelf_manager.archives_path(shelf)?;
        let full = archives_dir.join(relative_path);
        if full.exists() { Some(full) } else { None }
    }

    /// List all archive files in the shelf's archives/ directory (relative paths).
    pub fn list_archives(&self, shelf: &str) -> Result<Vec<String>> {
        let archives_dir = self.shelf_manager.archives_path(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        if !archives_dir.exists() {
            return Ok(Vec::new());
        }
        let mut results = Vec::new();
        list_archives_recursive(&archives_dir, &archives_dir, &mut results)?;
        results.sort();
        Ok(results)
    }

    // --- Backfill ---

    /// Generate embedding vectors for all entries that don't have one yet.
    /// Idempotent: entries that already have vectors are skipped.
    pub fn backfill_vectors(&mut self, shelf: &str) -> Result<BackfillStats> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;

        if !shelf_ref.embedder.is_available() {
            return Err(crate::error::HypatiaError::ModelUnavailable(
                "no embedding model found; place embedding_model.onnx and tokenizer.json in the shelf directory".to_string(),
            ));
        }

        let mut stats = BackfillStats { created: 0, skipped: 0, errors: 0 };

        // Backfill knowledge entries
        let knowledge_missing = shelf_ref.duckdb.knowledge_without_embeddings()?;
        stats.skipped += shelf_ref.duckdb.knowledge_with_embeddings()?.len();

        for (name, content_json) in knowledge_missing {
            let content = match Content::from_json_str(&content_json) {
                Ok(c) => c,
                Err(_) => { stats.errors += 1; continue; }
            };

            let text = content.embedding_text(&name);
            match shelf_ref.embedder.embed(&text) {
                Ok(vector) => {
                    match shelf_ref.duckdb.upsert_knowledge_embedding(&name, &vector) {
                        Ok(_) => stats.created += 1,
                        Err(_) => stats.errors += 1,
                    }
                }
                Err(_) => stats.errors += 1,
            }
        }

        // Backfill statement entries
        let stmt_missing = shelf_ref.duckdb.statements_without_embeddings()?;

        for (triple, content_json) in stmt_missing {
            let content = match Content::from_json_str(&content_json) {
                Ok(c) => c,
                Err(_) => { stats.errors += 1; continue; }
            };

            let text = content.embedding_text(&triple);
            match shelf_ref.embedder.embed(&text) {
                Ok(vector) => {
                    match shelf_ref.duckdb.upsert_statement_embedding(&triple, &vector) {
                        Ok(_) => stats.created += 1,
                        Err(_) => stats.errors += 1,
                    }
                }
                Err(_) => stats.errors += 1,
            }
        }

        Ok(stats)
    }
}

fn list_archives_recursive(base: &Path, dir: &Path, results: &mut Vec<String>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            list_archives_recursive(base, &path, results)?;
        } else {
            if let Ok(rel) = path.strip_prefix(base) {
                results.push(rel.to_string_lossy().to_string());
            }
        }
    }
    Ok(())
}

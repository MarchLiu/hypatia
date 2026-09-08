use chrono::NaiveDateTime;
use std::path::Path;

use crate::engine::Evaluator;
use crate::error::Result;
use crate::model::*;
use crate::service::{KnowledgeService, StatementService};
use crate::storage::{ShelfManager, Storage};

/// Statistics returned by backfill operation.
#[derive(Debug)]
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

    pub fn import_shelf(&mut self, name: &str, source: &Path, reembed: bool) -> Result<()> {
        self.shelf_manager.import(name, source, reembed)
    }

    // --- JSE Query ---

    pub fn query(&mut self, shelf_name: &str, jse: &serde_json::Value) -> Result<QueryResult> {
        let shelf = self.shelf_manager.get(shelf_name).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf_name}' is not connected"))
        })?;
        Evaluator::execute(jse, shelf)
    }

    // --- Knowledge CRUD ---

    pub fn create_knowledge(
        &mut self,
        shelf: &str,
        name: &str,
        content: Content,
    ) -> Result<Knowledge> {
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
        shelf_ref.backend.get_knowledge(name)
    }

    pub fn update_knowledge(
        &mut self,
        shelf: &str,
        name: &str,
        content: Content,
    ) -> Result<Knowledge> {
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
        shelf_ref.backend.get_statement(key)
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

    // --- Similarity search ---

    pub fn similar(
        &self,
        shelf: &str,
        query: &str,
        target: &str,
        limit: i64,
    ) -> Result<QueryResult> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;

        let opts = SearchOpts {
            catalog: None,
            offset: 0,
            limit,
        };

        match target {
            "knowledge" => shelf_ref.execute_similar(query, &opts, QueryTarget::Knowledge),
            "statement" => shelf_ref.execute_similar(query, &opts, QueryTarget::Statement),
            "both" => {
                let mut knowledge_rows = shelf_ref
                    .execute_similar(query, &opts, QueryTarget::Knowledge)?
                    .rows;
                let mut statement_rows = shelf_ref
                    .execute_similar(query, &opts, QueryTarget::Statement)?
                    .rows;

                for row in &mut knowledge_rows {
                    row.insert(
                        "catalog".to_string(),
                        serde_json::Value::String("knowledge".to_string()),
                    );
                }
                for row in &mut statement_rows {
                    row.insert(
                        "catalog".to_string(),
                        serde_json::Value::String("statement".to_string()),
                    );
                }

                let mut all_rows = knowledge_rows;
                all_rows.extend(statement_rows);
                all_rows.sort_by(|a, b| {
                    let da = a
                        .get("distance")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(f64::MAX);
                    let db = b
                        .get("distance")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(f64::MAX);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
                all_rows.truncate(limit as usize);
                Ok(QueryResult::new(all_rows))
            }
            _ => Err(crate::error::HypatiaError::Validation(format!(
                "invalid target '{target}': must be 'knowledge', 'statement', or 'both'"
            ))),
        }
    }

    // --- Archive files ---

    /// Store a file in the shelf's archives/ directory.
    /// `dest_relative` is the target path relative to archives/ (e.g., "euclid/fig1.png").
    /// Returns the absolute path of the stored file.
    pub fn store_archive(
        &self,
        shelf: &str,
        src: &Path,
        dest_relative: &str,
    ) -> Result<std::path::PathBuf> {
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
        self.backfill_vectors_with_reembed(shelf, false)
    }

    pub fn backfill_vectors_with_reembed(
        &mut self,
        shelf: &str,
        reembed: bool,
    ) -> Result<BackfillStats> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;

        if !shelf_ref.embedder.is_available() {
            return Err(crate::error::HypatiaError::ModelUnavailable(
                "no embedding model found; place embedding_model.onnx and tokenizer.json in the shelf directory".to_string(),
            ));
        }

        if !reembed
            && !shelf_ref.backend.embeddings_identified()?
            && shelf_ref.backend.embedding_row_count("knowledge")?
                + shelf_ref.backend.embedding_row_count("statement")?
                > 0
        {
            return Err(crate::error::HypatiaError::Config(
                "legacy vector identity is unknown; run backfill --reembed explicitly".into(),
            ));
        }
        if reembed {
            if !shelf_ref.settings.embedding.model_identity_trusted {
                return Err(crate::error::HypatiaError::Config(
                    "re-embedding requires an identifiable configured model".into(),
                ));
            }
            let metadata = crate::storage::transfer::EmbeddingMetadata {
                model: shelf_ref.settings.embedding.model_identity().into(),
                dimensions: shelf_ref.settings.embedding.dimensions(),
                metric: "cosine".into(),
            };
            shelf_ref.backend.reset_embeddings(&metadata)?;
        }
        let mut stats = BackfillStats {
            created: 0,
            skipped: 0,
            errors: 0,
        };

        for catalog in ["knowledge", "statement"] {
            stats.skipped += shelf_ref.backend.embedding_row_count(catalog)?;
            let mut after: Option<String> = None;
            loop {
                let page = shelf_ref
                    .backend
                    .missing_embeddings(catalog, after.as_deref(), 128)?;
                if page.is_empty() {
                    break;
                }
                for (key, content, version) in page {
                    after = Some(key.clone());
                    match shelf_ref.embedder.embed(&content.embedding_text(&key)) {
                        Ok(vector) => match shelf_ref
                            .backend
                            .install_embedding(catalog, &key, version, &vector)
                        {
                            Ok(true) => stats.created += 1,
                            Ok(false) => stats.skipped += 1,
                            Err(e) => {
                                eprintln!("backfill {catalog}/{key}: {e}");
                                stats.errors += 1;
                            }
                        },
                        Err(e) => {
                            eprintln!("backfill {catalog}/{key}: {e}");
                            stats.errors += 1;
                        }
                    }
                }
            }
        }
        shelf_ref.rebuild_vector_indexes()?;

        Ok(stats)
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::*;
    use crate::embedding::EmbeddingProvider;
    struct Fixed {
        fail: bool,
    }
    impl EmbeddingProvider for Fixed {
        fn embed(&self, _: &str) -> Result<Vec<f32>> {
            if self.fail {
                Err(crate::error::HypatiaError::Embedding(
                    "test network failure".into(),
                ))
            } else {
                Ok(vec![1., 0., 0.])
            }
        }
        fn dimensions(&self) -> usize {
            3
        }
        fn is_available(&self) -> bool {
            true
        }
    }
    #[test]
    fn saved_content_survives_provider_failure_and_existing_backfill_repairs_it() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("shelf.toml"),
            "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
        )
        .unwrap();
        let mut manager = ShelfManager::with_home(home.path().into()).unwrap();
        manager.connect(dir.path(), Some("test")).unwrap();
        manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: true });
        let mut lab = Lab {
            shelf_manager: manager,
        };
        assert_eq!(
            lab.create_knowledge("test", "key", Content::new("saved"))
                .unwrap()
                .content
                .data,
            "saved"
        );
        let key = StatementKey::new("a", "r", "b");
        lab.create_statement("test", &key, Content::new("edge"), None, None)
            .unwrap();
        assert_eq!(lab.backfill_vectors("test").unwrap().errors, 2);
        lab.shelf_manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: false });
        assert_eq!(lab.backfill_vectors("test").unwrap().created, 2);
        assert_eq!(lab.backfill_vectors("test").unwrap().skipped, 2);
        let legacy = crate::storage::SqliteStore::open(&dir.path().join("hypatia.sqlite")).unwrap();
        legacy
            .conn()
            .execute("DELETE FROM meta WHERE k='embedding_metadata'", [])
            .unwrap();
        assert!(
            lab.backfill_vectors("test")
                .unwrap_err()
                .to_string()
                .contains("--reembed")
        );
        assert_eq!(
            lab.backfill_vectors_with_reembed("test", true)
                .unwrap()
                .created,
            2
        );
        lab.shelf_manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: true });
        lab.update_knowledge("test", "key", Content::new("new saved"))
            .unwrap();
        assert_eq!(
            lab.get_knowledge("test", "key")
                .unwrap()
                .unwrap()
                .content
                .data,
            "new saved"
        );
        assert_eq!(
            lab.shelf_manager
                .get("test")
                .unwrap()
                .backend
                .embedding_row_count("knowledge")
                .unwrap(),
            0
        );
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

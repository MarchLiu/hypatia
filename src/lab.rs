use chrono::NaiveDateTime;
use std::path::Path;

use crate::engine::Evaluator;
use crate::error::Result;
use crate::model::*;
use crate::service::{
    CreatedStatement, KnowledgePatch, KnowledgeService, StatementService, UpdatedKnowledge,
};
use crate::storage::{ShelfManager, Storage};

/// Statistics returned by backfill operation.
#[derive(Debug)]
pub struct BackfillStats {
    pub created: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// What pointing a shelf at a local model did.
#[derive(Debug, PartialEq)]
pub enum Attached {
    /// shelf.toml now names the model, and the shelf was opened again with it. `kept` lists
    /// model settings shelf.toml already had, which may have been meant for another model.
    Configured {
        config: std::path::PathBuf,
        kept: Vec<String>,
    },
    /// The shelf already names the model.
    AlreadyConfigured,
    /// The shelf holds vectors of another model, which switching would strand; unchanged.
    HasVectors { config: std::path::PathBuf },
    /// The shelf embeds through a remote API; unchanged.
    Remote { config: std::path::PathBuf },
}

/// What works on a shelf, as `hypatia init` reports it.
#[derive(Debug)]
pub struct ShelfStatus {
    pub name: String,
    pub path: std::path::PathBuf,
    pub postgres: bool,
    /// What embeds the shelf's entries: a model name, shelf-directory files, or a remote API.
    pub embedder: String,
    /// Why semantic search is off, and how to turn it on; `None` when it is on.
    pub semantic_search_off: Option<String>,
    /// What stops vectors from being written although a model is there, and what to do.
    pub attention: Option<String>,
    pub debt: crate::storage::flush::EmbeddingDebt,
}

pub struct Lab {
    shelf_manager: ShelfManager,
}

impl Lab {
    pub fn new() -> Result<Self> {
        let shelf_manager = ShelfManager::new()?;
        Ok(Self { shelf_manager })
    }

    /// A lab over a given manager, so tests stay out of the real home directory.
    #[cfg(test)]
    pub(crate) fn from_manager(shelf_manager: ShelfManager) -> Self {
        Self { shelf_manager }
    }

    // --- Shelf operations ---

    pub fn connect_shelf(&mut self, path: &Path, name: Option<&str>) -> Result<String> {
        self.shelf_manager.connect(path, name)
    }

    pub fn disconnect_shelf(&mut self, name: &str) -> Result<()> {
        self.shelf_manager.disconnect(name)
    }

    /// Opens a registered shelf again; the error says why it cannot be opened.
    pub fn reopen_shelf(&mut self, name: &str) -> Result<()> {
        self.shelf_manager.reopen(name)
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
        let shelf = self.shelf_manager.get_mut(shelf_name).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf_name}' is not connected"))
        })?;
        if uses_similar(jse) {
            shelf.flush_before_similar();
        }
        Evaluator::execute(jse, &*shelf)
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

    /// Change only the fields a patch names; see [`KnowledgeService::patch`].
    pub fn patch_knowledge(
        &mut self,
        shelf: &str,
        name: &str,
        patch: &KnowledgePatch,
    ) -> Result<UpdatedKnowledge> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        KnowledgeService::new(shelf_ref).patch(name, patch)
    }

    pub fn delete_knowledge(&mut self, shelf: &str, name: &str) -> Result<()> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let mut svc = KnowledgeService::new(shelf_ref);
        svc.delete(name)
    }

    // --- Statement CRUD ---

    /// Idempotent: an existing triple comes back unchanged with `created: false`.
    pub fn create_statement(
        &mut self,
        shelf: &str,
        key: &StatementKey,
        content: Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<CreatedStatement> {
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
        &mut self,
        shelf: &str,
        query: &str,
        target: &str,
        limit: i64,
    ) -> Result<QueryResult> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.flush_before_similar();

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

    /// The shelf's embedding debt: what is pending, since when, and why it may be stuck.
    pub fn embedding_debt(&self, shelf: &str) -> Result<crate::storage::flush::EmbeddingDebt> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.embedding_debt()
    }

    /// When a command on `shelf` begins: pays embedding debt that has waited too long.
    pub fn flush_if_overdue(&mut self, shelf: &str) -> Result<()> {
        let shelf_ref = self.shelf_manager.get_mut(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        shelf_ref.flush_if_overdue()
    }

    /// Points `shelf` at the local model `model` by naming it in the shelf's shelf.toml, and
    /// opens the shelf again so it takes effect. A shelf holding vectors of another model, or
    /// embedding through a remote API, is only reported on. Comments and other settings in
    /// shelf.toml are kept.
    pub fn attach_model(&mut self, shelf: &str, model: &str) -> Result<Attached> {
        let dir = self
            .shelf_manager
            .list()
            .into_iter()
            .find(|(name, _, _)| *name == shelf)
            .map(|(_, path, _)| path.clone())
            .ok_or_else(|| {
                crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not registered"))
            })?;
        let config = dir.join("shelf.toml");
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        if shelf_ref.settings.embedding.provider == crate::embedding::config::ProviderKind::Remote {
            return Ok(Attached::Remote { config });
        }
        let original = match std::fs::read_to_string(&config) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        // Never echo the file: it may hold a PostgreSQL connection string.
        let mut document: toml_edit::DocumentMut = original
            .as_deref()
            .unwrap_or_default()
            .parse()
            .map_err(|e: toml_edit::TomlError| {
                crate::error::HypatiaError::Config(format!("invalid shelf.toml: {}", e.message()))
            })?;
        let current = document
            .get("embedding")
            .and_then(|embedding| embedding.get("model"))
            .and_then(|model| model.as_str());
        if current == Some(model) {
            return Ok(Attached::AlreadyConfigured);
        }
        let backend = &shelf_ref.backend;
        if backend.embedding_row_count("knowledge")? + backend.embedding_row_count("statement")? > 0
        {
            return Ok(Attached::HasVectors { config });
        }
        if document
            .get("embedding")
            .is_some_and(|embedding| !embedding.is_table_like())
        {
            return Err(crate::error::HypatiaError::Config(
                "shelf.toml: embedding must be a table".into(),
            ));
        }
        if document.get("embedding").is_none() {
            // A section of its own, where instructions tell users to add settings.
            document["embedding"] = toml_edit::table();
        }
        document["embedding"]["model"] = toml_edit::value(model);
        if let Some(embedding) = document["embedding"].as_table_like_mut() {
            // `model` takes their place; left behind they would still change the identity.
            embedding.remove("model_path");
            embedding.remove("tokenizer_path");
        }
        let kept: Vec<String> = ["dimensions", "pooling", "max_seq_length"]
            .into_iter()
            .filter(|key| document["embedding"].get(key).is_some())
            .map(String::from)
            .collect();
        write_atomically(&config, &document.to_string())?;
        if let Err(e) = self.shelf_manager.reopen(shelf) {
            // Put the shelf back as it was, and say so if even that fails.
            let restored = match &original {
                Some(text) => write_atomically(&config, text),
                None => std::fs::remove_file(&config).map_err(Into::into),
            }
            .and_then(|()| self.shelf_manager.reopen(shelf));
            return Err(match restored {
                Ok(()) => e,
                Err(restore) => crate::error::HypatiaError::Config(format!(
                    "{e}; restoring {} failed too: {restore}",
                    config.display()
                )),
            });
        }
        Ok(Attached::Configured { config, kept })
    }

    /// What works on `shelf`, for `hypatia init`.
    pub fn shelf_status(&self, shelf: &str) -> Result<ShelfStatus> {
        let shelf_ref = self.shelf_manager.get(shelf).ok_or_else(|| {
            crate::error::HypatiaError::Shelf(format!("shelf '{shelf}' is not connected"))
        })?;
        let embedding = &shelf_ref.settings.embedding;
        let embedder = match embedding.provider {
            crate::embedding::config::ProviderKind::Remote => {
                format!("the remote API model {}", embedding.remote.api_model)
            }
            crate::embedding::config::ProviderKind::Local => {
                embedding.model.clone().unwrap_or_else(|| {
                    format!("the model at {}", embedding.local.model_path.display())
                })
            }
        };
        let debt = shelf_ref.embedding_debt()?;
        // Only reported once semantic search is on; until then the way to turn it on comes first.
        let attention = match shelf_ref.backend.identity_mismatch() {
            // Its message already ends with the command that recovers the shelf.
            Some(mismatch) => Some(mismatch.to_string()),
            None => debt.blocked.as_ref().map(|blocked| blocked.message.clone()),
        };
        Ok(ShelfStatus {
            name: shelf.to_string(),
            path: shelf_ref.id.path.clone(),
            postgres: shelf_ref.settings.storage.backend
                == crate::storage::settings::BackendKind::Pgvector,
            embedder,
            semantic_search_off: shelf_ref.semantic_search_off(),
            attention,
            debt,
        })
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

        // Without a usable model nothing below can run, re-embedding included: say how to
        // set one up first.
        if let Some(off) = shelf_ref.semantic_search_off_error() {
            return Err(off);
        }
        if !reembed && let Some(m) = shelf_ref.backend.identity_mismatch() {
            return Err(m.to_error());
        }
        if !shelf_ref.embedder.is_available() {
            // The model is there but failed to load earlier in this process: embedding says why.
            return Err(shelf_ref.embedder.embed("").err().unwrap_or_else(|| {
                crate::error::HypatiaError::ModelUnavailable(
                    "the embedding model is unavailable".to_string(),
                )
            }));
        }

        if !reembed
            && !shelf_ref.backend.embeddings_identified()?
            && shelf_ref.backend.embedding_row_count("knowledge")?
                + shelf_ref.backend.embedding_row_count("statement")?
                > 0
        {
            return Err(crate::error::HypatiaError::Config(format!(
                "legacy vector identity is unknown; run `hypatia backfill --reembed -s {shelf}` explicitly"
            )));
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
                let Some((last, _, _)) = page.last() else {
                    break;
                };
                after = Some(last.clone());
                // One batch per page: a single request (remote) or a few forward passes (local).
                let texts: Vec<String> = page
                    .iter()
                    .map(|(key, content, _)| content.embedding_text(key))
                    .collect();
                let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
                let mut vectors = shelf_ref.embedder.embed_batch(&texts).into_iter();
                for (key, _, version) in page {
                    // A provider returning too few results must not silently drop rows.
                    let vector = vectors.next().unwrap_or_else(|| {
                        Err(crate::error::HypatiaError::Embedding(
                            "provider returned too few vectors".into(),
                        ))
                    });
                    match vector.and_then(|v| {
                        shelf_ref
                            .backend
                            .install_embedding(catalog, &key, version, &v)
                    }) {
                        Ok(true) => stats.created += 1,
                        Ok(false) => stats.skipped += 1,
                        Err(e) => {
                            eprintln!("backfill {catalog}/{key}: {e}");
                            stats.errors += 1;
                        }
                    }
                }
            }
        }
        shelf_ref.rebuild_vector_indexes()?;
        // Bookkeeping only: the vectors are in, so a failure here must not fail the backfill.
        if let Err(e) = shelf_ref
            .settle_pending()
            .and_then(|_| shelf_ref.record_backfill(stats.created, stats.errors))
        {
            eprintln!("warning: could not update embedding bookkeeping: {e}");
        }

        Ok(stats)
    }
}

/// Replaces a file in one step, so a crash never leaves it half written or empty. Its
/// permissions are kept (shelf.toml may hold a database password), and a symbolic link is
/// followed to the file it names.
fn write_atomically(path: &Path, text: &str) -> Result<()> {
    let target = if std::fs::symlink_metadata(path).is_ok_and(|meta| meta.file_type().is_symlink())
    {
        std::fs::canonicalize(path)?
    } else {
        path.to_path_buf()
    };
    let dir = target.parent().unwrap_or(Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(dir)?;
    if let Ok(meta) = std::fs::metadata(&target) {
        temporary.as_file().set_permissions(meta.permissions())?;
    }
    std::io::Write::write_all(temporary.as_file_mut(), text.as_bytes())?;
    temporary.as_file().sync_all()?;
    temporary.persist(&target).map_err(|e| e.error)?;
    Ok(())
}

/// Whether a JSE expression contains a `$similar` operator anywhere.
pub fn uses_similar(jse: &serde_json::Value) -> bool {
    match jse {
        serde_json::Value::Array(items) => {
            // An operator takes an argument; a lone "$similar" is data.
            (items.len() > 1 && items[0].as_str() == Some("$similar"))
                || items.iter().any(uses_similar)
        }
        serde_json::Value::Object(fields) => fields.values().any(uses_similar),
        _ => false,
    }
}

#[cfg(test)]
mod backfill_tests {
    use super::*;
    use crate::embedding::EmbeddingProvider;
    /// Records every batch size; rejects texts containing "bad".
    struct Batching {
        batches: std::rc::Rc<std::cell::RefCell<Vec<usize>>>,
    }
    impl EmbeddingProvider for Batching {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            if text.contains("bad") {
                Err(crate::error::HypatiaError::Embedding("rejected".into()))
            } else {
                Ok(vec![1., 0., 0.])
            }
        }
        fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>>> {
            self.batches.borrow_mut().push(texts.len());
            texts.iter().map(|t| self.embed(t)).collect()
        }
        fn dimensions(&self) -> usize {
            3
        }
        fn is_available(&self) -> bool {
            true
        }
    }
    #[test]
    fn status_and_similar_say_how_to_turn_semantic_search_on() {
        let home = tempfile::tempdir().unwrap();
        let bare = tempfile::tempdir().unwrap();
        let named = tempfile::tempdir().unwrap();
        std::fs::write(
            named.path().join("shelf.toml"),
            "[embedding]\nmodel = 'org/missing'\ndimensions = 3\n",
        )
        .unwrap();
        let mut manager = ShelfManager::with_home(home.path().into()).unwrap();
        manager.connect(bare.path(), Some("bare")).unwrap();
        manager.connect(named.path(), Some("named")).unwrap();
        let mut lab = Lab {
            shelf_manager: manager,
        };
        lab.create_knowledge("bare", "k", Content::new("x"))
            .unwrap();

        // No model at all: install the default one, or configure a remote API.
        let status = lab.shelf_status("bare").unwrap();
        let off = status.semantic_search_off.clone().unwrap();
        assert!(
            off.contains("hypatia model install BAAI/bge-m3 -s bare"),
            "{off}"
        );
        assert_eq!((status.debt.pending_knowledge, status.postgres), (1, false));
        let err = lab
            .similar("bare", "x", "knowledge", 5)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("semantic search is off") && err.contains("model install"),
            "{err}"
        );

        // A named model that is not installed: install that one.
        let off = lab
            .shelf_status("named")
            .unwrap()
            .semantic_search_off
            .unwrap();
        assert!(
            off.contains("`hypatia model install org/missing -s named`"),
            "{off}"
        );
        assert_eq!(lab.shelf_status("named").unwrap().embedder, "org/missing");

        // Legacy paths that point nowhere: name the files, not the default model.
        let legacy = tempfile::tempdir().unwrap();
        std::fs::write(
            legacy.path().join("shelf.toml"),
            "[embedding]\nmodel_path = '/nonexistent/m.onnx'\ndimensions = 3\n",
        )
        .unwrap();
        lab.connect_shelf(legacy.path(), Some("legacy")).unwrap();
        let off = lab
            .shelf_status("legacy")
            .unwrap()
            .semantic_search_off
            .unwrap();
        assert!(
            off.starts_with("embedding model files not found: /nonexistent/m.onnx"),
            "{off}"
        );

        // A remote API without its key: name the variable.
        let remote = tempfile::tempdir().unwrap();
        std::fs::write(
            remote.path().join("shelf.toml"),
            "[embedding]\nprovider = 'remote'\napi_key_env = 'HYPATIA_TEST_KEY_THAT_IS_NEVER_SET'\ndimensions = 3\n",
        )
        .unwrap();
        lab.connect_shelf(remote.path(), Some("remote")).unwrap();
        let off = lab
            .shelf_status("remote")
            .unwrap()
            .semantic_search_off
            .unwrap();
        assert!(
            off.ends_with("environment variable HYPATIA_TEST_KEY_THAT_IS_NEVER_SET"),
            "{off}"
        );

        // A model that works turns it on.
        lab.shelf_manager.get_mut("named").unwrap().embedder = Box::new(Fixed { fail: false });
        let status = lab.shelf_status("named").unwrap();
        assert_eq!((status.semantic_search_off, status.attention), (None, None));
    }
    #[test]
    fn backfill_embeds_a_page_per_batch_and_isolates_failures() {
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
        for i in 0..130 {
            let data = if i == 7 {
                "bad".to_string()
            } else {
                format!("entry {i}")
            };
            lab.create_knowledge("test", &format!("k{i:03}"), Content::new(&data))
                .unwrap();
        }
        let batches = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        lab.shelf_manager.get_mut("test").unwrap().embedder = Box::new(Batching {
            batches: batches.clone(),
        });
        let stats = lab.backfill_vectors("test").unwrap();
        assert_eq!((stats.created, stats.errors), (129, 1));
        assert_eq!(*batches.borrow(), [128, 2]);
        let debt = lab.embedding_debt("test").unwrap();
        assert_eq!(debt.pending_knowledge, 1);
        assert!(
            debt.pending_since.is_some(),
            "the failed entry is still owed"
        );
    }
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
        // Both hints name the shelf: without `-s`, backfill would rebuild the default one.
        let attention = lab.shelf_status("test").unwrap().attention.unwrap();
        assert!(
            attention.contains("`hypatia backfill --reembed -s test`"),
            "{attention}"
        );
        let err = lab.backfill_vectors("test").unwrap_err().to_string();
        assert!(
            err.contains("`hypatia backfill --reembed -s test`"),
            "{err}"
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
    #[test]
    fn changed_model_opens_degraded_until_reembed() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let write_model = |model: &str| {
            std::fs::write(
                dir.path().join("shelf.toml"),
                format!("[embedding]\nmodel='{model}'\ndimensions=3\n"),
            )
            .unwrap()
        };
        let open = || {
            let mut lab = Lab {
                shelf_manager: ShelfManager::with_home(home.path().into()).unwrap(),
            };
            lab.shelf_manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: false });
            lab
        };
        let mismatched = |lab: &Lab| {
            lab.shelf_manager
                .get("test")
                .unwrap()
                .backend
                .identity_mismatch()
                .is_some()
        };
        let vectors = |lab: &Lab| {
            lab.shelf_manager
                .get("test")
                .unwrap()
                .backend
                .embedding_row_count("knowledge")
                .unwrap()
        };
        write_model("model-a");
        ShelfManager::with_home(home.path().into())
            .unwrap()
            .connect(dir.path(), Some("test"))
            .unwrap();

        // Without vectors a new model simply rebinds.
        write_model("model-b");
        let mut lab = open();
        assert!(!mismatched(&lab));
        lab.create_knowledge("test", "kept", Content::new("saved"))
            .unwrap();
        lab.backfill_vectors("test").unwrap();
        assert_eq!(vectors(&lab), 1);
        drop(lab);

        // With vectors a different model still opens: CRUD works, vectors are refused.
        write_model("model-c");
        let mut lab = open();
        assert!(mismatched(&lab));
        assert_eq!(
            lab.get_knowledge("test", "kept")
                .unwrap()
                .unwrap()
                .content
                .data,
            "saved"
        );
        lab.create_knowledge("test", "added", Content::new("while degraded"))
            .unwrap();
        assert_eq!(vectors(&lab), 1, "no vectors from a second model");
        let refused = [
            lab.similar("test", "saved", "knowledge", 5).err().unwrap(),
            lab.query(
                "test",
                &serde_json::json!(["$knowledge", ["$similar", "saved"]]),
            )
            .err()
            .unwrap(),
            lab.backfill_vectors("test").err().unwrap(),
        ];
        for err in refused {
            assert!(err.to_string().contains("--reembed"), "{err}");
        }
        drop(lab);

        // A changed model that is not installed: installing it comes before re-embedding.
        write_model("org/missing");
        let mut unusable = Lab {
            shelf_manager: ShelfManager::with_home(home.path().into()).unwrap(),
        };
        let refused = [
            unusable
                .similar("test", "saved", "knowledge", 5)
                .err()
                .unwrap(),
            unusable
                .backfill_vectors_with_reembed("test", true)
                .err()
                .unwrap(),
        ];
        for err in refused {
            assert!(
                err.to_string()
                    .contains("`hypatia model install org/missing -s test`"),
                "{err}"
            );
        }
        drop(unusable);
        write_model("model-c");
        let mut lab = open();

        // An explicit reembed rebuilds every vector under the new identity.
        assert_eq!(
            lab.backfill_vectors_with_reembed("test", true)
                .unwrap()
                .created,
            2
        );
        assert!(!mismatched(&lab));
        assert_eq!(lab.embedding_debt("test").unwrap().pending_since, None);
        assert_eq!(
            lab.similar("test", "saved", "knowledge", 5)
                .unwrap()
                .rows
                .len(),
            2
        );
        drop(lab);
        assert!(!mismatched(&open()));
    }
    #[test]
    fn vectors_are_never_written_under_an_unknown_identity() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        // No model name and no model files: the identity is only a placeholder. Embedding on
        // write checks the synchronous path; the flush below checks the deferred one.
        std::fs::write(
            dir.path().join("shelf.toml"),
            "[embedding]\ndimensions=3\ndefer=false\n",
        )
        .unwrap();
        let mut manager = ShelfManager::with_home(home.path().into()).unwrap();
        manager.connect(dir.path(), Some("test")).unwrap();
        manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: false });
        let mut lab = Lab {
            shelf_manager: manager,
        };
        lab.create_knowledge("test", "k", Content::new("x"))
            .unwrap();
        let shelf = lab.shelf_manager.get_mut("test").unwrap();
        assert_eq!(shelf.flush_pending(128).unwrap().installed, 0);
        assert_eq!(shelf.backend.embedding_row_count("knowledge").unwrap(), 0);
    }
    #[test]
    fn semantic_reads_embed_recent_local_writes_first() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("shelf.toml"),
            "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
        )
        .unwrap();
        let mut manager = ShelfManager::with_home(home.path().into()).unwrap();
        manager.connect(dir.path(), Some("test")).unwrap();
        manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: false });
        let mut lab = Lab {
            shelf_manager: manager,
        };
        lab.create_knowledge("test", "one", Content::new("first"))
            .unwrap();
        // Far below the write threshold: the write only notes the debt.
        assert_eq!(lab.embedding_debt("test").unwrap().pending_knowledge, 1);
        assert_eq!(
            lab.similar("test", "first", "knowledge", 5)
                .unwrap()
                .rows
                .len(),
            1
        );
        lab.create_knowledge("test", "two", Content::new("second"))
            .unwrap();
        let query = serde_json::json!(["$knowledge", ["$similar", "second"]]);
        assert!(!lab.query("test", &query).unwrap().rows.is_empty());
        let debt = lab.embedding_debt("test").unwrap();
        assert_eq!((debt.pending_knowledge, debt.pending_since), (0, None));
    }
    #[test]
    fn similar_operators_are_found_anywhere_in_a_query() {
        use serde_json::json;
        assert!(uses_similar(&json!(["$knowledge", ["$similar", "x"]])));
        assert!(uses_similar(&json!([
            "$knowledge",
            ["$and", ["$eq", "name", "a"], ["$similar", "x"]]
        ])));
        assert!(uses_similar(&json!({"any": ["$similar", "x"]})));
        // The word as data is not the operator.
        assert!(!uses_similar(&json!([
            "$knowledge",
            ["$eq", "name", "$similar"]
        ])));
        assert!(!uses_similar(&json!(["$knowledge", ["$search", "x"]])));
        assert!(!uses_similar(&json!([
            "$knowledge",
            ["$has", "tags", ["$similar"]]
        ])));
    }
    #[test]
    fn a_model_is_attached_only_where_it_strands_nothing() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("shelf.toml");
        std::fs::write(
            &config,
            "# my shelf\n[embedding]\ndimensions = 3\ntokenizer_path = 'old-tokenizer.json'\n\n[storage]\nbackend = 'sqlite'\n",
        )
        .unwrap();
        // It may hold a database password: its permissions must survive the rewrite.
        #[cfg(unix)]
        std::fs::set_permissions(&config, std::os::unix::fs::PermissionsExt::from_mode(0o600))
            .unwrap();
        let mut manager = ShelfManager::with_home(home.path().into()).unwrap();
        manager.connect(dir.path(), Some("test")).unwrap();
        let mut lab = Lab {
            shelf_manager: manager,
        };
        lab.create_knowledge("test", "k", Content::new("x"))
            .unwrap();

        // No vectors yet: the model is named, and the reopened shelf uses it.
        assert_eq!(
            lab.attach_model("test", "org/model").unwrap(),
            Attached::Configured {
                config: config.clone(),
                kept: vec!["dimensions".to_string()],
            }
        );
        let text = std::fs::read_to_string(&config).unwrap();
        assert!(
            text.contains("# my shelf") && text.contains("[storage]"),
            "{text}"
        );
        assert!(text.contains("model = \"org/model\""), "{text}");
        assert!(!text.contains("tokenizer_path"), "{text}");
        #[cfg(unix)]
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(&config).unwrap().permissions()
            ) & 0o777,
            0o600
        );
        let identity = |lab: &Lab| {
            lab.shelf_manager
                .get("test")
                .unwrap()
                .settings
                .embedding
                .model_identity()
                .to_string()
        };
        assert!(identity(&lab).contains("org/model"));
        assert_eq!(
            lab.attach_model("test", "org/model").unwrap(),
            Attached::AlreadyConfigured
        );

        // With vectors of that model, another one is only reported on.
        lab.shelf_manager.get_mut("test").unwrap().embedder = Box::new(Fixed { fail: false });
        assert_eq!(lab.backfill_vectors("test").unwrap().created, 1);
        assert_eq!(
            lab.attach_model("test", "other/model").unwrap(),
            Attached::HasVectors {
                config: config.clone()
            }
        );
        assert!(
            !std::fs::read_to_string(&config)
                .unwrap()
                .contains("other/model")
        );
        assert!(identity(&lab).contains("org/model"));

        // A shelf without a shelf.toml gets one; a remote shelf is left alone.
        let bare = tempfile::tempdir().unwrap();
        lab.connect_shelf(bare.path(), Some("bare")).unwrap();
        assert!(matches!(
            lab.attach_model("bare", "org/model").unwrap(),
            Attached::Configured { .. }
        ));
        // A section of its own, not an inline table, so settings can be added beside it.
        let created = std::fs::read_to_string(bare.path().join("shelf.toml")).unwrap();
        assert!(
            created.contains("[embedding]") && created.contains("model = \"org/model\""),
            "{created}"
        );
        let remote = tempfile::tempdir().unwrap();
        let remote_config = remote.path().join("shelf.toml");
        let remote_text =
            "[embedding]\nprovider = 'remote'\napi_key_env = 'PATH'\ndimensions = 3\n";
        std::fs::write(&remote_config, remote_text).unwrap();
        lab.connect_shelf(remote.path(), Some("remote")).unwrap();
        assert_eq!(
            lab.attach_model("remote", "org/model").unwrap(),
            Attached::Remote {
                config: remote_config.clone()
            }
        );
        assert_eq!(
            std::fs::read_to_string(&remote_config).unwrap(),
            remote_text
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

//! Complete shelf backend boundary. Local vector files never escape LocalBackend.
use super::{
    SqliteStore, VectorFileIndex,
    flush::{Blocked, BlockedReason, FLUSH_STATE_KEY, FlushState},
    open_or_migrate,
    settings::{BackendKind, ShelfSettings},
    transfer::{EmbeddingMetadata, IdentityMismatch, Snapshot, validate_vector},
};
use crate::{
    engine::filter::SqlFilter,
    error::{HypatiaError, Result},
    model::{Content, Knowledge, QueryTarget, SearchOpts, ShelfConfig, Statement, StatementKey},
};
use chrono::NaiveDateTime;
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub struct ShelfBackend {
    inner: Backend,
    dims: usize,
    identity_trusted: bool,
    /// The configured identity; vectors are only written while it is the stored one.
    identity: EmbeddingMetadata,
    /// Stored vectors belong to a different model than configured.
    mismatch: Option<IdentityMismatch>,
    /// The shelf's name, for hints that name a command.
    shelf: String,
}
enum Backend {
    Local(LocalBackend),
    #[cfg(feature = "postgres-backend")]
    Postgres(super::postgres_store::PgStore),
}
struct LocalBackend {
    store: SqliteStore,
    vectors: std::cell::RefCell<HashMap<String, VectorFileIndex>>,
    path: PathBuf,
    dims: usize,
    cache_clock: std::cell::Cell<i64>,
    cache_identity: String,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheManifest {
    identity: String,
    clock: i64,
    dimensions: usize,
    catalogs: Vec<String>,
}
impl LocalBackend {
    fn clock(&self) -> Result<i64> {
        Ok(self.store.conn().query_row(
            "SELECT CAST(v AS INTEGER) FROM meta WHERE k='content_clock'",
            [],
            |r| r.get(0),
        )?)
    }
    fn cache_file(&self, clock: i64, catalog: &str) -> PathBuf {
        self.path.join(format!(
            "{}-{clock}-{}-{catalog}.usearch",
            self.cache_identity, self.dims
        ))
    }
    // Caller holds a DB read transaction; clock, vectors, and later hydrated
    // content all describe the same database snapshot.
    fn refresh_snapshot(&self, clock: i64, force: bool) -> Result<()> {
        if !force && self.cache_clock.get() == clock {
            return Ok(());
        }
        if !force {
            if let Ok(bytes) = std::fs::read(self.path.join("cache.json")) {
                if let Ok(manifest) = serde_json::from_slice::<CacheManifest>(&bytes) {
                    if manifest.identity == self.cache_identity
                        && manifest.clock == clock
                        && manifest.dimensions == self.dims
                        && manifest
                            .catalogs
                            .iter()
                            .all(|c| matches!(c.as_str(), "knowledge" | "statement"))
                    {
                        let loaded: Result<HashMap<_, _>> = manifest
                            .catalogs
                            .iter()
                            .map(|c| {
                                Ok((
                                    c.clone(),
                                    VectorFileIndex::load(&self.cache_file(clock, c), self.dims)?,
                                ))
                            })
                            .collect();
                        if let Ok(loaded) = loaded {
                            *self.vectors.borrow_mut() = loaded;
                            self.cache_clock.set(clock);
                            return Ok(());
                        }
                    }
                }
            }
        }
        let mut vectors = HashMap::new();
        for catalog in ["knowledge", "statement"] {
            let items: Vec<_> = self
                .store
                .embedding_pairs(catalog)?
                .into_iter()
                .map(|(id, b)| (id, super::sqlite_store::blob_to_vector(&b)))
                .filter(|(_, v)| validate_vector(v, self.dims).is_ok())
                .collect();
            if !items.is_empty() {
                vectors.insert(
                    catalog.into(),
                    VectorFileIndex::build(&self.cache_file(clock, catalog), self.dims, &items)?,
                );
            }
        }
        *self.vectors.borrow_mut() = vectors;
        self.cache_clock.set(clock);
        // Persistence is optional: usable in-memory indexes survive disk errors.
        if let Err(e) = self.flush() {
            eprintln!("warning: cannot persist vector cache; SQLite remains authoritative: {e}");
        }
        Ok(())
    }
    fn rebuild(&self) -> Result<()> {
        let _snapshot = self.store.conn().unchecked_transaction()?;
        let clock = self.clock()?;
        self.refresh_snapshot(clock, true)
    }
    fn flush(&self) -> Result<()> {
        if self.cache_clock.get() < 0 {
            return Ok(());
        }
        std::fs::create_dir_all(&self.path)?;
        let mut vectors = self.vectors.borrow_mut();
        for idx in vectors.values_mut() {
            idx.save()?;
        }
        let manifest = CacheManifest {
            identity: self.cache_identity.clone(),
            clock: self.cache_clock.get(),
            dimensions: self.dims,
            catalogs: vectors.keys().cloned().collect(),
        };
        super::vector_index::atomic_write(
            &self.path.join("cache.json"),
            &serde_json::to_vec(&manifest)?,
        )?;
        // Never remove another writer's newer generation. An older concurrent
        // reader whose file disappears can rebuild from its own DB snapshot.
        if let Ok(files) = std::fs::read_dir(&self.path) {
            let prefix = format!("{}-", self.cache_identity);
            for file in files.flatten() {
                let name = file.file_name();
                if let Some(rest) = name.to_str().and_then(|n| n.strip_prefix(&prefix)) {
                    if let Some((old, _)) = rest.split_once('-') {
                        if old.parse::<i64>().is_ok_and(|c| c < manifest.clock)
                            && name.to_string_lossy().ends_with(".usearch")
                        {
                            let _ = std::fs::remove_file(file.path());
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn search_vector(
        &self,
        target: QueryTarget,
        vector: &[f32],
        limit: i64,
        filter: Option<&SqlFilter>,
    ) -> Result<Vec<(String, String, f64)>> {
        // RAII rollback releases this read-only snapshot on every return path.
        // A writer cannot change the hydrated rows after our clock observation.
        let _snapshot = self.store.conn().unchecked_transaction()?;
        let clock = self.clock()?;
        if self.store.embedding_metadata()?.is_none()
            && self.store.embedding_row_count("knowledge")?
                + self.store.embedding_row_count("statement")?
                > 0
        {
            return Err(HypatiaError::Config("legacy vectors have unknown model identity; explicitly reembed before semantic search".into()));
        }
        if let Err(e) = self.refresh_snapshot(clock, false) {
            eprintln!("warning: vector cache unavailable; using exact SQLite search: {e}");
        }
        let catalog = target.table_name();
        if self.cache_clock.get() == clock {
            if let Some(idx) = self.vectors.borrow().get(catalog) {
                // Fixed before ranking, so the walk passes over the entries it excludes.
                let allowed = filter
                    .map(|f| self.store.filtered_doc_ids(target, f))
                    .transpose()?;
                let hits = match &allowed {
                    None => idx.search(vector, limit as usize),
                    Some(ids) => {
                        idx.filtered_search(vector, limit as usize, |id| ids.contains(&id))
                    }
                };
                if let Ok(hits) = hits {
                    // A filtered walk can stop short although enough entries qualify;
                    // then only the exact search below finds them all.
                    let enough = match &allowed {
                        None => !hits.is_empty(),
                        Some(ids) => hits.len() >= ids.len().min(limit as usize),
                    };
                    if enough {
                        let ids: Vec<_> = hits.iter().map(|(id, _)| *id).collect();
                        let mut rows: HashMap<_, _> = self
                            .store
                            .rows_by_doc_ids(catalog, &ids)?
                            .into_iter()
                            .map(|(id, k, c)| (id, (k, c)))
                            .collect();
                        return Ok(hits
                            .into_iter()
                            .filter_map(|(id, d)| rows.remove(&id).map(|(k, c)| (k, c, d)))
                            .collect());
                    }
                }
            }
        }
        match (filter, target) {
            (Some(f), _) => self.store.vector_search_where(target, vector, limit, f),
            (None, QueryTarget::Knowledge) => self.store.vector_search_knowledge(vector, limit),
            (None, QueryTarget::Statement) => self.store.vector_search_statements(vector, limit),
        }
    }
}
impl ShelfBackend {
    pub fn open(config: &ShelfConfig, settings: &ShelfSettings) -> Result<Self> {
        settings.validate()?;
        let dims = settings.embedding.dimensions();
        let configured = EmbeddingMetadata {
            model: settings.embedding.model_identity().into(),
            dimensions: dims,
            metric: "cosine".into(),
        };
        let mut stored = None;
        let inner = match settings.storage.backend {
            BackendKind::Sqlite => {
                // Vector files are a disposable cache; disk failure must not disable CRUD.
                let mut store = open_or_migrate(config)?;
                store.set_skip_tags(&settings.embedding.skip_tags);
                if settings.embedding.model_identity_trusted {
                    stored = store.configure_embedding(&configured)?;
                }
                store.conn().execute("INSERT OR IGNORE INTO meta(k,v) VALUES('cache_identity',lower(hex(randomblob(16))))",[])?;
                let cache_identity: String = store.conn().query_row(
                    "SELECT v FROM meta WHERE k='cache_identity'",
                    [],
                    |r| r.get(0),
                )?;
                if cache_identity.len() != 32
                    || !cache_identity.bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err(HypatiaError::Config("invalid local cache identity".into()));
                }
                let local = LocalBackend {
                    store,
                    vectors: std::cell::RefCell::new(HashMap::new()),
                    path: config.vectors_path.clone(),
                    dims,
                    cache_clock: std::cell::Cell::new(-1),
                    cache_identity,
                };
                // First search loads a matching persisted generation or rebuilds.
                // Plain CRUD startup never scans embeddings or needs usearch files.
                Backend::Local(local)
            }
            BackendKind::Pgvector => {
                #[cfg(feature = "postgres-backend")]
                {
                    let pg = super::postgres_store::PgStore::open(
                        settings
                            .storage
                            .postgres
                            .as_ref()
                            .expect("validated PG config"),
                        &settings.storage.vector,
                        &settings.embedding,
                    )?;
                    stored = pg.stored_identity_mismatch().cloned();
                    Backend::Postgres(pg)
                }
                #[cfg(not(feature = "postgres-backend"))]
                {
                    return Err(HypatiaError::Config("this binary does not support pgvector; rebuild with --features postgres-backend".into()));
                }
            }
        };
        Ok(Self {
            inner,
            dims,
            identity_trusted: settings.embedding.model_identity_trusted,
            shelf: config.id.name.clone(),
            mismatch: stored.map(|stored| IdentityMismatch {
                shelf: config.id.name.clone(),
                stored,
                configured: configured.clone(),
            }),
            identity: configured,
        })
    }
    pub fn is_local(&self) -> bool {
        matches!(self.inner, Backend::Local(_))
    }
    pub fn schema(&self) -> Option<&str> {
        match &self.inner {
            Backend::Local(_) => None,
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => Some(pg.schema()),
        }
    }
    /// Stored vectors were built with a different model: vector reads and writes are refused.
    pub fn identity_mismatch(&self) -> Option<&IdentityMismatch> {
        self.mismatch.as_ref()
    }
    pub fn vector_search(
        &self,
        target: QueryTarget,
        vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        self.vector_search_where(target, vector, limit, None)
    }
    /// Prepares `filter` against `target`'s table without running it.
    pub fn check_filter(&self, target: QueryTarget, filter: &SqlFilter) -> Result<()> {
        match &self.inner {
            Backend::Local(l) => l.store.check_filter(target, filter),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.check_filter(target, filter),
        }
    }
    /// [`vector_search`](Self::vector_search) among the entries `filter` accepts. The
    /// filter applies before ranking: `limit` entries come back whenever that many
    /// qualify, however many nearer ones it excludes.
    pub fn vector_search_where(
        &self,
        target: QueryTarget,
        vector: &[f32],
        limit: i64,
        filter: Option<&SqlFilter>,
    ) -> Result<Vec<(String, String, f64)>> {
        if let Some(m) = &self.mismatch {
            return Err(m.to_error());
        }
        validate_vector(vector, self.dims)?;
        if limit < 0 {
            return Err(HypatiaError::Validation("limit must be nonnegative".into()));
        }
        if limit == 0 {
            return Ok(vec![]);
        }
        match &self.inner {
            Backend::Local(l) => l.search_vector(target, vector, limit, filter),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => match (filter, target) {
                (Some(f), _) => pg.vector_search_where(target, vector, limit, f),
                (None, QueryTarget::Knowledge) => pg.vector_search_knowledge(vector, limit),
                (None, QueryTarget::Statement) => pg.vector_search_statements(vector, limit),
            },
        }
    }
    pub fn install_embedding(
        &mut self,
        catalog: &str,
        key: &str,
        version: i64,
        vector: &[f32],
    ) -> Result<bool> {
        if let Some(m) = &self.mismatch {
            return Err(m.to_error());
        }
        // An untrusted identity is only a placeholder; vectors written under it could never be matched.
        if !self.identity_trusted {
            return Err(HypatiaError::Config("embedding model identity is unknown; configure embedding.model or provide readable model files before writing vectors".into()));
        }
        validate_vector(vector, self.dims)?;
        match &mut self.inner {
            Backend::Local(l) => {
                if l.store.embedding_metadata()?.is_none() {
                    return Err(HypatiaError::Config("legacy vectors have unknown model identity; explicitly reembed before embedding writeback".into()));
                }
                let installed =
                    l.store
                        .install_embedding_as(&self.identity, catalog, key, version, vector)?;
                if installed {
                    // The next search lazily loads or rebuilds a coherent generation.
                    l.cache_clock.set(-1);
                }
                Ok(installed)
            }
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.install_embedding(catalog, key, version, vector),
        }
    }
    pub fn embeddings_identified(&self) -> Result<bool> {
        match &self.inner {
            Backend::Local(l) => Ok(l.store.embedding_metadata()?.is_some()),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(_) => Ok(true),
        }
    }
    pub fn pending_count(&self, catalog: &str) -> Result<usize> {
        match &self.inner {
            Backend::Local(l) => l.store.pending_count(catalog),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.pending_count(catalog),
        }
    }
    /// Whether this version of an entry still lacks a vector.
    pub fn is_pending(&self, catalog: &str, key: &str, version: i64) -> Result<bool> {
        match &self.inner {
            Backend::Local(l) => l.store.is_pending(catalog, key, version),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.is_pending(catalog, key, version),
        }
    }
    /// Pending entries across both catalogs, counted no further than `limit` in each, so the
    /// cost never grows with the debt.
    pub fn pending_up_to(&self, limit: usize) -> Result<usize> {
        match &self.inner {
            Backend::Local(l) => l.store.pending_up_to(limit as i64),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.pending_up_to(limit as i64),
        }
    }
    /// Why no vector can be written right now, if so. Checked before embedding anything, so
    /// a flush never computes vectors only to have them refused.
    pub fn vectors_blocked(&self) -> Result<Option<Blocked>> {
        let blocked = |reason: BlockedReason, message: String| -> Result<Option<Blocked>> {
            Ok(Some(Blocked { reason, message }))
        };
        if let Some(m) = &self.mismatch {
            return blocked(BlockedReason::IdentityMismatch, m.to_string());
        }
        if !self.identity_trusted {
            return blocked(
                BlockedReason::UnknownIdentity,
                "embedding model identity is unknown; configure embedding.model or provide readable model files".into(),
            );
        }
        // A trusted identity always binds while no vector exists, so no binding means
        // vectors from before identity tracking.
        let legacy = match &self.inner {
            Backend::Local(l) => l.store.embedding_metadata()?.is_none(),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(_) => false,
        };
        if legacy {
            return blocked(
                BlockedReason::LegacyVectors,
                format!(
                    "legacy vectors have unknown model identity; run `hypatia backfill --reembed -s {}`",
                    self.shelf
                ),
            );
        }
        Ok(None)
    }
    /// Up to `limit` pending entries across both catalogs, most recently written first.
    pub fn newest_missing_embeddings(
        &self,
        limit: i64,
    ) -> Result<Vec<(String, String, Content, i64)>> {
        match &self.inner {
            Backend::Local(l) => l.store.newest_missing_embeddings(limit),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.newest_missing_embeddings(limit),
        }
    }
    /// Embedding debt bookkeeping for the configured identity. The debt's start survives a
    /// model change (entries lack vectors either way); the breaker and learned batch size
    /// belong to the identity they were recorded under. Unreadable state reads as empty.
    pub fn flush_state(&self) -> Result<FlushState> {
        let json = match &self.inner {
            Backend::Local(l) => l.store.meta_value(FLUSH_STATE_KEY)?,
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.meta_value(FLUSH_STATE_KEY)?,
        };
        Ok(Self::read_flush_state(&self.identity.model, json))
    }
    /// Reads, changes and writes the flush state as one step, so concurrent processes never
    /// lose each other's updates. Writes only if `change` changed something.
    pub fn update_flush_state(
        &mut self,
        change: impl FnOnce(&mut FlushState),
    ) -> Result<FlushState> {
        let mut updated = FlushState::default();
        let identity = &self.identity.model;
        let update = |json: Option<String>| -> Result<Option<String>> {
            let mut state = Self::read_flush_state(identity, json);
            let before = state.clone();
            change(&mut state);
            let write = (state != before)
                .then(|| serde_json::to_string(&state))
                .transpose()?;
            updated = state;
            Ok(write)
        };
        match &self.inner {
            Backend::Local(l) => l.store.update_meta_value(FLUSH_STATE_KEY, update)?,
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.update_meta_value(FLUSH_STATE_KEY, update)?,
        }
        Ok(updated)
    }
    fn read_flush_state(identity: &str, json: Option<String>) -> FlushState {
        let mut state = FlushState::for_identity(identity);
        if let Some(stored) = json.and_then(|json| serde_json::from_str::<FlushState>(&json).ok()) {
            if stored.identity == state.identity {
                state = stored;
            } else {
                state.pending_since = stored.pending_since;
            }
        }
        state
    }

    /// Drops the vectors of entries the shelf no longer embeds; see the stores' own method.
    pub fn clear_skipped_embeddings(&self) -> Result<usize> {
        match &self.inner {
            Backend::Local(l) => {
                let cleared = l.store.clear_skipped_embeddings()?;
                if cleared > 0 {
                    l.cache_clock.set(-1);
                }
                Ok(cleared)
            }
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.clear_skipped_embeddings(),
        }
    }
    /// Explicit reembed: drops every vector and rebinds the identity to `metadata`.
    pub fn reset_embeddings(&mut self, metadata: &EmbeddingMetadata) -> Result<()> {
        match &mut self.inner {
            Backend::Local(l) => {
                l.store.reset_embeddings(metadata)?;
                l.cache_clock.set(-1);
            }
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.reset_embeddings(metadata)?,
        }
        self.mismatch = None;
        self.identity = metadata.clone();
        Ok(())
    }

    pub fn rebuild_indexes(&mut self) -> Result<()> {
        match &mut self.inner {
            Backend::Local(l) => {
                l.store.rebuild_json_index()?;
                l.rebuild()
            }
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.rebuild_indexes(),
        }
    }
    pub fn flush(&mut self) -> Result<()> {
        match &mut self.inner {
            Backend::Local(l) => l.flush(),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(_) => Ok(()),
        }
    }
    pub fn backup_local(&self, path: &Path) -> Result<()> {
        match &self.inner {
            Backend::Local(l) => l.store.backup_to(path),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(_) => Err(HypatiaError::Validation(
                "PostgreSQL requires logical export".into(),
            )),
        }
    }
}
/// Enumeration covers top-level fields only. A dotted path such as `synonyms.head`
/// is a path in the SQLite postings table but not a key of the PostgreSQL content
/// document, so the two backends would answer differently; refusing it keeps them
/// from disagreeing silently.
fn top_level_array_field(field: &str) -> Result<()> {
    if field.is_empty() || field.contains('.') {
        return Err(HypatiaError::Validation(format!(
            "cannot enumerate '{field}': only top-level fields such as tags and scopes"
        )));
    }
    Ok(())
}

impl Drop for ShelfBackend {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl ShelfBackend {
    pub fn insert_knowledge(&self, name: &str, content: &Content) -> Result<i64> {
        match &self.inner {
            Backend::Local(l) => l.store.insert_knowledge(name, content),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.insert_knowledge(name, content),
        }
    }
    pub fn get_knowledge(&self, name: &str) -> Result<Option<Knowledge>> {
        match &self.inner {
            Backend::Local(l) => l.store.get_knowledge(name),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.get_knowledge(name),
        }
    }
    pub fn update_knowledge(&self, name: &str, content: &Content) -> Result<i64> {
        match &self.inner {
            Backend::Local(l) => l.store.update_knowledge(name, content),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.update_knowledge(name, content),
        }
    }
    pub fn delete_knowledge(&self, name: &str) -> Result<()> {
        match &self.inner {
            Backend::Local(l) => l.store.delete_knowledge(name),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.delete_knowledge(name),
        }
    }
    pub fn query_knowledge(&self, sql: &str, params: Vec<Value>) -> Result<Vec<Knowledge>> {
        match &self.inner {
            Backend::Local(l) => l.store.query_knowledge(sql, params),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.query_knowledge(sql, params),
        }
    }
    /// Idempotent: `None` when the triple already exists and nothing was written.
    pub fn insert_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<Option<i64>> {
        match &self.inner {
            Backend::Local(l) => l.store.insert_statement(key, content, tr_start, tr_end),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.insert_statement(key, content, tr_start, tr_end),
        }
    }
    pub fn update_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<i64> {
        match &self.inner {
            Backend::Local(l) => l.store.update_statement(key, content, tr_start, tr_end),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.update_statement(key, content, tr_start, tr_end),
        }
    }
    pub fn get_statement(&self, key: &StatementKey) -> Result<Option<Statement>> {
        match &self.inner {
            Backend::Local(l) => l.store.get_statement(key),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.get_statement(key),
        }
    }
    pub fn delete_statement(&self, key: &StatementKey) -> Result<()> {
        match &self.inner {
            Backend::Local(l) => l.store.delete_statement(key),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.delete_statement(key),
        }
    }
    pub fn query_statements(&self, sql: &str, params: Vec<Value>) -> Result<Vec<Statement>> {
        match &self.inner {
            Backend::Local(l) => l.store.query_statements(sql, params),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.query_statements(sql, params),
        }
    }
    pub fn query_khop(
        &self,
        head: &str,
        relation: Option<&str>,
        depth: i64,
    ) -> Result<Vec<Statement>> {
        match &self.inner {
            Backend::Local(l) => l.store.query_khop(head, relation, depth),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.query_khop(head, relation, depth),
        }
    }
    /// Distinct values of a top-level array content field (`tags`, `scopes`,
    /// `figures`, `synonyms`) across knowledge and statements, with how many
    /// entries carry each.
    pub fn field_values(&self, field: &str) -> Result<Vec<(String, i64)>> {
        top_level_array_field(field)?;
        match &self.inner {
            Backend::Local(l) => l.store.field_values(field),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.field_values(field),
        }
    }
    pub fn field_value_exists(&self, field: &str, value: &str) -> Result<bool> {
        top_level_array_field(field)?;
        match &self.inner {
            Backend::Local(l) => l.store.field_value_exists(field, value),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.field_value_exists(field, value),
        }
    }
    pub fn search(&self, query: &str, opts: &SearchOpts) -> Result<Vec<super::FtsResult>> {
        match &self.inner {
            Backend::Local(l) => l.store.search(query, opts),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.search(query, opts),
        }
    }
    pub fn embedding_version(&self, catalog: &str, key: &str) -> Result<Option<i64>> {
        match &self.inner {
            Backend::Local(l) => l.store.embedding_version(catalog, key),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.embedding_version(catalog, key),
        }
    }
    pub fn missing_embeddings(
        &self,
        catalog: &str,
        after_key: Option<&str>,
        limit: i64,
    ) -> Result<Vec<(String, Content, i64)>> {
        match &self.inner {
            Backend::Local(l) => l.store.missing_embeddings(catalog, after_key, limit),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.missing_embeddings(catalog, after_key, limit),
        }
    }
    pub fn embedding_row_count(&self, catalog: &str) -> Result<usize> {
        match &self.inner {
            Backend::Local(l) => l.store.embedding_row_count(catalog),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.embedding_row_count(catalog),
        }
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        match &self.inner {
            Backend::Local(l) => l.store.snapshot(),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.snapshot(),
        }
    }
    pub fn import_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        match &self.inner {
            Backend::Local(l) => l.store.import_snapshot(snapshot),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.import_snapshot(snapshot),
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    fn local(dir: &Path) -> LocalBackend {
        let store = SqliteStore::open(&dir.join("db.sqlite")).unwrap();
        store
            .configure_embedding(&EmbeddingMetadata {
                model: "cache-test".into(),
                dimensions: 2,
                metric: "cosine".into(),
            })
            .unwrap();
        LocalBackend {
            store,
            path: dir.join("vectors"),
            dims: 2,
            cache_clock: std::cell::Cell::new(-1),
            vectors: std::cell::RefCell::new(HashMap::new()),
            cache_identity: "0123456789abcdef0123456789abcdef".into(),
        }
    }
    #[test]
    fn cache_loads_persisted_generation_and_refreshes_after_external_write() {
        let dir = tempfile::tempdir().unwrap();
        let first = local(dir.path());
        let version = first
            .store
            .insert_knowledge("doc", &Content::new("old"))
            .unwrap();
        first
            .store
            .install_embedding("knowledge", "doc", version, &[1., 0.])
            .unwrap();
        first
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
            .unwrap();
        let file = first.cache_file(first.cache_clock.get(), "knowledge");
        let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
        let second = local(dir.path());
        second
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
            .unwrap();
        assert_eq!(
            std::fs::metadata(&file).unwrap().modified().unwrap(),
            modified,
            "reopening must load rather than rewrite the index"
        );
        let version = first
            .store
            .update_knowledge("doc", &Content::new("new"))
            .unwrap();
        first
            .store
            .install_embedding("knowledge", "doc", version, &[0., 1.])
            .unwrap();
        let hits = second
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
            .unwrap();
        assert_eq!(Content::from_json_str(&hits[0].1).unwrap().data, "new");
        assert!(hits[0].2 > 0.99);
        assert_eq!(second.cache_clock.get(), second.clock().unwrap());
        assert!(!file.exists(), "obsolete generations should be collected");
    }
    #[test]
    fn a_filtered_walk_that_comes_back_short_is_ranked_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let l = local(dir.path());
        for (name, vector) in [("near", [1., 0.]), ("a", [1., 0.2]), ("b", [1., 0.4])] {
            let version = l.store.insert_knowledge(name, &Content::new(name)).unwrap();
            l.store
                .install_embedding("knowledge", name, version, &vector)
                .unwrap();
        }
        let names = |hits: Vec<(String, String, f64)>| {
            hits.into_iter()
                .map(|(name, _, _)| name)
                .collect::<Vec<_>>()
        };
        let not_near = SqlFilter {
            fragment: "name != ?".into(),
            params: vec![Value::from("near")],
        };
        let search = || {
            names(
                l.search_vector(QueryTarget::Knowledge, &[1., 0.], 2, Some(&not_near))
                    .unwrap(),
            )
        };
        assert_eq!(search(), ["a", "b"]);
        // An index missing `b` walks up one hit short of the two that qualify.
        let doc = |name| l.store.doc_id_by_key("knowledge", name).unwrap().unwrap();
        let partial = VectorFileIndex::build(
            &dir.path().join("partial.usearch"),
            2,
            &[(doc("near"), vec![1., 0.]), (doc("a"), vec![1., 0.2])],
        )
        .unwrap();
        l.vectors.borrow_mut().insert("knowledge".into(), partial);
        assert_eq!(search(), ["a", "b"]);
        assert_eq!(
            l.cache_clock.get(),
            l.clock().unwrap(),
            "the index was current, so the walk ran and was found short"
        );
    }
    #[test]
    fn unwritable_cache_and_corrupt_manifest_do_not_disable_search() {
        let dir = tempfile::tempdir().unwrap();
        let l = local(dir.path());
        let version = l
            .store
            .insert_knowledge("doc", &Content::new("hello"))
            .unwrap();
        l.store
            .install_embedding("knowledge", "doc", version, &[1., 0.])
            .unwrap();
        std::fs::write(&l.path, b"file blocks cache directory").unwrap();
        assert_eq!(
            l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_file(&l.path).unwrap();
        std::fs::create_dir_all(&l.path).unwrap();
        std::fs::write(l.path.join("cache.json"), b"not json").unwrap();
        l.cache_clock.set(-1);
        assert_eq!(
            l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
                .unwrap()
                .len(),
            1
        );
    }
    #[test]
    fn concurrent_content_updates_never_pair_new_rows_with_old_distances() {
        let dir = tempfile::tempdir().unwrap();
        let l = local(dir.path());
        let version = l.store.insert_knowledge("doc", &Content::new("a")).unwrap();
        l.store
            .install_embedding("knowledge", "doc", version, &[1., 0.])
            .unwrap();
        l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
            .unwrap();
        let db = dir.path().join("db.sqlite");
        let ready = std::sync::Arc::new(std::sync::Barrier::new(2));
        let writer_ready = ready.clone();
        let writer = std::thread::spawn(move || {
            let store = SqliteStore::open(&db).unwrap();
            writer_ready.wait();
            for i in 0..80 {
                let (data, v) = if i % 2 == 0 {
                    ("b", [0., 1.])
                } else {
                    ("a", [1., 0.])
                };
                let version = store.update_knowledge("doc", &Content::new(data)).unwrap();
                store
                    .install_embedding("knowledge", "doc", version, &v)
                    .unwrap();
            }
        });
        ready.wait();
        for _ in 0..80 {
            for (_, content, distance) in l
                .search_vector(QueryTarget::Knowledge, &[1., 0.], 1, None)
                .unwrap()
            {
                let data = Content::from_json_str(&content).unwrap().data;
                assert!(
                    (distance - if data == "a" { 0. } else { 1. }).abs() < 0.001,
                    "row {data} has stale distance {distance}"
                );
            }
        }
        writer.join().unwrap();
    }
}

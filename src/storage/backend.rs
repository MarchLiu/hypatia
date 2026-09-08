//! Complete shelf backend boundary. Local vector files never escape LocalBackend.
use super::{
    SqliteStore, VectorFileIndex, open_or_migrate,
    settings::{BackendKind, ShelfSettings},
    transfer::{EmbeddingMetadata, Snapshot, validate_vector},
};
use crate::{
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
                if let Ok(hits) = idx.search(vector, limit as usize) {
                    if !hits.is_empty() {
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
        match target {
            QueryTarget::Knowledge => self.store.vector_search_knowledge(vector, limit),
            QueryTarget::Statement => self.store.vector_search_statements(vector, limit),
        }
    }
}
impl ShelfBackend {
    pub fn open(config: &ShelfConfig, settings: &ShelfSettings) -> Result<Self> {
        settings.validate()?;
        let dims = settings.embedding.dimensions();
        let inner = match settings.storage.backend {
            BackendKind::Sqlite => {
                // Vector files are a disposable cache; disk failure must not disable CRUD.
                let store = open_or_migrate(config)?;
                if settings.embedding.model_identity_trusted {
                    store.configure_embedding(&EmbeddingMetadata {
                        model: settings.embedding.model_identity().into(),
                        dimensions: dims,
                        metric: "cosine".into(),
                    })?;
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
                    Backend::Postgres(super::postgres_store::PgStore::open(
                        settings
                            .storage
                            .postgres
                            .as_ref()
                            .expect("validated PG config"),
                        &settings.storage.vector,
                        &settings.embedding,
                    )?)
                }
                #[cfg(not(feature = "postgres-backend"))]
                {
                    return Err(HypatiaError::Config("this binary does not support pgvector; rebuild with --features postgres-backend".into()));
                }
            }
        };
        Ok(Self { inner, dims })
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
    pub fn vector_search(
        &self,
        target: QueryTarget,
        vector: &[f32],
        limit: i64,
    ) -> Result<Vec<(String, String, f64)>> {
        validate_vector(vector, self.dims)?;
        if limit < 0 {
            return Err(HypatiaError::Validation("limit must be nonnegative".into()));
        }
        if limit == 0 {
            return Ok(vec![]);
        }
        match &self.inner {
            Backend::Local(l) => l.search_vector(target, vector, limit),
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => match target {
                QueryTarget::Knowledge => pg.vector_search_knowledge(vector, limit),
                QueryTarget::Statement => pg.vector_search_statements(vector, limit),
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
        validate_vector(vector, self.dims)?;
        match &mut self.inner {
            Backend::Local(l) => {
                if l.store.embedding_metadata()?.is_none() {
                    return Err(HypatiaError::Config("legacy vectors have unknown model identity; explicitly reembed before embedding writeback".into()));
                }
                let installed = l.store.install_embedding(catalog, key, version, vector)?;
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

    pub fn reset_embeddings(&mut self, metadata: &EmbeddingMetadata) -> Result<()> {
        match &mut self.inner {
            Backend::Local(l) => {
                l.store.reset_embeddings(metadata)?;
                l.cache_clock.set(-1);
                Ok(())
            }
            #[cfg(feature = "postgres-backend")]
            Backend::Postgres(pg) => pg.clear_all_embeddings(),
        }
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
    pub fn insert_statement(
        &self,
        key: &StatementKey,
        content: &Content,
        tr_start: Option<NaiveDateTime>,
        tr_end: Option<NaiveDateTime>,
    ) -> Result<i64> {
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
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
            .unwrap();
        let file = first.cache_file(first.cache_clock.get(), "knowledge");
        let modified = std::fs::metadata(&file).unwrap().modified().unwrap();
        let second = local(dir.path());
        second
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
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
            .search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
            .unwrap();
        assert_eq!(Content::from_json_str(&hits[0].1).unwrap().data, "new");
        assert!(hits[0].2 > 0.99);
        assert_eq!(second.cache_clock.get(), second.clock().unwrap());
        assert!(!file.exists(), "obsolete generations should be collected");
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
            l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_file(&l.path).unwrap();
        std::fs::create_dir_all(&l.path).unwrap();
        std::fs::write(l.path.join("cache.json"), b"not json").unwrap();
        l.cache_clock.set(-1);
        assert_eq!(
            l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
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
        l.search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
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
                .search_vector(QueryTarget::Knowledge, &[1., 0.], 1)
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

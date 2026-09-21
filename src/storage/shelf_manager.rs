use crate::embedding::{
    BatchFailure, BatchOutcome, EmbeddingProvider, build_provider, config::ProviderKind,
};
use crate::error::{HypatiaError, Result};
use crate::model::{Content, QueryResult, QueryTarget, SearchOpts, ShelfConfig, ShelfId};
use crate::storage::{
    ShelfRegistry, Storage,
    backend::ShelfBackend,
    flush::{
        Blocked, BlockedReason, Breaker, EmbeddingDebt, FLUSH_BATCH, FlushState, FlushStats,
        LOCAL_WRITE_THRESHOLD, MAX_SKIPPED, Paused, REMOTE_MAX_DELAY_SECS, REMOTE_WRITE_THRESHOLD,
        SkippedEntry, now, seconds_since,
    },
    settings::ShelfSettings,
    transfer::validate_vector,
};
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
        // Say how to turn semantic search on first: re-embedding after a model change needs
        // a usable model too.
        if let Some(off) = self.semantic_search_off_error() {
            return Err(off);
        }
        // Refuse before embedding: the query vector could not be compared anyway.
        if let Some(m) = self.backend.identity_mismatch() {
            return Err(m.to_error());
        }
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
/// Why a row of an automatic batch got no vector.
enum RowFailure {
    /// The provider failed on this input alone, or gave it non-finite or zero values.
    Alone,
    /// The provider failed on this input along with the rest of the request.
    Provider,
    /// The vector has this many values instead of the configured dimensions.
    WrongSize(usize),
    /// Storing the vector failed; the rest of the batch was left untried.
    Storage,
}

impl OpenShelf {
    pub fn open(path: &Path, name: Option<&str>) -> Result<Self> {
        // Configuration errors must not create a SQLite file or local vector directory.
        let mut settings = ShelfSettings::load(path)?;
        let config = ShelfConfig::from_path(path, name);
        settings.embedding.for_shelf(&config.id.name);
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
    /// Content has committed, and its vector is owed. By default the write only notes the
    /// debt, and pays a batch of it once enough has built up; with `embedding.defer = false`
    /// it embeds right away. Never fails the write: vectors are a rebuildable cache.
    pub fn embed_saved(&mut self, catalog: &str, key: &str, content: &Content, version: i64) {
        // Skip before embedding: the vector would be refused, and embedding may load a model.
        if let Some(m) = self.backend.identity_mismatch() {
            eprintln!(
                "warning: {catalog}/{key}: content saved; embedding skipped because the embedding model changed; {}",
                m.hint()
            );
            self.note_pending();
            return;
        }
        if self.settings.embedding.defer {
            self.note_pending();
            let threshold = match self.settings.embedding.provider {
                ProviderKind::Local => LOCAL_WRITE_THRESHOLD,
                ProviderKind::Remote => REMOTE_WRITE_THRESHOLD,
            };
            self.auto_flush(threshold);
            return;
        }
        // Opted out of deferral. Embed only a vector that could be written: without a model,
        // or under an identity that refuses vectors, the entry stays owed, as when deferred.
        if self
            .vectors_blocked()
            .is_ok_and(|blocked| blocked.is_none())
        {
            let outcome = (|| -> Result<bool> {
                let vector = self.embedder.embed(&content.embedding_text(key))?;
                self.backend
                    .install_embedding(catalog, key, version, &vector)
            })();
            match outcome {
                Ok(true) => return,
                // The content changed meanwhile, and the newer save owes its own vector.
                Ok(false) => {}
                Err(e) => eprintln!(
                    "warning: {catalog}/{key}: content saved; embedding pending: {e}; run `hypatia backfill -s {}`",
                    self.id.name
                ),
            }
        }
        self.note_pending();
    }
    /// Records when the current embedding debt started, unless a start is already recorded.
    /// A start left over from a debt paid without bookkeeping (a delete, say) is cleared by
    /// `settle_pending` when the next command begins. Best-effort: bookkeeping must never
    /// fail a committed write.
    fn note_pending(&mut self) {
        let _ = (|| -> Result<()> {
            if self.backend.flush_state()?.pending_since.is_none() {
                self.backend.update_flush_state(|state| {
                    state.pending_since.get_or_insert_with(now);
                })?;
            }
            Ok(())
        })();
    }
    /// Brings `pending_since` in line with the debt automatic flushes can pay: cleared once
    /// none is left, however it was paid (entries passed over as failing on their own do not
    /// count), and started when payable entries predate the bookkeeping. The debt is looked up
    /// outside the update, so a write landing in between can lose its start; the next command
    /// starts it again.
    pub fn settle_pending(&mut self) -> Result<FlushState> {
        let state = self.backend.flush_state()?;
        let pending = self.payable_up_to(&state, 1)? > 0;
        let settled = |state: &FlushState| match (pending, state.pending_since.is_some()) {
            (false, true) => Some(None),
            (true, false) => Some(Some(now())),
            _ => None,
        };
        if settled(&state).is_none() {
            return Ok(state);
        }
        self.backend.update_flush_state(|state| {
            if let Some(since) = settled(&*state) {
                state.pending_since = since;
            }
        })
    }
    /// Pending entries automatic flushes can embed, counted no further than `limit`: entries
    /// passed over as failing on their own are debt they will not pay.
    fn payable_up_to(&self, state: &FlushState, limit: usize) -> Result<usize> {
        if state.skipped.is_empty() {
            return self.backend.pending_up_to(limit);
        }
        // At most `skipped.len()` of these rows are passed over, so the count reaches `limit`
        // whenever that many are payable.
        Ok(self
            .backend
            .newest_missing_embeddings((limit + state.skipped.len()) as i64)?
            .iter()
            .filter(|(catalog, key, _, version)| !state.is_skipped(catalog, key, *version))
            .count())
    }
    /// The entries passed over as failing on their own that are still pending as they were:
    /// not edited, deleted or embedded since.
    fn passed_over(&self, state: &FlushState) -> Result<Vec<SkippedEntry>> {
        let mut still = Vec::new();
        for entry in &state.skipped {
            if self
                .backend
                .is_pending(&entry.catalog, &entry.key, entry.version)?
            {
                still.push(entry.clone());
            }
        }
        Ok(still)
    }
    /// Why vectors cannot be written right now: the backend refuses them, or the provider
    /// cannot produce them.
    pub fn vectors_blocked(&self) -> Result<Option<Blocked>> {
        if let Some(blocked) = self.backend.vectors_blocked()? {
            return Ok(Some(blocked));
        }
        if self.embedder.is_available() {
            return Ok(None);
        }
        let embedding = &self.settings.embedding;
        let message = match embedding.provider {
            ProviderKind::Local => embedding
                .local_unavailable
                .clone()
                .unwrap_or_else(|| "embedding model files are missing or failed to load".into()),
            ProviderKind::Remote => format!(
                "environment variable {} is not set",
                embedding.remote.api_key_env
            ),
        };
        Ok(Some(Blocked {
            reason: BlockedReason::ProviderUnavailable,
            message,
        }))
    }
    /// Why semantic search is off on this shelf, and how to turn it on; `None` when a query
    /// can be embedded.
    pub fn semantic_search_off(&self) -> Option<String> {
        if self.embedder.is_available() {
            return None;
        }
        let embedding = &self.settings.embedding;
        match embedding.provider {
            ProviderKind::Remote => Some(format!(
                "the remote API needs its key in the environment variable {}",
                embedding.remote.api_key_env
            )),
            // Files that are there but fail to load: the embedding error itself says why.
            ProviderKind::Local if embedding.local_files_exist() => None,
            // A named model's message already says how to install it, or why it fails.
            ProviderKind::Local => Some(embedding.local_unavailable.clone().unwrap_or_else(|| {
                let local = &embedding.local;
                let missing: Vec<String> = [&local.model_path, &local.tokenizer_path]
                    .into_iter()
                    .filter(|p| !p.exists())
                    .map(|p| p.display().to_string())
                    .collect();
                let nothing_configured = missing.len() == 2
                    && local.model_path == self.id.path.join("embedding_model.onnx")
                    && local.tokenizer_path == self.id.path.join("tokenizer.json");
                if nothing_configured {
                    format!(
                        "no embedding model is set up; run `hypatia model install BAAI/bge-m3 -s {}` (about 2.3 GB), or configure a remote API in {}",
                        self.id.name,
                        self.id.path.join("shelf.toml").display()
                    )
                } else {
                    format!("embedding model files not found: {}", missing.join(", "))
                }
            })),
        }
    }
    /// `semantic_search_off` as the error a vector operation fails with.
    pub fn semantic_search_off_error(&self) -> Option<HypatiaError> {
        self.semantic_search_off().map(|off| {
            HypatiaError::ModelUnavailable(format!(
                "semantic search is off on shelf '{}': {off}",
                self.id.name
            ))
        })
    }
    /// Pays one batch of embedding debt, newest first, once at least `min_pending` entries
    /// are owed: one fail-fast attempt that writes vectors only (no index rebuild). Does
    /// nothing while vectors are blocked or automatic flushes are paused. A failure pauses
    /// later attempts and prints a warning, but never reaches the command that triggered it.
    pub fn auto_flush(&mut self, min_pending: usize) -> FlushStats {
        self.try_auto_flush(min_pending).unwrap_or_else(|e| {
            eprintln!("warning: automatic embedding skipped: {e}");
            FlushStats::default()
        })
    }
    fn try_auto_flush(&mut self, min_pending: usize) -> Result<FlushStats> {
        let mut stats = FlushStats::default();
        if self.vectors_blocked()?.is_some() {
            return Ok(stats);
        }
        let state = self.backend.flush_state()?;
        if state.breaker.as_ref().is_some_and(|b| !b.is_due()) {
            return Ok(stats);
        }
        if min_pending > 1 && self.payable_up_to(&state, min_pending)? < min_pending {
            return Ok(stats);
        }
        let limit = match self.settings.embedding.provider {
            ProviderKind::Local => FLUSH_BATCH,
            ProviderKind::Remote => state.remote_batch.unwrap_or(FLUSH_BATCH),
        };
        // Entries that fail on their own are passed over, so they never crowd out the rest.
        let rows: Vec<_> = self
            .backend
            .newest_missing_embeddings((limit + state.skipped.len()) as i64)?
            .into_iter()
            .filter(|(catalog, key, _, version)| !state.is_skipped(catalog, key, *version))
            .take(limit)
            .collect();
        if rows.is_empty() {
            return Ok(stats);
        }
        let sent = rows.len();
        let entries: Vec<SkippedEntry> = rows
            .iter()
            .map(|(catalog, key, _, version)| SkippedEntry {
                catalog: catalog.clone(),
                key: key.clone(),
                version: *version,
            })
            .collect();
        let texts: Vec<String> = rows
            .iter()
            .map(|(_, key, content, _)| content.embedding_text(key))
            .collect();
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let outcome = match self.embedder.try_embed_batch(&texts) {
            Ok(outcome) => outcome,
            Err(failure) => {
                // Nothing got through: pause, for good when retrying cannot help.
                stats.failed = sent;
                stats.error = Some(failure.message().to_string());
                let permanent = matches!(failure, BatchFailure::Refused(_));
                self.pause(permanent, failure.message().to_string());
                return Ok(stats);
            }
        };
        let accepted_size = outcome.accepted_size;
        let failures = self.install_vectors(rows, outcome, &mut stats);
        // When nothing at all got through a batch of several, the provider is failing, even if
        // it failed on each entry separately.
        let alone: Vec<SkippedEntry> = if stats.installed == 0 && sent > 1 {
            Vec::new()
        } else {
            failures
                .iter()
                .filter(|(_, failure)| matches!(failure, RowFailure::Alone))
                .map(|(index, _)| entries[*index].clone())
                .collect()
        };
        // Entries edited, deleted or embedded since they were passed over no longer count.
        let still_passed_over = (state.skipped.len() + alone.len() > MAX_SKIPPED)
            .then(|| self.passed_over(&state).ok())
            .flatten();
        let crowded = still_passed_over
            .as_ref()
            .is_some_and(|still| still.len() + alone.len() > MAX_SKIPPED);
        let wrong_size = failures.iter().find_map(|(_, failure)| match failure {
            RowFailure::WrongSize(len) => Some(*len),
            _ => None,
        });
        let dims = self.settings.embedding.dimensions();
        let pause = match wrong_size {
            // Every vector will have the wrong size: a configuration error.
            Some(len) => Some((
                true,
                format!(
                    "the provider returns vectors of {len} values, but the shelf expects {dims}"
                ),
            )),
            // Nothing stored, and not because of the entries: the provider is failing.
            None if stats.installed == 0 && stats.failed > alone.len() => {
                Some((false, stats.error.clone().unwrap_or_default()))
            }
            None if crowded => Some((
                false,
                format!("more than {MAX_SKIPPED} entries fail on their own"),
            )),
            None => None,
        };
        let recorded = self.backend.update_flush_state(|current| {
            if let Some(still) = &still_passed_over {
                // Forget the stale entries; ones another process passed over meanwhile stay.
                current
                    .skipped
                    .retain(|entry| still.contains(entry) || !state.skipped.contains(entry));
            }
            current.skip(alone.iter().cloned());
            if pause.is_none() {
                current.breaker = None;
                // A batch turned down as too large teaches the size, unless an entry failing
                // alone may be what was too large.
                if alone.is_empty()
                    && let Some(size) = accepted_size
                {
                    current.remote_batch = Some(size);
                }
            }
        });
        if let Err(e) = recorded {
            eprintln!("warning: could not record the outcome of automatic embedding: {e}");
        }
        let shelf = &self.id.name;
        let error = stats.error.as_deref().unwrap_or_default();
        if !alone.is_empty() {
            eprintln!(
                "warning: {} entries could not be embedded ({error}); automatic embedding passes over them until their content changes, and `hypatia backfill -s {shelf}` reports why",
                alone.len()
            );
        }
        if stats.installed > 0 && stats.failed > alone.len() {
            eprintln!(
                "warning: {} entries got no vector ({error}); they stay pending",
                stats.failed - alone.len()
            );
        }
        if let Some((permanent, reason)) = pause {
            self.pause(permanent, reason);
        }
        // Bookkeeping only: the vectors are in, so a failure here must not fail the flush.
        let _ = self.settle_pending();
        Ok(stats)
    }
    /// Pauses automatic flushes after a failure, and says so. Best-effort, like all of their
    /// bookkeeping.
    fn pause(&mut self, permanent: bool, reason: String) {
        let paused = self.backend.update_flush_state(|state| {
            state.breaker = Some(Breaker::trip(state.breaker.as_ref(), permanent, reason));
        });
        let shelf = &self.id.name;
        match paused.map(|state| state.breaker) {
            Ok(Some(b)) if b.permanent => eprintln!(
                "warning: automatic embedding paused: {}; fix the embedding configuration, then run `hypatia backfill -s {shelf}`",
                b.reason
            ),
            Ok(Some(b)) => eprintln!(
                "warning: automatic embedding failed: {}; entries stay pending until the next attempt after {}",
                b.reason,
                b.retry_after.as_deref().unwrap_or_default()
            ),
            Ok(None) => {}
            Err(e) => {
                eprintln!("warning: could not record the outcome of automatic embedding: {e}")
            }
        }
    }
    /// Embeds up to `limit` pending entries, newest first, with the provider's full retries
    /// and splitting, and writes their vectors only (no index rebuild). For explicit callers,
    /// such as a benchmark paying its debt before querying: paused automatic flushes do not
    /// stop it.
    pub fn flush_pending(&mut self, limit: usize) -> Result<FlushStats> {
        let mut stats = FlushStats::default();
        if self.vectors_blocked()?.is_some() {
            return Ok(stats);
        }
        let rows = self.backend.newest_missing_embeddings(limit as i64)?;
        let texts: Vec<String> = rows
            .iter()
            .map(|(_, key, content, _)| content.embedding_text(key))
            .collect();
        let texts: Vec<&str> = texts.iter().map(String::as_str).collect();
        let outcome = BatchOutcome::from_results(self.embedder.embed_batch(&texts));
        self.install_vectors(rows, outcome, &mut stats);
        // Bookkeeping only: the vectors are in, so a failure here must not fail the flush.
        let _ = self.settle_pending();
        Ok(stats)
    }
    /// Checks and installs one vector per row, in order, counting the outcomes. Returns the
    /// rows that got no vector, and why. A storage error stops the batch: the rows after it
    /// would most likely wait on the same lock, one by one.
    fn install_vectors(
        &mut self,
        rows: Vec<(String, String, Content, i64)>,
        outcome: BatchOutcome,
        stats: &mut FlushStats,
    ) -> Vec<(usize, RowFailure)> {
        let dims = self.settings.embedding.dimensions();
        let total = rows.len();
        let mut failures = Vec::new();
        let mut vectors = outcome.vectors.into_iter();
        for (index, (catalog, key, _, version)) in rows.into_iter().enumerate() {
            let checked = match vectors.next() {
                None => Err((
                    HypatiaError::Embedding("provider returned too few vectors".into()),
                    RowFailure::Provider,
                )),
                Some(Err(e)) if outcome.refused_alone.contains(&index) => {
                    Err((e, RowFailure::Alone))
                }
                Some(Err(e)) => Err((e, RowFailure::Provider)),
                Some(Ok(vector)) if vector.len() != dims => Err((
                    HypatiaError::Validation(format!(
                        "embedding has {} values, but the shelf expects {dims}",
                        vector.len()
                    )),
                    RowFailure::WrongSize(vector.len()),
                )),
                Some(Ok(vector)) => validate_vector(&vector, dims)
                    .map(|()| vector)
                    .map_err(|e| (e, RowFailure::Alone)),
            };
            let installed = checked.and_then(|vector| {
                self.backend
                    .install_embedding(&catalog, &key, version, &vector)
                    .map_err(|e| (e, RowFailure::Storage))
            });
            match installed {
                Ok(true) => stats.installed += 1,
                Ok(false) => stats.skipped += 1,
                Err((e, failure)) => {
                    stats.failed += 1;
                    stats.error.get_or_insert_with(|| e.to_string());
                    let storage = matches!(failure, RowFailure::Storage);
                    failures.push((index, failure));
                    if storage {
                        stats.failed += total - index - 1;
                        break;
                    }
                }
            }
        }
        failures
    }
    /// Before a semantic read. A local model is about to be loaded for the query anyway, so
    /// pay a batch of debt first, newest first, and recent writes are found at once. Remote
    /// debt is left to the write threshold and the time cap.
    pub fn flush_before_similar(&mut self) {
        if self.settings.embedding.provider == ProviderKind::Local {
            self.auto_flush(1);
        }
    }
    /// When a command on this shelf begins: settles the bookkeeping, and pays a batch of a
    /// remote debt older than `REMOTE_MAX_DELAY_SECS`. With no background process, an overdue
    /// debt can only be paid by the next command.
    pub fn flush_if_overdue(&mut self) -> Result<()> {
        let state = self.settle_pending()?;
        let overdue = state.pending_since.as_deref().is_some_and(|since| {
            seconds_since(since).is_none_or(|secs| secs >= REMOTE_MAX_DELAY_SECS)
        });
        if overdue && self.settings.embedding.provider == ProviderKind::Remote {
            self.auto_flush(1);
        }
        Ok(())
    }
    /// An explicit backfill has run. Once it embedded anything, or nothing failed, the
    /// provider evidently works: paused automatic flushes resume, and what they learned starts
    /// over, the batch size and the entries passed over alike (backfill has just tried those).
    /// A backfill that only failed proves nothing and changes nothing.
    pub fn record_backfill(&mut self, created: usize, errors: usize) -> Result<()> {
        self.backend.update_flush_state(|state| {
            if created > 0 || errors == 0 {
                state.breaker = None;
                state.remote_batch = None;
                state.skipped.clear();
            }
        })?;
        Ok(())
    }
    /// The shelf's embedding debt, for `backfill --status` and agent interfaces.
    pub fn embedding_debt(&self) -> Result<EmbeddingDebt> {
        let pending_knowledge = self.backend.pending_count("knowledge")?;
        let pending_statement = self.backend.pending_count("statement")?;
        let state = self.backend.flush_state()?;
        let passed_over = self.passed_over(&state)?.len();
        Ok(EmbeddingDebt {
            pending_knowledge,
            pending_statement,
            passed_over,
            // A paid debt has no start, whatever the bookkeeping last recorded.
            pending_since: state
                .pending_since
                .filter(|_| pending_knowledge + pending_statement > passed_over),
            blocked: self.vectors_blocked()?,
            // A pause past its retry time is over, although the next attempt has yet to run.
            paused: state
                .breaker
                .as_ref()
                .filter(|b| !b.is_due())
                .map(Paused::from),
        })
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

    /// Restore every registered shelf that is not connected yet.
    fn restore_registered(&mut self) {
        let entries: Vec<(String, std::path::PathBuf)> = self
            .registry
            .shelves
            .iter()
            // What is already open decides this, rather than the name "default":
            // assuming where a name lives is what let the registry go unread.
            .filter(|(name, _)| !self.shelves.contains_key(name.as_str()))
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

    /// Every connected shelf.
    pub fn open_shelves_mut(&mut self) -> impl Iterator<Item = &mut OpenShelf> {
        self.shelves.values_mut()
    }

    /// Opens a registered shelf again, so a changed shelf.toml takes effect.
    pub fn reopen(&mut self, name: &str) -> Result<()> {
        let path = self
            .registry
            .get(name)
            .cloned()
            .ok_or_else(|| HypatiaError::Shelf(format!("shelf '{name}' is not registered")))?;
        // Close it first: dropping a shelf saves its vector caches.
        self.shelves.remove(name);
        self.connect_internal(&path, Some(name))?;
        Ok(())
    }

    /// List all registered shelves with their paths.
    /// Returns (name, path, is_connected) tuples.
    ///
    /// A connected shelf reports the path it is open at rather than the registered one:
    /// when the two disagree, the path being written to is the one worth showing.
    pub fn list(&self) -> Vec<(&str, &std::path::PathBuf, bool)> {
        self.registry
            .list()
            .into_iter()
            .map(|(name, registered)| match self.shelves.get(name) {
                Some(shelf) => (name, &shelf.config.id.path, true),
                None => (name, registered, false),
            })
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
    ///
    /// The registry says where `default` lives; `~/.hypatia/default` is only where it is
    /// put when nothing is registered yet. Reading that registration back is the whole
    /// point: connecting to the built-in path regardless would send every `-s default`
    /// write to a shelf other than the one `list` reports.
    pub fn ensure_default(&mut self) -> Result<String> {
        if self.shelves.contains_key("default") {
            return Ok("default".to_string());
        }

        let Some(registered) = self.registry.get("default").cloned() else {
            let builtin = self.home.join(".hypatia").join("default");
            self.registry.register("default", &builtin);
            self.registry.save(&self.registry_path)?;
            return self.connect_internal(&builtin, Some("default"));
        };

        // A default that cannot be opened stops the run, as it always has. Falling back to
        // the built-in path would put this session's writes in a shelf other than the
        // registered one -- the split this registration is now read to prevent, and what
        // `docs/pgvector-backend.md` rules out for a shelf whose database is merely down.
        // Naming the path and where it is configured is what was missing.
        self.connect_internal(&registered, Some("default"))
            .map_err(|e| {
                HypatiaError::Shelf(format!(
                    "cannot open the default shelf at {}: {e}; that path is the \"default\" \
                     entry in {}",
                    registered.display(),
                    self.registry_path.display(),
                ))
            })
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

    /// Writes a `shelves.json` under `home` before a manager reads it.
    fn register(home: &TempDir, entries: &[(&str, &std::path::Path)]) {
        let shelves: serde_json::Map<String, serde_json::Value> = entries
            .iter()
            .map(|(name, path)| (name.to_string(), serde_json::json!(path)))
            .collect();
        let dir = home.path().join(".hypatia");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shelves.json"),
            serde_json::json!({ "shelves": shelves }).to_string(),
        )
        .unwrap();
    }

    #[test]
    fn default_connects_where_the_registry_says() {
        let home = TempDir::new().unwrap();
        let elsewhere = TempDir::new().unwrap();
        register(&home, &[("default", elsewhere.path())]);

        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        assert_eq!(mgr.get("default").unwrap().id.path, elsewhere.path());

        // The write follows, rather than landing in the built-in shelf unseen.
        mgr.get_mut("default")
            .unwrap()
            .backend
            .insert_knowledge("probe", &Content::new("which dir?"))
            .unwrap();
        assert!(elsewhere.path().join("hypatia.sqlite").exists());
        assert!(!home.path().join(".hypatia/default/hypatia.sqlite").exists());
    }

    #[test]
    fn default_registers_the_built_in_path_when_unregistered() {
        let home = TempDir::new().unwrap();
        let mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();

        let builtin = home.path().join(".hypatia").join("default");
        assert_eq!(mgr.get("default").unwrap().id.path, builtin);
        assert_eq!(mgr.registry.get("default"), Some(&builtin));
    }

    #[test]
    fn an_unopenable_registered_default_stops_the_run() {
        let home = TempDir::new().unwrap();
        let unopenable = TempDir::new().unwrap();
        // Refused by ShelfSettings::load, before anything is created on disk.
        std::fs::write(unopenable.path().join("shelf.toml"), "embedding = 1").unwrap();
        register(&home, &[("default", unopenable.path())]);

        let e = match ShelfManager::with_home(home.path().to_path_buf()) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an unopenable default must not be papered over"),
        };
        // Naming the path and the file holding it is the whole recovery hint.
        assert!(e.contains(unopenable.path().to_str().unwrap()), "{e}");
        assert!(e.contains("shelves.json"), "{e}");
        // Quietly opening the built-in shelf instead is the split being fixed here.
        assert!(!home.path().join(".hypatia/default").exists());
    }

    #[test]
    fn every_registered_shelf_opens_at_its_registered_path() {
        let home = TempDir::new().unwrap();
        let elsewhere = TempDir::new().unwrap();
        let other = TempDir::new().unwrap();
        register(
            &home,
            &[("default", elsewhere.path()), ("other", other.path())],
        );

        let mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        assert_eq!(mgr.get("default").unwrap().id.path, elsewhere.path());
        assert_eq!(mgr.get("other").unwrap().id.path, other.path());
        assert_eq!(mgr.list().len(), 2);
    }

    #[test]
    fn list_reports_the_path_the_shelf_is_open_at() {
        let open_at = TempDir::new().unwrap();
        let stale = TempDir::new().unwrap();
        let home = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.connect(open_at.path(), Some("s")).unwrap();
        // Only an edit behind the CLI's back can still part the two, but printing the
        // registry path beside a `[connected]` taken from the name is what hid the split.
        mgr.registry.register("s", &stale.path().to_path_buf());

        let shelves = mgr.list();
        let (_, path, connected) = shelves.iter().find(|(n, _, _)| *n == "s").unwrap();
        assert_eq!(**path, open_at.path());
        assert!(connected);
    }

    #[test]
    fn list_reports_the_registered_path_when_disconnected() {
        let home = TempDir::new().unwrap();
        let never_opened = TempDir::new().unwrap();
        let mut mgr = ShelfManager::with_home(home.path().to_path_buf()).unwrap();
        mgr.registry
            .register("gone", &never_opened.path().to_path_buf());

        let shelves = mgr.list();
        let (_, path, connected) = shelves.iter().find(|(n, _, _)| *n == "gone").unwrap();
        assert_eq!(**path, never_opened.path());
        assert!(!connected);
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

//! Shared domain contracts, including a real optional PostgreSQL target.
use hypatia::{
    embedding::{BatchFailure, BatchOutcome, config::ProviderKind},
    engine::Evaluator,
    model::{Content, QueryTarget, SearchOpts, StatementKey},
    service::KnowledgeService,
    storage::{
        OpenShelf, ShelfManager, Storage,
        flush::{BlockedReason, FlushStats},
    },
};
use serde_json::json;
use tempfile::TempDir;

fn local(dir: &TempDir) -> OpenShelf {
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
    )
    .unwrap();
    OpenShelf::open(dir.path(), Some("contract")).unwrap()
}
/// Embeds everything to the same vector; the flush test only cares which rows get one.
struct Unit;
impl hypatia::embedding::EmbeddingProvider for Unit {
    fn embed(&self, _: &str) -> Result<Vec<f32>, hypatia::error::HypatiaError> {
        Ok(vec![1., 0., 0.])
    }
    fn dimensions(&self) -> usize {
        3
    }
    fn is_available(&self) -> bool {
        true
    }
}

#[test]
fn flushing_pays_embedding_debt_newest_first() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    // The configured model is not installed, so every write leaves its entry pending.
    for name in ["first", "second", "third"] {
        KnowledgeService::new(&mut shelf)
            .create(name, Content::new(name))
            .unwrap();
    }
    let debt = shelf.embedding_debt().unwrap();
    assert_eq!(debt.pending_knowledge, 3);
    assert_eq!(
        debt.blocked.map(|b| b.reason),
        Some(BlockedReason::ProviderUnavailable)
    );
    assert!(debt.pending_since.is_some());
    assert_eq!(shelf.flush_pending(128).unwrap().installed, 0);

    shelf.embedder = Box::new(Unit);
    let stats = shelf.flush_pending(2).unwrap();
    assert_eq!((stats.installed, stats.failed), (2, 0));
    let left: Vec<String> = shelf
        .backend
        .newest_missing_embeddings(10)
        .unwrap()
        .into_iter()
        .map(|(_, key, _, _)| key)
        .collect();
    assert_eq!(left, ["first"], "the newest entries are paid first");
    assert!(shelf.embedding_debt().unwrap().pending_since.is_some());

    shelf.flush_pending(2).unwrap();
    let debt = shelf.embedding_debt().unwrap();
    assert_eq!(debt.pending_knowledge, 0);
    assert_eq!(debt.pending_since, None, "a paid debt has no start time");
}

#[test]
fn a_paid_debt_has_no_start_and_a_new_one_starts_fresh() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    KnowledgeService::new(&mut shelf)
        .create("owed", Content::new("x"))
        .unwrap();
    assert!(shelf.embedding_debt().unwrap().pending_since.is_some());
    // Deleting the last pending entry pays the debt, although no bookkeeping ran.
    KnowledgeService::new(&mut shelf).delete("owed").unwrap();
    assert_eq!(shelf.embedding_debt().unwrap().pending_since, None);
    // The next command settles the stale start, so it never dates the next debt.
    let stale = "2000-01-01T00:00:00Z";
    shelf
        .backend
        .update_flush_state(|state| state.pending_since = Some(stale.into()))
        .unwrap();
    shelf.flush_if_overdue().unwrap();
    KnowledgeService::new(&mut shelf)
        .create("new", Content::new("z"))
        .unwrap();
    let started = shelf.embedding_debt().unwrap().pending_since;
    assert!(started.is_some() && started.as_deref() != Some(stale));
}

#[test]
fn a_write_on_a_degraded_shelf_is_owed_a_vector() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    shelf.embedder = Box::new(Unit);
    KnowledgeService::new(&mut shelf)
        .create("embedded", Content::new("x"))
        .unwrap();
    // Writes defer embedding; pay it so the shelf holds a vector of this model.
    shelf.flush_pending(128).unwrap();
    drop(shelf);
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='another-model'\ndimensions=3\n",
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("contract")).unwrap();
    shelf.embedder = Box::new(Unit);
    KnowledgeService::new(&mut shelf)
        .create("later", Content::new("y"))
        .unwrap();
    let debt = shelf.embedding_debt().unwrap();
    assert_eq!(debt.pending_knowledge, 1);
    assert!(debt.pending_since.is_some());
    assert_eq!(
        debt.blocked.map(|b| b.reason),
        Some(BlockedReason::IdentityMismatch)
    );
    // Blocked: nothing is embedded, not even to be thrown away.
    assert_eq!(shelf.flush_pending(128).unwrap(), FlushStats::default());
}

#[test]
fn a_model_change_keeps_the_debt_start() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    KnowledgeService::new(&mut shelf)
        .create("entry", Content::new("x"))
        .unwrap();
    let started = shelf.embedding_debt().unwrap().pending_since;
    assert!(started.is_some());
    drop(shelf);
    // No vectors yet, so the new model simply rebinds; the entry is still owed a vector.
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='another-model'\ndimensions=3\n",
    )
    .unwrap();
    let shelf = OpenShelf::open(dir.path(), Some("contract")).unwrap();
    assert_eq!(shelf.embedding_debt().unwrap().pending_since, started);
}

/// What a `Scripted` provider has been asked, and how it answers automatic flushes:
/// `None` succeeds, and so does everything once the outcomes run out.
#[derive(Default)]
struct Script {
    outcomes: std::collections::VecDeque<Option<BatchFailure>>,
    batches: Vec<usize>,
    /// Inputs containing this text fail on their own.
    refuse: Option<&'static str>,
    /// Inputs containing this text get a vector of non-finite values.
    garble: Option<&'static str>,
    /// Answer with vectors of the wrong size.
    wrong_size: bool,
    /// Batch size the next successful batch reports as learned.
    accepted_size: Option<usize>,
    /// Single embeddings asked for.
    embeds: usize,
}
struct Scripted(std::rc::Rc<std::cell::RefCell<Script>>);
impl Scripted {
    fn answer(script: &Script, text: &str) -> Result<Vec<f32>, hypatia::error::HypatiaError> {
        match (script.refuse, script.garble) {
            (Some(marker), _) if text.contains(marker) => Err(
                hypatia::error::HypatiaError::Embedding(format!("refused {text}")),
            ),
            (_, Some(marker)) if text.contains(marker) => Ok(vec![f32::NAN, 0., 0.]),
            _ if script.wrong_size => Ok(vec![1., 0.]),
            _ => Ok(vec![1., 0., 0.]),
        }
    }
}
impl hypatia::embedding::EmbeddingProvider for Scripted {
    fn embed(&self, text: &str) -> Result<Vec<f32>, hypatia::error::HypatiaError> {
        let mut script = self.0.borrow_mut();
        script.embeds += 1;
        Self::answer(&script, text)
    }
    fn try_embed_batch(&self, texts: &[&str]) -> Result<BatchOutcome, BatchFailure> {
        let mut script = self.0.borrow_mut();
        script.batches.push(texts.len());
        if let Some(failure) = script.outcomes.pop_front().flatten() {
            return Err(failure);
        }
        let mut outcome = BatchOutcome::from_results(
            texts
                .iter()
                .map(|text| Self::answer(&script, text))
                .collect(),
        );
        outcome.accepted_size = script.accepted_size.take();
        Ok(outcome)
    }
    fn dimensions(&self) -> usize {
        3
    }
    fn is_available(&self) -> bool {
        true
    }
}

fn scripted(
    shelf: &mut OpenShelf,
    provider: ProviderKind,
) -> std::rc::Rc<std::cell::RefCell<Script>> {
    let script = std::rc::Rc::default();
    shelf.embedder = Box::new(Scripted(std::rc::Rc::clone(&script)));
    shelf.settings.embedding.provider = provider;
    script
}

fn write(shelf: &mut OpenShelf, names: impl IntoIterator<Item = String>) {
    for name in names {
        KnowledgeService::new(&mut *shelf)
            .create(&name, Content::new(&name))
            .unwrap();
    }
}

fn pending(shelf: &OpenShelf) -> usize {
    shelf.embedding_debt().unwrap().pending_knowledge
}

#[test]
fn a_local_write_flushes_once_64_entries_are_owed() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    write(&mut shelf, (0..63).map(|i| format!("k{i}")));
    assert!(script.borrow().batches.is_empty());
    assert_eq!(pending(&shelf), 63);
    write(&mut shelf, ["k63".to_string()]);
    assert_eq!(script.borrow().batches, [64]);
    let debt = shelf.embedding_debt().unwrap();
    assert_eq!((debt.pending_knowledge, debt.pending_since), (0, None));
}

#[test]
fn remote_debt_waits_for_a_full_batch_or_a_minute() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Remote);
    write(&mut shelf, (0..127).map(|i| format!("k{i}")));
    shelf.flush_if_overdue().unwrap();
    assert!(
        script.borrow().batches.is_empty(),
        "a fresh debt is not overdue"
    );
    write(&mut shelf, ["k127".to_string()]);
    assert_eq!(script.borrow().batches, [128]);
    write(&mut shelf, ["late".to_string()]);
    shelf.flush_if_overdue().unwrap();
    assert_eq!(script.borrow().batches, [128]);
    shelf
        .backend
        .update_flush_state(|state| state.pending_since = Some("2000-01-01T00:00:00Z".into()))
        .unwrap();
    shelf.flush_if_overdue().unwrap();
    assert_eq!(script.borrow().batches, [128, 1]);
    assert_eq!(pending(&shelf), 0);
}

#[test]
fn automatic_flush_failures_pause_learn_the_batch_size_or_pass_over_entries() {
    let dir = TempDir::new().unwrap();
    flush_transitions(&mut local(&dir));
}

/// Pause, batch-size and passed-over transitions of automatic flushes to a remote API.
fn flush_transitions(shelf: &mut OpenShelf) {
    let script = scripted(shelf, ProviderKind::Remote);
    write(shelf, (0..3).map(|i| format!("k{i}")));
    let fail_next = |failure: BatchFailure| script.borrow_mut().outcomes.push_back(Some(failure));
    let sent = || script.borrow().batches.clone();
    let make_due = |shelf: &mut OpenShelf| {
        shelf
            .backend
            .update_flush_state(|state| {
                state.breaker.as_mut().unwrap().retry_after = Some("2000-01-01T00:00:00Z".into())
            })
            .unwrap();
    };
    let state = |shelf: &OpenShelf| shelf.backend.flush_state().unwrap();

    // A request turned down as sent pauses automatic flushes, without passing over entries.
    fail_next(BatchFailure::Rejected("400".into()));
    assert_eq!(shelf.auto_flush(1).failed, 3);
    let paused = state(shelf);
    assert!(paused.skipped.is_empty() && paused.remote_batch.is_none());
    assert!(!shelf.embedding_debt().unwrap().paused.unwrap().permanent);
    shelf.auto_flush(1);
    assert_eq!(sent(), [3], "nothing is sent while paused");

    // A refused request pauses for good, though it is still retried once due. A pause past its
    // retry time is not reported.
    make_due(&mut *shelf);
    fail_next(BatchFailure::Refused("401".into()));
    shelf.auto_flush(1);
    let breaker = state(shelf).breaker.unwrap();
    assert_eq!((breaker.permanent, breaker.failures), (true, 2));
    make_due(&mut *shelf);
    assert_eq!(shelf.embedding_debt().unwrap().paused, None);

    // Success resumes, and keeps a batch size learned from a batch that was too large.
    script.borrow_mut().accepted_size = Some(2);
    assert_eq!(shelf.auto_flush(1).installed, 3);
    let resumed = state(shelf);
    assert_eq!((resumed.breaker, resumed.remote_batch), (None, Some(2)));
    write(
        shelf,
        ["k3".to_string(), "k4".to_string(), "k5".to_string()],
    );
    shelf.auto_flush(1);
    assert_eq!(sent(), [3, 3, 3, 2]);

    // An entry failing on its own is passed over until its content changes, and a debt of
    // nothing but such entries never makes a command overdue.
    script.borrow_mut().refuse = Some("bad");
    write(shelf, ["bad-entry".to_string()]);
    let stats = shelf.auto_flush(1);
    assert_eq!((stats.installed, stats.failed), (1, 1));
    let passed_over = state(shelf);
    assert_eq!(
        (passed_over.skipped.len(), passed_over.pending_since),
        (1, None)
    );
    assert_eq!(pending(shelf), 1);
    shelf.flush_if_overdue().unwrap();
    shelf.flush_if_overdue().unwrap();
    assert_eq!(sent().len(), 5);
    script.borrow_mut().refuse = None;
    KnowledgeService::new(&mut *shelf)
        .update("bad-entry", Content::new("fixed"))
        .unwrap();
    assert_eq!(shelf.auto_flush(1).installed, 1);
    assert_eq!(pending(shelf), 0);

    // Only a backfill that embeds something resumes paused flushes and starts over.
    write(shelf, ["k6".to_string()]);
    fail_next(BatchFailure::Transient("503".into()));
    shelf.auto_flush(1);
    shelf.record_backfill(0, 1).unwrap();
    let kept = state(shelf);
    assert!(kept.remote_batch.is_some() && kept.breaker.is_some() && !kept.skipped.is_empty());
    shelf.record_backfill(1, 0).unwrap();
    let cleared = state(shelf);
    assert_eq!(
        (cleared.remote_batch, cleared.breaker, cleared.skipped.len()),
        (None, None, 0)
    );
}

#[test]
fn entries_the_provider_fails_on_alone_are_passed_over() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().refuse = Some("bad");
    write(
        &mut shelf,
        ["bad-entry".to_string(), "good-entry".to_string()],
    );
    let stats = shelf.auto_flush(1);
    assert_eq!((stats.installed, stats.failed), (1, 1));
    let state = shelf.backend.flush_state().unwrap();
    assert_eq!((state.skipped.len(), state.breaker), (1, None));
    assert_eq!(shelf.auto_flush(1), FlushStats::default());
    // A passed-over entry does not count toward the write threshold.
    write(&mut shelf, (0..63).map(|i| format!("k{i}")));
    assert_eq!(script.borrow().batches, [2]);
    write(&mut shelf, ["k63".to_string()]);
    assert_eq!(script.borrow().batches, [2, 64]);
    // An explicit flush still tries it.
    assert_eq!(shelf.flush_pending(128).unwrap().failed, 1);
}

#[test]
fn provider_failures_pause_wrong_sizes_need_a_fix_and_bad_values_pass_over_one_entry() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().refuse = Some("entry");
    write(&mut shelf, ["entry-1".to_string(), "entry-2".to_string()]);
    shelf.auto_flush(1);
    let state = shelf.backend.flush_state().unwrap();
    assert!(
        state.skipped.is_empty(),
        "a failing provider is not the entries' fault"
    );
    assert!(!state.breaker.unwrap().permanent);

    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().wrong_size = true;
    write(&mut shelf, ["entry".to_string()]);
    shelf.auto_flush(1);
    assert!(shelf.embedding_debt().unwrap().paused.unwrap().permanent);

    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().garble = Some("entry");
    write(&mut shelf, ["entry".to_string()]);
    shelf.auto_flush(1);
    let state = shelf.backend.flush_state().unwrap();
    assert_eq!((state.skipped.len(), state.breaker), (1, None));

    // Past the cap, entries failing alone pause automatic flushes instead of churning.
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().refuse = Some("bad");
    write(&mut shelf, (0..40).map(|i| format!("bad-{i}")));
    write(&mut shelf, ["good".to_string()]);
    shelf.auto_flush(1);
    let state = shelf.backend.flush_state().unwrap();
    assert_eq!(state.skipped.len(), hypatia::storage::flush::MAX_SKIPPED);
    assert!(!state.breaker.unwrap().permanent);
}

#[test]
fn a_paused_flush_holds_back_writes_and_similar_leaves_remote_debt_alone() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Remote);
    write(&mut shelf, ["first".to_string()]);
    shelf.flush_before_similar();
    assert!(script.borrow().batches.is_empty());
    script
        .borrow_mut()
        .outcomes
        .push_back(Some(BatchFailure::Transient("503".into())));
    shelf.auto_flush(1);
    write(&mut shelf, (0..127).map(|i| format!("k{i}")));
    assert_eq!(
        script.borrow().batches,
        [1],
        "past the threshold, a paused flush still sends nothing"
    );
    assert_eq!(pending(&shelf), 128);
}

#[test]
fn only_entries_still_passed_over_count_toward_the_cap() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().refuse = Some("bad");
    let bad: Vec<String> = (0..32).map(|i| format!("bad-{i}")).collect();
    write(&mut shelf, bad.clone());
    write(&mut shelf, ["good".to_string()]);
    shelf.auto_flush(1);
    let debt = shelf.embedding_debt().unwrap();
    assert_eq!((debt.passed_over, debt.pending_since), (32, None));
    // Edited, they fail again as new versions, and the old versions no longer count.
    for name in &bad {
        KnowledgeService::new(&mut shelf)
            .update(name, Content::new("edited"))
            .unwrap();
    }
    write(&mut shelf, ["good-again".to_string()]);
    shelf.auto_flush(1);
    let state = shelf.backend.flush_state().unwrap();
    assert_eq!((state.skipped.len(), state.breaker), (32, None));
    assert_eq!(shelf.embedding_debt().unwrap().passed_over, 32);
}

#[test]
fn a_batch_of_nothing_but_invalid_vectors_pauses_without_passing_over() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Local);
    script.borrow_mut().garble = Some("entry");
    write(&mut shelf, ["entry-1".to_string(), "entry-2".to_string()]);
    shelf.auto_flush(1);
    let state = shelf.backend.flush_state().unwrap();
    assert!(state.skipped.is_empty());
    assert!(!state.breaker.unwrap().permanent);
}

#[test]
fn a_batch_size_is_not_learned_while_an_entry_fails_alone() {
    let dir = TempDir::new().unwrap();
    let mut shelf = local(&dir);
    let script = scripted(&mut shelf, ProviderKind::Remote);
    script.borrow_mut().refuse = Some("bad");
    script.borrow_mut().accepted_size = Some(1);
    write(
        &mut shelf,
        ["bad-entry".to_string(), "good-entry".to_string()],
    );
    assert_eq!(shelf.auto_flush(1).installed, 1);
    let state = shelf.backend.flush_state().unwrap();
    assert_eq!((state.skipped.len(), state.remote_batch), (1, None));
}

#[test]
fn opting_out_of_deferral_never_embeds_a_vector_that_would_be_refused() {
    let dir = TempDir::new().unwrap();
    // No model name and no model files: a vector would have no identity to be stored under.
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\ndimensions=3\ndefer=false\n",
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("contract")).unwrap();
    let script = scripted(&mut shelf, ProviderKind::Local);
    write(&mut shelf, ["entry".to_string()]);
    assert_eq!(script.borrow().embeds, 0);
    assert_eq!(pending(&shelf), 1);
}

#[test]
fn opting_out_of_deferral_embeds_each_write() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\ndefer=false\n",
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("contract")).unwrap();
    shelf.embedder = Box::new(Unit);
    write(&mut shelf, ["now".to_string()]);
    assert_eq!(pending(&shelf), 0);
    assert_eq!(shelf.backend.embedding_row_count("knowledge").unwrap(), 1);
}

#[test]
fn concurrent_meta_updates_are_never_lost() {
    let dir = TempDir::new().unwrap();
    drop(local(&dir));
    let path = dir.path().join("hypatia.sqlite");
    let stores: Vec<_> = (0..4)
        .map(|_| hypatia::storage::SqliteStore::open(&path).unwrap())
        .collect();
    std::thread::scope(|scope| {
        for store in stores {
            scope.spawn(move || {
                for _ in 0..25 {
                    store
                        .update_meta_value("counter", |current| {
                            let n: u32 = current.map_or(0, |v| v.parse().unwrap());
                            Ok(Some((n + 1).to_string()))
                        })
                        .unwrap();
                }
            });
        }
    });
    let store = hypatia::storage::SqliteStore::open(&path).unwrap();
    assert_eq!(store.meta_value("counter").unwrap().as_deref(), Some("100"));
}

fn contract(shelf: &mut OpenShelf) {
    let initial = Content::new(r#"{"n":12,"mixed":[1,"rust",null],"nested":{"x":true}}"#)
        .with_tags(vec!["rust".into()]);
    let version = shelf.backend.insert_knowledge("first", &initial).unwrap();
    assert!(shelf.backend.insert_knowledge("first", &initial).is_err());
    assert!(
        shelf
            .backend
            .install_embedding("knowledge", "first", version, &[1., 0., 0.])
            .unwrap()
    );
    let hits = shelf
        .backend
        .vector_search(QueryTarget::Knowledge, &[1., 0., 0.], 10)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].2.abs() < 1e-5);
    for v in [
        vec![0., 0., 0.],
        vec![1., 2.],
        vec![f32::NAN, 1., 0.],
        vec![f32::INFINITY, 0., 1.],
    ] {
        assert!(
            shelf
                .backend
                .install_embedding("knowledge", "first", version, &v)
                .is_err()
        );
    }
    for expression in [
        json!(["$knowledge", ["$has", "tags", "rust"]]),
        json!(["$knowledge", ["$eq", "$name", "first"]]),
        json!(["$knowledge", ["$has", "data.mixed", 1]]),
    ] {
        assert_eq!(
            Evaluator::execute(&expression, shelf).unwrap().rows.len(),
            1,
            "{expression}"
        );
    }
    let next = shelf
        .backend
        .update_knowledge("first", &Content::new("更新正文 storage"))
        .unwrap();
    assert_ne!(version, next);
    assert!(
        !shelf
            .backend
            .install_embedding("knowledge", "first", version, &[1., 0., 0.])
            .unwrap()
    );
    assert!(
        shelf
            .backend
            .vector_search(QueryTarget::Knowledge, &[1., 0., 0.], 10)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        shelf
            .backend
            .missing_embeddings("knowledge", None, 1)
            .unwrap()[0]
            .2,
        next
    );
    assert!(
        shelf
            .backend
            .install_embedding("knowledge", "first", next, &[0., 1., 0.])
            .unwrap()
    );
    shelf.backend.rebuild_indexes().unwrap();
    assert_eq!(
        shelf
            .execute_search("storage", &SearchOpts::default())
            .unwrap()
            .rows
            .len(),
        1
    );
    assert_eq!(
        shelf
            .execute_search("正文", &SearchOpts::default())
            .unwrap()
            .rows
            .len(),
        1
    );
    let a = StatementKey::new("first", "links", "second");
    let b = StatementKey::new("second", "links", "first");
    let date = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
        .unwrap()
        .and_hms_opt(1, 2, 3)
        .unwrap();
    shelf
        .backend
        .insert_statement(&a, &Content::new("edge"), Some(date), None)
        .unwrap();
    shelf
        .backend
        .insert_statement(&b, &Content::new("cycle"), None, None)
        .unwrap();
    assert_eq!(
        shelf.backend.get_statement(&a).unwrap().unwrap().tr_start,
        Some(date)
    );
    assert_eq!(
        shelf
            .backend
            .query_khop("first", Some("links"), 3)
            .unwrap()
            .len(),
        2
    );
    shelf.backend.delete_knowledge("first").unwrap();
    assert!(shelf.backend.delete_knowledge("first").is_err());
    let recreated = shelf.backend.insert_knowledge("first", &initial).unwrap();
    assert_ne!(recreated, next);
    assert!(
        !shelf
            .backend
            .install_embedding("knowledge", "first", next, &[0., 1., 0.])
            .unwrap()
    );
    assert!(
        shelf
            .backend
            .vector_search(QueryTarget::Knowledge, &[1., 0., 0.], 10)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn sqlite_shared_contract() {
    let dir = TempDir::new().unwrap();
    contract(&mut local(&dir));
}
#[test]
fn invalid_config_never_creates_local_storage() {
    for config in [
        "[storage]\nbackend='pgvector'\n",
        "[storage]\nbacked='pgvector'\n",
        "not toml",
    ] {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("shelf.toml"), config).unwrap();
        assert!(OpenShelf::open(dir.path(), None).is_err());
        assert!(!dir.path().join("hypatia.sqlite").exists());
        assert!(!dir.path().join("vectors").exists());
    }
}
#[test]
fn stale_sqlite_writer_cannot_reinstall_after_delete_or_update() {
    let dir = TempDir::new().unwrap();
    let mut one = local(&dir);
    let two = OpenShelf::open(dir.path(), None).unwrap();
    let version = one
        .backend
        .insert_knowledge("key", &Content::new("old"))
        .unwrap();
    one.backend
        .install_embedding("knowledge", "key", version, &[1., 0., 0.])
        .unwrap();
    one.backend.rebuild_indexes().unwrap();
    two.backend
        .update_knowledge("key", &Content::new("new"))
        .unwrap();
    assert!(
        !one.backend
            .install_embedding("knowledge", "key", version, &[1., 0., 0.])
            .unwrap()
    );
    assert!(
        one.backend
            .vector_search(QueryTarget::Knowledge, &[1., 0., 0.], 1)
            .unwrap()
            .is_empty()
    );
}
#[test]
fn logical_export_import_preserves_content_time_vectors_and_archives() {
    let home = TempDir::new().unwrap();
    let source = TempDir::new().unwrap();
    let target = TempDir::new().unwrap();
    let export = TempDir::new().unwrap();
    for dir in [&source, &target] {
        std::fs::write(
            dir.path().join("shelf.toml"),
            "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
        )
        .unwrap();
    }
    let mut mgr = ShelfManager::with_home(home.path().into()).unwrap();
    mgr.connect(source.path(), Some("source")).unwrap();
    mgr.connect(target.path(), Some("target")).unwrap();
    let src = mgr.get_mut("source").unwrap();
    std::fs::write(src.config.archives_path.join("figure.txt"), "attachment").unwrap();
    let version = src
        .backend
        .insert_knowledge(
            "key",
            &Content::new("body").with_figures(vec!["archive://figure.txt".into()]),
        )
        .unwrap();
    src.backend
        .install_embedding("knowledge", "key", version, &[1., 0., 0.])
        .unwrap();
    let expected = src.backend.snapshot().unwrap();
    mgr.export("source", export.path()).unwrap();
    assert!(export.path().join("hypatia.sqlite").exists());
    mgr.import("target", export.path(), false).unwrap();
    assert_eq!(
        mgr.get("target").unwrap().backend.snapshot().unwrap(),
        expected
    );
    assert_eq!(
        std::fs::read(target.path().join("archives/figure.txt")).unwrap(),
        b"attachment"
    );
    assert!(mgr.import("target", export.path(), false).is_err());
}
#[cfg(not(feature = "postgres-backend"))]
#[test]
fn missing_feature_is_explicit() {
    let dir = TempDir::new().unwrap();
    std::fs::write(dir.path().join("shelf.toml"),"[storage]\nbackend='pgvector'\n[storage.postgres]\nurl_env='HYPATIA_TEST_POSTGRES_URL'\nschema='feature_test'\n[embedding]\nmodel='hypatia-contract-test'\n").unwrap();
    let err = OpenShelf::open(dir.path(), None).err().unwrap().to_string();
    assert!(err.contains("--features postgres-backend"));
    assert!(!dir.path().join("hypatia.sqlite").exists());
}

#[cfg(feature = "postgres-backend")]
#[test]
#[ignore = "requires disposable pgvector database"]
fn sqlite_pg_sqlite_migration_includes_new_pg_writes() {
    std::env::var("HYPATIA_TEST_POSTGRES_URL").expect("HYPATIA_TEST_POSTGRES_URL required");
    let root = TempDir::new().unwrap();
    let source = root.path().join("source");
    let pgdir = root.path().join("pg");
    let returned = root.path().join("returned");
    for dir in [&source, &pgdir, &returned] {
        std::fs::create_dir_all(dir).unwrap();
    }
    let model = "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n";
    std::fs::write(source.join("shelf.toml"), model).unwrap();
    std::fs::write(returned.join("shelf.toml"), model).unwrap();
    let schema = format!(
        "roundtrip_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    std::fs::write(pgdir.join("shelf.toml"),format!("[storage]\nbackend='pgvector'\n[storage.postgres]\nurl_env='HYPATIA_TEST_POSTGRES_URL'\nschema='{schema}'\n{model}")).unwrap();
    let mut mgr = ShelfManager::with_home(root.path().join("home")).unwrap();
    mgr.connect(&source, Some("source")).unwrap();
    mgr.connect(&pgdir, Some("pg")).unwrap();
    mgr.connect(&returned, Some("returned")).unwrap();
    let src = mgr.get_mut("source").unwrap();
    std::fs::write(
        src.config.archives_path.join("figure.txt"),
        b"local attachment",
    )
    .unwrap();
    let v = src
        .backend
        .insert_knowledge(
            "original",
            &Content::new("original text").with_figures(vec!["archive://figure.txt".into()]),
        )
        .unwrap();
    src.backend
        .install_embedding("knowledge", "original", v, &[1., 0., 0.])
        .unwrap();
    let key = StatementKey::new("original", "linked", "later");
    src.backend
        .insert_statement(&key, &Content::new("relation"), None, None)
        .unwrap();
    let first = root.path().join("to-pg");
    mgr.export("source", &first).unwrap();
    mgr.import("pg", &first, false).unwrap();
    let pg = mgr.get_mut("pg").unwrap();
    let v = pg
        .backend
        .insert_knowledge("added-in-pg", &Content::new("after switch"))
        .unwrap();
    pg.backend
        .install_embedding("knowledge", "added-in-pg", v, &[0., 1., 0.])
        .unwrap();
    let second = root.path().join("to-sqlite");
    mgr.export("pg", &second).unwrap();
    assert!(!second.join("hypatia.sqlite").exists());
    assert!(!second.join("shelf.toml").exists());
    mgr.import("returned", &second, false).unwrap();
    assert!(
        mgr.get("returned")
            .unwrap()
            .backend
            .get_knowledge("added-in-pg")
            .unwrap()
            .is_some()
    );
    assert!(
        mgr.get("source")
            .unwrap()
            .backend
            .get_knowledge("added-in-pg")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        mgr.get("returned")
            .unwrap()
            .backend
            .embedding_row_count("knowledge")
            .unwrap(),
        2
    );
    assert_eq!(
        std::fs::read(returned.join("archives/figure.txt")).unwrap(),
        b"local attachment"
    );
    // Tampering is rejected before target data is written.
    std::fs::write(second.join("archives/figure.txt"), b"tampered").unwrap();
    let fresh = root.path().join("fresh");
    std::fs::create_dir_all(&fresh).unwrap();
    std::fs::write(fresh.join("shelf.toml"), model).unwrap();
    mgr.connect(&fresh, Some("fresh")).unwrap();
    assert!(
        mgr.import("fresh", &second, false)
            .unwrap_err()
            .to_string()
            .contains("checksum")
    );
    assert!(
        mgr.get("fresh")
            .unwrap()
            .backend
            .snapshot()
            .unwrap()
            .knowledge
            .is_empty()
    );
    drop(mgr);
    let mut admin = postgres::Client::connect(
        &std::env::var("HYPATIA_TEST_POSTGRES_URL").unwrap(),
        postgres::NoTls,
    )
    .unwrap();
    admin
        .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
        .unwrap();
}

#[cfg(feature = "postgres-backend")]
#[test]
#[ignore = "requires HYPATIA_TEST_POSTGRES_URL pointing at disposable pgvector database"]
fn postgres_shared_contract_and_mixed_shelf_isolation() {
    std::env::var("HYPATIA_TEST_POSTGRES_URL").expect("HYPATIA_TEST_POSTGRES_URL required");
    let home = TempDir::new().unwrap();
    let one = TempDir::new().unwrap();
    let two = TempDir::new().unwrap();
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let schemas = [
        format!("contract_{suffix}_a"),
        format!("contract_{suffix}_b"),
    ];
    let direct_url = std::env::var("HYPATIA_TEST_POSTGRES_URL").unwrap();
    for (dir, schema) in [(&one, &schemas[0]), (&two, &schemas[1])] {
        let connection = if dir.path() == one.path() {
            "url_env='HYPATIA_TEST_POSTGRES_URL'\n".to_string()
        } else {
            toml::to_string(&std::collections::BTreeMap::from([("url", &direct_url)])).unwrap()
        };
        std::fs::write(dir.path().join("shelf.toml"),format!("[storage]\nbackend='pgvector'\n[storage.postgres]\n{connection}schema='{schema}'\n[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n")).unwrap();
    }
    let mut mgr = ShelfManager::with_home(home.path().into()).unwrap();
    mgr.connect(one.path(), Some("pg-a")).unwrap();
    mgr.connect(two.path(), Some("pg-b")).unwrap();
    contract(mgr.get_mut("pg-a").unwrap());
    mgr.get("pg-b")
        .unwrap()
        .backend
        .insert_knowledge("first", &Content::new("isolated"))
        .unwrap();
    assert_eq!(
        mgr.get("pg-b")
            .unwrap()
            .backend
            .get_knowledge("first")
            .unwrap()
            .unwrap()
            .content
            .data,
        "isolated"
    );
    assert!(
        mgr.get("default")
            .unwrap()
            .backend
            .get_knowledge("first")
            .unwrap()
            .is_none()
    );
    let export = TempDir::new().unwrap();
    mgr.export("pg-b", export.path()).unwrap();
    assert!(!export.path().join("shelf.toml").exists());
    for name in ["snapshot.json", "manifest.json"] {
        assert!(
            !std::fs::read_to_string(export.path().join(name))
                .unwrap()
                .contains(&direct_url)
        );
    }
    for dir in [&one, &two] {
        assert!(!dir.path().join("hypatia.sqlite").exists());
        assert!(!dir.path().join("vectors").exists());
    }
    drop(mgr);
    let mut client = postgres::Client::connect(
        &std::env::var("HYPATIA_TEST_POSTGRES_URL").unwrap(),
        postgres::NoTls,
    )
    .unwrap();
    for schema in schemas {
        client
            .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
            .unwrap();
    }
}

#[cfg(feature = "postgres-backend")]
#[test]
#[ignore = "requires HYPATIA_TEST_POSTGRES_URL pointing at disposable pgvector database"]
fn postgres_automatic_flush_transitions() {
    let url =
        std::env::var("HYPATIA_TEST_POSTGRES_URL").expect("HYPATIA_TEST_POSTGRES_URL required");
    let dir = TempDir::new().unwrap();
    let schema = format!(
        "flush_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    std::fs::write(dir.path().join("shelf.toml"), format!("[storage]\nbackend='pgvector'\n[storage.postgres]\nurl_env='HYPATIA_TEST_POSTGRES_URL'\nschema='{schema}'\n[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n")).unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("pg-flush")).unwrap();
    flush_transitions(&mut shelf);
    drop(shelf);
    postgres::Client::connect(&url, postgres::NoTls)
        .unwrap()
        .batch_execute(&format!("DROP SCHEMA \"{schema}\" CASCADE"))
        .unwrap();
}

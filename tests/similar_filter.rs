//! `similar` narrowed by a JSE filter: the filter applies before ranking, so the
//! nearest qualifying entries come back however many nearer ones it excludes.
use std::{cell::Cell, rc::Rc};

use hypatia::{
    engine::filter::{compile, similar_filter},
    model::{Content, QueryTarget, SearchOpts, StatementKey},
    storage::OpenShelf,
};
use serde_json::{Value, json};
use tempfile::TempDir;

/// Embeds every query to the same vector and counts the calls.
struct Query(Rc<Cell<usize>>);
impl hypatia::embedding::EmbeddingProvider for Query {
    fn embed(&self, _: &str) -> Result<Vec<f32>, hypatia::error::HypatiaError> {
        self.0.set(self.0.get() + 1);
        Ok(vec![1., 0., 0.])
    }
    fn dimensions(&self) -> usize {
        3
    }
    fn is_available(&self) -> bool {
        true
    }
}

fn knowledge(shelf: &mut OpenShelf, name: &str, content: Content, vector: [f32; 3]) {
    let version = shelf.backend.insert_knowledge(name, &content).unwrap();
    assert!(
        shelf
            .backend
            .install_embedding("knowledge", name, version, &vector)
            .unwrap()
    );
}

fn names(shelf: &OpenShelf, limit: i64, target: QueryTarget, filter: Option<Value>) -> Vec<String> {
    let opts = SearchOpts {
        limit,
        ..SearchOpts::default()
    };
    shelf
        .similar_where("anything", &opts, target, filter.as_ref())
        .unwrap()
        .rows
        .iter()
        .map(|row| row[target.key_column()].as_str().unwrap().to_string())
        .collect()
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

/// A shelf whose session log sits nearer every query than its knowledge: forty
/// messages at the query itself, and six rules further out, rule-1 nearest.
fn session_log_shelf(shelf: &mut OpenShelf) -> Rc<Cell<usize>> {
    for i in 1..=40 {
        knowledge(
            shelf,
            &format!("msg-{i:02}"),
            Content::new("what the user said").with_tags(vec!["message".into()]),
            [1., i as f32 * 0.01, 0.],
        );
    }
    for j in 1..=6 {
        let content = Content::new("a rule distilled from it").with_tags(vec!["rule".into()]);
        let content = if j <= 2 {
            content.with_scopes(vec!["project-a".into()])
        } else {
            content
        };
        knowledge(
            shelf,
            &format!("rule-{j}"),
            content,
            [1., 0.5 + j as f32 * 0.1, 0.],
        );
    }
    for (relation, tail, vector) in [
        ("knows", "Bob", [1., 0.001, 0.]),
        ("likes", "Tea", [1., 0.5, 0.]),
    ] {
        let key = StatementKey::new("Alice", relation, tail);
        let version = shelf
            .backend
            .insert_statement(&key, &Content::new(""), None, None)
            .unwrap()
            .unwrap();
        assert!(
            shelf
                .backend
                .install_embedding("statement", &key.to_csv_key(), version, &vector)
                .unwrap()
        );
    }
    let calls = Rc::new(Cell::new(0));
    shelf.embedder = Box::new(Query(calls.clone()));
    calls
}

fn contract(shelf: &mut OpenShelf) {
    let calls = session_log_shelf(shelf);
    let k = QueryTarget::Knowledge;
    let rules = |n: usize| (1..=n).map(|j| format!("rule-{j}")).collect::<Vec<_>>();

    // The issue: unfiltered, the session log takes every place.
    assert_eq!(
        names(shelf, 5, k, None),
        ["msg-01", "msg-02", "msg-03", "msg-04", "msg-05"]
    );
    // Excluding it, or asking for the rules, gives the nearest rules, nearest first.
    let exclude = similar_filter(&[], &strings(&["message", "summary", "session"]), None);
    assert_eq!(names(shelf, 5, k, exclude.clone()), rules(5));
    assert_eq!(
        names(shelf, 5, k, similar_filter(&strings(&["rule"]), &[], None)),
        rules(5)
    );
    // Fewer qualify than the limit: all of them, and nothing else.
    assert_eq!(names(shelf, 10, k, exclude), rules(6));
    // Any JSE condition narrows the same way.
    assert_eq!(
        names(
            shelf,
            5,
            k,
            Some(json!(["$contains", "scopes", "project-a"]))
        ),
        rules(2)
    );
    assert_eq!(
        names(
            shelf,
            3,
            k,
            similar_filter(&[], &[], Some(json!(["$like", "name", "rule-%"])))
        ),
        rules(3)
    );
    assert!(names(shelf, 5, k, Some(json!(["$contains", "tags", "absent"]))).is_empty());
    // Statements are filtered by their own columns.
    assert_eq!(
        names(
            shelf,
            5,
            QueryTarget::Statement,
            Some(json!(["$eq", "relation", "likes"]))
        ),
        [StatementKey::new("Alice", "likes", "Tea").to_csv_key()]
    );

    // A filter that is not a condition fails before it costs an embedding.
    let before = calls.get();
    for bad in [
        json!("message"),
        json!(["$knowledge", ["$contains", "tags", "rule"]]),
        json!(["$similar", "rules"]),
        json!(["$not", ["$search", "rules"]]),
    ] {
        assert!(
            shelf
                .similar_where("anything", &SearchOpts::default(), k, Some(&bad))
                .is_err(),
            "{bad}"
        );
    }
    // So does one naming a field the target's table lacks, in either direction.
    for (target, field) in [
        (QueryTarget::Statement, json!(["$like", "name", "rule-%"])),
        (k, json!(["$eq", "relation", "likes"])),
    ] {
        let err = shelf
            .similar_where("anything", &SearchOpts::default(), target, Some(&field))
            .unwrap_err()
            .to_string();
        assert!(err.contains("entries do not have"), "{field}: {err}");
    }
    assert_eq!(calls.get(), before);
}

#[test]
fn sqlite_similar_filter_contract() {
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("similar-filter")).unwrap();
    contract(&mut shelf);

    // The approximate index and the exact scan it falls back to agree.
    let filter = compile(
        &similar_filter(&[], &strings(&["message"]), None).unwrap(),
        QueryTarget::Knowledge,
        &shelf,
    )
    .unwrap();
    let store = hypatia::storage::SqliteStore::open(&dir.path().join("hypatia.sqlite")).unwrap();
    let exact: Vec<_> = store
        .vector_search_where(QueryTarget::Knowledge, &[1., 0., 0.], 5, &filter)
        .unwrap()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    assert_eq!(
        exact,
        names(
            &shelf,
            5,
            QueryTarget::Knowledge,
            Some(json!(["$not", ["$contains", "tags", "message"]]))
        )
    );
    assert_eq!(
        store
            .filtered_doc_ids(QueryTarget::Knowledge, &filter)
            .unwrap()
            .len(),
        6
    );
}

#[test]
fn a_filter_that_crowds_the_index_out_still_fills_the_limit() {
    // Hundreds of excluded vectors surround the query. usearch keeps walking past them
    // until it has the qualifying ones, so the index fills the limit by itself; the
    // exact fallback for a walk that does not is tested in `storage::backend`.
    let dir = TempDir::new().unwrap();
    std::fs::write(
        dir.path().join("shelf.toml"),
        "[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n",
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("crowded")).unwrap();
    for i in 0..600 {
        let angle = i as f32 * 0.0005;
        knowledge(
            &mut shelf,
            &format!("msg-{i:03}"),
            Content::new("chatter").with_tags(vec!["message".into()]),
            [angle.cos(), angle.sin(), 0.],
        );
    }
    for j in 0..8 {
        let angle = 1.2 + j as f32 * 0.01;
        knowledge(
            &mut shelf,
            &format!("rule-{j}"),
            Content::new("rule").with_tags(vec!["rule".into()]),
            [angle.cos(), 0., angle.sin()],
        );
    }
    shelf.embedder = Box::new(Query(Rc::new(Cell::new(0))));
    let found = names(
        &shelf,
        8,
        QueryTarget::Knowledge,
        similar_filter(&[], &strings(&["message"]), None),
    );
    assert_eq!(
        found,
        (0..8).map(|j| format!("rule-{j}")).collect::<Vec<_>>()
    );
}

#[cfg(feature = "postgres-backend")]
#[test]
#[ignore = "requires HYPATIA_TEST_POSTGRES_URL pointing at disposable pgvector database"]
fn postgres_similar_filter_contract() {
    std::env::var("HYPATIA_TEST_POSTGRES_URL").expect("HYPATIA_TEST_POSTGRES_URL required");
    let dir = TempDir::new().unwrap();
    let schema = format!(
        "similar_filter_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    // HNSW, the default index, is the path that can come back short.
    std::fs::write(
        dir.path().join("shelf.toml"),
        format!("[storage]\nbackend='pgvector'\n[storage.postgres]\nurl_env='HYPATIA_TEST_POSTGRES_URL'\nschema='{schema}'\n[embedding]\nmodel='hypatia-contract-test'\ndimensions=3\n"),
    )
    .unwrap();
    let mut shelf = OpenShelf::open(dir.path(), Some("pg-similar-filter")).unwrap();
    contract(&mut shelf);
}

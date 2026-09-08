//! Shared domain contracts, including a real optional PostgreSQL target.
use hypatia::{
    engine::Evaluator,
    model::{Content, QueryTarget, SearchOpts, StatementKey},
    storage::{OpenShelf, ShelfManager, Storage},
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

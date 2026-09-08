#![cfg(feature = "postgres-backend")]
//! Explicit live PostgreSQL tests: run with HYPATIA_TEST_POSTGRES_URL set and --ignored.
//! Missing configuration is a failure, never a successful silent skip.
use hypatia::{
    embedding::EmbeddingConfig,
    model::{Content, SearchOpts, StatementKey},
    storage::{
        postgres_store::PgStore,
        settings::{PostgresSettings, ShelfSettings, VectorSettings},
    },
};
use postgres::{Client, NoTls};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
#[test]
fn invalid_direct_url_does_not_echo_credentials() {
    let settings = ShelfSettings::parse(
        "[storage]\nbackend='pgvector'\n[storage.postgres]\nschema='redaction_test'\nurl='postgres://user:secret-test-password@localhost/db?sslmode=unsupported'\n[embedding]\nmodel='test-model'\n",
        std::path::Path::new("."),
    ).unwrap();
    let err = PgStore::open(
        settings.storage.postgres.as_ref().unwrap(),
        &settings.storage.vector,
        &settings.embedding,
    )
    .err()
    .unwrap();
    assert!(
        err.to_string()
            .contains("Invalid PostgreSQL connection configuration")
    );
    assert!(!format!("{err:?}").contains("secret-test-password"));
}

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    config: PostgresSettings,
    embedding: EmbeddingConfig,
    vector: VectorSettings,
}
impl Fixture {
    fn new(index: &str) -> Self {
        std::env::var("HYPATIA_TEST_POSTGRES_URL").expect("set HYPATIA_TEST_POSTGRES_URL to a disposable PostgreSQL database with pgvector installed");
        let settings = ShelfSettings::parse(
            "[embedding]\nprovider='remote'\napi_model='pg-store-test'\ndimensions=3",
            std::path::Path::new("."),
        )
        .unwrap();
        Self {
            config: PostgresSettings {
                url_env: Some("HYPATIA_TEST_POSTGRES_URL".into()),
                url: None,
                schema: format!(
                    "pgstore_test_{}_{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ),
                connect_timeout_seconds: 5,
                statement_timeout_ms: 10000,
            },
            embedding: settings.embedding,
            vector: VectorSettings {
                index: index.into(),
                metric: "cosine".into(),
            },
        }
    }
    fn open(&self) -> PgStore {
        PgStore::open(&self.config, &self.vector, &self.embedding).unwrap()
    }
    fn admin(&self) -> Client {
        Client::connect(&std::env::var("HYPATIA_TEST_POSTGRES_URL").unwrap(), NoTls).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if let Ok(url) = std::env::var("HYPATIA_TEST_POSTGRES_URL") {
            if let Ok(mut c) = Client::connect(&url, NoTls) {
                let _ = c.batch_execute(&format!(
                    "DROP SCHEMA IF EXISTS \"{}\" CASCADE",
                    self.config.schema
                ));
            }
        }
    }
}
#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn crud_versions_fts_vectors_and_isolation() {
    let f = Fixture::new("hnsw");
    let a = f.open();
    let b = f.open();
    let v1 = a
        .insert_knowledge("stable", &Content::new("oldword 北京知识图谱"))
        .unwrap();
    let id = a.doc_id_by_key("knowledge", "stable").unwrap().unwrap();
    assert!(
        b.install_embedding("knowledge", "stable", v1, &[1., 0., 0.])
            .unwrap()
    );
    let v2 = a
        .update_knowledge("stable", &Content::new("newword 北京知识图谱"))
        .unwrap();
    assert_ne!(v1, v2);
    assert_eq!(a.embedding_row_count("knowledge").unwrap(), 0);
    assert!(
        !b.install_embedding("knowledge", "stable", v1, &[1., 0., 0.])
            .unwrap()
    );
    assert_eq!(a.doc_id_by_key("knowledge", "stable").unwrap(), Some(id));
    assert!(
        a.search("oldword", &SearchOpts::default())
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        a.search("newword", &SearchOpts::default()).unwrap().len(),
        1
    );
    assert_eq!(a.search("北京", &SearchOpts::default()).unwrap().len(), 1);
    assert_eq!(
        a.search("missing OR newword", &SearchOpts::default())
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        a.search("newword AND 北京", &SearchOpts::default())
            .unwrap()
            .len(),
        1
    );
    assert!(a.search("newword", &SearchOpts::default()).unwrap()[0].rank < 0.0);
    assert!(
        a.install_embedding("knowledge", "stable", v2, &[1., 0., 0.])
            .unwrap()
    );
    assert!(
        a.install_embedding("knowledge", "stable", v2, &[0., 0., 0.])
            .is_err()
    );
    assert!(
        a.install_embedding("knowledge", "stable", v2, &[f32::NAN, 0., 1.])
            .is_err()
    );
    assert!(
        a.install_embedding("knowledge", "stable", v2, &[1., 0.])
            .is_err()
    );
    let other = a
        .insert_knowledge("opposite", &Content::new("other"))
        .unwrap();
    a.install_embedding("knowledge", "opposite", other, &[-1., 0., 0.])
        .unwrap();
    let hits = b.vector_search_knowledge(&[1., 0., 0.], 2).unwrap();
    assert_eq!(hits[0].0, "stable");
    assert!(hits[0].2.abs() < 1e-6);
    assert!((hits[1].2 - 2.0).abs() < 1e-6);
    a.delete_knowledge("stable").unwrap();
    let v3 = a
        .insert_knowledge("stable", &Content::new("recreated"))
        .unwrap();
    assert_ne!(v2, v3);
    assert!(
        !b.install_embedding("knowledge", "stable", v2, &[1., 0., 0.])
            .unwrap()
    );
    assert_ne!(a.doc_id_by_key("knowledge", "stable").unwrap(), Some(id));
    let isolated = Fixture::new("none");
    assert!(isolated.open().get_knowledge("stable").unwrap().is_none());
    assert_eq!(
        a.missing_embeddings("knowledge", None, 1).unwrap()[0].0,
        "stable"
    );
    assert!(
        a.missing_embeddings("knowledge", Some("stable"), 1)
            .unwrap()
            .is_empty()
    );
    a.rebuild_indexes().unwrap();
    assert_eq!(
        a.search("recreated", &SearchOpts::default()).unwrap().len(),
        1
    );
}
#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn graph_native_parameters_payload_and_snapshots() {
    let f = Fixture::new("none");
    let a = f.open();
    let content = Content::new(r#"{"n":3,"tokens":[1e-7,"x"]}"#);
    a.insert_knowledge("json", &content).unwrap();
    let sql = format!(
        "SELECT name,content,created_at FROM \"{}\".knowledge WHERE payload->>'n'=$1::text AND $2::float8=3 AND $3::bool AND $4::int4=4 LIMIT $5::int8",
        f.config.schema
    );
    assert_eq!(
        a.query_knowledge(
            &sql,
            vec![json!("3"), json!(3), json!(true), json!(4), json!(10)]
        )
        .unwrap()[0]
            .content,
        content
    );
    assert!(
        a.query_knowledge(
            &sql,
            vec![
                json!("3"),
                json!("not number"),
                json!(true),
                json!(4),
                json!(10)
            ]
        )
        .is_err()
    );
    let dt =
        chrono::NaiveDateTime::parse_from_str("2024-01-02 03:04:05", "%Y-%m-%d %H:%M:%S").unwrap();
    let ab = StatementKey::new("a", "r", "b");
    let bc = StatementKey::new("b", "r", "c");
    let ca = StatementKey::new("c", "r", "a");
    let token = a
        .insert_statement(&ab, &Content::new("edge"), Some(dt), None)
        .unwrap();
    a.install_embedding("statement", &ab.to_csv_key(), token, &[0., 1., 0.])
        .unwrap();
    a.insert_statement(&bc, &Content::new("edge2"), None, None)
        .unwrap();
    a.insert_statement(&ca, &Content::new("cycle"), None, None)
        .unwrap();
    assert_eq!(a.query_khop("a", Some("r"), 2).unwrap().len(), 2);
    assert_eq!(a.query_khop("a", None, 10).unwrap().len(), 3);
    assert!(a.query_khop("a", None, 0).unwrap().is_empty());
    assert_eq!(a.get_statement(&ab).unwrap().unwrap().tr_start, Some(dt));
    let snap = a.snapshot().unwrap();
    let dest = Fixture::new("none");
    let target = dest.open();
    target.import_snapshot(&snap).unwrap();
    assert_eq!(target.snapshot().unwrap(), snap);
    assert!(target.import_snapshot(&snap).is_err());
    // Database rejects NUL text after the first valid record; whole import rolls back.
    let rollback = Fixture::new("none");
    let empty = rollback.open();
    let mut bad = snap.clone();
    bad.knowledge.push(snap.knowledge[0].clone());
    bad.knowledge.last_mut().unwrap().knowledge.name = "bad\0key".into();
    assert!(empty.import_snapshot(&bad).is_err());
    assert!(empty.snapshot().unwrap().knowledge.is_empty());
    assert!(empty.snapshot().unwrap().statements.is_empty());
    let mut wrong = snap.clone();
    wrong.embedding.as_mut().unwrap().model = "wrong-model".into();
    assert!(empty.import_snapshot(&wrong).is_err());
}
#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn metadata_validation_and_concurrent_initialization() {
    let f = Fixture::new("hnsw");
    std::thread::scope(|scope| {
        let threads: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    drop(f.open());
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    });
    let a = f.open();
    let mut changed = f.embedding.clone();
    changed.remote.dimensions = 4;
    assert!(PgStore::open(&f.config, &f.vector, &changed).is_err());
    changed = f.embedding.clone();
    changed.model_identity = "different-provider-model".into();
    assert!(PgStore::open(&f.config, &f.vector, &changed).is_err());
    let mut admin = f.admin();
    admin
        .batch_execute(&format!(
            "ALTER TABLE \"{}\".knowledge ALTER COLUMN embedding TYPE public.vector(4)",
            f.config.schema
        ))
        .unwrap();
    assert!(PgStore::open(&f.config, &f.vector, &f.embedding).is_err());
    admin
        .batch_execute(&format!(
            "ALTER TABLE \"{}\".knowledge ALTER COLUMN embedding TYPE public.vector(3)",
            f.config.schema
        ))
        .unwrap();
    admin
        .batch_execute(&format!(
            "ALTER TABLE \"{}\".knowledge DROP COLUMN tokens",
            f.config.schema
        ))
        .unwrap();
    assert!(PgStore::open(&f.config, &f.vector, &f.embedding).is_err());
    drop(a);
    let mut hnsw = f.embedding.clone();
    hnsw.remote.dimensions = 2001;
    assert!(PgStore::open(&f.config, &f.vector, &hnsw).is_err());
}

#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn concurrent_embedding_install_cannot_survive_content_update() {
    let f = Fixture::new("none");
    let store = f.open();
    for iteration in 0..16 {
        let key = format!("race-{iteration}");
        let token = store.insert_knowledge(&key, &Content::new("old")).unwrap();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let install = scope.spawn(|| {
                let c = f.open();
                barrier.wait();
                c.install_embedding("knowledge", &key, token, &[1., 0., 0.])
                    .unwrap()
            });
            let update = scope.spawn(|| {
                let c = f.open();
                barrier.wait();
                c.update_knowledge(&key, &Content::new("new")).unwrap()
            });
            install.join().unwrap();
            update.join().unwrap();
        });
        assert_eq!(
            store.get_knowledge(&key).unwrap().unwrap().content.data,
            "new"
        );
        assert_eq!(store.embedding_row_count("knowledge").unwrap(), 0);
    }
}

#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn connection_and_statement_timeouts_are_explicit_errors() {
    let mut f = Fixture::new("none");
    if std::env::var("HYPATIA_PG_FAULT_SCENARIO").as_deref() == Ok("unreachable") {
        f.config.connect_timeout_seconds = 1;
        assert!(PgStore::open(&f.config, &f.vector, &f.embedding).is_err());
        return;
    }
    f.config.statement_timeout_ms = 25;
    let store = f.open();
    store
        .insert_knowledge("slow", &Content::new("data"))
        .unwrap();
    let result=store.query_knowledge(&format!("SELECT name,content,created_at FROM \"{}\".knowledge CROSS JOIN pg_catalog.pg_sleep(0.2)",f.config.schema),vec![]);
    assert!(result.unwrap_err().to_string().contains("57014"));
    assert!(store.get_knowledge("slow").unwrap().is_some());
    run_fault_child(
        "connection_and_statement_timeouts_are_explicit_errors",
        "unreachable",
        &test_connection(None, None, Some(1)),
    );
}

#[test]
#[ignore = "requires a disposable PostgreSQL database with pgvector"]
fn representative_1024_dimension_hnsw_and_exact_plans() {
    let mut f = Fixture::new("hnsw");
    f.embedding.remote.dimensions = 1024;
    let store = f.open();
    let mut vector = vec![0.; 1024];
    vector[0] = 1.;
    for n in 0..32 {
        let name = format!("vector-{n}");
        let token = store
            .insert_knowledge(&name, &Content::new("sample"))
            .unwrap();
        store
            .install_embedding("knowledge", &name, token, &vector)
            .unwrap();
    }
    assert_eq!(store.vector_search_knowledge(&vector, 5).unwrap().len(), 5);
    let mut admin = f.admin();
    let extension:String=admin.query_one("SELECT n.nspname FROM pg_catalog.pg_extension e JOIN pg_catalog.pg_namespace n ON n.oid=e.extnamespace WHERE e.extname='vector'",&[]).unwrap().get(0);
    let op = format!("OPERATOR(\"{}\".<=>)", extension.replace('"', "\"\""));
    let query = format!(
        "EXPLAIN (FORMAT JSON) SELECT name FROM \"{}\".knowledge WHERE embedding IS NOT NULL ORDER BY embedding {op} $1 LIMIT 5",
        f.config.schema
    );
    let v = pgvector::Vector::from(vector.clone());
    // Small fixtures prefer sequential scans naturally. Disable them to verify
    // the ANN access path is available; this is not a throughput benchmark.
    admin.batch_execute("SET enable_seqscan=off").unwrap();
    let plan: serde_json::Value = admin.query_one(&query, &[&v]).unwrap().get(0);
    assert!(plan.to_string().contains("knowledge_embedding_hnsw_idx"));
    eprintln!("1024D HNSW plan: {plan}");
    let mut exact = Fixture::new("none");
    exact.embedding.remote.dimensions = 1024;
    let target = exact.open();
    target.import_snapshot(&store.snapshot().unwrap()).unwrap();
    let exact_query = query.replace(&f.config.schema, &exact.config.schema);
    admin.batch_execute("SET enable_seqscan=on").unwrap();
    let plan: serde_json::Value = admin.query_one(&exact_query, &[&v]).unwrap().get(0);
    assert!(plan.to_string().contains("Seq Scan"));
    assert!(!plan.to_string().contains("hnsw_idx"));
    assert_eq!(target.vector_search_knowledge(&vector, 5).unwrap().len(), 5);
    eprintln!("1024D exact plan: {plan}");
}

// Fault scenarios run in child test processes: no process-wide environment
// mutation races with PostgreSQL clients or other Rust tests.
fn run_fault_child(test: &str, scenario: &str, url: &str) {
    let result = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", test, "--nocapture"])
        .env("HYPATIA_PG_FAULT_SCENARIO", scenario)
        .env("HYPATIA_TEST_POSTGRES_URL", url)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "fault child failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}
fn test_connection(database: Option<&str>, role: Option<&str>, port: Option<u16>) -> String {
    let original: postgres::Config = std::env::var("HYPATIA_TEST_POSTGRES_URL")
        .unwrap()
        .parse()
        .unwrap();
    let host = match &original.get_hosts()[0] {
        postgres::config::Host::Tcp(h) => h.clone(),
        #[cfg(unix)]
        postgres::config::Host::Unix(p) => p.to_string_lossy().into_owned(),
    };
    let quote = |s: &str| format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"));
    let user = role.or(original.get_user()).unwrap_or("postgres");
    let password = if role.is_some() {
        "hypatia-role-test-only"
    } else {
        std::str::from_utf8(original.get_password().unwrap_or_default()).unwrap()
    };
    format!(
        "host={} port={} user={} password={} dbname={} sslmode=disable",
        quote(&host),
        port.unwrap_or_else(|| original.get_ports().first().copied().unwrap_or(5432)),
        quote(user),
        quote(password),
        quote(database.or(original.get_dbname()).unwrap_or("postgres"))
    )
}
#[test]
#[ignore = "requires PostgreSQL admin privileges in a disposable test cluster"]
fn missing_extension_and_permission_errors() {
    let f = Fixture::new("none");
    if let Ok(scenario) = std::env::var("HYPATIA_PG_FAULT_SCENARIO") {
        let error = match PgStore::open(&f.config, &f.vector, &f.embedding) {
            Ok(_) => panic!("fault scenario unexpectedly opened"),
            Err(e) => e.to_string(),
        };
        assert!(
            error.contains(if scenario == "extension" {
                "preinstalled"
            } else {
                "42501"
            }),
            "{error}"
        );
        return;
    }
    struct Resources {
        admin: Client,
        database: String,
        role: String,
    }
    impl Drop for Resources {
        fn drop(&mut self) {
            let _ = self.admin.batch_execute(&format!(
                "DROP DATABASE IF EXISTS \"{}\" WITH (FORCE)",
                self.database
            ));
            let _ = self
                .admin
                .batch_execute(&format!("DROP ROLE IF EXISTS \"{}\"", self.role));
        }
    }
    let mut resources = Resources {
        admin: f.admin(),
        database: format!("{}_db", f.config.schema),
        role: format!("{}_role", f.config.schema),
    };
    resources
        .admin
        .batch_execute(&format!(
            "CREATE DATABASE \"{}\" TEMPLATE template0",
            resources.database
        ))
        .unwrap();
    run_fault_child(
        "missing_extension_and_permission_errors",
        "extension",
        &test_connection(Some(&resources.database), None, None),
    );
    resources
        .admin
        .batch_execute(&format!(
            "CREATE ROLE \"{}\" LOGIN PASSWORD 'hypatia-role-test-only'",
            resources.role
        ))
        .unwrap();
    run_fault_child(
        "missing_extension_and_permission_errors",
        "permission",
        &test_connection(None, Some(&resources.role), None),
    );
}

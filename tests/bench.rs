//! Hypatia Benchmark
//!
//! Measures ingestion throughput, FTS search recall/latency, and JSE query latency.
//! Inspired by MemPalace's benchmark methodology.
//!
//! Usage:
//!   cargo test --test bench
//!   BENCH_SCALE=medium cargo test --test bench
//!   BENCH_REPORT=report.json cargo test --test bench

mod bench_data;

use std::env;
use std::time::{Duration, Instant};

use serde_json::json;

use bench_data::{BenchDataGenerator, LatencyStats, ScaleConfig};
use hypatia::model::{Content, SearchOpts, StatementKey};
use hypatia::storage::{ShelfManager, Storage};

// ── Main benchmark ────────────────────────────────────────────────────

#[test]
fn run_benchmark() {
    let scale_name = env::var("BENCH_SCALE").unwrap_or_else(|_| "small".to_string());
    let report_path = env::var("BENCH_REPORT").ok();
    let config = ScaleConfig::from_name(&scale_name);

    println!();
    println!("{}", "═".repeat(58));
    println!("  Hypatia Benchmark");
    println!("{}", "═".repeat(58));
    println!("  Scale:       {scale_name}");
    println!("  Knowledge:   {}", config.n_knowledge);
    println!("  Statements:  {}", config.n_statements);
    println!("  Needles:     {}", config.n_needles);
    println!("  Queries:     {}", config.n_queries);
    println!("{}", "─".repeat(58));
    println!();

    // Generate data
    println!("  Generating synthetic data...");
    let mut generator = BenchDataGenerator::new(config);
    generator.generate();
    println!("  Generated {} knowledge, {} statements, {} needles",
        generator.knowledge.len(), generator.statements.len(), generator.needles.len());

    // Setup shelf in temp directory
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let shelf_path = tmp_dir.path().join("bench_shelf");
    let mut mgr = ShelfManager::new().expect("create shelf manager");
    let shelf_name = mgr.connect(&shelf_path, Some("bench")).expect("connect shelf");

    // ── Phase 1: Ingest ────────────────────────────────────────────
    println!("\n  Phase 1: Ingestion...");

    let t0 = Instant::now();
    let mut knowledge_count = 0usize;
    for entry in &generator.knowledge {
        let tags = entry.tags.clone();
        let content = Content::new(&entry.data).with_tags(tags);
        let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
        let mut svc = hypatia::service::KnowledgeService::new(shelf);
        svc.create(&entry.name, content).expect("create knowledge");
        knowledge_count += 1;
        if knowledge_count % 500 == 0 {
            print!("    {knowledge_count}/{} knowledge entries\r", generator.knowledge.len());
        }
    }
    let knowledge_time = t0.elapsed();

    let t1 = Instant::now();
    let mut stmt_count = 0usize;
    for entry in &generator.statements {
        let key = StatementKey::new(&entry.subject, &entry.predicate, &entry.object);
        let content = Content::new(&entry.data);
        let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
        let mut svc = hypatia::service::StatementService::new(shelf);
        svc.create(&key, content, None, None).expect("create statement");
        stmt_count += 1;
        if stmt_count % 1000 == 0 {
            print!("    {stmt_count}/{} statements\r", generator.statements.len());
        }
    }
    let statement_time = t1.elapsed();

    let total_ingest = t0.elapsed();
    let knowledge_per_sec = knowledge_count as f64 / knowledge_time.as_secs_f64();
    let statement_per_sec = stmt_count as f64 / statement_time.as_secs_f64();

    println!("    Knowledge: {knowledge_count} entries in {:.2}s ({knowledge_per_sec:.0}/s)", knowledge_time.as_secs_f64());
    println!("    Statements: {stmt_count} entries in {:.2}s ({statement_per_sec:.0}/s)", statement_time.as_secs_f64());
    println!("    Total ingest: {:.2}s", total_ingest.as_secs_f64());

    // ── Phase 2: FTS Search Recall ─────────────────────────────────
    println!("\n  Phase 2: FTS Search Recall...");

    let shelf = mgr.get(&shelf_name).expect("get shelf");
    let mut recall_hits_at_1 = 0usize;
    let mut recall_hits_at_5 = 0usize;
    let mut recall_hits_at_10 = 0usize;
    let mut needle_count = 0usize;

    for query in &generator.queries {
        if !query.is_needle {
            continue;
        }
        needle_count += 1;
        let expected = match &query.expected_name {
            Some(n) => n.clone(),
            None => continue,
        };

        // Search with limit=10 to measure recall at multiple K values
        let opts = SearchOpts {
            catalog: Some("knowledge".to_string()),
            limit: 10,
            offset: 0,
        };
        let result = match shelf.execute_search(&query.query, &opts) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("    WARNING: search failed for '{}': {}", query.query, e);
                continue;
            }
        };

        let found_names: Vec<String> = result.rows.iter()
            .filter_map(|row| row.get("key").and_then(|v| v.as_str()).map(String::from))
            .collect();

        if found_names.iter().take(1).any(|n| n == &expected) {
            recall_hits_at_1 += 1;
        }
        if found_names.iter().take(5).any(|n| n == &expected) {
            recall_hits_at_5 += 1;
        }
        if found_names.iter().take(10).any(|n| n == &expected) {
            recall_hits_at_10 += 1;
        }
    }

    let recall_at_1 = recall_hits_at_1 as f64 / needle_count as f64;
    let recall_at_5 = recall_hits_at_5 as f64 / needle_count as f64;
    let recall_at_10 = recall_hits_at_10 as f64 / needle_count as f64;

    println!("    Needle queries: {needle_count}");
    println!("    Recall@1:  {:.1}% ({}/{needle_count})", recall_at_1 * 100.0, recall_hits_at_1);
    println!("    Recall@5:  {:.1}% ({}/{needle_count})", recall_at_5 * 100.0, recall_hits_at_5);
    println!("    Recall@10: {:.1}% ({}/{needle_count})", recall_at_10 * 100.0, recall_hits_at_10);

    // ── Phase 3: FTS Search Latency ────────────────────────────────
    println!("\n  Phase 3: FTS Search Latency...");

    let shelf = mgr.get(&shelf_name).expect("get shelf");
    let mut search_durations: Vec<Duration> = Vec::new();

    for query in &generator.queries {
        let opts = SearchOpts {
            catalog: None,
            limit: 10,
            offset: 0,
        };
        let t = Instant::now();
        match shelf.execute_search(&query.query, &opts) {
            Ok(_) => search_durations.push(t.elapsed()),
            Err(_) => continue, // Skip FTS parsing failures
        }
    }

    let search_stats = LatencyStats::from_durations(&search_durations);
    println!("    Queries: {}", generator.queries.len());
    println!("    p50: {:.0} µs", search_stats.p50_us);
    println!("    p99: {:.0} µs", search_stats.p99_us);
    println!("    max: {:.0} µs", search_stats.max_us);

    // ── Phase 4: JSE Query Latency ─────────────────────────────────
    println!("\n  Phase 4: JSE Query Latency...");

    let shelf = mgr.get(&shelf_name).expect("get shelf");

    // Build a set of JSE queries that exercise different operators
    let jse_queries = vec![
        // Knowledge: basic
        r#"["$knowledge"]"#,
        r#"["$knowledge", ["$eq", "name", "knowledge_000000"]]"#,
        r#"["$knowledge", ["$contains", "data", "authentication"]]"#,
        r#"["$knowledge", ["$contains", "tags", "benchmark"]]"#,
        r#"["$knowledge", ["$search", "database migration"]]"#,
        r#"["$knowledge", ["$and", ["$contains", "tags", "backend"], ["$contains", "data", "API"]]]"#,
        // Knowledge: $like and $content
        r#"["$knowledge", ["$like", "name", "knowledge_000%"]]"#,
        r#"["$knowledge", ["$like", "created_at", "2025-%"]]"#,
        r#"["$knowledge", ["$content", {"format": "markdown"}]]"#,
        r#"["$knowledge", ["$and", ["$content", {"format": "markdown"}], ["$contains", "data", "API"]]]"#,
        // Statement: $contains (legacy)
        r#"["$statement"]"#,
        r#"["$statement", ["$contains", "triple", "Alice"]]"#,
        r#"["$statement", ["$contains", "triple", "works_on"]]"#,
        r#"["$statement", ["$search", "Alice"]]"#,
        r#"["$statement", ["$and", ["$contains", "triple", "Bob"], ["$contains", "triple", "manages"]]]"#,
        // Statement: $triple (indexed)
        r#"["$statement", ["$triple", "$*", "manages", "$*"]]"#,
        r#"["$statement", ["$triple", "Alice", "$*", "$*"]]"#,
        r#"["$statement", ["$triple", "$*", "$*", "Bob"]]"#,
        // Statement: $like
        r#"["$statement", ["$like", "subject", "Alice%"]]"#,
        // Statement: $content
        r#"["$statement", ["$content", {"format": "markdown"}]]"#,
    ];

    let mut jse_durations: Vec<Duration> = Vec::new();
    // Run each query 3 times for more stable measurements
    for _ in 0..3 {
        for jse_str in &jse_queries {
            let jse: serde_json::Value = serde_json::from_str(jse_str).expect("parse JSE");
            let t = Instant::now();
            let _ = hypatia::engine::Evaluator::execute(&jse, shelf).expect("execute JSE");
            jse_durations.push(t.elapsed());
        }
    }

    let jse_stats = LatencyStats::from_durations(&jse_durations);
    println!("    Queries: {} ({} unique × 3 runs)", jse_durations.len(), jse_queries.len());
    println!("    p50: {:.0} µs", jse_stats.p50_us);
    println!("    p99: {:.0} µs", jse_stats.p99_us);
    println!("    max: {:.0} µs", jse_stats.max_us);

    // ── Summary ────────────────────────────────────────────────────
    println!("\n{}", "═".repeat(58));
    println!("  SUMMARY");
    println!("{}", "═".repeat(58));
    println!("  Ingest:    {:.0} knowledge/s, {:.0} statements/s", knowledge_per_sec, statement_per_sec);
    println!("  Recall@1:  {:.1}%", recall_at_1 * 100.0);
    println!("  Recall@5:  {:.1}%", recall_at_5 * 100.0);
    println!("  Recall@10: {:.1}%", recall_at_10 * 100.0);
    println!("  Search p50: {:.0} µs", search_stats.p50_us);
    println!("  JSE p50:    {:.0} µs", jse_stats.p50_us);
    println!("{}", "═".repeat(58));
    println!();

    // ── JSON Report ────────────────────────────────────────────────
    let report = json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "scale": scale_name,
        "config": {
            "n_knowledge": config.n_knowledge,
            "n_statements": config.n_statements,
            "n_needles": config.n_needles,
            "n_queries": config.n_queries,
        },
        "results": {
            "ingest": {
                "knowledge_count": knowledge_count,
                "statement_count": stmt_count,
                "knowledge_per_sec": (knowledge_per_sec as u64),
                "statement_per_sec": (statement_per_sec as u64),
                "total_sec": (total_ingest.as_secs_f64() as u64),
            },
            "search_recall": {
                "needle_count": needle_count,
                "recall_at_1": format!("{:.3}", recall_at_1),
                "recall_at_5": format!("{:.3}", recall_at_5),
                "recall_at_10": format!("{:.3}", recall_at_10),
            },
            "search_latency_us": {
                "p50": search_stats.p50_us,
                "p99": search_stats.p99_us,
                "max": search_stats.max_us,
                "min": search_stats.min_us,
            },
            "jse_query_latency_us": {
                "p50": jse_stats.p50_us,
                "p99": jse_stats.p99_us,
                "max": jse_stats.max_us,
                "min": jse_stats.min_us,
            },
        }
    });

    let report_json = serde_json::to_string_pretty(&report).unwrap();
    println!("{}", report_json);

    if let Some(path) = report_path {
        std::fs::write(&path, &report_json).expect("write report");
        println!("\n  Report saved to: {path}");
    }
}

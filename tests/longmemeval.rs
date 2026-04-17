//! LongMemEval Benchmark for Hypatia
//!
//! Loads the LongMemEval M benchmark (500 questions, 7 types, 5 abilities),
//! ingests conversation sessions into Hypatia, runs FTS + vector retrieval
//! for all questions, and outputs JSONL results.
//!
//! Usage:
//!   LONGMEMEVAL_DATA=longmemeval_m.json LONGMEMEVAL_RESULTS=longmemeval_m_results.jsonl \
//!     cargo test --test longmemeval --release -- --nocapture

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;

use hypatia::model::{Content, QueryTarget, SearchOpts};
use hypatia::storage::{sanitize_fts_query, ShelfManager, Storage};

// ── Helper: accept both string and integer for "answer" field ────────

fn deserialize_answer_to_string<'de, D: serde::Deserializer<'de>>(
    de: D,
) -> Result<String, D::Error> {
    use serde::de::{self, Visitor};
    struct AnswerVisitor;
    impl Visitor<'_> for AnswerVisitor {
        type Value = String;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string or a number")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_u64<E: de::Error>(self, v: u64) -> Result<String, E> {
            Ok(v.to_string())
        }
        fn visit_i64<E: de::Error>(self, v: i64) -> Result<String, E> {
            Ok(v.to_string())
        }
    }
    de.deserialize_any(AnswerVisitor)
}

// ── LongMemEval data structures ─────────────────────────────────────

#[allow(dead_code)]
#[derive(Debug, Deserialize)]

struct LongMemEvalData(Vec<EvalInstance>);

#[derive(Debug, Deserialize)]
struct EvalInstance {
    question_id: String,
    question_type: String,
    question: String,
    #[serde(deserialize_with = "deserialize_answer_to_string")]
    answer: String,
    question_date: String,

    #[serde(default)]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_dates: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<Turn>>,

    #[serde(default)]
    answer_session_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Turn {
    role: String,
    content: String,
    #[serde(default)]
    has_answer: Option<bool>,
}

// ── Question type → ability mapping ─────────────────────────────────

fn question_type_to_ability(qtype: &str) -> &'static str {
    if qtype.contains("abs") || qtype == "abstention" {
        "abstention"
    } else {
        match qtype {
            "single-session-user" | "single-session-assistant" | "single-session-preference" => {
                "information_extraction"
            }
            "multi-session" => "multi_session_reasoning",
            "knowledge-update" => "knowledge_updates",
            "temporal-reasoning" => "temporal_reasoning",
            _ => "unknown",
        }
    }
}

fn canonical_question_type(qtype: &str) -> String {
    // Strip _abs suffix for grouping
    qtype.strip_suffix("_abs").unwrap_or(qtype).to_string()
}

// ── Shelf setup (reused from locomo.rs) ─────────────────────────────

fn default_shelf_dir() -> std::path::PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(".hypatia")
        .join("default")
}

fn setup_model_files(shelf_path: &std::path::Path) -> bool {
    let src_dir = default_shelf_dir();
    let model_src = src_dir.join("embedding_model.onnx");
    let tokenizer_src = src_dir.join("tokenizer.json");

    if !model_src.exists() || !tokenizer_src.exists() {
        return false;
    }

    if std::fs::create_dir_all(shelf_path).is_err() {
        return false;
    }

    let model_dest = shelf_path.join("embedding_model.onnx");
    let tokenizer_dest = shelf_path.join("tokenizer.json");

    if std::os::unix::fs::symlink(&model_src, &model_dest).is_err() {
        if std::fs::copy(&model_src, &model_dest).is_err() {
            return false;
        }
    }
    if std::os::unix::fs::symlink(&tokenizer_src, &tokenizer_dest).is_err() {
        if std::fs::copy(&tokenizer_src, &tokenizer_dest).is_err() {
            return false;
        }
    }

    // Handle external data files
    for candidate in [
        src_dir.join("model.onnx_data"),
        src_dir.join("model_quantized.onnx_data"),
        src_dir.join("embedding_model.onnx_data"),
        src_dir.join("embedding_model.onnx.data"),
    ] {
        if candidate.exists() {
            let dest_name = candidate.file_name().unwrap().to_string_lossy().to_string();
            let dest = shelf_path.join(&dest_name);
            if std::os::unix::fs::symlink(&candidate, &dest).is_err() {
                if std::fs::copy(&candidate, &dest).is_err() {
                    return false;
                }
            }
        }
    }

    // Copy shelf.toml if it exists
    let shelf_toml_src = src_dir.join("shelf.toml");
    if shelf_toml_src.exists() {
        let shelf_toml_dest = shelf_path.join("shelf.toml");
        if std::os::unix::fs::symlink(&shelf_toml_src, &shelf_toml_dest).is_err() {
            if std::fs::copy(&shelf_toml_src, &shelf_toml_dest).is_err() {
                eprintln!("    WARN: Could not copy shelf.toml");
            }
        }
    }

    shelf_path.join("embedding_model.onnx").exists() && shelf_path.join("tokenizer.json").exists()
}

// ── Result record ───────────────────────────────────────────────────

struct EvalResult {
    question_id: String,
    question_type: String,
    ability: String,
    question: String,
    answer: String,
    question_date: String,
    answer_session_ids: Vec<String>,
    answer_turn_ids: Vec<String>,

    // FTS results
    fts_top_keys: Vec<String>,
    fts_retrieved_sessions: Vec<String>,
    fts_latency_us: u64,

    // Vector results
    vec_top_keys: Option<Vec<String>>,
    vec_retrieved_sessions: Option<Vec<String>>,
    vec_latency_us: Option<u64>,
}

fn extract_session_id_from_key(key: &str) -> Option<String> {
    // Key format: "{question_id}__sess_{idx}__turn_{turn_idx}"
    let parts: Vec<&str> = key.split("__").collect();
    if parts.len() >= 2 {
        Some(parts[1].to_string())
    } else {
        None
    }
}

#[allow(dead_code)]
fn compute_recall(top_keys: &[String], expected: &[String], k: usize) -> bool {
    expected
        .iter()
        .any(|exp| top_keys.iter().take(k).any(|k_| k_ == exp))
}

fn compute_session_recall(top_sessions: &[String], expected_sessions: &[String], k: usize) -> bool {
    let top_k_sessions: Vec<&String> = top_sessions.iter().take(k).collect();
    expected_sessions
        .iter()
        .any(|exp| top_k_sessions.contains(&exp))
}

// ── Main benchmark ──────────────────────────────────────────────────

#[test]
fn run_longmemeval_benchmark() {
    let data_path =
        env::var("LONGMEMEVAL_DATA").unwrap_or_else(|_| "longmemeval_m.json".to_string());
    let results_path = env::var("LONGMEMEVAL_RESULTS")
        .unwrap_or_else(|_| "longmemeval_m_results.jsonl".to_string());

    println!();
    println!("{}", "═".repeat(70));
    println!("  LongMemEval Benchmark for Hypatia");
    println!("{}", "═".repeat(70));

    // Load data
    let file = File::open(&data_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: Cannot open {data_path}: {e}");
        eprintln!("  Download: python3 scripts/longmemeval_download.py --variant m");
        panic!("Data file not found");
    });
    let reader = BufReader::new(file);
    let instances: Vec<EvalInstance> =
        serde_json::from_reader(reader).unwrap_or_else(|e| {
            eprintln!("  ERROR: Failed to parse {data_path}: {e}");
            panic!("Parse error");
        });

    // Support subset via LONGMEMEVAL_MAX_QUESTIONS env var
    let max_questions: usize = env::var("LONGMEMEVAL_MAX_QUESTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let instances = if max_questions > 0 && max_questions < instances.len() {
        instances.into_iter().take(max_questions).collect()
    } else {
        instances
    };

    let total_questions = instances.len();

    // Count by type and ability
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut ability_counts: HashMap<String, usize> = HashMap::new();
    for inst in &instances {
        let ct = canonical_question_type(&inst.question_type);
        *type_counts.entry(ct).or_insert(0) += 1;
        let ability = question_type_to_ability(&inst.question_type);
        *ability_counts.entry(ability.to_string()).or_insert(0) += 1;
    }

    println!("  Questions: {total_questions}");
    println!("  Data: {data_path}");
    println!();
    println!("  By type:");
    for (t, c) in &type_counts {
        println!("    {t}: {c}");
    }
    println!();
    println!("  By ability:");
    for (a, c) in &ability_counts {
        println!("    {a}: {c}");
    }
    println!("{}", "─".repeat(70));

    // Setup temp shelf
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let shelf_path = tmp_dir.path().join("longmemeval_shelf");

    let has_model = setup_model_files(&shelf_path);
    if has_model {
        println!("  Embedding model: AVAILABLE");
    } else {
        println!("  Embedding model: NOT FOUND (vector search disabled)");
    }

    let mut mgr = ShelfManager::new().expect("create shelf manager");
    let shelf_name = mgr
        .connect(&shelf_path, Some("longmemeval"))
        .expect("connect shelf");

    // ── Phase 1: Ingest ──────────────────────────────────────────────
    println!("\n  Phase 1: Loading sessions into Hypatia...");

    let t0 = Instant::now();
    let mut total_entries = 0usize;
    let mut total_sessions = 0usize;
    let mut total_turns_with_answer = 0usize;

    // We ingest ALL sessions from ALL 500 questions into a single shelf.
    // Each question shares a common set of haystack sessions.
    // To avoid duplicate ingestion, we track ingested session keys.
    let mut ingested_keys: std::collections::HashSet<String> = std::collections::HashSet::new();

    for inst in &instances {
        let qid = &inst.question_id;

        for (sess_idx, session) in inst.haystack_sessions.iter().enumerate() {
            let sess_id = inst
                .haystack_session_ids
                .get(sess_idx)
                .cloned()
                .unwrap_or_else(|| format!("sess_{sess_idx}"));
            let timestamp = inst
                .haystack_dates
                .get(sess_idx)
                .cloned()
                .unwrap_or_default();

            let is_answer_session = inst.answer_session_ids.contains(&sess_id);

            for (turn_idx, turn) in session.iter().enumerate() {
                let entry_name = format!("{qid}__{sess_id}__turn_{turn_idx}");

                // Deduplicate across questions (LongMemEval M shares sessions)
                let dedup_key = format!("{sess_id}__turn_{turn_idx}");
                if ingested_keys.contains(&dedup_key) {
                    continue;
                }
                ingested_keys.insert(dedup_key);

                let data = format!("[{}] {}", turn.role, turn.content);
                let mut tags = vec![
                    qid.clone(),
                    sess_id.clone(),
                    format!("turn_{turn_idx}"),
                    turn.role.clone(),
                ];

                if !timestamp.is_empty() {
                    tags.push(timestamp.clone());
                }
                if is_answer_session || turn.has_answer.unwrap_or(false) {
                    tags.push("evidence".to_string());
                    total_turns_with_answer += 1;
                }

                let content = Content::new(&data).with_tags(tags);
                let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
                let mut svc = hypatia::service::KnowledgeService::new(shelf);
                if let Err(e) = svc.create(&entry_name, content) {
                    eprintln!("    WARN: Failed to create {entry_name}: {e}");
                    continue;
                }
                total_entries += 1;
            }

            total_sessions += 1;

            if total_entries % 1000 == 0 {
                println!("    {total_entries} entries ({total_sessions} sessions) loaded");
            }
        }
    }

    let ingest_time = t0.elapsed();
    println!(
        "    Loaded {total_entries} entries ({total_sessions} sessions) in {:.2}s",
        ingest_time.as_secs_f64()
    );
    println!("    Evidence turns: {total_turns_with_answer}");

    // ── Phase 2: Retrieval ───────────────────────────────────────────
    println!("\n  Phase 2: Running retrieval for {total_questions} questions...");
    println!("    Methods: FTS (BM25) + Vector (cosine similarity)");

    let shelf = mgr.get(&shelf_name).expect("get shelf");
    let mut eval_results: Vec<EvalResult> = Vec::new();
    let mut eval_count = 0usize;
    let mut fts_latencies: Vec<u64> = Vec::new();
    let mut vec_latencies: Vec<u64> = Vec::new();

    for inst in &instances {
        eval_count += 1;

        let qid = &inst.question_id;
        let fts_query = sanitize_fts_query(&inst.question);

        // Build expected turn keys for answer sessions
        let answer_turn_ids: Vec<String> = inst
            .answer_session_ids
            .iter()
            .map(|sid| format!("{qid}__{sid}"))
            .collect();

        // --- FTS search ---
        let fts_opts = SearchOpts {
            catalog: Some("knowledge".to_string()),
            limit: 50,
            offset: 0,
        };

        let (fts_top_keys, fts_sessions, fts_lat) = {
            let t = Instant::now();
            let result = shelf
                .execute_search(&fts_query, &fts_opts)
                .unwrap_or_else(|e| {
                    eprintln!("    WARN: FTS search failed for '{:?}': {}", inst.question, e);
                    hypatia::model::QueryResult::new(Vec::new())
                });
            let lat = t.elapsed().as_micros() as u64;
            let keys: Vec<String> = result
                .rows
                .iter()
                .filter_map(|row: &serde_json::Map<String, serde_json::Value>| {
                    row.get("key").and_then(|v| v.as_str()).map(String::from)
                })
                .collect();
            let sessions: Vec<String> = keys
                .iter()
                .filter_map(|k| extract_session_id_from_key(k))
                .collect();
            (keys, sessions, lat)
        };
        fts_latencies.push(fts_lat);

        // --- Vector search ---
        let (vec_top_keys, vec_sessions, vec_lat) = if has_model {
            let vec_opts = SearchOpts {
                catalog: None,
                limit: 50,
                offset: 0,
            };
            let t = Instant::now();
            match shelf.execute_similar(&inst.question, &vec_opts, QueryTarget::Knowledge) {
                Ok(result) => {
                    let lat = t.elapsed().as_micros() as u64;
                    let keys: Vec<String> = result
                        .rows
                        .iter()
                        .filter_map(|row: &serde_json::Map<String, serde_json::Value>| {
                            row.get("name").and_then(|v| v.as_str()).map(String::from)
                        })
                        .collect();
                    let sessions: Vec<String> = keys
                        .iter()
                        .filter_map(|k| extract_session_id_from_key(k))
                        .collect();
                    (Some(keys), Some(sessions), Some(lat))
                }
                Err(e) => {
                    eprintln!("    WARN: Vector search failed: {e}");
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };

        if let Some(lat) = vec_lat {
            vec_latencies.push(lat);
        }

        eval_results.push(EvalResult {
            question_id: inst.question_id.clone(),
            question_type: canonical_question_type(&inst.question_type),
            ability: question_type_to_ability(&inst.question_type).to_string(),
            question: inst.question.clone(),
            answer: inst.answer.clone(),
            question_date: inst.question_date.clone(),
            answer_session_ids: inst.answer_session_ids.clone(),
            answer_turn_ids,
            fts_top_keys,
            fts_retrieved_sessions: fts_sessions,
            fts_latency_us: fts_lat,
            vec_top_keys,
            vec_retrieved_sessions: vec_sessions,
            vec_latency_us: vec_lat,
        });

        if eval_count % 50 == 0 {
            println!("    {eval_count}/{total_questions} queries processed");
        }
    }

    println!("    {eval_count}/{total_questions} queries processed");

    // ── Write JSONL results ──────────────────────────────────────────
    let results_file = File::create(&results_path).expect("create results file");
    let mut writer = std::io::BufWriter::new(results_file);

    for r in &eval_results {
        let record = json!({
            "question_id": r.question_id,
            "question_type": r.question_type,
            "ability": r.ability,
            "question": r.question,
            "answer": r.answer,
            "question_date": r.question_date,
            "answer_session_ids": r.answer_session_ids,
            "answer_turn_ids": r.answer_turn_ids,
            "fts_top_keys": r.fts_top_keys,
            "fts_retrieved_sessions": r.fts_retrieved_sessions,
            "fts_latency_us": r.fts_latency_us,
            "vec_top_keys": r.vec_top_keys,
            "vec_retrieved_sessions": r.vec_retrieved_sessions,
            "vec_latency_us": r.vec_latency_us,
        });
        writeln!(writer, "{}", record).ok();
    }
    writer.flush().ok();

    // ── Compute summary stats ────────────────────────────────────────
    let mut fts_by_type: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut vec_by_type: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut fts_by_ability: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    let mut vec_by_ability: HashMap<String, (usize, usize, usize, usize)> = HashMap::new();
    // (count, r@1, r@5, r@10)

    for r in &eval_results {
        // Skip abstention for retrieval metrics (no ground truth answer location)
        if r.ability == "abstention" {
            continue;
        }

        // Session-level recall
        let expected_sessions = &r.answer_session_ids;

        // FTS
        let fts_r1 = compute_session_recall(&r.fts_retrieved_sessions, expected_sessions, 1);
        let fts_r5 = compute_session_recall(&r.fts_retrieved_sessions, expected_sessions, 5);
        let fts_r10 = compute_session_recall(&r.fts_retrieved_sessions, expected_sessions, 10);

        let fts_type = fts_by_type.entry(r.question_type.clone()).or_insert((0, 0, 0, 0));
        fts_type.0 += 1;
        if fts_r1 { fts_type.1 += 1; }
        if fts_r5 { fts_type.2 += 1; }
        if fts_r10 { fts_type.3 += 1; }

        let fts_ab = fts_by_ability.entry(r.ability.clone()).or_insert((0, 0, 0, 0));
        fts_ab.0 += 1;
        if fts_r1 { fts_ab.1 += 1; }
        if fts_r5 { fts_ab.2 += 1; }
        if fts_r10 { fts_ab.3 += 1; }

        // Vector
        if let Some(ref vec_sessions) = r.vec_retrieved_sessions {
            let vec_r1 = compute_session_recall(vec_sessions, expected_sessions, 1);
            let vec_r5 = compute_session_recall(vec_sessions, expected_sessions, 5);
            let vec_r10 = compute_session_recall(vec_sessions, expected_sessions, 10);

            let vec_type = vec_by_type.entry(r.question_type.clone()).or_insert((0, 0, 0, 0));
            vec_type.0 += 1;
            if vec_r1 { vec_type.1 += 1; }
            if vec_r5 { vec_type.2 += 1; }
            if vec_r10 { vec_type.3 += 1; }

            let vec_ab = vec_by_ability.entry(r.ability.clone()).or_insert((0, 0, 0, 0));
            vec_ab.0 += 1;
            if vec_r1 { vec_ab.1 += 1; }
            if vec_r5 { vec_ab.2 += 1; }
            if vec_r10 { vec_ab.3 += 1; }
        }
    }

    fts_latencies.sort();
    vec_latencies.sort();

    let fts_p50 = fts_latencies.get(fts_latencies.len() / 2).copied().unwrap_or(0);
    let fts_p99 = fts_latencies
        .get(fts_latencies.len() * 99 / 100)
        .copied()
        .unwrap_or(0);
    let vec_p50 = vec_latencies.get(vec_latencies.len() / 2).copied().unwrap_or(0);
    let vec_p99 = vec_latencies
        .get(vec_latencies.len() * 99 / 100)
        .copied()
        .unwrap_or(0);

    // ── Print summary ────────────────────────────────────────────────
    println!("\n\n{}", "═".repeat(70));
    println!("  RESULTS");
    println!("{}", "═".repeat(70));
    println!("  Entries loaded: {total_entries}");
    println!("  Sessions:       {total_sessions}");
    println!("  Questions:      {eval_count}");
    println!();

    // By ability
    println!("  RETRIEVAL — By Ability (session-level, excl. abstention)");
    println!("  {:25} {:>5} {:>8} {:>8} {:>8}", "Ability", "N", "R@1", "R@5", "R@10");
    println!("  {}", "─".repeat(58));

    let ability_order = [
        "information_extraction",
        "multi_session_reasoning",
        "knowledge_updates",
        "temporal_reasoning",
    ];

    let mut fts_overall = (0usize, 0usize, 0usize, 0usize);

    for ability in &ability_order {
        if let Some(&(n, r1, r5, r10)) = fts_by_ability.get(*ability) {
            fts_overall.0 += n;
            fts_overall.1 += r1;
            fts_overall.2 += r5;
            fts_overall.3 += r10;

            let vec_n = vec_by_ability.get(*ability).map(|v| v.0).unwrap_or(0);
            let vec_r5 = vec_by_ability.get(*ability).map(|v| v.2).unwrap_or(0);
            let display_name = ability.replace('_', " ");
            println!(
                "  {:25} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%  (vec R@5: {})",
                display_name,
                n,
                r1 as f64 / n as f64 * 100.0,
                r5 as f64 / n as f64 * 100.0,
                r10 as f64 / n as f64 * 100.0,
                if vec_n > 0 {
                    format!("{:.1}%", vec_r5 as f64 / vec_n as f64 * 100.0)
                } else {
                    "N/A".to_string()
                }
            );
        }
    }
    println!("  {}", "─".repeat(58));
    if fts_overall.0 > 0 {
        println!(
            "  {:25} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
            "OVERALL",
            fts_overall.0,
            fts_overall.1 as f64 / fts_overall.0 as f64 * 100.0,
            fts_overall.2 as f64 / fts_overall.0 as f64 * 100.0,
            fts_overall.3 as f64 / fts_overall.0 as f64 * 100.0,
        );
    }

    // By question type
    println!();
    println!("  RETRIEVAL — By Question Type (session-level)");
    println!(
        "  {:30} {:>5} {:>8} {:>8} {:>8}",
        "Type", "N", "R@1", "R@5", "R@10"
    );
    println!("  {}", "─".repeat(63));

    let type_order = [
        "single-session-user",
        "single-session-assistant",
        "single-session-preference",
        "multi-session",
        "knowledge-update",
        "temporal-reasoning",
    ];

    for qtype in &type_order {
        if let Some(&(n, r1, r5, r10)) = fts_by_type.get(*qtype) {
            println!(
                "  {:30} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
                qtype,
                n,
                r1 as f64 / n as f64 * 100.0,
                r5 as f64 / n as f64 * 100.0,
                r10 as f64 / n as f64 * 100.0,
            );
        }
    }

    // Latency
    println!();
    println!("  LATENCY");
    println!("  FTS search p50:  {fts_p50} µs");
    println!("  FTS search p99:  {fts_p99} µs");
    if has_model {
        println!("  Vec search p50:  {vec_p50} µs");
        println!("  Vec search p99:  {vec_p99} µs");
    }
    println!("{}", "═".repeat(70));
    println!("\n  Results saved to: {results_path}");
    println!("  Next: python3 scripts/longmemeval_eval.py --results {results_path} --retrieval-only");
    println!();
}

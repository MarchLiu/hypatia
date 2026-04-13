//! LoCoMo Benchmark for Hypatia
//!
//! Loads the LoCoMo long-term conversational memory benchmark into Hypatia,
//! runs FTS + vector searches for all QA pairs, and outputs results as JSONL.
//!
//! Usage:
//!   LOCOMO_DATA=locomo10.json LOCOMO_RESULTS=locomo_results.jsonl \
//!     cargo test --test locomo --release -- --nocapture

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufReader, Write};
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;

use hypatia::model::{Content, QueryTarget, SearchOpts};
use hypatia::storage::{ShelfManager, Storage};

// ── LoCoMo data structures ───────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct LoCoMoData(Vec<Conversation>);

#[derive(Debug, Deserialize)]
struct Conversation {
    sample_id: String,
    conversation: ConversationData,
    #[serde(default)]
    session_summary: HashMap<String, serde_json::Value>,
    #[serde(default)]
    event_summary: HashMap<String, serde_json::Value>,
    #[serde(default)]
    observation: HashMap<String, serde_json::Value>,
    #[serde(default)]
    qa: Vec<QaEntry>,
}

#[derive(Debug, Deserialize)]
struct ConversationData {
    speaker_a: String,
    speaker_b: String,
    #[serde(flatten)]
    sessions: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct QaEntry {
    question: String,
    #[serde(default, deserialize_with = "string_or_none")]
    answer: Option<String>,
    category: u32,
    #[serde(default)]
    evidence: Vec<String>,
}

fn string_or_none<'de, D>(de: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: Option<serde_json::Value> = Option::deserialize(de)?;
    Ok(val.and_then(|v| match v {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }))
}

#[derive(Debug, Deserialize)]
struct Turn {
    speaker: String,
    dia_id: String,
    text: String,
}

// ── FTS sanitization ─────────────────────────────────────────────────

use hypatia::storage::sanitize_fts_query;

// ── Extract sessions from conversation data ──────────────────────────

fn extract_sessions(conv_data: &ConversationData) -> Vec<(usize, String, Vec<Turn>)> {
    let mut sessions: Vec<(usize, String, Vec<Turn>)> = Vec::new();

    for (key, value) in &conv_data.sessions {
        if let Some(rest) = key.strip_prefix("session_") {
            if rest.contains('_') || rest.contains(' ') {
                continue;
            }
            if let Ok(session_num) = rest.parse::<usize>() {
                let date_key = format!("session_{session_num}_date_time");
                let date = conv_data
                    .sessions
                    .get(&date_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown date")
                    .to_string();

                let turns: Vec<Turn> = serde_json::from_value(value.clone())
                    .unwrap_or_default();

                sessions.push((session_num, date, turns));
            }
        }
    }

    sessions.sort_by_key(|(num, _, _)| *num);
    sessions
}

// ── Helper: find model files ─────────────────────────────────────────

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

    // Ensure shelf directory exists before copying
    if std::fs::create_dir_all(shelf_path).is_err() {
        return false;
    }

    // Copy/symlink model files
    let model_dest = shelf_path.join("embedding_model.onnx");
    let tokenizer_dest = shelf_path.join("tokenizer.json");
    // Use symlink for large ONNX files, copy for small tokenizer
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

    // Handle external data file — use symlink to avoid copying large files
    for candidate in [
        src_dir.join("model.onnx_data"),
        src_dir.join("model_quantized.onnx_data"),
        src_dir.join("embedding_model.onnx_data"),
        src_dir.join("embedding_model.onnx.data"),
    ] {
        if candidate.exists() {
            let dest_name = candidate.file_name().unwrap().to_string_lossy().to_string();
            let dest = shelf_path.join(&dest_name);
            // Prefer symlink for large files, fall back to copy
            if std::os::unix::fs::symlink(&candidate, &dest).is_err() {
                if std::fs::copy(&candidate, &dest).is_err() {
                    return false;
                }
            }
        }
    }

    // Copy shelf.toml if it exists (for embedding config: dimensions, pooling strategy, etc.)
    let shelf_toml_src = src_dir.join("shelf.toml");
    if shelf_toml_src.exists() {
        let shelf_toml_dest = shelf_path.join("shelf.toml");
        if std::os::unix::fs::symlink(&shelf_toml_src, &shelf_toml_dest).is_err() {
            if std::fs::copy(&shelf_toml_src, &shelf_toml_dest).is_err() {
                // Non-fatal: will fall back to defaults
                eprintln!("    WARN: Could not copy shelf.toml");
            }
        }
    }

    // Verify files are in place
    shelf_path.join("embedding_model.onnx").exists() && shelf_path.join("tokenizer.json").exists()
}

// ── Result record ────────────────────────────────────────────────────

struct EvalResult {
    sample_id: String,
    question: String,
    answer: String,
    category: u32,
    evidence_names: Vec<String>,
    fts_query: String,

    // FTS results
    fts_top_keys: Vec<String>,
    fts_recall_at_1: bool,
    fts_recall_at_5: bool,
    fts_recall_at_10: bool,
    fts_latency_us: u64,

    // Vector results (None if model unavailable)
    vec_top_keys: Option<Vec<String>>,
    vec_recall_at_1: Option<bool>,
    vec_recall_at_5: Option<bool>,
    vec_recall_at_10: Option<bool>,
    vec_latency_us: Option<u64>,
}

fn compute_recall(top_keys: &[String], expected: &[String], k: usize) -> bool {
    expected.iter().any(|exp| top_keys.iter().take(k).any(|k_| k_ == exp))
}

// ── Main benchmark ───────────────────────────────────────────────────

#[test]
fn run_locomo_benchmark() {
    let data_path =
        env::var("LOCOMO_DATA").unwrap_or_else(|_| "locomo10.json".to_string());
    let results_path = env::var("LOCOMO_RESULTS")
        .unwrap_or_else(|_| "locomo_results.jsonl".to_string());

    // Load LoCoMo data
    println!();
    println!("{}", "═".repeat(60));
    println!("  LoCoMo Benchmark for Hypatia");
    println!("{}", "═".repeat(60));

    let file = File::open(&data_path).unwrap_or_else(|e| {
        eprintln!("  ERROR: Cannot open {data_path}: {e}");
        eprintln!("  Download: curl -sL https://huggingface.co/datasets/Percena/locomo-mc10/resolve/main/raw/locomo10.json -o locomo10.json");
        panic!("Data file not found");
    });
    let reader = BufReader::new(file);
    let conversations: Vec<Conversation> =
        serde_json::from_reader(reader).expect("parse locomo10.json");

    let total_qa: usize = conversations.iter().map(|c| c.qa.len()).sum();
    let non_adversarial: usize = conversations
        .iter()
        .flat_map(|c| c.qa.iter())
        .filter(|q| q.category != 5)
        .count();

    println!("  Conversations: {}", conversations.len());
    println!("  Total QA pairs: {total_qa}");
    println!("  Evaluated (non-adversarial): {non_adversarial}");
    println!("  Data: {data_path}");
    println!("{}", "─".repeat(60));

    // Setup temp shelf
    let tmp_dir = tempfile::tempdir().expect("create temp dir");
    let shelf_path = tmp_dir.path().join("locomo_shelf");

    let has_model = setup_model_files(&shelf_path);
    if has_model {
        println!("  Embedding model: AVAILABLE");
    } else {
        println!("  Embedding model: NOT FOUND (vector search disabled)");
    }

    let mut mgr = ShelfManager::new();
    let shelf_name = mgr
        .connect(&shelf_path, Some("locomo"))
        .expect("connect shelf");

    // ── Phase 1: Ingest ──────────────────────────────────────────────
    println!("\n  Phase 1: Loading conversations into Hypatia...");

    let t0 = Instant::now();
    let mut total_entries = 0usize;

    for conv in &conversations {
        let sid = &conv.sample_id;
        let speaker_a = &conv.conversation.speaker_a;
        let speaker_b = &conv.conversation.speaker_b;

        let sessions = extract_sessions(&conv.conversation);
        for (session_num, date, turns) in &sessions {
            for turn in turns {
                let name = format!("{sid}__{}", turn.dia_id.replace(':', "_"));
                let data = format!("[{}] {}", turn.speaker, turn.text);
                let tags = vec![
                    sid.clone(),
                    format!("session_{session_num}"),
                    turn.speaker.clone(),
                    speaker_a.clone(),
                    speaker_b.clone(),
                    date.clone(),
                ];
                let content = Content::new(&data).with_tags(tags);
                let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
                let mut svc = hypatia::service::KnowledgeService::new(shelf);
                if let Err(e) = svc.create(&name, content) {
                    eprintln!("    WARN: Failed to create {name}: {e}");
                    continue;
                }
                total_entries += 1;
            }

            // Load session summary
            let summary_key = format!("session_{session_num}");
            if let Some(summary) = conv.session_summary.get(&summary_key) {
                if let Some(text) = summary.as_str() {
                    let name = format!("{sid}__summary_{session_num}");
                    let tags = vec![sid.clone(), format!("session_{session_num}"), "summary".into()];
                    let content = Content::new(text).with_tags(tags);
                    let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
                    let mut svc = hypatia::service::KnowledgeService::new(shelf);
                    if svc.create(&name, content).is_ok() {
                        total_entries += 1;
                    }
                }
            }

            // Load event summary
            let event_key = format!("events_session_{session_num}");
            if let Some(events) = conv.event_summary.get(&event_key) {
                let name = format!("{sid}__events_{session_num}");
                let text = serde_json::to_string(events).unwrap_or_default();
                let tags = vec![sid.clone(), format!("session_{session_num}"), "events".into()];
                let content = Content::new(&text).with_tags(tags);
                let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
                let mut svc = hypatia::service::KnowledgeService::new(shelf);
                if svc.create(&name, content).is_ok() {
                    total_entries += 1;
                }
            }

            // Load observations
            let obs_key = format!("session_{session_num}_observation");
            if let Some(obs) = conv.observation.get(&obs_key) {
                let name = format!("{sid}__obs_{session_num}");
                let text = serde_json::to_string(obs).unwrap_or_default();
                let tags = vec![sid.clone(), format!("session_{session_num}"), "observation".into()];
                let content = Content::new(&text).with_tags(tags);
                let shelf = mgr.get_mut(&shelf_name).expect("get shelf");
                let mut svc = hypatia::service::KnowledgeService::new(shelf);
                if svc.create(&name, content).is_ok() {
                    total_entries += 1;
                }
            }
        }

        if total_entries % 500 == 0 {
            print!("    {total_entries} entries loaded\r");
        }
    }

    let ingest_time = t0.elapsed();
    println!(
        "    Loaded {total_entries} entries in {:.2}s",
        ingest_time.as_secs_f64()
    );

    // ── Phase 2: Search (FTS + Vector) ──────────────────────────────
    println!("\n  Phase 2: Running searches for {} QA pairs...", non_adversarial);
    println!("    Methods: FTS (BM25) + Vector (cosine similarity)");

    let shelf = mgr.get(&shelf_name).expect("get shelf");
    let mut eval_results: Vec<EvalResult> = Vec::new();

    let mut eval_count = 0usize;
    let mut fts_latencies: Vec<u64> = Vec::new();
    let mut vec_latencies: Vec<u64> = Vec::new();

    for conv in &conversations {
        let sid = &conv.sample_id;

        for qa in &conv.qa {
            if qa.category == 5 {
                continue;
            }

            let answer = match &qa.answer {
                Some(a) if !a.is_empty() => a.clone(),
                _ => continue,
            };

            eval_count += 1;

            let fts_query = sanitize_fts_query(&qa.question);
            let expected_names: Vec<String> = qa
                .evidence
                .iter()
                .map(|dia_id| format!("{sid}__{}", dia_id.replace(':', "_")))
                .collect();

            // --- FTS search ---
            let fts_opts = SearchOpts {
                catalog: Some("knowledge".to_string()),
                limit: 10,
                offset: 0,
            };

            let (fts_top_keys, fts_r1, fts_r5, fts_r10, fts_lat) = {
                let t = Instant::now();
                let result = shelf.execute_search(&fts_query, &fts_opts)
                    .unwrap_or_else(|e| {
                        eprintln!("    WARN: FTS search failed for '{}': {}", qa.question, e);
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
                let r1 = compute_recall(&keys, &expected_names, 1);
                let r5 = compute_recall(&keys, &expected_names, 5);
                let r10 = compute_recall(&keys, &expected_names, 10);
                (keys, r1, r5, r10, lat)
            };
            fts_latencies.push(fts_lat);

            // --- Vector search ---
            let (vec_top_keys, vec_r1, vec_r5, vec_r10, vec_lat) = if has_model {
                let vec_opts = SearchOpts {
                    catalog: None,
                    limit: 10,
                    offset: 0,
                };
                let t = Instant::now();
                match shelf.execute_similar(&qa.question, &vec_opts, QueryTarget::Knowledge) {
                    Ok(result) => {
                        let lat = t.elapsed().as_micros() as u64;
                        let keys: Vec<String> = result
                            .rows
                            .iter()
                            .filter_map(|row: &serde_json::Map<String, serde_json::Value>| {
                                row.get("name").and_then(|v| v.as_str()).map(String::from)
                            })
                            .collect();
                        let r1 = compute_recall(&keys, &expected_names, 1);
                        let r5 = compute_recall(&keys, &expected_names, 5);
                        let r10 = compute_recall(&keys, &expected_names, 10);
                        (Some(keys), Some(r1), Some(r5), Some(r10), Some(lat))
                    }
                    Err(e) => {
                        eprintln!("    WARN: Vector search failed: {e}");
                        (None, None, None, None, None)
                    }
                }
            } else {
                (None, None, None, None, None)
            };

            if let Some(lat) = vec_lat {
                vec_latencies.push(lat);
            }

            eval_results.push(EvalResult {
                sample_id: sid.clone(),
                question: qa.question.clone(),
                answer,
                category: qa.category,
                evidence_names: expected_names,
                fts_query,
                fts_top_keys,
                fts_recall_at_1: fts_r1,
                fts_recall_at_5: fts_r5,
                fts_recall_at_10: fts_r10,
                fts_latency_us: fts_lat,
                vec_top_keys,
                vec_recall_at_1: vec_r1,
                vec_recall_at_5: vec_r5,
                vec_recall_at_10: vec_r10,
                vec_latency_us: vec_lat,
            });

            if eval_count % 100 == 0 {
                print!("    {eval_count}/{non_adversarial} queries processed\r");
            }
        }
    }

    println!("    {eval_count}/{non_adversarial} queries processed");

    // ── Write JSONL results ──────────────────────────────────────────
    let results_file = File::create(&results_path).expect("create results file");
    let mut writer = std::io::BufWriter::new(results_file);

    for r in &eval_results {
        let record = json!({
            "sample_id": r.sample_id,
            "question": r.question,
            "category": r.category,
            "fts_query": r.fts_query,
            "fts_top_keys": r.fts_top_keys,
            "fts_recall_at_1": r.fts_recall_at_1,
            "fts_recall_at_5": r.fts_recall_at_5,
            "fts_recall_at_10": r.fts_recall_at_10,
            "fts_latency_us": r.fts_latency_us,
            "vec_top_keys": r.vec_top_keys,
            "vec_recall_at_1": r.vec_recall_at_1,
            "vec_recall_at_5": r.vec_recall_at_5,
            "vec_recall_at_10": r.vec_recall_at_10,
            "vec_latency_us": r.vec_latency_us,
        });
        writeln!(writer, "{}", record).ok();
    }
    writer.flush().ok();

    // ── Compute summary stats ────────────────────────────────────────
    let mut fts_by_cat: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new();
    let mut vec_by_cat: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new();

    for r in &eval_results {
        let fts = fts_by_cat.entry(r.category).or_insert((0, 0, 0, 0));
        fts.0 += 1;
        if r.fts_recall_at_1 { fts.1 += 1; }
        if r.fts_recall_at_5 { fts.2 += 1; }
        if r.fts_recall_at_10 { fts.3 += 1; }

        if let (Some(vr1), Some(vr5), Some(vr10)) = (r.vec_recall_at_1, r.vec_recall_at_5, r.vec_recall_at_10) {
            let vec = vec_by_cat.entry(r.category).or_insert((0, 0, 0, 0));
            vec.0 += 1;
            if vr1 { vec.1 += 1; }
            if vr5 { vec.2 += 1; }
            if vr10 { vec.3 += 1; }
        }
    }

    fts_latencies.sort();
    vec_latencies.sort();

    let fts_p50 = fts_latencies.get(fts_latencies.len() / 2).copied().unwrap_or(0);
    let fts_p99 = fts_latencies.get(fts_latencies.len() * 99 / 100).copied().unwrap_or(0);
    let vec_p50 = vec_latencies.get(vec_latencies.len() / 2).copied().unwrap_or(0);
    let vec_p99 = vec_latencies.get(vec_latencies.len() * 99 / 100).copied().unwrap_or(0);

    // ── Summary ──────────────────────────────────────────────────────
    println!("\n\n{}", "═".repeat(70));
    println!("  RESULTS");
    println!("{}", "═".repeat(70));
    println!("  Entries loaded: {total_entries}");
    println!("  QA evaluated:   {eval_count}");
    println!();

    let cat_names = [(4u32, "Single-hop"), (1, "Multi-hop"), (2, "Temporal"), (3, "Open-domain")];

    // FTS table
    println!("  FTS (BM25, top-K)");
    println!("  {:20} {:>5} {:>8} {:>8} {:>8}", "Category", "N", "R@1", "R@5", "R@10");
    println!("  {}", "-".repeat(53));
    let mut fts_total = (0usize, 0usize, 0usize, 0usize);
    for (cat, name) in &cat_names {
        if let Some(&(n, r1, r5, r10)) = fts_by_cat.get(cat) {
            fts_total.0 += n; fts_total.1 += r1; fts_total.2 += r5; fts_total.3 += r10;
            println!("  {:20} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
                name, n,
                r1 as f64 / n as f64 * 100.0,
                r5 as f64 / n as f64 * 100.0,
                r10 as f64 / n as f64 * 100.0,
            );
        }
    }
    println!("  {}", "-".repeat(53));
    println!("  {:20} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
        "OVERALL", fts_total.0,
        fts_total.1 as f64 / fts_total.0 as f64 * 100.0,
        fts_total.2 as f64 / fts_total.0 as f64 * 100.0,
        fts_total.3 as f64 / fts_total.0 as f64 * 100.0,
    );

    // Vector table
    if has_model && !vec_by_cat.is_empty() {
        println!();
        println!("  Vector (cosine similarity, top-K)");
        println!("  {:20} {:>5} {:>8} {:>8} {:>8}", "Category", "N", "R@1", "R@5", "R@10");
        println!("  {}", "-".repeat(53));
        let mut vec_total = (0usize, 0usize, 0usize, 0usize);
        for (cat, name) in &cat_names {
            if let Some(&(n, r1, r5, r10)) = vec_by_cat.get(cat) {
                vec_total.0 += n; vec_total.1 += r1; vec_total.2 += r5; vec_total.3 += r10;
                println!("  {:20} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
                    name, n,
                    r1 as f64 / n as f64 * 100.0,
                    r5 as f64 / n as f64 * 100.0,
                    r10 as f64 / n as f64 * 100.0,
                );
            }
        }
        println!("  {}", "-".repeat(53));
        if vec_total.0 > 0 {
            println!("  {:20} {:>5} {:>7.1}% {:>7.1}% {:>7.1}%",
                "OVERALL", vec_total.0,
                vec_total.1 as f64 / vec_total.0 as f64 * 100.0,
                vec_total.2 as f64 / vec_total.0 as f64 * 100.0,
                vec_total.3 as f64 / vec_total.0 as f64 * 100.0,
            );

            // Delta
            println!();
            println!("  IMPROVEMENT (Vector vs FTS)");
            println!("  {:20} {:>8} {:>8} {:>8}", "Category", "Δ R@1", "Δ R@5", "Δ R@10");
            println!("  {}", "-".repeat(48));
            for (cat, name) in &cat_names {
                let fts = fts_by_cat.get(cat).copied().unwrap_or((0,0,0,0));
                let vec_ = vec_by_cat.get(cat).copied().unwrap_or((0,0,0,0));
                if fts.0 > 0 && vec_.0 > 0 {
                    let d1 = vec_.1 as f64 / vec_.0 as f64 - fts.1 as f64 / fts.0 as f64;
                    let d5 = vec_.2 as f64 / vec_.0 as f64 - fts.2 as f64 / fts.0 as f64;
                    let d10 = vec_.3 as f64 / vec_.0 as f64 - fts.3 as f64 / fts.0 as f64;
                    println!("  {:20} {:>+7.1}% {:>+7.1}% {:>+7.1}%",
                        name, d1 * 100.0, d5 * 100.0, d10 * 100.0);
                }
            }
            if fts_total.0 > 0 && vec_total.0 > 0 {
                let d1 = vec_total.1 as f64 / vec_total.0 as f64 - fts_total.1 as f64 / fts_total.0 as f64;
                let d5 = vec_total.2 as f64 / vec_total.0 as f64 - fts_total.2 as f64 / fts_total.0 as f64;
                let d10 = vec_total.3 as f64 / vec_total.0 as f64 - fts_total.3 as f64 / fts_total.0 as f64;
                println!("  {}", "-".repeat(48));
                println!("  {:20} {:>+7.1}% {:>+7.1}% {:>+7.1}%",
                    "OVERALL", d1 * 100.0, d5 * 100.0, d10 * 100.0);
            }
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
    println!();
}

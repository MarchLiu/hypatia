//! LoCoMo Benchmark for Hypatia
//!
//! Loads the LoCoMo long-term conversational memory benchmark into Hypatia,
//! runs FTS searches for all QA pairs, and outputs results as JSONL.
//!
//! Usage:
//!   LOCOMO_DATA=locomo10.json LOCOMO_RESULTS=locomo_results.jsonl \
//!     cargo test --test locomo --release -- --nocapture

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;

use serde::Deserialize;
use serde_json::json;

use hypatia::model::{Content, SearchOpts};
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
    use serde::de;
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

fn sanitize_fts_query(query: &str) -> String {
    let sanitized: String = query
        .chars()
        .map(|c| {
            matches!(
                c,
                ':' | '"' | '\'' | '*' | '^' | '+' | '-' | '(' | ')' | '.' | '?' | '!' | ','
            )
            .then_some(' ')
            .unwrap_or(c)
        })
        .collect();
    let mut result = String::new();
    let mut prev_space = false;
    for c in sanitized.chars() {
        if c == ' ' {
            if !prev_space {
                result.push(c);
            }
            prev_space = true;
        } else {
            result.push(c);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

// ── Extract sessions from conversation data ──────────────────────────

fn extract_sessions(conv_data: &ConversationData) -> Vec<(usize, String, Vec<Turn>)> {
    let mut sessions: Vec<(usize, String, Vec<Turn>)> = Vec::new();

    for (key, value) in &conv_data.sessions {
        // Match "session_N" keys (not "session_N_date_time")
        if let Some(rest) = key.strip_prefix("session_") {
            if rest.contains('_') || rest.contains(' ') {
                continue; // Skip "session_1_date_time" etc.
            }
            if let Ok(session_num) = rest.parse::<usize>() {
                // Get date for this session
                let date_key = format!("session_{session_num}_date_time");
                let date = conv_data
                    .sessions
                    .get(&date_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown date")
                    .to_string();

                // Parse turns
                let turns: Vec<Turn> = serde_json::from_value(value.clone())
                    .unwrap_or_default();

                sessions.push((session_num, date, turns));
            }
        }
    }

    sessions.sort_by_key(|(num, _, _)| *num);
    sessions
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

        // Load dialogue turns
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

    // ── Phase 2: Search ──────────────────────────────────────────────
    println!("\n  Phase 2: Running FTS searches for {} QA pairs...", non_adversarial);

    let shelf = mgr.get(&shelf_name).expect("get shelf");
    let results_file = File::create(&results_path).expect("create results file");
    let mut results_writer = std::io::BufWriter::new(results_file);

    let mut eval_count = 0usize;
    let mut search_latencies: Vec<u64> = Vec::new();

    for conv in &conversations {
        let sid = &conv.sample_id;

        for qa in &conv.qa {
            // Skip adversarial questions (category 5)
            if qa.category == 5 {
                continue;
            }

            let answer = match &qa.answer {
                Some(a) if !a.is_empty() => a.clone(),
                _ => continue,
            };

            eval_count += 1;

            // Sanitize question for FTS5
            let fts_query = sanitize_fts_query(&qa.question);

            // Run search
            let opts = SearchOpts {
                catalog: Some("knowledge".to_string()),
                limit: 10,
                offset: 0,
            };

            let t = Instant::now();
            let result = match shelf.execute_search(&fts_query, &opts) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("    WARN: Search failed for '{}': {}", qa.question, e);
                    // Still record the failure
                    let record = json!({
                        "sample_id": sid,
                        "question": qa.question,
                        "answer": answer,
                        "category": qa.category,
                        "evidence_dia_ids": qa.evidence,
                        "fts_query": fts_query,
                        "top_keys": [],
                        "top_texts": [],
                        "recall_at_1": false,
                        "recall_at_5": false,
                        "recall_at_10": false,
                        "search_latency_us": 0u64,
                        "error": format!("{e}"),
                    });
                    writeln!(results_writer, "{}", record).ok();
                    continue;
                }
            };
            let latency_us = t.elapsed().as_micros() as u64;
            search_latencies.push(latency_us);

            // Extract results
            let top_keys: Vec<String> = result
                .rows
                .iter()
                .filter_map(|row| row.get("key").and_then(|v| v.as_str()).map(String::from))
                .collect();

            let top_texts: Vec<String> = result
                .rows
                .iter()
                .filter_map(|row| {
                    row.get("data")
                        .or_else(|| row.get("content"))
                        .and_then(|v| v.as_str())
                        .map(String::from)
                })
                .collect();

            // Compute recall: check if any evidence dia_id maps to a result key
            let expected_names: Vec<String> = qa
                .evidence
                .iter()
                .map(|dia_id| format!("{sid}__{}", dia_id.replace(':', "_")))
                .collect();

            let recall_at_1 = expected_names.iter().any(|exp| top_keys.iter().take(1).any(|k| k == exp));
            let recall_at_5 = expected_names.iter().any(|exp| top_keys.iter().take(5).any(|k| k == exp));
            let recall_at_10 = expected_names.iter().any(|exp| top_keys.iter().take(10).any(|k| k == exp));

            let record = json!({
                "sample_id": sid,
                "question": qa.question,
                "answer": answer,
                "category": qa.category,
                "evidence_dia_ids": qa.evidence,
                "evidence_names": expected_names,
                "fts_query": fts_query,
                "top_keys": top_keys,
                "top_texts": top_texts,
                "recall_at_1": recall_at_1,
                "recall_at_5": recall_at_5,
                "recall_at_10": recall_at_10,
                "search_latency_us": latency_us,
            });

            writeln!(results_writer, "{}", record).ok();

            if eval_count % 200 == 0 {
                print!("    {eval_count}/{non_adversarial} queries processed\r");
            }
        }
    }

    results_writer.flush().ok();

    // Compute summary stats
    let mut r1_hits = 0usize;
    let mut r5_hits = 0usize;
    let mut r10_hits = 0usize;

    // Re-read results for summary
    let results_file = File::open(&results_path).expect("open results");
    let reader = BufReader::new(results_file);
    let mut by_category: HashMap<u32, (usize, usize, usize, usize)> = HashMap::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let record: serde_json::Value = serde_json::from_str(&line).unwrap();
        let cat = record["category"].as_u64().unwrap() as u32;
        let r1 = record["recall_at_1"].as_bool().unwrap();
        let r5 = record["recall_at_5"].as_bool().unwrap();
        let r10 = record["recall_at_10"].as_bool().unwrap();

        if r1 { r1_hits += 1; }
        if r5 { r5_hits += 1; }
        if r10 { r10_hits += 1; }

        let entry = by_category.entry(cat).or_insert((0, 0, 0, 0));
        entry.0 += 1;
        if r1 { entry.1 += 1; }
        if r5 { entry.2 += 1; }
        if r10 { entry.3 += 1; }
    }

    search_latencies.sort();
    let p50 = search_latencies.get(search_latencies.len() / 2).copied().unwrap_or(0);
    let p99 = search_latencies.get(search_latencies.len() * 99 / 100).copied().unwrap_or(0);

    // ── Summary ──────────────────────────────────────────────────────
    println!("\n\n{}", "═".repeat(60));
    println!("  RESULTS");
    println!("{}", "═".repeat(60));
    println!("  Entries loaded: {total_entries}");
    println!("  QA evaluated:   {eval_count}");
    println!();
    println!("  RETRIEVAL (FTS, top-K)");
    println!("  {:20} {:>5} {:>8} {:>8} {:>8}", "Category", "N", "R@1", "R@5", "R@10");
    println!("  {}", "-".repeat(53));

    let cat_names = [(4u32, "Single-hop"), (1, "Multi-hop"), (2, "Temporal"), (3, "Open-domain")];
    for (cat, name) in &cat_names {
        if let Some(&(n, r1, r5, r10)) = by_category.get(cat) {
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
        "OVERALL", eval_count,
        r1_hits as f64 / eval_count as f64 * 100.0,
        r5_hits as f64 / eval_count as f64 * 100.0,
        r10_hits as f64 / eval_count as f64 * 100.0,
    );

    println!();
    println!("  LATENCY");
    println!("  Search p50: {p50} µs");
    println!("  Search p99: {p99} µs");
    println!("{}", "═".repeat(60));
    println!("\n  Results saved to: {results_path}");
    println!();
}

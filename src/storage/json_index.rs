//! JSON path-tree postings: content → (doc_id, path, kind, value, array_index).
//!
//! Indexing rules (docs/sqlite-refactor-plan.md §3):
//! - Top-level Content fields become paths (`tags`, `scopes`, `format`, …);
//!   arrays index every element via `array_index` (exact membership).
//! - `synonyms` objects recurse one level (`synonyms.head`, `synonyms.relation`, …).
//! - `data` is NOT indexed as a leaf (free text; queried via JSON1). When
//!   `format == "json"` the data string is parsed and its direct children are
//!   indexed as `data.<key>` — that is the promised "complex JSON access".
//! - Containers beyond the recursion budget are stored as opaque leaves
//!   (kind=object/array, value = compact JSON text).

use serde_json::Value;

use crate::error::{Result, StorageError};

/// How deep to recurse below the top-level content object.
const RECURSE_BUDGET: u32 = 2;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Posting {
    pub path: String,
    pub kind: &'static str,
    pub value: Option<String>,
    pub array_index: Option<i64>,
}

/// Canonical string token for a scalar JSON value. Indexing and query side
/// must agree on this exact representation.
pub fn scalar_token(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null => Some(String::new()),
        _ => None,
    }
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Walk a content JSON document and produce its postings.
pub fn content_postings(content_json: &str) -> Vec<Posting> {
    let mut out = Vec::new();
    let root: Value = match serde_json::from_str(content_json) {
        Ok(v) => v,
        Err(_) => return out, // unparsable content: nothing to index
    };

    if let Value::Object(map) = &root {
        for (key, val) in map {
            if key == "data" {
                // Free text: never a leaf. Unpack JSON payloads instead.
                unpack_data(&mut out, val);
                continue;
            }
            walk(&mut out, key.clone(), val, RECURSE_BUDGET, None);
        }
    }
    out
}

/// `format == "json"` payloads: index `data.<key>` for direct children.
fn unpack_data(out: &mut Vec<Posting>, data_val: &Value) {
    let Value::String(s) = data_val else { return };
    let Ok(parsed) = serde_json::from_str::<Value>(s) else { return };
    let Value::Object(map) = parsed else { return };
    for (key, val) in &map {
        let path = format!("data.{key}");
        match val {
            Value::Array(elems) => {
                for (i, e) in elems.iter().enumerate() {
                    if let Some(tok) = scalar_token(e) {
                        out.push(Posting {
                            path: path.clone(),
                            kind: kind_of(e),
                            value: Some(tok),
                            array_index: Some(i as i64),
                        });
                    }
                }
            }
            _ => {
                if let Some(tok) = scalar_token(val) {
                    out.push(Posting {
                        path,
                        kind: kind_of(val),
                        value: Some(tok),
                        array_index: None,
                    });
                } else {
                    out.push(Posting {
                        path,
                        kind: kind_of(val),
                        value: Some(val.to_string()),
                        array_index: None,
                    });
                }
            }
        }
    }
}

fn walk(out: &mut Vec<Posting>, path: String, val: &Value, budget: u32, idx: Option<i64>) {
    match val {
        Value::Array(elems) => {
            if budget == 0 {
                out.push(Posting {
                    path,
                    kind: "array",
                    value: Some(val.to_string()),
                    array_index: idx,
                });
                return;
            }
            for (i, e) in elems.iter().enumerate() {
                walk(out, path.clone(), e, budget - 1, Some(i as i64));
            }
        }
        Value::Object(map) => {
            if budget == 0 || map.is_empty() {
                out.push(Posting {
                    path,
                    kind: "object",
                    value: Some(val.to_string()),
                    array_index: idx,
                });
                return;
            }
            for (k, v) in map {
                walk(out, format!("{path}.{k}"), v, budget - 1, idx);
            }
        }
        _ => {
            if let Some(tok) = scalar_token(val) {
                out.push(Posting {
                    path,
                    kind: kind_of(val),
                    value: Some(tok),
                    array_index: idx,
                });
            }
        }
    }
}

/// Replace all postings for one document inside an open transaction.
pub(crate) fn replace_postings_in(
    tx: &rusqlite::Transaction<'_>,
    doc_id: i64,
    content_json: &str,
) -> Result<()> {
    tx.execute("DELETE FROM json_index WHERE doc_id = ?1", rusqlite::params![doc_id])
        .map_err(StorageError::from)?;
    let postings = content_postings(content_json);
    for p in postings {
        tx.execute(
            "INSERT OR IGNORE INTO json_index(doc_id, path, kind, value, array_index)
             VALUES(?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![doc_id, p.path, p.kind, p.value, p.array_index],
        )
        .map_err(StorageError::from)?;
    }
    Ok(())
}

/// Rebuild the whole json_index from knowledge/statement content.
/// Returns the number of indexed documents.
pub fn rebuild_all(conn: &rusqlite::Connection) -> Result<usize> {
    let tx = conn.unchecked_transaction().map_err(StorageError::from)?;
    tx.execute("DELETE FROM json_index", []).map_err(StorageError::from)?;
    let mut docs = 0usize;
    {
        let mut stmt = tx
            .prepare(
                "SELECT d.id, COALESCE(k.content, s.content, '') \
                 FROM docs d \
                 LEFT JOIN knowledge k ON d.catalog='knowledge' AND k.name=d.key \
                 LEFT JOIN statement s ON d.catalog='statement' AND s.triple=d.key",
            )
            .map_err(StorageError::from)?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))
            .map_err(StorageError::from)?;
        let pairs: Vec<(i64, String)> = rows
            .filter_map(|r| r.ok())
            .collect();
        for (id, content_json) in pairs {
            replace_postings_in(&tx, id, &content_json)?;
            docs += 1;
        }
    }
    tx.commit().map_err(StorageError::from)?;
    Ok(docs)
}

// ── json_contains: PG `@>` semantics (jq `contains()` as behavioural base) ──

/// Recursive JSON containment (PG `@>` / jq `contains()`).
/// Strings require EQUALITY (PG jsonb semantics, not jq substring).
pub fn json_contains(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (_, Value::Null) => lhs == rhs,
        (Value::Object(a), Value::Object(b)) => b
            .iter()
            .all(|(k, rv)| a.get(k).map(|lv| json_contains(lv, rv)).unwrap_or(false)),
        (Value::Array(a), Value::Array(b)) => b.iter().all(|rv| {
            a.iter().any(|lv| json_contains(lv, rv))
        }),
        // PG: scalar containment requires type AND value equality.
        _ => lhs == rhs,
    }
}

/// `json_contains(content_json, rhs_json)` over raw JSON texts; either side
/// unparsable or NULL → false.
pub fn json_contains_str(lhs: &str, rhs: &str) -> bool {
    match (serde_json::from_str::<Value>(lhs), serde_json::from_str::<Value>(rhs)) {
        (Ok(l), Ok(r)) => json_contains(&l, &r),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn postings_cover_arrays_and_unpack_data() {
        let content = r#"{
            "format": "json",
            "data": "{\"filename\":\"fig.png\",\"mime_type\":\"image/png\"}",
            "tags": ["archive", "image"],
            "scopes": ["proj"],
            "synonyms": {"head": ["Bob"], "relation": ["leads"]}
        }"#;
        let p = content_postings(content);
        let has = |path: &str, value: &str| {
            p.iter()
                .any(|x| x.path == path && x.value.as_deref() == Some(value))
        };
        assert!(has("format", "json"));
        assert!(has("tags", "archive") && has("tags", "image"));
        assert!(has("scopes", "proj"));
        assert!(has("synonyms.head", "Bob"));
        assert!(has("synonyms.relation", "leads"));
        // unpacked data children
        assert!(has("data.filename", "fig.png"));
        assert!(has("data.mime_type", "image/png"));
        // data itself never indexed as a giant leaf
        assert!(!p.iter().any(|x| x.path == "data"));
    }

    #[test]
    fn deep_containers_stored_opaque() {
        let content = r#"{"meta":{"a":{"b":[1,2]}}}"#;
        let p = content_postings(content);
        // budget 2: root(1) -> meta(2) -> a would exceed → opaque leaf
        // budget exhausted one level deeper: array stored opaque
        assert!(p
            .iter()
            .any(|x| x.path == "meta.a.b" && x.kind == "array"));
    }

    #[test]
    fn json_contains_pg_semantics() {
        assert!(json_contains(
            &json!({"author":{"country":"CN"}}),
            &json!({"author":{"country":"CN"}})
        ));
        assert!(json_contains(
            &json!({"author":{"country":"CN","age":20}}),
            &json!({"author":{"country":"CN"}})
        ));
        assert!(!json_contains(
            &json!({"author":{"country":"US"}}),
            &json!({"author":{"country":"CN"}})
        ));
        // array containment: every rhs element matched somewhere in lhs
        assert!(json_contains(&json!({"tags":["a","b"]}), &json!({"tags":["a"]})));
        assert!(!json_contains(&json!({"tags":["a"]}), &json!({"tags":["a","b"]})));
        // strings require equality (PG), not substring (jq)
        assert!(!json_contains(&json!("abc"), &json!("b")));
        assert!(json_contains(&json!("abc"), &json!("abc")));
    }
}

//! MCP resources: shelf status, and single entries by URI.
//!
//! URIs are `hypatia://{shelf}/status`, `hypatia://{shelf}/knowledge/{name}` and
//! `hypatia://{shelf}/statement/{head}/{relation}/{tail}`, with each segment
//! percent-encoded. Only shelf status is listed: listing entries would not scale.

use serde_json::{Value, json};

use super::{INTERNAL_ERROR, INVALID_PARAMS, RESOURCE_NOT_FOUND, RpcError};
use crate::error::HypatiaError;
use crate::lab::Lab;
use crate::model::StatementKey;

const JSON: &str = "application/json";

pub(super) fn list(lab: &Lab) -> Value {
    let resources: Vec<Value> = lab
        .list_shelves()
        .into_iter()
        .filter(|(_, _, connected)| *connected)
        .map(|(name, _, _)| {
            json!({
                "uri": format!("hypatia://{}/status", encode(name)),
                "name": format!("{name} status"),
                "description": "What works on this shelf, and its embedding debt",
                "mimeType": JSON
            })
        })
        .collect();
    json!({ "resources": resources })
}

pub(super) fn templates() -> Vec<Value> {
    vec![
        json!({
            "uriTemplate": "hypatia://{shelf}/status",
            "name": "shelf-status",
            "description": "What works on a shelf, and its embedding debt",
            "mimeType": JSON
        }),
        json!({
            "uriTemplate": "hypatia://{shelf}/knowledge/{name}",
            "name": "knowledge",
            "description": "One knowledge entry by name",
            "mimeType": JSON
        }),
        json!({
            "uriTemplate": "hypatia://{shelf}/statement/{head}/{relation}/{tail}",
            "name": "statement",
            "description": "One statement by its triple",
            "mimeType": JSON
        }),
    ]
}

pub(super) fn read(lab: &mut Lab, params: &Value) -> Result<Value, RpcError> {
    let uri = params
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(INVALID_PARAMS, "resources/read needs a uri"))?;
    let path = uri
        .strip_prefix("hypatia://")
        .ok_or_else(|| not_found(uri))?;
    if path.contains(['?', '#']) {
        return Err(RpcError::new(
            INVALID_PARAMS,
            format!("query strings and fragments are not supported: {uri}"),
        ));
    }
    let segments: Vec<String> = path
        .split('/')
        .map(decode)
        .collect::<Option<_>>()
        .ok_or_else(|| {
            RpcError::new(
                INVALID_PARAMS,
                format!("malformed percent-encoding in {uri}"),
            )
        })?;
    let segments: Vec<&str> = segments.iter().map(String::as_str).collect();
    let body = match segments.as_slice() {
        [shelf, "status"] => status_json(lab, shelf).map_err(|e| lab_error(uri, e))?,
        [shelf, "knowledge", name] => {
            let k = lab
                .get_knowledge(shelf, name)
                .map_err(|e| lab_error(uri, e))?
                .ok_or_else(|| not_found(uri))?;
            json!({ "name": k.name, "content": k.content, "created_at": k.created_at.to_string() })
        }
        [shelf, "statement", head, relation, tail] => {
            let key = StatementKey::new(*head, *relation, *tail);
            let s = lab
                .get_statement(shelf, &key)
                .map_err(|e| lab_error(uri, e))?
                .ok_or_else(|| not_found(uri))?;
            json!({
                "head": s.key.head,
                "relation": s.key.relation,
                "tail": s.key.tail,
                "content": s.content,
                "created_at": s.created_at.to_string(),
                "tr_start": s.tr_start.map(|t| t.to_string()),
                "tr_end": s.tr_end.map(|t| t.to_string())
            })
        }
        _ => return Err(not_found(uri)),
    };
    Ok(json!({
        "contents": [{
            "uri": uri,
            "mimeType": JSON,
            "text": serde_json::to_string_pretty(&body).unwrap_or_default()
        }]
    }))
}

/// What works on a shelf and its embedding debt: the status resource and the
/// `shelf_status` tool return this same document.
pub(super) fn status_json(lab: &Lab, shelf: &str) -> Result<Value, HypatiaError> {
    let status = lab.shelf_status(shelf)?;
    Ok(json!({
        "name": status.name,
        "path": status.path,
        "postgres": status.postgres,
        "embedder": status.embedder,
        "semantic_search_off": status.semantic_search_off,
        "attention": status.attention,
        "debt": status.debt
    }))
}

fn not_found(uri: &str) -> RpcError {
    RpcError::new(RESOURCE_NOT_FOUND, format!("resource not found: {uri}"))
}

/// A shelf that is not connected is a missing resource; anything else is the server's fault.
fn lab_error(uri: &str, e: HypatiaError) -> RpcError {
    match e {
        HypatiaError::Shelf(message) => RpcError::new(
            RESOURCE_NOT_FOUND,
            format!("resource not found: {uri}: {message}"),
        ),
        HypatiaError::NotFound { .. } => not_found(uri),
        HypatiaError::Validation(message) | HypatiaError::Parse(message) => {
            RpcError::new(INVALID_PARAMS, message)
        }
        other => RpcError::new(INTERNAL_ERROR, other.to_string()),
    }
}

/// Percent-encodes everything but RFC 3986 unreserved characters.
fn encode(segment: &str) -> String {
    let mut out = String::new();
    for b in segment.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn decode(segment: &str) -> Option<String> {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = segment.get(i + 1..i + 3)?;
            if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return None;
            }
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_round_trip_through_percent_encoding() {
        for segment in ["plain", "my note", "a/b", "中文", "100%", "x,y"] {
            assert_eq!(decode(&encode(segment)).as_deref(), Some(segment));
        }
        assert_eq!(decode("bad%zz"), None);
        assert_eq!(decode("cut%4"), None);
        assert_eq!(decode("%+5"), None);
    }
}

//! `hypatia mcp`: serve the knowledge graph to an agent over the Model Context Protocol.
//!
//! stdio transport: one JSON-RPC message per line on stdin, one reply per line on stdout,
//! handled in order by a synchronous loop (docs/agent-interfaces.md §3.1). stdout carries
//! protocol messages only; diagnostics go to stderr. The tools are thin graph primitives for
//! the data plane. Administration (connecting shelves, init, model install, export and
//! import, re-embedding) stays in the CLI, and judgement stays in the skills.

mod resources;
mod tools;

use std::io::{BufRead, Write};

use serde_json::{Value, json};

use crate::lab::Lab;

/// Protocol versions of the `initialize`-handshake era this server speaks, newest first. The
/// 2026-07-28 revision drops the handshake for per-request versions; hosts still use the
/// handshake, so that model is not implemented.
const PROTOCOL_VERSIONS: &[&str] = &["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];

const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
pub(super) const INVALID_PARAMS: i64 = -32602;
pub(super) const INTERNAL_ERROR: i64 = -32603;
pub(super) const RESOURCE_NOT_FOUND: i64 = -32002;

const INSTRUCTIONS: &str = "Hypatia is a local knowledge graph of knowledge entries (named \
notes) and statements (head, relation, tail triples). These tools are thin graph primitives: \
when and how to remember, link and consolidate is decided by the hypatia skills, not by this \
server. The shelf argument defaults to \"default\". Writes return the shelf's embedding debt: \
vectors are generated later, so a fresh entry can be missing from `similar` until the debt is \
paid. Connecting shelves, installing models, export and import are done with the hypatia CLI; restart this server afterwards, since it reads shelf \
configuration once at start.";

/// A JSON-RPC error.
pub(super) struct RpcError {
    code: i64,
    message: String,
}

impl RpcError {
    pub(super) fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// Runs the server until stdin closes.
pub(crate) fn serve(mut lab: Lab) -> crate::error::Result<()> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut line = Vec::new();
    loop {
        line.clear();
        // Bytes, not `lines()`: a line that is not UTF-8 gets a parse error, not a dead server.
        if stdin.read_until(b'\n', &mut line)? == 0 {
            return Ok(());
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if let Some(reply) = handle_line(&mut lab, &line) {
            serde_json::to_writer(&mut stdout, &reply)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
        }
    }
}

fn handle_line(lab: &mut Lab, line: &[u8]) -> Option<Value> {
    let message: Value = match serde_json::from_slice(line) {
        Ok(message) => message,
        Err(e) => {
            return Some(error_reply(
                Value::Null,
                RpcError::new(PARSE_ERROR, format!("parse error: {e}")),
            ));
        }
    };
    if message.is_array() {
        return Some(error_reply(
            Value::Null,
            RpcError::new(INVALID_REQUEST, "batched requests are not supported"),
        ));
    }
    if !message.is_object() {
        return Some(error_reply(
            Value::Null,
            RpcError::new(INVALID_REQUEST, "a message must be a JSON object"),
        ));
    }
    handle_message(lab, &message)
}

fn handle_message(lab: &mut Lab, message: &Value) -> Option<Value> {
    let id = message.get("id").cloned();
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        // Replies to requests this server never sends are ignored.
        return match id {
            Some(id) if message.get("result").is_none() && message.get("error").is_none() => Some(
                error_reply(id, RpcError::new(INVALID_REQUEST, "missing method")),
            ),
            _ => None,
        };
    };
    // Notifications (`notifications/initialized`, `notifications/cancelled`, ...) get no
    // reply. Requests run to completion, so there is nothing to cancel.
    let id = id?;
    if !(id.is_string() || id.is_i64() || id.is_u64()) {
        return Some(error_reply(
            Value::Null,
            RpcError::new(
                INVALID_REQUEST,
                "a request id must be a string or an integer",
            ),
        ));
    }
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let outcome = match method {
        "initialize" => Ok(initialize(&params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tools::definitions() })),
        "tools/call" => tools::call(lab, &params),
        "resources/list" => Ok(resources::list(lab)),
        "resources/templates/list" => Ok(json!({ "resourceTemplates": resources::templates() })),
        "resources/read" => resources::read(lab, &params),
        _ => Err(RpcError::new(
            METHOD_NOT_FOUND,
            format!("method not found: {method}"),
        )),
    };
    if matches!(method, "tools/call" | "resources/read") {
        // A long-lived process keeps no local model between requests (§3.5).
        lab.release_embedders();
    }
    Some(match outcome {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(e) => error_reply(id, e),
    })
}

fn initialize(params: &Value) -> Value {
    let version = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .filter(|v| PROTOCOL_VERSIONS.contains(v))
        .unwrap_or(PROTOCOL_VERSIONS[0]);
    json!({
        "protocolVersion": version,
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "subscribe": false, "listChanged": false }
        },
        "serverInfo": { "name": "hypatia", "title": "Hypatia", "version": env!("CARGO_PKG_VERSION") },
        "instructions": INSTRUCTIONS
    })
}

fn error_reply(id: Value, error: RpcError) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": error.code, "message": error.message }
    })
}

//! Protocol-level test for `hypatia mcp`: drive the binary over stdio as a host would.
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{Value, json};
use tempfile::TempDir;

/// Feeds `input` to `hypatia mcp`, closes stdin, and returns every reply in order.
fn run(home: &TempDir, input: &[u8]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .arg("mcp")
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hypatia mcp");
    let mut stdin = child.stdin.take().unwrap();
    // Write from a thread so a large input can never deadlock against a full stdout pipe.
    let input = input.to_vec();
    let writer = std::thread::spawn(move || {
        stdin.write_all(&input).unwrap();
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert!(
        out.status.success(),
        "exit {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            // stdout must carry protocol messages only.
            let reply: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("non-JSON on stdout: {line}: {e}"));
            assert_eq!(reply["jsonrpc"], "2.0", "{reply}");
            reply
        })
        .collect()
}

/// Sends each line and returns replies with integer ids by id, and the rest in order.
fn session(home: &TempDir, lines: &[String]) -> (HashMap<i64, Value>, Vec<Value>) {
    let mut input = Vec::new();
    for line in lines {
        input.extend_from_slice(line.as_bytes());
        input.push(b'\n');
    }
    let mut by_id = HashMap::new();
    let mut others = Vec::new();
    for reply in run(home, &input) {
        match reply["id"].as_i64() {
            Some(id) => {
                by_id.insert(id, reply);
            }
            None => others.push(reply),
        }
    }
    (by_id, others)
}

fn request(id: i64, method: &str, params: Value) -> String {
    json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string()
}

fn call(id: i64, tool: &str, arguments: Value) -> String {
    request(
        id,
        "tools/call",
        json!({ "name": tool, "arguments": arguments }),
    )
}

fn initialize(id: i64, version: &str) -> String {
    request(
        id,
        "initialize",
        json!({ "protocolVersion": version, "capabilities": {}, "clientInfo": { "name": "test", "version": "0" } }),
    )
}

fn ok(replies: &HashMap<i64, Value>, id: i64) -> Value {
    let reply = replies
        .get(&id)
        .unwrap_or_else(|| panic!("no reply to {id}"));
    assert!(reply.get("error").is_none(), "{reply}");
    let result = reply["result"].clone();
    if result.get("isError").is_some() {
        assert_eq!(result["isError"], false, "{result}");
        return result["structuredContent"].clone();
    }
    result
}

fn tool_error(replies: &HashMap<i64, Value>, id: i64) -> String {
    let result = &replies[&id]["result"];
    assert_eq!(result["isError"], true, "{}", replies[&id]);
    result["content"][0]["text"].as_str().unwrap().to_string()
}

fn rpc_error(replies: &HashMap<i64, Value>, id: i64) -> i64 {
    replies[&id]["error"]["code"]
        .as_i64()
        .unwrap_or_else(|| panic!("{}", replies[&id]))
}

fn resource_text(replies: &HashMap<i64, Value>, id: i64) -> Value {
    serde_json::from_str(ok(replies, id)["contents"][0]["text"].as_str().unwrap()).unwrap()
}

#[test]
fn a_host_can_drive_the_data_plane_over_stdio() {
    let home = TempDir::new().unwrap();
    let upload = home.path().join("upload.txt");
    std::fs::write(&upload, "hello").unwrap();
    let upload = upload.to_str().unwrap();
    let initialized =
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }).to_string();
    let lines = vec![
        initialize(1, "2025-06-18"),
        initialized,
        request(2, "tools/list", json!({})),
        call(
            3,
            "knowledge_create",
            json!({ "name": "my note", "data": "original", "tags": ["a", " "], "scopes": ["p", ""] }),
        ),
        call(4, "knowledge_get", json!({ "name": "my note" })),
        call(
            5,
            "knowledge_update",
            json!({ "name": "my note", "data": "changed" }),
        ),
        call(
            6,
            "knowledge_update",
            json!({ "name": "my note", "data": "changed" }),
        ),
        call(
            7,
            "statement_create",
            json!({ "head": "my note", "relation": "about", "tail": "mcp" }),
        ),
        call(
            8,
            "statement_create",
            json!({ "head": "my note", "relation": "about", "tail": "mcp" }),
        ),
        call(
            9,
            "search",
            json!({ "query": "changed", "catalog": "knowledge" }),
        ),
        call(10, "knowledge_get", json!({ "name": "nope" })),
        call(
            11,
            "knowledge_create",
            json!({ "name": "x", "tagz": ["typo"] }),
        ),
        call(12, "knowledge_create", json!({ "name": "my note" })),
        call(13, "similar", json!({ "query": "anything" })),
        call(14, "backfill", json!({})),
        call(15, "knowledge_update", json!({ "name": "my note" })),
        request(16, "resources/list", json!({})),
        request(
            17,
            "resources/read",
            json!({ "uri": "hypatia://default/status" }),
        ),
        request(
            18,
            "resources/read",
            json!({ "uri": "hypatia://default/knowledge/my%20note" }),
        ),
        request(
            19,
            "resources/read",
            json!({ "uri": "hypatia://default/statement/my%20note/about/mcp" }),
        ),
        request(
            20,
            "resources/read",
            json!({ "uri": "hypatia://default/knowledge/nope" }),
        ),
        request(21, "resources/templates/list", json!({})),
        request(22, "no/such/method", json!({})),
        call(23, "no_such_tool", json!({})),
        request(24, "ping", json!({})),
        call(25, "list_shelves", json!({})),
        call(26, "session_current", json!({ "scope": "p" })),
        call(27, "shelf_status", json!({})),
        call(
            28,
            "query",
            json!({ "jse": "[\"$knowledge\", [\"$eq\", \"$name\", \"my note\"]]" }),
        ),
        call(
            29,
            "query",
            json!({ "jse": ["$knowledge", ["$eq", "$name", "my note"]] }),
        ),
        call(30, "search", json!({ "query": "changed", "limit": -1 })),
        call(
            31,
            "archive_store",
            json!({ "file": upload, "name": "../escape.txt" }),
        ),
        call(
            32,
            "archive_store",
            json!({ "file": upload, "name": "docs/a.txt" }),
        ),
        call(33, "archive_get", json!({ "name": "docs/a.txt" })),
        call(34, "archive_get", json!({ "name": "/etc/passwd" })),
        call(35, "archive_list", json!({})),
        call(
            36,
            "statement_delete",
            json!({ "head": "my note", "relation": "about", "tail": "mcp" }),
        ),
        call(37, "knowledge_create", json!({ "name": "temp" })),
        call(38, "knowledge_delete", json!({ "name": "temp" })),
        call(
            39,
            "statement_create",
            json!({ "head": "a", "relation": "r", "tail": "b", "synonyms": { "head": [" alias ", ""], "relation": [] } }),
        ),
        request(
            40,
            "resources/read",
            json!({ "uri": "hypatia://default/statement/a/r/b" }),
        ),
        request(
            41,
            "resources/read",
            json!({ "uri": "hypatia://default/knowledge/bad%zz" }),
        ),
        request(
            42,
            "resources/read",
            json!({ "uri": "hypatia://no-such-shelf/status" }),
        ),
        call(43, "backfill", json!({ "limit": 0 })),
        "{ not json".to_string(),
        json!([{ "jsonrpc": "2.0", "id": 99, "method": "ping" }]).to_string(),
    ];
    let (replies, others) = session(&home, &lines);

    let init = ok(&replies, 1);
    assert_eq!(init["protocolVersion"], "2025-06-18");
    assert_eq!(init["serverInfo"]["name"], "hypatia");
    assert!(
        init["capabilities"]["tools"].is_object() && init["capabilities"]["resources"].is_object()
    );

    let tools: Vec<&str> = replies[&2]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for expected in [
        "query",
        "search",
        "similar",
        "knowledge_create",
        "knowledge_update",
        "statement_create",
        "backfill",
        "shelf_status",
    ] {
        assert!(tools.contains(&expected), "{tools:?}");
    }
    assert!(!tools.iter().any(|t| t.contains("connect")), "{tools:?}");

    let created = ok(&replies, 3);
    assert_eq!(created["created"], true);
    assert!(created["embedding"].is_object(), "{created}");
    let got = ok(&replies, 4);
    assert_eq!(got["content"]["tags"], json!(["a"]));
    assert_eq!(got["content"]["scopes"], json!(["p", ""]));

    assert_eq!(ok(&replies, 5)["changed"], true);
    assert_eq!(ok(&replies, 6)["changed"], false);
    assert_eq!(ok(&replies, 7)["created"], true);
    assert_eq!(ok(&replies, 8)["created"], false);
    assert!(!ok(&replies, 9)["rows"].as_array().unwrap().is_empty());

    assert!(tool_error(&replies, 10).contains("not found"));
    assert!(tool_error(&replies, 11).contains("unknown field"));
    assert!(tool_error(&replies, 12).contains("already exists"));
    assert!(tool_error(&replies, 13).contains("semantic search is off"));
    assert!(tool_error(&replies, 14).contains("semantic search is off"));
    assert!(tool_error(&replies, 15).contains("nothing to update"));

    let listed = ok(&replies, 16);
    assert!(
        listed["resources"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["uri"] == "hypatia://default/status"),
        "{listed}"
    );
    let status = resource_text(&replies, 17);
    assert_eq!(status["name"], "default");
    assert!(status["debt"].is_object(), "{status}");
    assert_eq!(resource_text(&replies, 18)["content"]["data"], "changed");
    assert_eq!(resource_text(&replies, 19)["tail"], "mcp");
    assert_eq!(rpc_error(&replies, 20), -32002);
    assert_eq!(
        ok(&replies, 21)["resourceTemplates"]
            .as_array()
            .unwrap()
            .len(),
        3
    );

    assert_eq!(rpc_error(&replies, 22), -32601);
    assert_eq!(rpc_error(&replies, 23), -32602);
    assert_eq!(ok(&replies, 24), json!({}));
    assert!(ok(&replies, 25)["shelves"].as_array().is_some());
    assert!(ok(&replies, 26)["rows"].is_array());
    assert_eq!(
        ok(&replies, 27),
        status,
        "the tool and the resource return one document"
    );

    // JSE as a JSON string and as JSON.
    assert_eq!(ok(&replies, 28)["rows"].as_array().unwrap().len(), 1);
    assert_eq!(ok(&replies, 29)["rows"].as_array().unwrap().len(), 1);
    assert!(tool_error(&replies, 30).contains("limit must be at least 1"));

    // Archive names stay inside the shelf.
    assert!(tool_error(&replies, 31).contains("relative path"));
    assert!(
        !home
            .path()
            .join(".hypatia")
            .join("default")
            .join("escape.txt")
            .exists()
    );
    assert_eq!(ok(&replies, 32)["uri"], "archive://docs/a.txt");
    assert!(std::path::Path::new(ok(&replies, 33)["path"].as_str().unwrap()).exists());
    assert!(tool_error(&replies, 34).contains("not found"));
    assert!(
        ok(&replies, 35)["files"]
            .as_array()
            .unwrap()
            .iter()
            .any(|f| f == "archive://docs/a.txt")
    );

    assert_eq!(ok(&replies, 36)["deleted"], true);
    assert_eq!(ok(&replies, 38)["deleted"], true);
    // Positional synonyms are cleaned, and empty positions dropped.
    assert_eq!(
        resource_text(&replies, 40)["content"]["synonyms"],
        json!({ "head": ["alias"] })
    );
    assert_eq!(rpc_error(&replies, 41), -32602);
    assert_eq!(rpc_error(&replies, 42), -32002);
    assert!(tool_error(&replies, 43).contains("between 1 and"));

    // The notification got no reply; the parse error and the batch got id-less errors.
    assert!(!replies.contains_key(&99));
    let codes: Vec<i64> = others
        .iter()
        .map(|r| r["error"]["code"].as_i64().unwrap())
        .collect();
    assert_eq!(codes, [-32700, -32600]);
}

#[test]
fn an_agent_can_read_the_vocabulary_of_a_shelf_before_writing_into_it() {
    let home = TempDir::new().unwrap();
    let lines = vec![
        call(
            1,
            "knowledge_create",
            json!({ "name": "alpha", "tags": ["rule", "memory"], "scopes": ["proj-a", ""] }),
        ),
        call(
            2,
            "knowledge_create",
            json!({ "name": "beta", "tags": ["rule"], "scopes": ["proj-a"] }),
        ),
        call(
            3,
            "statement_create",
            json!({ "head": "alpha", "relation": "relatesTo", "tail": "beta", "scopes": ["proj-b"] }),
        ),
        call(4, "scope_list", json!({})),
        call(5, "tag_list", json!({})),
        call(6, "scope_exists", json!({ "value": "proj-a" })),
        call(7, "scope_exists", json!({ "value": "" })),
        call(8, "scope_exists", json!({ "value": "proj_a" })),
        call(9, "tag_exists", json!({ "value": "rule" })),
        call(10, "tag_exists", json!({ "value": "proj-a" })),
        call(11, "scope_list", json!({ "shelf": "no-such-shelf" })),
        call(12, "tag_exists", json!({ "name": "rule" })),
    ];
    let (replies, _) = session(&home, &lines);

    // The global scope stays the empty string here: this output is read by a model,
    // which passes it straight back to knowledge_create.
    assert_eq!(
        ok(&replies, 4),
        json!({
            "field": "scopes",
            "total_count": 3,
            "values": [
                { "value": "", "entries": 1 },
                { "value": "proj-a", "entries": 2 },
                { "value": "proj-b", "entries": 1 }
            ]
        }),
        "statements are enumerated with knowledge, and counts are per entry"
    );
    assert_eq!(
        ok(&replies, 5)["values"],
        json!([
            { "value": "memory", "entries": 1 },
            { "value": "rule", "entries": 2 }
        ])
    );

    assert_eq!(ok(&replies, 6)["exists"], true);
    assert_eq!(ok(&replies, 7)["exists"], true);
    // A near miss is reported, not guessed at: this is the check that stops an
    // agent from inventing a second spelling of a scope it already has.
    assert_eq!(
        ok(&replies, 8),
        json!({ "field": "scopes", "value": "proj_a", "exists": false })
    );
    assert_eq!(ok(&replies, 9)["exists"], true);
    assert_eq!(ok(&replies, 10)["exists"], false, "a scope is not a tag");

    assert!(tool_error(&replies, 11).contains("no-such-shelf"));
    assert!(tool_error(&replies, 12).contains("unknown field"));
}

#[test]
fn malformed_messages_get_errors_and_the_server_keeps_serving() {
    let home = TempDir::new().unwrap();
    let mut input = Vec::new();
    for line in [
        json!({ "jsonrpc": "2.0", "id": "abc", "method": "ping" }).to_string(),
        json!({ "jsonrpc": "2.0", "id": null, "method": "ping" }).to_string(),
        "42".to_string(),
        // A tools/call without an id is a notification: it must run nothing.
        json!({ "jsonrpc": "2.0", "method": "tools/call", "params": { "name": "knowledge_create", "arguments": { "name": "ghost" } } }).to_string(),
    ] {
        input.extend_from_slice(line.as_bytes());
        input.push(b'\n');
    }
    // Invalid UTF-8 inside an otherwise valid request.
    input.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\",\"x\":\"\xff\"}\n");
    input.extend_from_slice(
        json!({ "jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": { "name": "knowledge_get", "arguments": { "name": "ghost" } } })
            .to_string()
            .as_bytes(),
    );
    input.push(b'\n');

    let replies = run(&home, &input);
    assert_eq!(replies.len(), 5, "{replies:#?}");
    assert_eq!(replies[0]["id"], "abc");
    assert_eq!(replies[0]["result"], json!({}));
    let id_less: Vec<i64> = replies[1..4]
        .iter()
        .map(|r| {
            assert!(r["id"].is_null(), "{r}");
            r["error"]["code"].as_i64().unwrap()
        })
        .collect();
    assert_eq!(id_less, [-32600, -32600, -32700]);
    assert_eq!(replies[4]["id"], 8);
    assert_eq!(
        replies[4]["result"]["isError"], true,
        "the notification created nothing"
    );
}

#[test]
fn an_unknown_protocol_version_gets_the_newest_one() {
    let home = TempDir::new().unwrap();
    let (replies, _) = session(&home, &[initialize(1, "1999-01-01")]);
    assert_eq!(ok(&replies, 1)["protocolVersion"], "2025-11-25");
}

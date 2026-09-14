//! CLI contract for `knowledge-update`: only the named fields change, an empty value
//! clears a field, `created_at` is kept, and an update that changes nothing writes nothing.
use std::process::{Command, Output};
use tempfile::TempDir;

fn hypatia(home: &TempDir, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .args(args)
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .output()
        .expect("spawn hypatia")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn assert_ok(out: &Output) {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        stdout(out),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn get(home: &TempDir, name: &str) -> serde_json::Value {
    let out = hypatia(home, &["knowledge-get", name]);
    assert_ok(&out);
    serde_json::from_str(&stdout(&out)).expect("knowledge-get prints JSON")
}

#[test]
fn update_changes_only_named_fields_and_keeps_created_at() {
    let home = TempDir::new().unwrap();
    assert_ok(&hypatia(
        &home,
        &[
            "knowledge-create",
            "k",
            "-d",
            "original",
            "-t",
            "a,b",
            "--scopes",
            "p,",
            "--synonyms",
            "alias",
            "--figures",
            "archive://f.png",
        ],
    ));
    let before = get(&home, "k");

    let out = hypatia(&home, &["knowledge-update", "k", "-d", "changed"]);
    assert_ok(&out);
    assert_eq!(stdout(&out), "Updated knowledge: k\n");
    let after = get(&home, "k");
    assert_eq!(after["content"]["data"], "changed");
    assert_eq!(after["content"]["tags"], serde_json::json!(["a", "b"]));
    assert_eq!(after["content"]["scopes"], serde_json::json!(["p", ""]));
    assert_eq!(after["created_at"], before["created_at"]);
    assert_eq!(after["content"]["synonyms"], serde_json::json!(["alias"]));
    assert_eq!(
        after["content"]["figures"],
        serde_json::json!(["archive://f.png"])
    );

    // The full-text index follows the new content.
    let hit = hypatia(&home, &["search", "changed", "-c", "knowledge"]);
    assert!(stdout(&hit).contains("changed"), "{}", stdout(&hit));
    let miss = hypatia(&home, &["search", "original", "-c", "knowledge"]);
    assert!(
        stdout(&miss).contains("No results found."),
        "{}",
        stdout(&miss)
    );

    // An empty value clears that field and leaves the others alone.
    assert_ok(&hypatia(&home, &["knowledge-update", "k", "-t", ""]));
    let cleared = get(&home, "k");
    assert!(
        cleared["content"]["tags"]
            .as_array()
            .is_none_or(|tags| tags.is_empty()),
        "{cleared}"
    );
    assert_eq!(cleared["content"]["scopes"], serde_json::json!(["p", ""]));
    assert_eq!(cleared["content"]["data"], "changed");

    // Scopes are parsed as on create: a trailing comma keeps the global marker.
    assert_ok(&hypatia(
        &home,
        &["knowledge-update", "k", "--scopes", "q,"],
    ));
    assert_eq!(
        get(&home, "k")["content"]["scopes"],
        serde_json::json!(["q", ""])
    );

    let same = hypatia(&home, &["knowledge-update", "k", "-d", "changed"]);
    assert_ok(&same);
    assert_eq!(stdout(&same), "Knowledge unchanged: k\n");
}

#[test]
fn update_rejects_missing_entries_and_empty_patches() {
    let home = TempDir::new().unwrap();
    let missing = hypatia(&home, &["knowledge-update", "nope", "-d", "x"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&missing.stderr).contains("not found: knowledge 'nope'"),
        "{}",
        String::from_utf8_lossy(&missing.stderr)
    );

    assert_ok(&hypatia(&home, &["knowledge-create", "k", "-d", "x"]));
    let empty = hypatia(&home, &["knowledge-update", "k"]);
    assert_eq!(empty.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&empty.stderr).contains("nothing to update"));
    assert_eq!(get(&home, "k")["content"]["data"], "x");
}

//! CLI contract for `scope` and `tag`: an agent checks the vocabulary of a shelf
//! before writing into it, and `exists` answers through the exit code so a script
//! can branch on it without parsing anything.
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

fn assert_ok(out: &Output) -> String {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        stdout(out),
        String::from_utf8_lossy(&out.stderr)
    );
    stdout(out)
}

fn seed(home: &TempDir) {
    assert_ok(&hypatia(
        home,
        &[
            "knowledge-create",
            "alpha",
            "-d",
            "a",
            "-t",
            "rule,memory",
            "--scopes",
            "proj-a,",
        ],
    ));
    assert_ok(&hypatia(
        home,
        &[
            "knowledge-create",
            "beta",
            "-d",
            "b",
            "-t",
            "rule",
            "--scopes",
            "proj-a",
        ],
    ));
    // No tags and no scopes at all. Both are stored as JSON null, whose token is
    // the empty string, so this entry must not turn up as a global scope.
    assert_ok(&hypatia(home, &["knowledge-create", "bare", "-d", "c"]));
    // Statements carry scopes too, and must be counted with knowledge.
    assert_ok(&hypatia(
        home,
        &[
            "statement-create",
            "alpha",
            "relatesTo",
            "beta",
            "--scopes",
            "proj-b",
        ],
    ));
}

#[test]
fn list_enumerates_both_catalogs_and_names_the_global_scope() {
    let home = TempDir::new().unwrap();
    assert_eq!(
        assert_ok(&hypatia(&home, &["scope", "list"])),
        "No scopes in shelf 'default'.\n"
    );

    seed(&home);

    let scopes = assert_ok(&hypatia(&home, &["scope", "list"]));
    assert_eq!(scopes, "  (global)\n  proj-a\n  proj-b\n  (3 scopes)\n");

    // --count reports entries, not postings: proj-a is on two entries.
    let counted = assert_ok(&hypatia(&home, &["scope", "list", "--count"]));
    assert_eq!(
        counted,
        "  (global)  1\n  proj-a    2\n  proj-b    1\n  (3 scopes)\n"
    );

    let tags = assert_ok(&hypatia(&home, &["tag", "list", "--count"]));
    assert_eq!(tags, "  memory  1\n  rule    2\n  (2 tags)\n");
}

#[test]
fn json_output_carries_the_values_verbatim() {
    let home = TempDir::new().unwrap();
    assert_eq!(
        assert_ok(&hypatia(&home, &["tag", "list", "--json"])),
        "[]\n",
        "an empty shelf is an empty list, not prose"
    );

    seed(&home);
    // A scope whose name is the label `list` prints for the global scope: the two
    // are indistinguishable in the terminal, which is what --json is for.
    assert_ok(&hypatia(
        &home,
        &[
            "knowledge-create",
            "trap",
            "-d",
            "t",
            "--scopes",
            "(global)",
        ],
    ));

    let json: serde_json::Value =
        serde_json::from_str(&assert_ok(&hypatia(&home, &["scope", "list", "--json"]))).unwrap();
    assert_eq!(
        json,
        serde_json::json!([
            { "value": "", "entries": 1 },
            { "value": "(global)", "entries": 1 },
            { "value": "proj-a", "entries": 2 },
            { "value": "proj-b", "entries": 1 }
        ])
    );

    // Both are in use, and each answers only for itself.
    assert!(hypatia(&home, &["scope", "exists", ""]).status.success());
    let literal = hypatia(&home, &["scope", "exists", "(global)"]);
    assert!(literal.status.success());
    assert_eq!(stdout(&literal), "\"(global)\"\n");
    assert_eq!(stdout(&hypatia(&home, &["scope", "exists", ""])), "\"\"\n");
}

#[test]
fn exists_answers_through_the_exit_code() {
    let home = TempDir::new().unwrap();
    seed(&home);

    let found = hypatia(&home, &["scope", "exists", "proj-a"]);
    assert!(found.status.success());
    assert_eq!(stdout(&found), "\"proj-a\"\n");

    // The global scope is the empty string, and it really is in use here.
    let global = hypatia(&home, &["scope", "exists", ""]);
    assert!(global.status.success());
    assert_eq!(stdout(&global), "\"\"\n");

    let missing = hypatia(&home, &["scope", "exists", "proj_a"]);
    assert_eq!(missing.status.code(), Some(1));
    assert_eq!(stdout(&missing), "", "the answer is the exit code");
    let hint = String::from_utf8_lossy(&missing.stderr);
    assert!(hint.contains("no such scope: \"proj_a\""), "{hint}");
    assert!(hint.contains("hypatia scope list"), "{hint}");

    assert!(hypatia(&home, &["tag", "exists", "rule"]).status.success());
    assert_eq!(
        hypatia(&home, &["tag", "exists", "proj-a"]).status.code(),
        Some(1),
        "a scope is not a tag"
    );
}

#[test]
fn deleted_entries_take_their_values_out_of_the_listing() {
    let home = TempDir::new().unwrap();
    seed(&home);
    assert_ok(&hypatia(&home, &["knowledge-delete", "alpha"]));

    // "memory" was only on alpha; "rule" and proj-a survive on beta.
    assert_eq!(
        assert_ok(&hypatia(&home, &["tag", "list"])),
        "  rule\n  (1 tags)\n"
    );
    assert_eq!(
        assert_ok(&hypatia(&home, &["scope", "list"])),
        "  proj-a\n  proj-b\n  (2 scopes)\n"
    );
    assert_eq!(
        hypatia(&home, &["tag", "exists", "memory"]).status.code(),
        Some(1)
    );
}

#[test]
fn an_unknown_shelf_is_an_error_rather_than_an_empty_listing() {
    let home = TempDir::new().unwrap();
    let out = hypatia(&home, &["scope", "list", "-s", "no-such-shelf"]);
    assert_eq!(out.status.code(), Some(1));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("no-such-shelf"), "{err}");
}

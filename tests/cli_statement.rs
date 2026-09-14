//! CLI contract for `statement-create`: a duplicate triple is a no-op with exit 0,
//! so callers can replay a half-finished multi-step graph write.
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

#[test]
fn duplicate_statement_create_is_a_noop_with_exit_zero() {
    let home = TempDir::new().unwrap();

    let first = hypatia(
        &home,
        &[
            "statement-create",
            "Alice",
            "knows",
            "Bob",
            "--data=original",
        ],
    );
    assert_ok(&first);
    assert_eq!(stdout(&first), "Created statement: (Alice, knows, Bob)\n");

    let again = hypatia(
        &home,
        &[
            "statement-create",
            "Alice",
            "knows",
            "Bob",
            "--data=changed",
        ],
    );
    assert_ok(&again);
    assert_eq!(
        stdout(&again),
        "Statement already exists: (Alice, knows, Bob)\n"
    );

    // The stored statement and its FTS doc were not overwritten.
    let kept = hypatia(&home, &["search", "original", "-c", "statement"]);
    assert_ok(&kept);
    assert!(stdout(&kept).contains("Alice"), "{}", stdout(&kept));
    let replaced = hypatia(&home, &["search", "changed", "-c", "statement"]);
    assert_ok(&replaced);
    assert!(
        stdout(&replaced).contains("No results found."),
        "{}",
        stdout(&replaced)
    );
}

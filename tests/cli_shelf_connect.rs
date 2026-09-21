//! CLI contract for connecting shelves: a directory is one shelf, under one name. `connect`
//! and `init` refuse a second name for it alike, and a second name registered by an older
//! version is reported at every start until `disconnect` removes it.
use std::path::Path;
use std::process::{Command, Output};
use tempfile::TempDir;

fn hypatia(home: &TempDir, args: &[&str]) -> Output {
    hypatia_in(home, home.path(), args)
}

fn hypatia_in(home: &TempDir, cwd: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .args(args)
        .current_dir(cwd)
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .output()
        .expect("spawn hypatia")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_ok(out: &Output) {
    assert!(
        out.status.success(),
        "exit {:?}\nstdout: {}\nstderr: {}",
        out.status.code(),
        stdout(out),
        stderr(out)
    );
}

fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// The names `hypatia list` shows, each with whether it is connected.
fn listed(home: &TempDir) -> Vec<(String, bool)> {
    let out = hypatia(home, &["list"]);
    assert_ok(&out);
    stdout(&out)
        .lines()
        .map(|line| {
            let name = line.split_whitespace().next().unwrap().to_string();
            (name, line.ends_with("[connected]"))
        })
        .collect()
}

#[test]
fn connect_and_init_refuse_a_second_name_for_one_directory() {
    let home = TempDir::new().unwrap();
    let shelf = TempDir::new().unwrap();
    assert_ok(&hypatia(&home, &["connect", path(shelf.path()), "-n", "a"]));

    let out = hypatia(&home, &["connect", path(shelf.path()), "-n", "b"]);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    assert!(
        stderr(&out).contains("is already connected as 'a'"),
        "{}",
        stderr(&out)
    );

    // `init` says the same about the same state.
    let out = hypatia(&home, &["init", path(shelf.path()), "-n", "c"]);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    assert!(
        stderr(&out).contains("is already connected as 'a'"),
        "{}",
        stderr(&out)
    );

    assert_eq!(
        listed(&home),
        [("a".to_string(), true), ("default".to_string(), true)]
    );
}

#[test]
fn a_second_name_left_by_an_older_version_is_reported_and_removable() {
    let home = TempDir::new().unwrap();
    let shelf = TempDir::new().unwrap();
    let registry = home.path().join(".hypatia");
    std::fs::create_dir_all(&registry).unwrap();
    std::fs::write(
        registry.join("shelves.json"),
        serde_json::json!({ "shelves": { "b": shelf.path(), "a": shelf.path() } }).to_string(),
    )
    .unwrap();

    let out = hypatia(
        &home,
        &["knowledge-create", "probe", "-d", "written to a", "-s", "a"],
    );
    assert_ok(&out);
    // Every run says which name is not opened, and how to be rid of it.
    let warning = stderr(&out);
    assert!(
        warning.contains("failed to restore shelf 'b'")
            && warning.contains("already connected as 'a'")
            && warning.contains("hypatia disconnect b"),
        "{warning}"
    );

    // The second name no longer reaches the first one's data.
    let out = hypatia(&home, &["knowledge-get", "probe", "-s", "b"]);
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    assert!(
        stderr(&out).contains("shelf 'b' is not connected"),
        "{}",
        stderr(&out)
    );

    assert_ok(&hypatia(&home, &["disconnect", "b"]));
    let out = hypatia(&home, &["knowledge-get", "probe", "-s", "a"]);
    assert_ok(&out);
    assert!(stdout(&out).contains("written to a"));
    assert!(
        !stderr(&out).contains("failed to restore"),
        "{}",
        stderr(&out)
    );
    assert_eq!(
        listed(&home),
        [("a".to_string(), true), ("default".to_string(), true)]
    );
}

#[test]
fn a_relative_path_names_the_same_directory_from_anywhere() {
    let home = TempDir::new().unwrap();
    let work = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    std::fs::create_dir(work.path().join("s")).unwrap();
    assert_ok(&hypatia_in(
        &home,
        work.path(),
        &["connect", "s", "-n", "a"],
    ));

    // Registered absolute, so the check does not depend on where it runs.
    let out = hypatia_in(
        &home,
        elsewhere.path(),
        &["connect", path(&work.path().join("s")), "-n", "b"],
    );
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    assert!(
        stderr(&out).contains("is already connected as 'a'"),
        "{}",
        stderr(&out)
    );
    // Another directory of the same relative name is another shelf.
    std::fs::create_dir(elsewhere.path().join("s")).unwrap();
    assert_ok(&hypatia_in(
        &home,
        elsewhere.path(),
        &["connect", "s", "-n", "c"],
    ));
    assert_eq!(
        listed(&home),
        [
            ("a".to_string(), true),
            ("c".to_string(), true),
            ("default".to_string(), true)
        ]
    );
}

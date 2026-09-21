//! CLI contract for the `default` shelf: where the registry points is where `-s default`
//! reads and writes, and `hypatia list` names that same directory. When the two could
//! disagree the CLI says so, rather than reporting one shelf and writing to another.
use std::path::Path;
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

/// Point `default` at `path`, as an older version or a hand edit would leave it.
fn point_default_at(home: &TempDir, path: &Path) {
    let dir = home.path().join(".hypatia");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("shelves.json"),
        serde_json::json!({ "shelves": { "default": path } }).to_string(),
    )
    .unwrap();
}

#[test]
fn default_reads_and_writes_the_registered_directory() {
    let home = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    point_default_at(&home, elsewhere.path());

    let out = hypatia(
        &home,
        &[
            "knowledge-create",
            "probe",
            "-d",
            "which dir?",
            "-s",
            "default",
        ],
    );
    assert_ok(&out);

    assert!(elsewhere.path().join("hypatia.sqlite").exists());
    assert!(
        !home.path().join(".hypatia/default/hypatia.sqlite").exists(),
        "the write landed in the built-in shelf instead of the registered one"
    );

    // Readable back through the same name, from a second process.
    let out = hypatia(&home, &["knowledge-get", "probe", "-s", "default"]);
    assert_ok(&out);
    assert!(stdout(&out).contains("which dir?"));
}

#[test]
fn list_names_the_directory_default_writes_to() {
    let home = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    point_default_at(&home, elsewhere.path());

    assert_ok(&hypatia(
        &home,
        &[
            "knowledge-create",
            "probe",
            "-d",
            "which dir?",
            "-s",
            "default",
        ],
    ));
    // This line alone does not discriminate: the buggy build printed the registry path
    // beside `[connected]` too. The `init` cross-check below is what pins the fix.
    let listed = stdout(&hypatia(&home, &["list"]));
    let line = listed
        .lines()
        .find(|l| l.split_whitespace().next() == Some("default"))
        .unwrap_or_else(|| panic!("no default in: {listed}"));
    assert!(
        line.contains(elsewhere.path().to_str().unwrap()) && line.contains("[connected]"),
        "list should name the shelf being written to: {line}"
    );

    // `init` reads the open shelf, and the two agreeing is the point.
    let reported = stdout(&hypatia(&home, &["init"]));
    assert!(
        reported.contains(elsewhere.path().to_str().unwrap()),
        "init and list disagree: {reported}"
    );
}

#[test]
fn an_unopenable_registered_default_fails_loudly_instead_of_writing_elsewhere() {
    let home = TempDir::new().unwrap();
    let unopenable = TempDir::new().unwrap();
    std::fs::write(unopenable.path().join("shelf.toml"), "embedding = 1").unwrap();
    point_default_at(&home, unopenable.path());

    let out = hypatia(
        &home,
        &["knowledge-create", "probe", "-d", "x", "-s", "default"],
    );
    assert!(!out.status.success(), "stdout: {}", stdout(&out));
    let message = stderr(&out);
    assert!(
        message.contains(unopenable.path().to_str().unwrap()) && message.contains("shelves.json"),
        "the error should name the shelf and where its path is set: {message}"
    );

    // The built-in shelf is not quietly stood up to receive the write instead:
    // `docs/pgvector-backend.md` rules that out for a database that is merely down.
    assert!(!home.path().join(".hypatia/default").exists());
}

#[test]
fn the_documented_recovery_repoints_default() {
    let home = TempDir::new().unwrap();
    let elsewhere = TempDir::new().unwrap();
    point_default_at(&home, elsewhere.path());

    assert_ok(&hypatia(&home, &["disconnect", "default"]));
    let listed = stdout(&hypatia(&home, &["list"]));
    let builtin = home.path().join(".hypatia").join("default");
    assert!(
        listed.contains(builtin.to_str().unwrap()),
        "default should be rebuilt at the built-in path: {listed}"
    );

    let out = hypatia(&home, &["connect", elsewhere.path().to_str().unwrap()]);
    assert_ok(&out);
    let name = elsewhere.path().file_name().unwrap().to_str().unwrap();
    assert!(stdout(&hypatia(&home, &["list"])).contains(name));
}

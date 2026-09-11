//! CLI contract for `hypatia skill`: bundled skills install byte-for-byte, reruns are
//! idempotent, and copies the user edited are never overwritten without `--force`.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const SKILLS: &[&str] = &["hypatia", "hypatia-memory", "hypatia-dream"];

fn hypatia(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .args(args)
        .env("HOME", home)
        .env("USERPROFILE", home)
        // Decoys: the code must ignore these, and the tests assert the decoy
        // dirs stay untouched, so a regression cannot write into a real config.
        .env("CLAUDE_CONFIG_DIR", home.join("decoy-claude"))
        .env("CODEX_HOME", home.join("decoy-codex"))
        .env_remove("OPENCODE_CONFIG_DIR")
        .env_remove("XDG_CONFIG_HOME")
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

fn source(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("skills")
        .join(name)
        .join("SKILL.md");
    std::fs::read_to_string(path).unwrap()
}

fn installed(dir: &Path, name: &str) -> PathBuf {
    dir.join(name).join("SKILL.md")
}

#[test]
fn install_writes_bundled_skills_and_reruns_are_idempotent() {
    let home = TempDir::new().unwrap();
    let dir = home.path().join("skills");
    let d = dir.to_str().unwrap();

    let first = hypatia(home.path(), &["skill", "install", "--dir", d]);
    assert_ok(&first);
    for name in SKILLS {
        assert_eq!(
            std::fs::read_to_string(installed(&dir, name)).unwrap(),
            source(name)
        );
        assert!(
            stdout(&first).contains(&format!("installed dir: {name}")),
            "{}",
            stdout(&first)
        );
    }
    // Skill management must not open or create any shelf.
    assert!(!home.path().join(".hypatia").join("default").exists());

    let again = hypatia(home.path(), &["skill", "install", "--dir", d]);
    assert_ok(&again);
    assert_eq!(stdout(&again).matches("unchanged ").count(), SKILLS.len());

    let status = hypatia(home.path(), &["skill", "status", "--dir", d]);
    assert_ok(&status);
    assert_eq!(stdout(&status).matches("current ").count(), SKILLS.len());
}

#[test]
fn edited_and_foreign_copies_need_force() {
    let home = TempDir::new().unwrap();
    let dir = home.path().join("skills");
    let d = dir.to_str().unwrap();
    assert_ok(&hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    ));

    // Edited after install.
    std::fs::write(installed(&dir, "hypatia"), "my local notes").unwrap();
    // Copied in by hand, never installed by hypatia.
    std::fs::create_dir_all(dir.join("hypatia-dream")).unwrap();
    std::fs::write(installed(&dir, "hypatia-dream"), "an old manual copy").unwrap();

    let status = hypatia(home.path(), &["skill", "status", "--dir", d]);
    assert_ok(&status);
    let s = stdout(&status);
    assert!(s.contains("modified  dir: hypatia "), "{s}");
    assert!(s.contains("unmanaged dir: hypatia-dream "), "{s}");
    assert!(s.contains("missing   dir: hypatia-memory "), "{s}");

    let refused = hypatia(home.path(), &["skill", "install", "--dir", d]);
    assert!(!refused.status.success(), "{}", stdout(&refused));
    assert_eq!(
        std::fs::read_to_string(installed(&dir, "hypatia")).unwrap(),
        "my local notes"
    );
    assert_eq!(
        std::fs::read_to_string(installed(&dir, "hypatia-dream")).unwrap(),
        "an old manual copy"
    );
    // Refusing is stable: a second run without --force must not treat the refused
    // files as hypatia's own (which would let a third run overwrite them).
    let refused_again = hypatia(home.path(), &["skill", "install", "--dir", d]);
    assert!(!refused_again.status.success());
    assert_eq!(
        std::fs::read_to_string(installed(&dir, "hypatia")).unwrap(),
        "my local notes"
    );
    let manifest =
        std::fs::read_to_string(home.path().join(".hypatia").join("skill-installs.json")).unwrap();
    assert!(
        !manifest.contains(
            &installed(&dir, "hypatia-dream")
                .to_string_lossy()
                .into_owned()
        ),
        "{manifest}"
    );
    assert!(
        stdout(&refused).contains("skipped   dir: hypatia "),
        "{}",
        stdout(&refused)
    );
    // The skill that was safe to write still got written.
    assert_eq!(
        std::fs::read_to_string(installed(&dir, "hypatia-memory")).unwrap(),
        source("hypatia-memory")
    );

    let forced = hypatia(home.path(), &["skill", "install", "--dir", d, "--force"]);
    assert_ok(&forced);
    for name in SKILLS {
        assert_eq!(
            std::fs::read_to_string(installed(&dir, name)).unwrap(),
            source(name)
        );
    }
}

#[test]
fn copy_from_an_older_hypatia_upgrades_without_force() {
    let home = TempDir::new().unwrap();
    let dir = home.path().join("skills");
    let d = dir.to_str().unwrap();
    assert_ok(&hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    ));

    // Simulate an older bundled version: the file and the install record agree
    // with each other but not with this binary.
    let path = installed(&dir, "hypatia");
    std::fs::write(&path, "older bundled text").unwrap();
    let manifest_path = home.path().join(".hypatia").join("skill-installs.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let sha = {
        use sha2::{Digest, Sha256};
        Sha256::digest(b"older bundled text")
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let key = path.to_string_lossy().into_owned();
    manifest["installs"][&key]["sha256"] = serde_json::Value::String(sha);
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();

    let status = hypatia(home.path(), &["skill", "status", "--dir", d]);
    assert!(
        stdout(&status).contains("outdated  dir: hypatia "),
        "{}",
        stdout(&status)
    );

    let upgrade = hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    );
    assert_ok(&upgrade);
    assert!(
        stdout(&upgrade).contains("updated   dir: hypatia "),
        "{}",
        stdout(&upgrade)
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), source("hypatia"));
}

#[test]
fn agent_claude_installs_under_home_and_unknown_skill_is_rejected() {
    let home = TempDir::new().unwrap();
    let out = hypatia(
        home.path(),
        &[
            "skill", "install", "--agent", "claude", "--skill", "hypatia",
        ],
    );
    assert_ok(&out);
    let path = home
        .path()
        .join(".claude")
        .join("skills")
        .join("hypatia")
        .join("SKILL.md");
    assert_eq!(std::fs::read_to_string(path).unwrap(), source("hypatia"));

    let bad = hypatia(
        home.path(),
        &["skill", "install", "--agent", "claude", "--skill", "nope"],
    );
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("unknown skill 'nope'"));

    let no_target = hypatia(home.path(), &["skill", "install"]);
    assert_eq!(
        no_target.status.code(),
        Some(2),
        "clap usage error, not a panic"
    );
    assert!(String::from_utf8_lossy(&no_target.stderr).contains("--agent"));
}

#[test]
fn each_agent_resolves_to_its_verified_user_skills_dir() {
    let home = TempDir::new().unwrap();
    let out = hypatia(
        home.path(),
        &[
            "skill",
            "install",
            "--agent",
            "claude,codex",
            "--agent",
            "opencode",
            "--skill",
            "hypatia",
        ],
    );
    assert_ok(&out);
    for dir in [
        home.path().join(".claude").join("skills"),
        home.path().join(".agents").join("skills"),
        home.path().join(".config").join("opencode").join("skills"),
    ] {
        assert_eq!(
            std::fs::read_to_string(installed(&dir, "hypatia")).unwrap(),
            source("hypatia"),
            "{}",
            dir.display()
        );
    }
    // CLAUDE_CONFIG_DIR and CODEX_HOME are ignored for installing.
    assert!(!home.path().join("decoy-claude").exists());
    assert!(!home.path().join("decoy-codex").exists());
    // Every copy is current, so there is nothing to warn about.
    assert!(!stdout(&out).contains("warning"), "{}", stdout(&out));

    // OPENCODE_CONFIG_DIR replaces OpenCode's whole config directory.
    let custom = home.path().join("oc");
    let out = Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .args([
            "skill", "install", "--agent", "opencode", "--skill", "hypatia",
        ])
        .env("HOME", home.path())
        .env("USERPROFILE", home.path())
        .env("OPENCODE_CONFIG_DIR", &custom)
        .env_remove("XDG_CONFIG_HOME")
        .output()
        .unwrap();
    assert_ok(&out);
    assert!(installed(&custom.join("skills"), "hypatia").exists());
}

#[test]
fn identical_hand_copy_is_adopted_and_corrupt_manifest_explains_reset() {
    let home = TempDir::new().unwrap();
    let dir = home.path().join("skills");
    let d = dir.to_str().unwrap();
    std::fs::create_dir_all(dir.join("hypatia")).unwrap();
    std::fs::write(installed(&dir, "hypatia"), source("hypatia")).unwrap();

    let out = hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    );
    assert_ok(&out);
    assert!(
        stdout(&out).contains("unchanged dir: hypatia "),
        "{}",
        stdout(&out)
    );
    let manifest_path = home.path().join(".hypatia").join("skill-installs.json");
    assert!(
        std::fs::read_to_string(&manifest_path)
            .unwrap()
            .contains("\"installs\"")
    );

    std::fs::write(&manifest_path, "{ truncated").unwrap();
    let broken = hypatia(home.path(), &["skill", "status", "--dir", d]);
    assert!(!broken.status.success());
    assert!(String::from_utf8_lossy(&broken.stderr).contains("Remove it to reset"));
}

#[test]
fn refuses_without_a_home_directory() {
    let dir = TempDir::new().unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_hypatia"))
        .args(["skill", "install", "--dir", dir.path().to_str().unwrap()])
        .env_remove("HOME")
        .env_remove("USERPROFILE")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot locate the home directory"));
    assert!(
        !dir.path().join(".hypatia").exists(),
        "must not fall back to the cwd"
    );
    assert!(!dir.path().join("hypatia").exists());
}

#[cfg(unix)]
#[test]
fn outdated_symlink_is_not_replaced_without_force() {
    let home = TempDir::new().unwrap();
    let dir = home.path().join("skills");
    let d = dir.to_str().unwrap();
    assert_ok(&hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    ));

    // The installed file now lives in a "dotfiles" location, linked back in, and
    // is an older bundled version (file and install record agree).
    let path = installed(&dir, "hypatia");
    let real = home.path().join("dotfiles-SKILL.md");
    std::fs::write(&real, "older bundled text").unwrap();
    std::fs::remove_file(&path).unwrap();
    std::os::unix::fs::symlink(&real, &path).unwrap();
    let manifest_path = home.path().join(".hypatia").join("skill-installs.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let sha = {
        use sha2::{Digest, Sha256};
        Sha256::digest(b"older bundled text")
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    manifest["installs"][path.to_string_lossy().as_ref()]["sha256"] =
        serde_json::Value::String(sha);
    std::fs::write(&manifest_path, manifest.to_string()).unwrap();

    let out = hypatia(
        home.path(),
        &["skill", "install", "--dir", d, "--skill", "hypatia"],
    );
    assert!(!out.status.success());
    assert!(
        std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(
        std::fs::read_to_string(&real).unwrap(),
        "older bundled text"
    );
}

#[test]
fn stale_copies_the_host_also_loads_are_reported() {
    let home = TempDir::new().unwrap();
    // A stale hand copy where OpenCode also looks, and one in Codex's deprecated dir.
    let claude = home.path().join(".claude").join("skills");
    let codex_old = home.path().join("decoy-codex").join("skills");
    for dir in [&claude, &codex_old] {
        std::fs::create_dir_all(dir.join("hypatia")).unwrap();
        std::fs::write(installed(dir, "hypatia"), "stale").unwrap();
    }

    let out = hypatia(
        home.path(),
        &[
            "skill",
            "install",
            "--agent",
            "opencode,codex",
            "--skill",
            "hypatia",
        ],
    );
    assert_ok(&out);
    let s = stdout(&out);
    let claude_copy = installed(&claude, "hypatia");
    let codex_copy = installed(&codex_old, "hypatia");
    assert!(
        s.contains(&format!(
            "warning   opencode: hypatia  {}",
            claude_copy.display()
        )),
        "{s}"
    );
    assert!(
        s.contains(&format!(
            "warning   codex: hypatia  {}",
            codex_copy.display()
        )),
        "{s}"
    );
    // Warnings never touch the other copies.
    assert_eq!(std::fs::read_to_string(claude_copy).unwrap(), "stale");

    let status = hypatia(home.path(), &["skill", "status", "--agent", "opencode"]);
    assert!(
        stdout(&status).contains("warning   opencode: hypatia"),
        "{}",
        stdout(&status)
    );
}

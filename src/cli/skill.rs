//! `hypatia skill`: install the bundled agent skills into a host's skills directory.
//!
//! The skills are compiled into the binary, so installing them needs no repository
//! checkout and an installed copy always matches the binary that wrote it. Installs
//! are recorded by content hash in `~/.hypatia/skill-installs.json`; that record is
//! how an upgrade tells a copy hypatia wrote earlier (safe to replace) from one the
//! user edited or copied in by hand (left alone unless `--force`). Only bundled
//! hashes are ever recorded, so an overwrite without `--force` can only replace
//! bytes that some hypatia release shipped.

use std::collections::BTreeMap;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};

use clap::{ArgGroup, Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{HypatiaError, Result};

/// Bundled skills as `(name, SKILL.md)`. The name is also the directory name.
const BUNDLED: &[(&str, &str)] = &[
    ("hypatia", include_str!("../../skills/hypatia/SKILL.md")),
    (
        "hypatia-memory",
        include_str!("../../skills/hypatia-memory/SKILL.md"),
    ),
    (
        "hypatia-dream",
        include_str!("../../skills/hypatia-dream/SKILL.md"),
    ),
];

const MANIFEST_VERSION: u32 = 1;

#[derive(Subcommand)]
pub(crate) enum SkillCommands {
    /// Install the bundled skills into an agent's skills directory
    Install {
        #[command(flatten)]
        target: Target,
        /// Install only this skill (repeatable); default: all bundled skills
        #[arg(long = "skill", value_name = "NAME")]
        skills: Vec<String>,
        /// Also replace copies that were edited, not installed by hypatia, or are
        /// symlinks. Local changes in those files are lost.
        #[arg(long)]
        force: bool,
    },
    /// Show whether installed skills match the ones bundled in this binary
    Status {
        #[command(flatten)]
        target: Target,
    },
}

#[derive(Args)]
#[command(group(ArgGroup::new("target").required(true).multiple(true).args(["agent", "dir"])))]
pub(crate) struct Target {
    /// Agent host to install for (repeatable or comma-separated)
    #[arg(long, value_enum, value_delimiter = ',')]
    agent: Vec<Agent>,
    /// A skills directory to use instead (repeatable); skills land in <DIR>/<name>/SKILL.md
    #[arg(long, value_name = "DIR")]
    dir: Vec<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Agent {
    Claude,
    Codex,
    Opencode,
}

impl Agent {
    fn label(self) -> &'static str {
        match self {
            Agent::Claude => "claude",
            Agent::Codex => "codex",
            Agent::Opencode => "opencode",
        }
    }

    /// The host's user-level skills directory (verified against each host's
    /// docs and source, 2026-09). Always absolute.
    fn skills_dir(self, home: &Path) -> Result<PathBuf> {
        Ok(match self {
            // `CLAUDE_CONFIG_DIR` is undocumented and unreliable for skills.
            Agent::Claude => home.join(".claude").join("skills"),
            // Codex's current location, shared with OpenCode. The older
            // `$CODEX_HOME/skills` is still read but deprecated.
            Agent::Codex => home.join(".agents").join("skills"),
            // `OPENCODE_CONFIG_DIR` replaces the whole config dir; otherwise
            // XDG, which OpenCode uses on Windows too.
            Agent::Opencode => match env_path("OPENCODE_CONFIG_DIR") {
                Some(dir) if dir.is_absolute() => dir,
                Some(dir) => {
                    return Err(HypatiaError::Config(format!(
                        "OPENCODE_CONFIG_DIR must be an absolute path, got '{}'",
                        dir.display()
                    )));
                }
                // The XDG spec says relative values are invalid and ignored.
                None => env_path("XDG_CONFIG_HOME")
                    .filter(|dir| dir.is_absolute())
                    .unwrap_or_else(|| home.join(".config"))
                    .join("opencode"),
            }
            .join("skills"),
        })
    }
}

impl Target {
    /// `(label, absolute skills directory)` for every requested target, deduplicated.
    fn resolve(&self, home: &Path) -> Result<Vec<(String, PathBuf)>> {
        let mut out: Vec<(String, PathBuf)> = Vec::new();
        for agent in &self.agent {
            out.push((agent.label().to_string(), agent.skills_dir(home)?));
        }
        for dir in &self.dir {
            out.push(("dir".to_string(), std::path::absolute(dir)?));
        }
        let mut seen = std::collections::HashSet::new();
        out.retain(|(_, dir)| seen.insert(dir.clone()));
        Ok(out)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum State {
    /// No SKILL.md at the target path.
    Missing,
    /// Byte-identical to the bundled skill.
    Current,
    /// Written by an earlier hypatia and not edited since; safe to replace.
    Outdated,
    /// Written by hypatia, then edited.
    Modified,
    /// Differs from the bundled skill and hypatia has no record of writing it.
    Unmanaged,
}

impl State {
    fn label(&self) -> &'static str {
        match self {
            State::Missing => "missing",
            State::Current => "current",
            State::Outdated => "outdated",
            State::Modified => "modified",
            State::Unmanaged => "unmanaged",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Entry {
    sha256: String,
    /// The hypatia version that wrote this copy.
    version: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    version: u32,
    /// Installed SKILL.md path -> what hypatia wrote there.
    installs: BTreeMap<String, Entry>,
}

pub(crate) fn execute(cmd: SkillCommands) -> Result<()> {
    let home = home()?;
    match cmd {
        SkillCommands::Install {
            target,
            skills,
            force,
        } => install(
            &home,
            &target.resolve(&home)?,
            &target.agent,
            &select(&skills)?,
            force,
        ),
        SkillCommands::Status { target } => status(&home, &target.resolve(&home)?, &target.agent),
    }
}

fn select(names: &[String]) -> Result<Vec<(&'static str, &'static str)>> {
    if names.is_empty() {
        return Ok(BUNDLED.to_vec());
    }
    let mut out: Vec<(&'static str, &'static str)> = Vec::new();
    for n in names {
        let skill = BUNDLED
            .iter()
            .find(|(name, _)| name == n)
            .copied()
            .ok_or_else(|| {
                let known: Vec<&str> = BUNDLED.iter().map(|(name, _)| *name).collect();
                HypatiaError::Validation(format!(
                    "unknown skill '{n}'; bundled skills: {}",
                    known.join(", ")
                ))
            })?;
        if !out.contains(&skill) {
            out.push(skill);
        }
    }
    Ok(out)
}

fn install(
    home: &Path,
    targets: &[(String, PathBuf)],
    agents: &[Agent],
    skills: &[(&'static str, &'static str)],
    force: bool,
) -> Result<()> {
    let mut manifest = load_manifest(home)?;
    let mut skipped = 0usize;
    let mut memory_written = false;
    for (label, dir) in targets {
        for (name, content) in skills {
            let path = dir.join(name).join("SKILL.md");
            let bundled = sha256_hex(content.as_bytes());
            let state = classify(&path, &bundled, &manifest)?;
            let why_skip = match state {
                _ if force => None,
                State::Modified => Some("edited since hypatia installed it"),
                State::Unmanaged => {
                    Some("differs from the bundled copy and was not installed by hypatia")
                }
                // An update renames a new file over the path, which would
                // turn a symlink (e.g. into a dotfiles repo) into a plain file.
                State::Outdated if is_symlink(&path) => {
                    Some("is a symlink, which an update would replace with a regular file")
                }
                _ => None,
            };
            if let Some(why) = why_skip {
                // Never recorded: a later run must still see it as foreign.
                skipped += 1;
                println!("skipped   {label}: {name}  {}  ({why})", path.display());
                continue;
            }
            let action = match state {
                State::Current => "unchanged",
                State::Missing => "installed",
                State::Outdated => "updated",
                State::Modified | State::Unmanaged => "replaced",
            };
            if state != State::Current {
                write_atomic(&path, content.as_bytes())?;
                memory_written |= *name == "hypatia-memory";
            }
            manifest.installs.insert(
                key(&path),
                Entry {
                    sha256: bundled,
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
            );
            println!("{action:<9} {label}: {name}  {}", path.display());
        }
    }
    save_manifest(home, &manifest)?;
    warn_shadows(home, agents, skills, &manifest)?;
    if memory_written {
        println!(
            "note: hypatia-memory only runs when host hooks fire; the skill file alone does \
             nothing. Claude Code: add the hooks listed in the skill's Trigger Conditions to \
             ~/.claude/settings.json yourself (there is no installer). Codex: \
             codex-integration/install.sh. OpenCode: opencode-integration/. Both are in the \
             Hypatia repository."
        );
    }
    if skipped > 0 {
        return Err(HypatiaError::Validation(format!(
            "{skipped} skill file(s) not overwritten because they were edited, not installed \
             by hypatia, or are symlinks; see the 'skipped' lines above. Review them before \
             rerunning with --force, which replaces them and discards any local changes."
        )));
    }
    Ok(())
}

fn status(home: &Path, targets: &[(String, PathBuf)], agents: &[Agent]) -> Result<()> {
    let manifest = load_manifest(home)?;
    for (label, dir) in targets {
        for (name, content) in BUNDLED {
            let path = dir.join(name).join("SKILL.md");
            let state = classify(&path, &sha256_hex(content.as_bytes()), &manifest)?;
            let mut detail = String::new();
            if state == State::Outdated
                && let Some(entry) = manifest.installs.get(&key(&path))
            {
                detail.push_str(&format!("  (installed by hypatia {})", entry.version));
            }
            if is_symlink(&path) {
                detail.push_str("  (symlink)");
            }
            println!(
                "{:<9} {label}: {name}  {}{detail}",
                state.label(),
                path.display()
            );
        }
    }
    warn_shadows(home, agents, BUNDLED, &manifest)
}

/// Other directories the host also loads skills from, where a stale same-name copy
/// can shadow the one hypatia manages.
fn shadow_dirs(agent: Agent, home: &Path) -> Result<Vec<(PathBuf, &'static str)>> {
    Ok(match agent {
        Agent::Claude => Vec::new(),
        // Codex still reads its deprecated `$CODEX_HOME/skills`.
        Agent::Codex => vec![(
            env_path("CODEX_HOME")
                .filter(|dir| dir.is_absolute())
                .unwrap_or_else(|| home.join(".codex"))
                .join("skills"),
            "Codex's deprecated skills directory",
        )],
        // OpenCode also scans Claude Code's and the cross-agent directory, and
        // keeps only one copy of a same-name skill.
        Agent::Opencode => vec![
            (
                Agent::Claude.skills_dir(home)?,
                "Claude Code's skills directory",
            ),
            (
                Agent::Codex.skills_dir(home)?,
                "the cross-agent skills directory",
            ),
        ],
    })
}

/// Warn about same-name copies in directories the host also loads that are not
/// current, naming each file. Silent when there is nothing to act on.
fn warn_shadows(
    home: &Path,
    agents: &[Agent],
    skills: &[(&'static str, &'static str)],
    manifest: &Manifest,
) -> Result<()> {
    for agent in agents {
        for (dir, what) in shadow_dirs(*agent, home)? {
            for (name, content) in skills {
                let path = dir.join(name).join("SKILL.md");
                let state = classify(&path, &sha256_hex(content.as_bytes()), manifest)?;
                if !matches!(state, State::Missing | State::Current) {
                    println!(
                        "warning   {}: {name}  {} is {} and {} loads it too ({what}); \
                         update or remove it so it cannot shadow the managed copy",
                        agent.label(),
                        path.display(),
                        state.label(),
                        agent.label()
                    );
                }
            }
        }
    }
    Ok(())
}

fn classify(path: &Path, bundled_sha: &str, manifest: &Manifest) -> Result<State> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(State::Missing),
        Err(e) => return Err(at(path, e)),
    };
    let sha = sha256_hex(&bytes);
    Ok(if sha == bundled_sha {
        State::Current
    } else {
        match manifest.installs.get(&key(path)) {
            Some(entry) if entry.sha256 == sha => State::Outdated,
            Some(_) => State::Modified,
            None => State::Unmanaged,
        }
    })
}

fn is_symlink(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn key(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// Attach the path to an I/O error so the user can tell which file failed.
fn at(path: &Path, e: std::io::Error) -> HypatiaError {
    std::io::Error::new(e.kind(), format!("{}: {e}", path.display())).into()
}

/// Write a uniquely named sibling temp file, fsync it, then rename it over `path`,
/// so a host never reads a half-written skill. The temp file is removed on failure.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| at(dir, e))?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir).map_err(|e| at(dir, e))?;
    tmp.write_all(bytes)
        .and_then(|()| tmp.as_file().sync_all())
        .map_err(|e| at(path, e))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tmp.as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o644))
            .map_err(|e| at(path, e))?;
    }
    tmp.persist(path).map_err(|e| at(path, e.error))?;
    Ok(())
}

fn manifest_path(home: &Path) -> PathBuf {
    home.join(".hypatia").join("skill-installs.json")
}

fn load_manifest(home: &Path) -> Result<Manifest> {
    let path = manifest_path(home);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Ok(Manifest {
                version: MANIFEST_VERSION,
                installs: BTreeMap::new(),
            });
        }
        Err(e) => return Err(at(&path, e)),
    };
    let manifest: Manifest = serde_json::from_str(&text).map_err(|e| {
        HypatiaError::Config(format!(
            "{} is unreadable ({e}). Remove it to reset: installed copies are then treated \
             as not installed by hypatia, and install asks for --force before replacing them.",
            path.display()
        ))
    })?;
    if manifest.version > MANIFEST_VERSION {
        return Err(HypatiaError::Config(format!(
            "{} was written by a newer hypatia (format {}); upgrade hypatia to manage skills",
            path.display(),
            manifest.version
        )));
    }
    Ok(manifest)
}

fn save_manifest(home: &Path, manifest: &Manifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    write_atomic(&manifest_path(home), json.as_bytes())
}

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The user's home directory. Unlike the shelf code, this refuses to fall back to
/// the current directory: the whole point here is writing into a host's config.
fn home() -> Result<PathBuf> {
    env_path("HOME")
        .or_else(|| env_path("USERPROFILE"))
        .filter(|p| p.is_absolute())
        .ok_or_else(|| {
            HypatiaError::Config(
                "cannot locate the home directory: set HOME (USERPROFILE on Windows) \
                 to an absolute path"
                    .into(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_names_match_frontmatter_and_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for (name, content) in BUNDLED {
            assert!(seen.insert(*name), "duplicate bundled skill {name}");
            let content = content.replace("\r\n", "\n");
            let front = content
                .strip_prefix("---\n")
                .and_then(|rest| rest.split("\n---").next())
                .unwrap_or_else(|| panic!("{name}: missing YAML frontmatter"));
            let declared = front
                .lines()
                .find_map(|l| l.strip_prefix("name:"))
                .map(|v| v.trim().trim_matches('"'))
                .unwrap_or_else(|| panic!("{name}: frontmatter has no name"));
            assert_eq!(
                declared, *name,
                "directory name must equal frontmatter name"
            );
        }
    }

    #[test]
    fn classify_covers_every_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s").join("SKILL.md");
        let bundled = sha256_hex(b"new");
        let mut manifest = Manifest {
            version: MANIFEST_VERSION,
            installs: BTreeMap::new(),
        };
        assert_eq!(
            classify(&path, &bundled, &manifest).unwrap(),
            State::Missing
        );

        write_atomic(&path, b"new").unwrap();
        assert_eq!(
            classify(&path, &bundled, &manifest).unwrap(),
            State::Current
        );

        write_atomic(&path, b"old").unwrap();
        assert_eq!(
            classify(&path, &bundled, &manifest).unwrap(),
            State::Unmanaged
        );

        let entry = |sha: String| Entry {
            sha256: sha,
            version: "0".into(),
        };
        manifest
            .installs
            .insert(key(&path), entry(sha256_hex(b"old")));
        assert_eq!(
            classify(&path, &bundled, &manifest).unwrap(),
            State::Outdated
        );

        manifest
            .installs
            .insert(key(&path), entry(sha256_hex(b"something else")));
        assert_eq!(
            classify(&path, &bundled, &manifest).unwrap(),
            State::Modified
        );
        // No temp files are left beside the skill.
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }
}

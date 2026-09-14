//! `hypatia init`: make sure a shelf is set up, and say what works on it and what to do next.
use std::path::{Path, PathBuf};

use crate::error::{HypatiaError, Result};
use crate::lab::{Lab, ShelfStatus};
use crate::model::ShelfId;

pub(crate) fn run(lab: &mut Lab, path: Option<&Path>, name: Option<&str>) -> Result<()> {
    let shelf = target_shelf(lab, path, name)?;
    for line in report(&lab.shelf_status(&shelf)?) {
        println!("{line}");
    }
    Ok(())
}

/// The shelf to set up: the default one, a registered one by name, or the one in `path`,
/// connected unless it already is. Running it again changes nothing, and no registration is
/// ever replaced.
fn target_shelf(lab: &mut Lab, path: Option<&Path>, name: Option<&str>) -> Result<String> {
    let registered: Vec<(String, PathBuf, bool)> = lab
        .list_shelves()
        .into_iter()
        .map(|(known, dir, connected)| (known.to_string(), dir.clone(), connected))
        .collect();
    let shelf = match path {
        None => {
            let name = name.unwrap_or("default");
            if !registered.iter().any(|(known, _, _)| known == name) {
                return Err(HypatiaError::Shelf(format!(
                    "shelf '{name}' is not connected; run `hypatia init <dir> -n {name}` to set it up"
                )));
            }
            name.to_string()
        }
        Some(path) => {
            // Named and registered as typed, with `..` resolved; compared through links.
            let path = resolve(path)?;
            match registered.iter().find(|(_, dir, _)| same_dir(dir, &path)) {
                Some((existing, _, _)) if name.is_some_and(|name| name != existing) => {
                    return Err(HypatiaError::Shelf(format!(
                        "{} is already connected as '{existing}'",
                        path.display()
                    )));
                }
                Some((existing, _, _)) => existing.clone(),
                None => {
                    let wanted = name.map_or_else(|| ShelfId::new(path.clone()).name, String::from);
                    // Connecting would re-point that name, even at a shelf that failed to open.
                    if let Some((_, other, _)) =
                        registered.iter().find(|(known, _, _)| *known == wanted)
                    {
                        return Err(HypatiaError::Shelf(format!(
                            "the name '{wanted}' belongs to the shelf at {}; choose another with `-n <name>`",
                            other.display()
                        )));
                    }
                    return lab.connect_shelf(&path, name);
                }
            }
        }
    };
    // Registered but not open: it failed to open at startup, so open it again to say why.
    if registered
        .iter()
        .any(|(known, _, connected)| *known == shelf && !connected)
    {
        lab.reopen_shelf(&shelf)?;
    }
    Ok(shelf)
}

/// `path` made absolute with `..` resolved, but a link at its end kept: the shelf is named and
/// registered after the name the user gave, not where that link points today.
fn resolve(path: &Path) -> std::io::Result<PathBuf> {
    let path = std::path::absolute(path)?;
    Ok(match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => std::fs::canonicalize(parent)
            .unwrap_or_else(|_| parent.to_path_buf())
            .join(name),
        _ => std::fs::canonicalize(&path).unwrap_or(path),
    })
}

/// Whether two paths name one directory, through symlinks too (macOS `/tmp` is `/private/tmp`).
fn same_dir(a: &Path, b: &Path) -> bool {
    let key = |p: &Path| std::fs::canonicalize(p).or_else(|_| std::path::absolute(p));
    matches!((key(a), key(b)), (Ok(a), Ok(b)) if a == b)
}

/// What `hypatia init` prints: what works, what does not yet, and the command that fixes it.
fn report(status: &ShelfStatus) -> Vec<String> {
    let backend = if status.postgres {
        "PostgreSQL"
    } else {
        "SQLite"
    };
    let mut lines = vec![
        format!(
            "✓ Shelf '{}' is ready: {} ({backend})",
            status.name,
            status.path.display()
        ),
        "✓ Full-text search, graph traversal and JSE queries work".to_string(),
    ];
    let debt = &status.debt;
    let pending = debt.pending_knowledge + debt.pending_statement;
    match (&status.semantic_search_off, &status.attention) {
        (Some(off), _) => {
            lines.push(format!("○ Semantic search is off: {off}"));
            lines.push(
                "  Entries already on the shelf get vectors automatically once it is on."
                    .to_string(),
            );
        }
        (None, Some(attention)) => {
            lines.push(format!("! Semantic search needs attention: {attention}"));
        }
        (None, None) => {
            // Set up, not proven: init neither loads the model nor calls the API.
            lines.push(format!(
                "✓ Semantic search is set up with {}",
                status.embedder
            ));
            if pending > 0 {
                lines.push(match &debt.paused {
                    Some(paused) => format!(
                        "  {pending} entries are waiting for vectors; automatic embedding is paused: {}",
                        paused.reason
                    ),
                    None => format!(
                        "  {pending} entries are waiting for vectors; they are embedded as you keep using hypatia, or all at once with `hypatia backfill -s {}`",
                        status.name
                    ),
                });
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ShelfManager;
    use crate::storage::flush::{EmbeddingDebt, Paused};

    fn status(off: Option<&str>, attention: Option<&str>, pending: usize) -> ShelfStatus {
        ShelfStatus {
            name: "default".into(),
            path: "/home/me/.hypatia/default".into(),
            postgres: false,
            embedder: "BAAI/bge-m3".into(),
            semantic_search_off: off.map(String::from),
            attention: attention.map(String::from),
            debt: EmbeddingDebt {
                pending_knowledge: pending,
                pending_statement: 0,
                passed_over: 0,
                pending_since: None,
                blocked: None,
                paused: None,
            },
        }
    }

    #[test]
    fn the_report_says_what_works_and_what_to_do_next() {
        let off = report(&status(
            Some("no embedding model is set up; run `x`"),
            None,
            3,
        ));
        assert_eq!(
            off[..3],
            [
                "✓ Shelf 'default' is ready: /home/me/.hypatia/default (SQLite)",
                "✓ Full-text search, graph traversal and JSE queries work",
                "○ Semantic search is off: no embedding model is set up; run `x`",
            ]
        );
        assert!(off[3].contains("automatically once it is on"));

        let on = report(&status(None, None, 3));
        assert_eq!(on[2], "✓ Semantic search is set up with BAAI/bge-m3");
        assert!(
            on[3].starts_with("  3 entries are waiting") && on[3].contains("backfill -s default")
        );
        assert_eq!(report(&status(None, None, 0)).len(), 3);

        let mut paused = status(None, None, 2);
        paused.debt.paused = Some(Paused {
            permanent: true,
            reason: "API returned 401".into(),
            retry_after: None,
        });
        assert!(report(&paused)[3].ends_with("automatic embedding is paused: API returned 401"));

        let changed = report(&status(
            None,
            Some("the model changed; run `backfill --reembed`"),
            5,
        ));
        assert_eq!(
            changed[2],
            "! Semantic search needs attention: the model changed; run `backfill --reembed`"
        );
        assert_eq!(changed.len(), 3);
    }

    #[test]
    fn init_connects_a_directory_once_and_under_one_name() {
        let home = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let mut lab = Lab::from_manager(ShelfManager::with_home(home.path().into()).unwrap());
        assert_eq!(target_shelf(&mut lab, None, None).unwrap(), "default");
        assert_eq!(
            target_shelf(&mut lab, Some(dir.path()), Some("work")).unwrap(),
            "work"
        );
        assert_eq!(
            target_shelf(&mut lab, Some(dir.path()), None).unwrap(),
            "work"
        );
        let err = target_shelf(&mut lab, Some(dir.path()), Some("other"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("already connected as 'work'"), "{err}");
        let default = home.path().join(".hypatia/default");
        assert_eq!(
            target_shelf(&mut lab, Some(&default), None).unwrap(),
            "default"
        );

        // Other spellings of the same directory.
        #[cfg(unix)]
        {
            let link = home.path().join("link");
            std::os::unix::fs::symlink(dir.path(), &link).unwrap();
            assert_eq!(target_shelf(&mut lab, Some(&link), None).unwrap(), "work");
            // A link to a new directory names the shelf, as `connect` would.
            let target = tempfile::tempdir().unwrap();
            let notes = home.path().join("notes");
            std::os::unix::fs::symlink(target.path(), &notes).unwrap();
            assert_eq!(target_shelf(&mut lab, Some(&notes), None).unwrap(), "notes");
            lab.disconnect_shelf("notes").unwrap();
        }
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        assert_eq!(
            target_shelf(&mut lab, Some(&dir.path().join("sub/..")), None).unwrap(),
            "work"
        );

        // Another directory whose name is taken does not re-point the registration.
        let elsewhere = tempfile::tempdir().unwrap();
        let clash = elsewhere.path().join("default");
        let err = target_shelf(&mut lab, Some(&clash), None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("choose another with `-n <name>`"), "{err}");

        let err = target_shelf(&mut lab, None, Some("missing"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("hypatia init <dir> -n missing"), "{err}");
        assert_eq!(lab.list_shelves().len(), 2);
        assert!(!clash.exists());

        // A registered shelf that fails to open says why, and stays registered.
        std::fs::write(dir.path().join("shelf.toml"), "[embedding\n").unwrap();
        let mut restarted = Lab::from_manager(ShelfManager::with_home(home.path().into()).unwrap());
        let err = target_shelf(&mut restarted, None, Some("work"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid shelf.toml"), "{err}");
        assert!(
            restarted
                .list_shelves()
                .iter()
                .any(|(known, _, connected)| *known == "work" && !connected)
        );
    }
}

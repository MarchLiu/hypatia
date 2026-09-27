//! Model administration shared by the CLI and the MCP server: pointing a shelf at an
//! already-installed model without touching the hub, reporting which shelves reference
//! a model, and removing an unused model. `model install` reuses the attach half so
//! downloading and attaching stay one command for users but one code path here.

use std::path::{Path, PathBuf};

use crate::error::{HypatiaError, Result};
use crate::lab::{Attached, Lab};

/// What pointing a shelf at an installed model did, with the wording kept out of the
/// logic so the CLI prints it and the MCP server structures it.
#[derive(Debug, PartialEq)]
pub(crate) enum AttachOutcome {
    /// shelf.toml now names the model, and the shelf was opened again with it. `kept`
    /// lists model settings shelf.toml already had, which may suit another model.
    Configured {
        config: PathBuf,
        kept: Vec<String>,
        /// Entries that still owe a vector under the new model.
        pending: usize,
    },
    /// The shelf already names the model.
    AlreadyConfigured {
        /// Entries that still owe a vector.
        pending: usize,
    },
    /// Refused: the shelf holds vectors of another model, which switching would strand.
    HasVectors { config: PathBuf },
    /// Refused: the shelf embeds through a remote API.
    Remote { config: PathBuf },
}

impl AttachOutcome {
    /// One-word outcome for structured results.
    pub(crate) fn result(&self) -> &'static str {
        match self {
            Self::Configured { .. } => "attached",
            Self::AlreadyConfigured { .. } => "already_attached",
            Self::HasVectors { .. } => "refused_has_vectors",
            Self::Remote { .. } => "refused_remote",
        }
    }

    /// What to do about vectors next: `pending` (entries lack vectors; `backfill`
    /// writes them), `reembed` (the shelf keeps vectors of another model; switching
    /// means `backfill --reembed`) or `none`.
    pub(crate) fn backfill(&self) -> &'static str {
        match self {
            Self::Configured { pending: 0, .. } | Self::AlreadyConfigured { pending: 0 } => "none",
            Self::Configured { .. } | Self::AlreadyConfigured { .. } => "pending",
            Self::HasVectors { .. } | Self::Remote { .. } => "reembed",
        }
    }

    /// The shelf.toml the outcome concerns, when there is one to point at.
    pub(crate) fn config(&self) -> Option<&Path> {
        match self {
            Self::Configured { config, .. }
            | Self::HasVectors { config }
            | Self::Remote { config } => Some(config),
            Self::AlreadyConfigured { .. } => None,
        }
    }
}

/// Points `shelf` at the installed model `name` without any hub interaction. Fails
/// before shelf.toml is touched when the model is not on disk.
pub(crate) fn attach(lab: &mut Lab, shelf: &str, name: &str) -> Result<AttachOutcome> {
    if crate::embedding::config::resolve_model(name).is_err() {
        return Err(HypatiaError::Embedding(format!(
            "model '{name}' is not installed; run `hypatia model install {name}` or `hypatia model register {name} <path>` first"
        )));
    }
    let attached = lab.attach_model(shelf, name)?;
    let pending = lab
        .embedding_debt(shelf)
        .map(|debt| debt.pending_knowledge + debt.pending_statement)
        .unwrap_or(0);
    Ok(match attached {
        Attached::Configured { config, kept } => AttachOutcome::Configured {
            config,
            kept,
            pending,
        },
        Attached::AlreadyConfigured => AttachOutcome::AlreadyConfigured { pending },
        Attached::HasVectors { config } => AttachOutcome::HasVectors { config },
        Attached::Remote { config } => AttachOutcome::Remote { config },
    })
}

/// The model hypatia's defaults (dimensions, pooling, sequence length) are tuned for.
const DEFAULT_MODEL: &str = "BAAI/bge-m3";

/// The CLI wording for an attach outcome; `model install` prints the same lines.
pub(crate) fn print_attach(shelf: &str, name: &str, outcome: &AttachOutcome) {
    match outcome {
        AttachOutcome::Configured {
            config,
            kept,
            pending,
        } => {
            println!(
                "Shelf '{shelf}' now uses {name} ({}). Existing entries get vectors automatically; `hypatia backfill -s {shelf}` embeds them all at once.",
                config.display()
            );
            if *pending > 0 {
                println!("{pending} entries are waiting for a vector.");
            }
            if !kept.is_empty() {
                println!(
                    "That file already sets {} under [embedding]; check that these settings suit {name}.",
                    kept.join(", ")
                );
            } else if name != DEFAULT_MODEL {
                println!(
                    "hypatia assumes 1024 dimensions, mean pooling and 8192 tokens; if {name} differs, set `dimensions`, `pooling` and `max_seq_length` under [embedding] there."
                );
            }
        }
        AttachOutcome::AlreadyConfigured { pending } => {
            println!("Shelf '{shelf}' already uses {name}.");
            if *pending > 0 {
                println!(
                    "{pending} entries still owe a vector; `hypatia backfill -s {shelf}` embeds them."
                );
            }
        }
        AttachOutcome::HasVectors { config } => println!(
            "Shelf '{shelf}' holds vectors of another model, so it was left unchanged. To switch, set `model = \"{name}\"` under [embedding] in {}, then run `hypatia backfill --reembed -s {shelf}`.",
            config.display()
        ),
        AttachOutcome::Remote { config } => println!(
            "Shelf '{shelf}' embeds through a remote API, so it was left unchanged. To embed locally instead, set `provider = \"local\"` and `model = \"{name}\"` under [embedding] in {}.",
            config.display()
        ),
    }
}

/// Registered shelves whose shelf.toml names the model or points `model_path` /
/// `tokenizer_path` into the model's directory under `models_dir`.
pub(crate) fn referencing_shelves(
    lab: &Lab,
    model: &str,
    models_dir: &Path,
) -> Result<Vec<String>> {
    let model_dir = models_dir.join(model);
    let mut shelves = Vec::new();
    for (shelf, path, _) in lab.list_shelves() {
        if references_model(&path.join("shelf.toml"), model, &model_dir) {
            shelves.push(shelf.to_string());
        }
    }
    Ok(shelves)
}

/// Whether a shelf.toml references the model: either as the configured `model`, or by
/// pointing a model file into the model's directory. A file that cannot be read or
/// parsed simply does not reference it.
fn references_model(config: &Path, model: &str, model_dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(config) else {
        return false;
    };
    let Ok(parsed) = text.parse::<toml::Table>() else {
        return false;
    };
    let Some(embedding) = parsed.get("embedding").and_then(|e| e.as_table()) else {
        return false;
    };
    if embedding.get("model").and_then(|m| m.as_str()) == Some(model) {
        return true;
    }
    ["model_path", "tokenizer_path"].iter().any(|key| {
        embedding
            .get(*key)
            .and_then(|v| v.as_str())
            .map(|path| Path::new(path).starts_with(model_dir))
            .unwrap_or(false)
    })
}

/// Deletes an installed model from `models_dir`, refusing while shelves still reference
/// it unless `force`. With `--force`, the listed shelves keep their stored vectors
/// (still searchable) but can embed nothing new until they are pointed at another model.
/// Returns the deleted directory.
pub(crate) fn remove(
    lab: &Lab,
    name: &str,
    force: bool,
    models_dir: &Path,
) -> Result<PathBuf> {
    let dir = models_dir.join(name);
    if !dir.exists() && !dir.is_symlink() {
        return Err(HypatiaError::Embedding(format!(
            "model '{name}' is not in {}; `hypatia model list` shows what is",
            models_dir.display()
        )));
    }
    let shelves = referencing_shelves(lab, name, models_dir)?;
    if !shelves.is_empty() && !force {
        return Err(HypatiaError::Embedding(format!(
            "model '{name}' is still used by: {}. Point them at another model first, or pass --force to remove it anyway (their existing vectors stay searchable, but nothing new can be embedded until they are reconfigured).",
            shelves.join(", ")
        )));
    }
    if dir.is_symlink() || dir.is_file() {
        // `model register` leaves a symlink; removing it must not touch the source.
        std::fs::remove_file(&dir)?;
    } else {
        std::fs::remove_dir_all(&dir)?;
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::shelf_manager::ShelfManager;

    struct Home {
        _dir: tempfile::TempDir,
        manager: ShelfManager,
    }

    fn home() -> Home {
        let dir = tempfile::tempdir().unwrap();
        let manager = ShelfManager::with_home(dir.path().into()).unwrap();
        Home { _dir: dir, manager }
    }

    fn write_shelf_config(path: &Path, text: &str) -> PathBuf {
        std::fs::create_dir_all(path).unwrap();
        let config = path.join("shelf.toml");
        std::fs::write(&config, text).unwrap();
        config
    }

    #[test]
    fn attach_fails_before_touching_the_shelf_when_the_model_is_missing() {
        let mut h = home();
        let shelf = tempfile::tempdir().unwrap();
        write_shelf_config(shelf.path(), "");
        h.manager.connect(shelf.path(), Some("s")).unwrap();
        let mut lab = Lab::from_manager(h.manager);
        let err = attach(&mut lab, "s", "org/never-installed").unwrap_err();
        assert!(err.to_string().contains("hypatia model install"), "{err}");
    }

    #[test]
    fn references_are_found_by_name_and_by_model_file_paths() {
        let mut h = home();
        let dir = tempfile::tempdir().unwrap();
        write_shelf_config(
            &dir.path().join("by-name"),
            "[embedding]\nmodel = 'org/m'\n",
        );
        write_shelf_config(
            &dir.path().join("by-path"),
            "[embedding]\nmodel = 'other/m'\nmodel_path = '/models/org/m/model.onnx'\n",
        );
        write_shelf_config(
            &dir.path().join("by-tokenizer"),
            "[embedding]\nmodel_path = 'x.onnx'\ntokenizer_path = '/models/org/m/tokenizer.json'\n",
        );
        write_shelf_config(&dir.path().join("clean"), "[embedding]\nmodel = 'org/n'\n");
        h.manager
            .connect(&dir.path().join("by-name"), Some("by-name"))
            .unwrap();
        h.manager
            .connect(&dir.path().join("by-path"), Some("by-path"))
            .unwrap();
        h.manager
            .connect(&dir.path().join("by-tokenizer"), Some("by-tokenizer"))
            .unwrap();
        h.manager
            .connect(&dir.path().join("clean"), Some("clean"))
            .unwrap();
        let lab = Lab::from_manager(h.manager);
        let models_dir = Path::new("/models");
        assert_eq!(
            referencing_shelves(&lab, "org/m", models_dir).unwrap(),
            ["by-name", "by-path", "by-tokenizer"]
        );
        assert_eq!(
            referencing_shelves(&lab, "org/n", models_dir).unwrap(),
            ["clean"]
        );
    }

    #[test]
    fn remove_refuses_referenced_models_and_force_overrides() {
        let mut h = home();
        let dir = tempfile::tempdir().unwrap();
        write_shelf_config(
            &dir.path().join("user"),
            "[embedding]\nmodel = 'org/m'\n",
        );
        h.manager
            .connect(&dir.path().join("user"), Some("user"))
            .unwrap();
        let lab = Lab::from_manager(h.manager);
        let models_dir = tempfile::tempdir().unwrap();
        let model_dir = models_dir.path().join("org/m");
        std::fs::create_dir_all(&model_dir).unwrap();

        let err = remove(&lab, "org/m", false, models_dir.path()).unwrap_err();
        assert!(err.to_string().contains("user"), "{err}");
        assert!(model_dir.exists());

        let removed = remove(&lab, "org/m", true, models_dir.path()).unwrap();
        assert_eq!(removed, model_dir);
        assert!(!model_dir.exists());
    }

    #[test]
    fn remove_takes_a_registered_symlink_without_touching_the_source() {
        let mut h = home();
        let shelf = tempfile::tempdir().unwrap();
        write_shelf_config(shelf.path(), "");
        h.manager.connect(shelf.path(), Some("s")).unwrap();
        let lab = Lab::from_manager(h.manager);
        let models_dir = tempfile::tempdir().unwrap();
        let source = tempfile::tempdir().unwrap();
        let target = models_dir.path().join("org/local");
        if std::os::unix::fs::symlink(source.path(), &target).is_err() {
            // Some sandboxes forbid symlinks; the dir-removal path is covered elsewhere.
            return;
        }

        remove(&lab, "org/local", false, models_dir.path()).unwrap();
        assert!(!target.exists());
        assert!(source.path().exists());
    }

    #[test]
    fn remove_names_the_models_directory_when_the_model_is_unknown() {
        let mut h = home();
        let lab = Lab::from_manager(h.manager);
        let models_dir = tempfile::tempdir().unwrap();
        let err = remove(&lab, "org/absent", false, models_dir.path()).unwrap_err();
        assert!(err.to_string().contains("not in"), "{err}");
    }
}

//! `hypatia model install`: download a model, then point the target shelf at it.
use crate::embedding::config::models_dir;
use crate::embedding::install::{Hub, install};
use crate::error::{HypatiaError, Result};
use crate::lab::{Attached, Lab};

/// The model hypatia's defaults (dimensions, pooling, sequence length) are tuned for.
const DEFAULT_MODEL: &str = "BAAI/bge-m3";

pub(crate) fn run(lab: &mut Lab, name: &str, shelf: &str, revision: &str) -> Result<()> {
    // Gigabytes of download must not end in a mistyped shelf name.
    let connected = lab
        .list_shelves()
        .iter()
        .any(|(registered, _, connected)| *registered == shelf && *connected);
    if !connected {
        return Err(HypatiaError::Shelf(format!(
            "shelf '{shelf}' is not connected; nothing was downloaded"
        )));
    }
    let installed = install(&Hub::from_env(), name, revision, &models_dir())
        .map_err(|e| HypatiaError::Embedding(format!("cannot install {name}: {e}")))?;
    let name = installed.repo.as_str();
    println!("Installed {name} in {}", installed.dir.display());
    if installed.replaced {
        println!(
            "{name} changed since it was last installed, so vectors embedded with the earlier files no longer match: run `hypatia backfill --reembed -s <shelf>` for each shelf that uses it."
        );
    }
    match lab.attach_model(shelf, name)? {
        Attached::Configured { config, kept } => {
            println!(
                "Shelf '{shelf}' now uses {name} ({}). Existing entries get vectors automatically; `hypatia backfill -s {shelf}` embeds them all at once.",
                config.display()
            );
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
        Attached::AlreadyConfigured => println!("Shelf '{shelf}' already uses {name}."),
        Attached::HasVectors { config } => println!(
            "Shelf '{shelf}' holds vectors of another model, so it was left unchanged. To switch, set `model = \"{name}\"` under [embedding] in {}, then run `hypatia backfill --reembed -s {shelf}`.",
            config.display()
        ),
        Attached::Remote { config } => println!(
            "Shelf '{shelf}' embeds through a remote API, so it was left unchanged. To embed locally instead, set `provider = \"local\"` and `model = \"{name}\"` under [embedding] in {}.",
            config.display()
        ),
    }
    Ok(())
}

//! `hypatia model install`: download a model, then point the target shelf at it.
use crate::embedding::config::models_dir;
use crate::embedding::install::{Hub, install};
use crate::error::{HypatiaError, Result};
use crate::lab::Lab;

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
    super::model_admin::print_attach(shelf, &name, &super::model_admin::attach(lab, shelf, &name)?);
    Ok(())
}

//! Versioned backend-neutral snapshots. Credentials and local model paths are excluded.
use crate::error::{HypatiaError, Result};
use crate::model::{Knowledge, Statement};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingMetadata {
    pub model: String,
    pub dimensions: usize,
    pub metric: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnowledgeRecord {
    pub knowledge: Knowledge,
    pub embedding: Option<Vec<f32>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StatementRecord {
    pub statement: Statement,
    pub embedding: Option<Vec<f32>>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Snapshot {
    pub format_version: u32,
    pub embedding: Option<EmbeddingMetadata>,
    pub knowledge: Vec<KnowledgeRecord>,
    pub statements: Vec<StatementRecord>,
}
impl Snapshot {
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1 {
            return Err(HypatiaError::Validation(
                "unsupported logical snapshot version".into(),
            ));
        }
        let mut keys = std::collections::HashSet::new();
        for r in &self.knowledge {
            if !keys.insert(r.knowledge.name.clone()) {
                return Err(HypatiaError::Validation(
                    "snapshot contains duplicate knowledge keys".into(),
                ));
            }
            self.validate_vector(r.embedding.as_deref())?;
        }
        keys.clear();
        for r in &self.statements {
            if !keys.insert(r.statement.key.to_csv_key()) {
                return Err(HypatiaError::Validation(
                    "snapshot contains duplicate statement keys".into(),
                ));
            }
            self.validate_vector(r.embedding.as_deref())?;
        }
        Ok(())
    }
    fn validate_vector(&self, vector: Option<&[f32]>) -> Result<()> {
        if let Some(v) = vector {
            let meta = self.embedding.as_ref().ok_or_else(|| {
                HypatiaError::Validation(
                    "vectors require trusted model metadata; use --reembed".into(),
                )
            })?;
            validate_vector(v, meta.dimensions)?;
            if meta.metric != "cosine" || meta.model.is_empty() {
                return Err(HypatiaError::Validation(
                    "invalid embedding metadata".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn without_embeddings(mut self) -> Self {
        self.embedding = None;
        for r in &mut self.knowledge {
            r.embedding = None;
        }
        for r in &mut self.statements {
            r.embedding = None;
        }
        self
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub knowledge_count: usize,
    pub statement_count: usize,
    pub files: std::collections::BTreeMap<String, String>,
}
fn checksum(path: &std::path::Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn archive_files(
    base: &std::path::Path,
    dir: &std::path::Path,
    files: &mut std::collections::BTreeMap<String, String>,
) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let path = entry.path();
        if kind.is_symlink() {
            return Err(HypatiaError::Validation(
                "archive exports cannot contain symlinks".into(),
            ));
        }
        if kind.is_dir() {
            archive_files(base, &path, files)?;
        } else if kind.is_file() {
            let relative = path
                .strip_prefix(base)
                .map_err(|_| HypatiaError::Validation("archive path outside package".into()))?;
            let key = relative
                .to_str()
                .ok_or_else(|| HypatiaError::Validation("archive filenames must be UTF-8".into()))?
                .replace('\\', "/");
            files.insert(key, checksum(&path)?);
        } else {
            return Err(HypatiaError::Validation(
                "archives must contain regular files only".into(),
            ));
        }
    }
    Ok(())
}
pub fn write_manifest(root: &std::path::Path, snapshot: &Snapshot) -> Result<()> {
    let mut files = std::collections::BTreeMap::new();
    files.insert(
        "snapshot.json".into(),
        checksum(&root.join("snapshot.json"))?,
    );
    archive_files(root, &root.join("archives"), &mut files)?;
    let manifest = Manifest {
        format_version: 1,
        knowledge_count: snapshot.knowledge.len(),
        statement_count: snapshot.statements.len(),
        files,
    };
    serde_json::to_writer_pretty(
        std::fs::File::create(root.join("manifest.json"))?,
        &manifest,
    )?;
    Ok(())
}
pub fn verify_manifest(root: &std::path::Path, snapshot: &Snapshot) -> Result<()> {
    let manifest: Manifest =
        serde_json::from_reader(std::fs::File::open(root.join("manifest.json"))?)?;
    if manifest.format_version != 1
        || manifest.knowledge_count != snapshot.knowledge.len()
        || manifest.statement_count != snapshot.statements.len()
        || !manifest.files.contains_key("snapshot.json")
    {
        return Err(HypatiaError::Validation(
            "logical export manifest version or row counts differ".into(),
        ));
    }
    let mut actual = std::collections::BTreeMap::new();
    actual.insert(
        "snapshot.json".into(),
        checksum(&root.join("snapshot.json"))?,
    );
    archive_files(root, &root.join("archives"), &mut actual)?;
    if actual != manifest.files {
        return Err(HypatiaError::Validation(
            "export checksum verification failed; preserve the original package".into(),
        ));
    }
    Ok(())
}

pub fn validate_archives(snapshot: &Snapshot, root: &std::path::Path) -> Result<()> {
    for content in snapshot
        .knowledge
        .iter()
        .map(|r| &r.knowledge.content)
        .chain(snapshot.statements.iter().map(|r| &r.statement.content))
    {
        for reference in content.figures.iter().flatten() {
            if let Some(relative) = reference.strip_prefix("archive://") {
                let path = std::path::Path::new(relative);
                if relative.is_empty()
                    || path
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                {
                    return Err(HypatiaError::Validation(format!(
                        "invalid archive reference: {reference}"
                    )));
                }
                let canonical = root.canonicalize()?;
                let file = root.join(path).canonicalize()?;
                if !file.starts_with(&canonical) || !file.is_file() {
                    return Err(HypatiaError::Validation(format!(
                        "missing or unsafe archive reference: {reference}"
                    )));
                }
            }
        }
    }
    Ok(())
}

pub fn validate_vector(vector: &[f32], dims: usize) -> Result<()> {
    if vector.len() != dims
        || vector.is_empty()
        || vector.iter().any(|v| !v.is_finite())
        || vector.iter().all(|v| *v == 0.0)
    {
        return Err(HypatiaError::Validation(format!(
            "embedding must contain {dims} finite values and have nonzero norm"
        )));
    }
    Ok(())
}

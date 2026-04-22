use std::path::{Path, PathBuf};

/// Embedding configuration loaded from `shelf.toml` (or defaults).
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Which provider to use: "local" (ONNX) or "remote" (HTTP API).
    pub provider: ProviderKind,
    /// Local ONNX settings.
    pub local: LocalConfig,
    /// Remote API settings.
    pub remote: RemoteConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderKind {
    Local,
    Remote,
}

/// Pooling strategy for extracting embeddings from model output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    /// Mean pooling over non-padding tokens (default for BGE-M3).
    #[default]
    Mean,
    /// CLS token at position 0.
    Cls,
    /// Last non-padding token's hidden state (used by Jina v5).
    LastToken,
}

#[derive(Debug, Clone)]
pub struct LocalConfig {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dimensions: usize,
    pub max_seq_length: usize,
    pub pooling: PoolingStrategy,
}

#[derive(Debug, Clone)]
pub struct RemoteConfig {
    pub api_url: String,
    pub api_key_env: String,
    pub api_model: String,
    pub dimensions: usize,
}

/// Top-level `shelf.toml` structure (currently only `[embedding]` section).
#[derive(Debug, Clone, serde::Deserialize)]
struct ShelfToml {
    #[serde(default)]
    embedding: EmbeddingToml,
}

/// Parsed `[embedding]` section from `shelf.toml`.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
struct EmbeddingToml {
    provider: String,
    /// HuggingFace-style model reference (e.g. "BAAI/bge-m3") or absolute path.
    model: Option<String>,
    /// Explicit model file path (backward compat).
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    dimensions: Option<usize>,
    max_seq_length: Option<usize>,
    pooling: Option<PoolingStrategy>,
    api_url: Option<String>,
    api_key_env: Option<String>,
    api_model: Option<String>,
}

impl Default for EmbeddingToml {
    fn default() -> Self {
        Self {
            provider: "local".into(),
            model: None,
            model_path: None,
            tokenizer_path: None,
            dimensions: None,
            max_seq_length: None,
            pooling: None,
            api_url: None,
            api_key_env: None,
            api_model: None,
        }
    }
}

// ── Model resolution ──────────────────────────────────────────────────

/// Result of resolving a model reference to file paths.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub model_dir: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

/// Resolve a model reference to concrete file paths.
///
/// Resolution order:
/// 1. Absolute path → use directly as model directory
/// 2. HF name like "BAAI/bge-m3" → look in `~/.hypatia/models/BAAI/bge-m3/`
/// 3. Not found → look in HF cache `~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/<hash>/`
pub fn resolve_model(model_ref: &str) -> Result<ResolvedModel, String> {
    let path = PathBuf::from(model_ref);

    // Case 1: Absolute path
    if path.is_absolute() {
        return resolve_model_dir(&path, model_ref);
    }

    // Case 2: ~/.hypatia/models/<org>/<name>/
    let hypatia_model_dir = dirs_home()
        .join(".hypatia")
        .join("models")
        .join(model_ref);
    if hypatia_model_dir.is_dir() {
        return resolve_model_dir(&hypatia_model_dir, model_ref);
    }

    // Case 3: HuggingFace cache
    // HF cache uses "models--Org--Name" format for the directory name
    let cache_entry = format!("models--{}", model_ref.replace('/', "--"));
    let hf_cache = dirs_home()
        .join(".cache")
        .join("huggingface")
        .join("hub")
        .join(&cache_entry);

    if hf_cache.is_dir() {
        // Look for the latest snapshot
        let snapshots_dir = hf_cache.join("snapshots");
        if snapshots_dir.is_dir() {
            let latest = find_latest_snapshot(&snapshots_dir)?;
            return resolve_model_dir(&latest, model_ref);
        }
    }

    Err(format!(
        "model '{}' not found in ~/.hypatia/models/ or HuggingFace cache",
        model_ref
    ))
}

/// Within a resolved model directory, find the ONNX model and tokenizer files.
fn resolve_model_dir(dir: &Path, _model_ref: &str) -> Result<ResolvedModel, String> {
    if !dir.is_dir() {
        return Err(format!("model directory does not exist: {}", dir.display()));
    }

    let model_path = find_onnx_file(dir).ok_or_else(|| {
        format!(
            "no ONNX model file found in {} (looked for embedding_model.onnx, model.onnx, *.onnx)",
            dir.display()
        )
    })?;

    let tokenizer_path = dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!(
            "no tokenizer.json found in {}",
            dir.display()
        ));
    }

    Ok(ResolvedModel {
        model_dir: dir.to_path_buf(),
        model_path,
        tokenizer_path,
    })
}

/// Find the ONNX model file in a directory, preferring specific names.
fn find_onnx_file(dir: &Path) -> Option<PathBuf> {
    // Priority order for model file names
    let preferred = ["embedding_model.onnx", "model.onnx", "model_quantized.onnx"];
    for name in &preferred {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Fallback: any .onnx file (excluding .onnx.data and similar external data)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".onnx") && !name.contains(".onnx.") {
                    return Some(path);
                }
            }
        }
    }

    None
}

/// Find the latest snapshot directory in HF cache.
fn find_latest_snapshot(snapshots_dir: &Path) -> Result<PathBuf, String> {
    let mut latest: Option<(PathBuf, std::time::SystemTime)> = None;

    let entries = std::fs::read_dir(snapshots_dir)
        .map_err(|e| format!("failed to read snapshots dir: {e}"))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let modified = path
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

            match &latest {
                Some((_, prev_time)) if modified <= *prev_time => {}
                _ => latest = Some((path, modified)),
            }
        }
    }

    latest
        .map(|(p, _)| p)
        .ok_or_else(|| "no snapshots found in HuggingFace cache".to_string())
}

/// List all models in ~/.hypatia/models/.
/// Returns (name, path) pairs, e.g. ("BAAI/bge-m3", "~/.hypatia/models/BAAI/bge-m3").
pub fn list_local_models() -> Vec<(String, PathBuf)> {
    let models_dir = dirs_home().join(".hypatia").join("models");
    let mut results = Vec::new();

    if !models_dir.is_dir() {
        return results;
    }

    // Walk two levels: Org/Name/
    if let Ok(orgs) = std::fs::read_dir(&models_dir) {
        for org_entry in orgs.flatten() {
            if !org_entry.path().is_dir() {
                continue;
            }
            let org_name = match org_entry.file_name().to_str() {
                Some(n) => n.to_string(),
                None => continue,
            };
            let org_dir = org_entry.path();
            if let Ok(models) = std::fs::read_dir(&org_dir) {
                for model_entry in models.flatten() {
                    if !model_entry.path().is_dir() {
                        continue;
                    }
                    let model_name = match model_entry.file_name().to_str() {
                        Some(n) => n.to_string(),
                        None => continue,
                    };
                    // Check it looks like a model dir (has at least one .onnx or tokenizer.json)
                    if is_model_dir(&model_entry.path()) {
                        results.push((
                            format!("{}/{}", org_name, model_name),
                            model_entry.path(),
                        ));
                    }
                }
            }
        }
    }

    // Also check single-level models (no org prefix)
    if let Ok(entries) = std::fs::read_dir(&models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(name) = entry.file_name().to_str() {
                // Skip if already listed as part of an org/name pair
                if !name.contains('/') && !results.iter().any(|(n, _)| n.starts_with(&format!("{}/", name))) {
                    if is_model_dir(&path) && !path.join("tokenizer.json").exists() {
                        // This is an org dir, not a flat model
                    } else if is_model_dir(&path) && path.join("tokenizer.json").exists() {
                        // Single-level model (no org)
                        // Only add if it wasn't already picked up as an org
                        let already = results.iter().any(|(n, _)| n.starts_with(name));
                        if !already {
                            results.push((name.to_string(), path));
                        }
                    }
                }
            }
        }
    }

    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

/// Check if a directory looks like a model directory (has ONNX or tokenizer files).
fn is_model_dir(dir: &Path) -> bool {
    find_onnx_file(dir).is_some() || dir.join("tokenizer.json").exists()
}

/// Register a model by creating a symlink from ~/.hypatia/models/<name> to the source path.
pub fn register_model(name: &str, source_path: &Path) -> Result<PathBuf, String> {
    let target = dirs_home().join(".hypatia").join("models").join(name);

    if !source_path.is_dir() {
        return Err(format!("source path is not a directory: {}", source_path.display()));
    }

    if !is_model_dir(source_path) {
        return Err(format!(
            "source directory does not contain model files (no .onnx or tokenizer.json): {}",
            source_path.display()
        ));
    }

    // Create parent directory
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create models directory: {e}"))?;
    }

    // Remove existing symlink/dir if present
    if target.exists() || target.is_symlink() {
        std::fs::remove_dir_all(&target)
            .map_err(|e| format!("failed to remove existing model: {e}"))?;
    }

    // Create symlink (relative if possible for portability)
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(source_path, &target)
            .map_err(|e| format!("failed to create symlink: {e}"))?;
    }
    #[cfg(not(unix))]
    {
        // Fallback: copy the directory for non-Unix systems
        copy_dir_recursive(source_path, &target)
            .map_err(|e| format!("failed to copy model: {e}"))?;
    }

    Ok(target)
}

/// Get model info: file listing and total size.
pub fn model_info(name: &str) -> Result<ModelInfo, String> {
    let resolved = resolve_model(name)?;

    let mut files = Vec::new();
    let mut total_size: u64 = 0;

    if let Ok(entries) = std::fs::read_dir(&resolved.model_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(meta) = path.metadata() {
                let file_name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string();
                let size = meta.len();
                total_size += size;
                files.push(ModelFile { name: file_name, size_bytes: size });
            }
        }
    }

    files.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    Ok(ModelInfo {
        name: name.to_string(),
        directory: resolved.model_dir,
        model_path: resolved.model_path,
        tokenizer_path: resolved.tokenizer_path,
        files,
        total_size_bytes: total_size,
    })
}

/// Information about a registered model.
#[derive(Debug)]
pub struct ModelInfo {
    pub name: String,
    pub directory: PathBuf,
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub files: Vec<ModelFile>,
    pub total_size_bytes: u64,
}

/// A single file in a model directory.
#[derive(Debug)]
pub struct ModelFile {
    pub name: String,
    pub size_bytes: u64,
}

// ── Helpers ───────────────────────────────────────────────────────────

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(not(unix))]
fn copy_dir_recursive(src: &Path, dest: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("{e}"))?;
    for entry in std::fs::read_dir(src).map_err(|e| format!("{e}"))? {
        let entry = entry.map_err(|e| format!("{e}"))?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dest_path)?;
        } else {
            std::fs::copy(&src_path, &dest_path).map_err(|e| format!("{e}"))?;
        }
    }
    Ok(())
}

// ── EmbeddingConfig construction ──────────────────────────────────────

impl EmbeddingConfig {
    /// Load from a shelf directory.
    /// Reads `shelf.toml` if present, otherwise uses defaults for BGE-M3.
    pub fn from_shelf_dir(shelf_dir: &Path) -> Self {
        let toml_path = shelf_dir.join("shelf.toml");
        let toml = if toml_path.exists() {
            let content = std::fs::read_to_string(&toml_path).unwrap_or_default();
            toml::from_str::<ShelfToml>(&content)
                .map(|s| s.embedding)
                .unwrap_or_default()
        } else {
            EmbeddingToml::default()
        };

        let provider = match toml.provider.as_str() {
            "remote" => ProviderKind::Remote,
            _ => ProviderKind::Local,
        };

        let default_dims = 1024;
        let dimensions = toml.dimensions.unwrap_or(default_dims);

        // Resolve local model paths with priority: model > model_path > shelf_dir defaults
        let (model_path, tokenizer_path) = if let Some(model_ref) = &toml.model {
            // New-style: resolve by HF name, absolute path, or fallback
            match resolve_model(model_ref) {
                Ok(resolved) => (resolved.model_path, resolved.tokenizer_path),
                Err(e) => {
                    eprintln!("warning: model resolution failed for '{}': {}", model_ref, e);
                    // Fall through to model_path or defaults
                    resolve_legacy_paths(&toml, shelf_dir)
                }
            }
        } else {
            resolve_legacy_paths(&toml, shelf_dir)
        };

        let local = LocalConfig {
            model_path,
            tokenizer_path,
            dimensions,
            max_seq_length: toml.max_seq_length.unwrap_or(8192),
            pooling: toml.pooling.unwrap_or_default(),
        };

        let remote = RemoteConfig {
            api_url: toml
                .api_url
                .unwrap_or_else(|| "https://api.openai.com/v1/embeddings".into()),
            api_key_env: toml
                .api_key_env
                .unwrap_or_else(|| "OPENAI_API_KEY".into()),
            api_model: toml
                .api_model
                .unwrap_or_else(|| "text-embedding-3-small".into()),
            dimensions,
        };

        Self {
            provider,
            local,
            remote,
        }
    }

    /// Effective dimensions for the active provider.
    pub fn dimensions(&self) -> usize {
        match self.provider {
            ProviderKind::Local => self.local.dimensions,
            ProviderKind::Remote => self.remote.dimensions,
        }
    }

    /// Check if the local model files exist.
    pub fn local_files_exist(&self) -> bool {
        self.local.model_path.exists() && self.local.tokenizer_path.exists()
    }
}

/// Resolve paths from legacy model_path/tokenizer_path fields or shelf directory defaults.
fn resolve_legacy_paths(toml: &EmbeddingToml, shelf_dir: &Path) -> (PathBuf, PathBuf) {
    let model_path = toml
        .model_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| shelf_dir.join("embedding_model.onnx"));
    let tokenizer_path = toml
        .tokenizer_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| shelf_dir.join("tokenizer.json"));
    (model_path, tokenizer_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_onnx_file_prefers_embedding_model() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create model.onnx but NOT embedding_model.onnx
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        let found = find_onnx_file(dir.path()).unwrap();
        assert!(found.file_name().unwrap().to_str().unwrap().contains("model.onnx"));
    }

    #[test]
    fn find_onnx_file_embedding_model_first() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("embedding_model.onnx"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        let found = find_onnx_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "embedding_model.onnx");
    }

    #[test]
    fn find_onnx_file_ignores_data_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("model.onnx.data"), b"fake").unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        let found = find_onnx_file(dir.path()).unwrap();
        assert_eq!(found.file_name().unwrap(), "model.onnx");
    }

    #[test]
    fn find_onnx_file_none_when_empty() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(find_onnx_file(dir.path()).is_none());
    }

    #[test]
    fn resolve_model_absolute_path() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("embedding_model.onnx"), b"fake").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let resolved = resolve_model(dir.path().to_str().unwrap()).unwrap();
        assert!(resolved.model_path.ends_with("embedding_model.onnx"));
    }

    #[test]
    fn resolve_model_absolute_path_missing_tokenizer() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        let result = resolve_model(dir.path().to_str().unwrap());
        assert!(result.is_err());
    }

    #[test]
    fn resolve_model_nonexistent_hf_name() {
        let result = resolve_model("nonexistent-org/nonexistent-model-xyz");
        assert!(result.is_err());
    }

    #[test]
    fn from_shelf_dir_defaults_to_shelf_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = EmbeddingConfig::from_shelf_dir(dir.path());
        assert_eq!(config.local.model_path, dir.path().join("embedding_model.onnx"));
        assert_eq!(config.local.tokenizer_path, dir.path().join("tokenizer.json"));
    }

    #[test]
    fn from_shelf_dir_with_model_field() {
        let dir = tempfile::TempDir::new().unwrap();
        // Use a unique name to avoid collision with parallel tests
        let test_id = std::thread::current().id();
        let model_name = format!("TestOrg/test-model-{:?}", test_id);
        // Create model in ~/.hypatia/models/TestOrg/test-model-<thread_id>/
        let model_dir = dirs_home()
            .join(".hypatia")
            .join("models")
            .join(&model_name);
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("embedding_model.onnx"), b"fake").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

        // Create shelf.toml with model reference
        let toml_content = format!(
            r#"
[embedding]
provider = "local"
model = "{}"
dimensions = 768
"#,
            model_name
        );
        std::fs::write(dir.path().join("shelf.toml"), &toml_content).unwrap();

        let config = EmbeddingConfig::from_shelf_dir(dir.path());
        assert_eq!(config.local.dimensions, 768);
        assert!(config.local.model_path.to_str().unwrap().contains("TestOrg"));
        assert!(config.local.tokenizer_path.to_str().unwrap().contains("TestOrg"));

        // Cleanup
        std::fs::remove_dir_all(model_dir.parent().unwrap().parent().unwrap()).ok();
    }

    #[test]
    fn from_shelf_dir_backward_compat_model_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let toml_content = r#"
[embedding]
provider = "local"
model_path = "/custom/path/model.onnx"
tokenizer_path = "/custom/path/tokenizer.json"
dimensions = 512
"#;
        std::fs::write(dir.path().join("shelf.toml"), toml_content).unwrap();

        let config = EmbeddingConfig::from_shelf_dir(dir.path());
        assert_eq!(config.local.model_path, PathBuf::from("/custom/path/model.onnx"));
        assert_eq!(config.local.tokenizer_path, PathBuf::from("/custom/path/tokenizer.json"));
        assert_eq!(config.local.dimensions, 512);
    }

    #[test]
    fn is_model_dir_detects_onnx() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(!is_model_dir(dir.path()));
        std::fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        assert!(is_model_dir(dir.path()));
    }

    #[test]
    fn is_model_dir_detects_tokenizer() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        assert!(is_model_dir(dir.path()));
    }

    #[test]
    fn register_model_creates_symlink() {
        let source = tempfile::TempDir::new().unwrap();
        std::fs::write(source.path().join("model.onnx"), b"fake").unwrap();
        std::fs::write(source.path().join("tokenizer.json"), b"{}").unwrap();

        let target = register_model("UnitTestOrg/test-reg-model", source.path()).unwrap();
        assert!(target.is_dir() || target.is_symlink());

        // Verify we can resolve it
        let resolved = resolve_model("UnitTestOrg/test-reg-model").unwrap();
        assert!(resolved.model_path.exists());

        // Cleanup
        let models_base = target.parent().unwrap().parent().unwrap();
        std::fs::remove_dir_all(models_base).ok();
    }

    #[test]
    fn model_info_returns_files() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("embedding_model.onnx"), b"fake-model").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

        let info = model_info(dir.path().to_str().unwrap()).unwrap();
        assert!(!info.files.is_empty());
        assert!(info.total_size_bytes > 0);
    }
}

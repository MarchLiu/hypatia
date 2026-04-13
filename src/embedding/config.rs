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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    /// Mean pooling over non-padding tokens (default for BGE-M3).
    Mean,
    /// CLS token at position 0.
    Cls,
    /// Last non-padding token's hidden state (used by Jina v5).
    LastToken,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
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

        let local = LocalConfig {
            model_path: toml
                .model_path
                .map(PathBuf::from)
                .unwrap_or_else(|| shelf_dir.join("embedding_model.onnx")),
            tokenizer_path: toml
                .tokenizer_path
                .map(PathBuf::from)
                .unwrap_or_else(|| shelf_dir.join("tokenizer.json")),
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

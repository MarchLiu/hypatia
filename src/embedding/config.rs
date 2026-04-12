use std::path::PathBuf;

/// Configuration for the embedded vector search model.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Path to the tokenizer.json file.
    pub tokenizer_path: PathBuf,
    /// Embedding vector dimensions (default: 768 for EmbeddingGemma-300M).
    pub dimensions: usize,
    /// Maximum sequence length for the tokenizer (default: 8192).
    pub max_seq_length: usize,
}

impl EmbeddingConfig {
    /// Create config from a shelf directory, using default file names.
    pub fn from_shelf_dir(shelf_dir: &std::path::Path) -> Self {
        Self {
            model_path: shelf_dir.join("embedding_model.onnx"),
            tokenizer_path: shelf_dir.join("tokenizer.json"),
            dimensions: 768,
            max_seq_length: 8192,
        }
    }

    /// Check if the required model files exist.
    pub fn files_exist(&self) -> bool {
        self.model_path.exists() && self.tokenizer_path.exists()
    }
}

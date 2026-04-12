use std::cell::RefCell;
use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::HypatiaError;
use super::config::EmbeddingConfig;

/// Inner state of the embedder, managed via RefCell for interior mutability.
enum EmbedderInner {
    /// Model files not found at configured paths.
    Unavailable { reason: String },
    /// Files exist but model not yet loaded. Will load on first embed() call.
    Pending { config: EmbeddingConfig },
    /// Model loaded and ready for inference.
    Ready {
        session: Session,
        tokenizer: tokenizers::Tokenizer,
        config: EmbeddingConfig,
    },
}

/// Lazy-loading embedding model wrapper.
pub struct Embedder {
    inner: RefCell<EmbedderInner>,
}

impl Embedder {
    /// Create a new embedder. Checks if model files exist and sets initial state.
    pub fn new(config: EmbeddingConfig) -> Self {
        let inner = if config.files_exist() {
            EmbedderInner::Pending { config }
        } else {
            EmbedderInner::Unavailable {
                reason: format!(
                    "embedding model files not found: {} or {}",
                    config.model_path.display(),
                    config.tokenizer_path.display()
                ),
            }
        };
        Self { inner: RefCell::new(inner) }
    }

    /// Create an embedder that is always unavailable (no model support).
    pub fn unavailable() -> Self {
        Self {
            inner: RefCell::new(EmbedderInner::Unavailable {
                reason: "embedding model not configured".to_string(),
            }),
        }
    }

    /// Check if the embedder is available (has model files).
    pub fn is_available(&self) -> bool {
        match &*self.inner.borrow() {
            EmbedderInner::Unavailable { .. } => false,
            EmbedderInner::Pending { .. } | EmbedderInner::Ready { .. } => true,
        }
    }

    /// Generate an embedding vector for the given text.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError> {
        self.ensure_loaded()?;

        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            EmbedderInner::Ready { session, tokenizer, config, .. } => {
                let vector = run_inference(session, tokenizer, text, config.max_seq_length)?;
                Ok(vector)
            }
            _ => unreachable!("ensure_loaded should guarantee Ready state"),
        }
    }

    /// Generate embeddings for multiple texts in batch.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HypatiaError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Try to generate an embedding. Returns Ok(None) if model is unavailable.
    pub fn maybe_embed(&self, text: &str) -> Result<Option<Vec<f32>>, HypatiaError> {
        if !self.is_available() {
            return Ok(None);
        }
        match self.embed(text) {
            Ok(v) => Ok(Some(v)),
            Err(HypatiaError::ModelUnavailable(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn ensure_loaded(&self) -> Result<(), HypatiaError> {
        let needs_load = match &*self.inner.borrow() {
            EmbedderInner::Pending { .. } => true,
            EmbedderInner::Unavailable { reason } => {
                return Err(HypatiaError::ModelUnavailable(reason.clone()));
            }
            EmbedderInner::Ready { .. } => false,
        };

        if needs_load {
            let mut inner = self.inner.borrow_mut();
            let old = std::mem::replace(
                &mut *inner,
                EmbedderInner::Unavailable { reason: "loading...".to_string() },
            );

            match old {
                EmbedderInner::Pending { config } => {
                    match load_model(&config.model_path, &config.tokenizer_path) {
                        Ok((session, tokenizer)) => {
                            *inner = EmbedderInner::Ready { session, tokenizer, config };
                            Ok(())
                        }
                        Err(e) => {
                            *inner = EmbedderInner::Unavailable {
                                reason: format!("failed to load model: {e}"),
                            };
                            Err(HypatiaError::Embedding(format!("failed to load ONNX model: {e}")))
                        }
                    }
                }
                other => {
                    *inner = other;
                    Ok(())
                }
            }
        } else {
            Ok(())
        }
    }
}

/// Load ONNX model and tokenizer from files.
fn load_model(
    model_path: &Path,
    tokenizer_path: &Path,
) -> Result<(Session, tokenizers::Tokenizer), String> {
    let session = Session::builder()
        .map_err(|e| format!("failed to create session builder: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("failed to load ONNX model: {e}"))?;

    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
        .map_err(|e| format!("failed to load tokenizer: {e}"))?;

    Ok((session, tokenizer))
}

/// Run model inference on a single text input.
fn run_inference(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    max_seq_length: usize,
) -> Result<Vec<f32>, HypatiaError> {
    // Tokenize
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| HypatiaError::Embedding(format!("tokenization failed: {e}")))?;

    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    // Truncate to max_seq_length
    let len = input_ids.len().min(max_seq_length);
    let input_ids = &input_ids[..len];
    let attention_mask_u32 = &attention_mask[..len];

    let seq_len = input_ids.len();
    let input_ids_data: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let attention_mask_data: Vec<i64> = attention_mask_u32.iter().map(|&m| m as i64).collect();

    // Create input tensors: shape [1, seq_len]
    let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids_data)
        .map_err(|e| HypatiaError::Embedding(format!("failed to create input_ids array: {e}")))?;

    let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask_data)
        .map_err(|e| HypatiaError::Embedding(format!("failed to create attention_mask array: {e}")))?;

    let input_ids_tensor = TensorRef::from_array_view(input_ids_array.view())
        .map_err(|e| HypatiaError::Embedding(format!("failed to create input_ids tensor: {e}")))?;

    let attention_mask_tensor = TensorRef::from_array_view(attention_mask_array.view())
        .map_err(|e| HypatiaError::Embedding(format!("failed to create attention_mask tensor: {e}")))?;

    // Run inference
    let outputs = session.run(ort::inputs![input_ids_tensor, attention_mask_tensor])
        .map_err(|e| HypatiaError::Embedding(format!("inference failed: {e}")))?;

    // Extract output
    let output = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| HypatiaError::Embedding(format!("failed to extract output: {e}")))?;

    let embedding = mean_pool(&output, attention_mask_u32);
    Ok(l2_normalize(&embedding))
}

/// Mean pooling over non-padding tokens using attention mask.
fn mean_pool(hidden_states: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::IxDyn>, attention_mask: &[u32]) -> Vec<f32> {
    let shape = hidden_states.shape();

    if shape.len() == 3 {
        let seq_len = shape[1];
        let hidden_dim = shape[2];
        let mut result = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;

        for i in 0..seq_len {
            if attention_mask[i] == 1 {
                count += 1.0;
                for j in 0..hidden_dim {
                    result[j] += hidden_states[[0, i, j]];
                }
            }
        }

        if count > 0.0 {
            for v in result.iter_mut() {
                *v /= count;
            }
        }
        result
    } else if shape.len() == 2 {
        // Model already did pooling: [1, hidden_dim]
        let hidden_dim = shape[1];
        let mut result = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            result[j] = hidden_states[[0, j]];
        }
        result
    } else {
        panic!("unexpected output shape: {shape:?}");
    }
}

/// L2 normalize a vector.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod model_tests {
    use super::*;
    use std::path::PathBuf;

    fn shelf_dir() -> PathBuf {
        dirs_home().join(".hypatia").join("default")
    }

    fn dirs_home() -> PathBuf {
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."))
    }

    fn model_available() -> bool {
        let dir = shelf_dir();
        dir.join("embedding_model.onnx").exists() && dir.join("tokenizer.json").exists()
    }

    #[test]
    fn tokenizer_loads() {
        if !model_available() { return; }
        let tokenizer_path = shelf_dir().join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("tokenizer should load");
        assert!(tokenizer.get_vocab_size(false) > 0);
    }

    #[test]
    fn tokenizer_encodes_multilingual() {
        if !model_available() { return; }
        let tokenizer_path = shelf_dir().join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap();

        let enc_en = tokenizer.encode("Hello world", true).unwrap();
        assert!(!enc_en.get_ids().is_empty());

        let enc_zh = tokenizer.encode("你好世界", true).unwrap();
        assert!(!enc_zh.get_ids().is_empty());
    }

    #[test]
    fn onnx_model_loads_with_ort() {
        if !model_available() { return; }
        let model_path = shelf_dir().join("embedding_model.onnx");
        let session = Session::builder()
            .expect("builder")
            .commit_from_file(&model_path)
            .expect("ort should load the ONNX model");

        for input in session.inputs() {
            eprintln!("  input: {}", input.name());
        }
        for output in session.outputs() {
            eprintln!("  output: {}", output.name());
        }
    }

    #[test]
    fn embedder_full_pipeline() {
        if !model_available() { return; }
        let config = EmbeddingConfig {
            model_path: shelf_dir().join("embedding_model.onnx"),
            tokenizer_path: shelf_dir().join("tokenizer.json"),
            dimensions: 768,
            max_seq_length: 8192,
        };
        let embedder = Embedder::new(config);
        assert!(embedder.is_available());

        let vector = embedder.embed("Hello, world! 你好世界").expect("embed should succeed");
        assert!(!vector.is_empty(), "embedding should not be empty");

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "L2 norm should be ~1.0, got {norm}");
    }

    #[test]
    fn embedder_semantic_similarity() {
        if !model_available() { return; }
        let config = EmbeddingConfig {
            model_path: shelf_dir().join("embedding_model.onnx"),
            tokenizer_path: shelf_dir().join("tokenizer.json"),
            dimensions: 768,
            max_seq_length: 8192,
        };
        let embedder = Embedder::new(config);

        let v_cat = embedder.embed("The cat sat on the mat").unwrap();
        let v_kitten = embedder.embed("A kitten is sitting on a rug").unwrap();
        let v_code = embedder.embed("Rust programming language compiler").unwrap();

        let sim_sim = cosine_similarity(&v_cat, &v_kitten);
        let sim_diff = cosine_similarity(&v_cat, &v_code);

        eprintln!("cat vs kitten similarity: {sim_sim:.4}");
        eprintln!("cat vs code similarity:   {sim_diff:.4}");
        assert!(sim_sim > sim_diff, "semantically similar texts should have higher cosine similarity");
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }
}

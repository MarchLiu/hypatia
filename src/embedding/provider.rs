use std::cell::RefCell;
use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

use crate::error::HypatiaError;
use super::config::{EmbeddingConfig, LocalConfig, PoolingStrategy, ProviderKind, RemoteConfig};

/// Trait for embedding providers (local ONNX or remote API).
pub trait EmbeddingProvider {
    /// Generate an embedding vector for the given text.
    fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError>;

    /// Generate embeddings for multiple texts.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HypatiaError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Vector dimensions of this provider.
    fn dimensions(&self) -> usize;

    /// Whether the provider is available for use.
    fn is_available(&self) -> bool;

    /// Try to embed, returning Ok(None) if unavailable.
    fn maybe_embed(&self, text: &str) -> Result<Option<Vec<f32>>, HypatiaError> {
        if !self.is_available() {
            return Ok(None);
        }
        match self.embed(text) {
            Ok(v) => Ok(Some(v)),
            Err(HypatiaError::ModelUnavailable(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ── ONNX Provider ─────────────────────────────────────────────────────

/// Local ONNX embedding provider using onnxruntime.
pub struct OnnxProvider {
    inner: RefCell<OnnxInner>,
    dimensions: usize,
    max_seq_length: usize,
    pooling: PoolingStrategy,
}

#[allow(clippy::large_enum_variant)]
enum OnnxInner {
    Unavailable { reason: String },
    Pending { model_path: std::path::PathBuf, tokenizer_path: std::path::PathBuf },
    Ready {
        session: Session,
        tokenizer: tokenizers::Tokenizer,
    },
}

impl OnnxProvider {
    pub fn new(config: &LocalConfig) -> Self {
        let inner = if config.model_path.exists() && config.tokenizer_path.exists() {
            OnnxInner::Pending {
                model_path: config.model_path.clone(),
                tokenizer_path: config.tokenizer_path.clone(),
            }
        } else {
            OnnxInner::Unavailable {
                reason: format!(
                    "embedding model files not found: {} or {}",
                    config.model_path.display(),
                    config.tokenizer_path.display()
                ),
            }
        };
        Self {
            inner: RefCell::new(inner),
            dimensions: config.dimensions,
            max_seq_length: config.max_seq_length,
            pooling: config.pooling,
        }
    }

    pub fn unavailable() -> Self {
        Self {
            inner: RefCell::new(OnnxInner::Unavailable {
                reason: "embedding model not configured".to_string(),
            }),
            dimensions: 0,
            max_seq_length: 0,
            pooling: PoolingStrategy::Mean,
        }
    }

    fn ensure_loaded(&self) -> Result<(), HypatiaError> {
        let needs_load = match &*self.inner.borrow() {
            OnnxInner::Pending { .. } => true,
            OnnxInner::Unavailable { reason } => {
                return Err(HypatiaError::ModelUnavailable(reason.clone()));
            }
            OnnxInner::Ready { .. } => false,
        };

        if needs_load {
            let mut inner = self.inner.borrow_mut();
            let old = std::mem::replace(
                &mut *inner,
                OnnxInner::Unavailable { reason: "loading...".to_string() },
            );

            match old {
                OnnxInner::Pending { model_path, tokenizer_path } => {
                    match load_onnx_model(&model_path, &tokenizer_path) {
                        Ok((session, tokenizer)) => {
                            *inner = OnnxInner::Ready { session, tokenizer };
                            Ok(())
                        }
                        Err(e) => {
                            *inner = OnnxInner::Unavailable {
                                reason: format!("failed to load model: {e}"),
                            };
                            Err(HypatiaError::Embedding(format!(
                                "failed to load ONNX model: {e}"
                            )))
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

impl EmbeddingProvider for OnnxProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError> {
        self.ensure_loaded()?;

        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            OnnxInner::Ready { session, tokenizer, .. } => {
                run_onnx_inference(session, tokenizer, text, self.max_seq_length, self.pooling)
            }
            _ => unreachable!("ensure_loaded should guarantee Ready state"),
        }
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        matches!(
            &*self.inner.borrow(),
            OnnxInner::Pending { .. } | OnnxInner::Ready { .. }
        )
    }
}

/// Load ONNX model and tokenizer from files.
fn load_onnx_model(
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

/// Run ONNX model inference on a single text input.
fn run_onnx_inference(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    max_seq_length: usize,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, HypatiaError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| HypatiaError::Embedding(format!("tokenization failed: {e}")))?;

    let input_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();

    let len = input_ids.len().min(max_seq_length);
    let input_ids = &input_ids[..len];
    let attention_mask_u32 = &attention_mask[..len];

    let seq_len = input_ids.len();
    let input_ids_data: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
    let attention_mask_data: Vec<i64> = attention_mask_u32.iter().map(|&m| m as i64).collect();

    let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids_data)
        .map_err(|e| HypatiaError::Embedding(format!("failed to create input_ids array: {e}")))?;

    let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask_data)
        .map_err(|e| HypatiaError::Embedding(format!("failed to create attention_mask array: {e}")))?;

    let input_ids_tensor = TensorRef::from_array_view(input_ids_array.view())
        .map_err(|e| HypatiaError::Embedding(format!("failed to create input_ids tensor: {e}")))?;

    let attention_mask_tensor = TensorRef::from_array_view(attention_mask_array.view())
        .map_err(|e| HypatiaError::Embedding(format!("failed to create attention_mask tensor: {e}")))?;

    let outputs = session.run(ort::inputs![input_ids_tensor, attention_mask_tensor])
        .map_err(|e| HypatiaError::Embedding(format!("inference failed: {e}")))?;

    // Prefer sentence_embedding (index 1) if available, else token_embeddings (index 0)
    let idx = if outputs.len() > 1 { 1 } else { 0 };
    let output = outputs[idx]
        .try_extract_array::<f32>()
        .map_err(|e| HypatiaError::Embedding(format!("failed to extract output: {e}")))?;

    let embedding = extract_embedding(&output, attention_mask_u32, idx == 0, pooling);
    Ok(l2_normalize(&embedding))
}

/// Extract embedding from model output.
/// `needs_pooling` is true when using token_embeddings (index 0), false for sentence_embedding (index 1).
/// `pooling` determines how to extract from 3D output.
fn extract_embedding(
    hidden_states: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::IxDyn>,
    attention_mask: &[u32],
    needs_pooling: bool,
    pooling: PoolingStrategy,
) -> Vec<f32> {
    let shape = hidden_states.shape();

    if shape.len() == 3 && needs_pooling {
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        match pooling {
            PoolingStrategy::Mean => {
                // Mean pooling over non-padding tokens
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
            }
            PoolingStrategy::Cls => {
                // CLS token: take position 0
                let mut result = vec![0.0f32; hidden_dim];
                for j in 0..hidden_dim {
                    result[j] = hidden_states[[0, 0, j]];
                }
                result
            }
            PoolingStrategy::LastToken => {
                // Last non-padding token
                let mut last_pos = 0;
                for (i, &mask) in attention_mask.iter().enumerate().take(seq_len) {
                    if mask == 1 {
                        last_pos = i;
                    }
                }
                let mut result = vec![0.0f32; hidden_dim];
                for j in 0..hidden_dim {
                    result[j] = hidden_states[[0, last_pos, j]];
                }
                result
            }
        }
    } else if shape.len() == 3 {
        // sentence_embedding output but 3D: take position 0 (CLS)
        let hidden_dim = shape[2];
        let mut result = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            result[j] = hidden_states[[0, 0, j]];
        }
        result
    } else if shape.len() == 2 {
        // Already pooled: [1, hidden_dim]
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

// ── Remote API Provider ───────────────────────────────────────────────

const API_TIMEOUT_SECS: u64 = 60;
const MAX_RETRIES: u32 = 3;
const RETRY_BASE_DELAY_MS: u64 = 500;

/// Remote embedding API provider (OpenAI-compatible).
pub struct RemoteApiProvider {
    api_url: String,
    api_key_env: String,
    api_model: String,
    dimensions: usize,
}

impl RemoteApiProvider {
    pub fn new(config: &RemoteConfig) -> Self {
        Self {
            api_url: config.api_url.clone(),
            api_key_env: config.api_key_env.clone(),
            api_model: config.api_model.clone(),
            dimensions: config.dimensions,
        }
    }

    fn api_key(&self) -> Result<String, HypatiaError> {
        std::env::var(&self.api_key_env).map_err(|_| {
            HypatiaError::Embedding(format!(
                "environment variable {} not set",
                self.api_key_env
            ))
        })
    }

    /// Send an embedding request with timeout and retry.
    fn request_with_retry(&self, input: &serde_json::Value) -> Result<serde_json::Value, HypatiaError> {
        let api_key = self.api_key()?;

        let mut request_body = serde_json::json!({
            "model": self.api_model,
            "input": input,
        });
        if self.dimensions > 0 {
            request_body["dimensions"] = serde_json::json!(self.dimensions);
        }

        let timeout = std::time::Duration::from_secs(API_TIMEOUT_SECS);
        let mut last_err = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(
                    RETRY_BASE_DELAY_MS * 2u64.pow(attempt - 1),
                );
                eprintln!(
                    "    [remote-embed] retry {attempt}/{MAX_RETRIES} after {}ms",
                    delay.as_millis()
                );
                std::thread::sleep(delay);
            }

            let result = ureq::post(&self.api_url)
                .header("Authorization", &format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .config()
                .timeout_per_call(Some(timeout))
                .http_status_as_error(false)
                .build()
                .send_json(&request_body);

            match result {
                Ok(mut response) => {
                    let status = response.status();
                    if status.as_u16() >= 400 {
                        let msg = response.body_mut().read_to_string().unwrap_or_default();
                        if status.as_u16() == 429 {
                            last_err = Some(format!("rate limited (429): {msg}"));
                            continue;
                        }
                        return Err(HypatiaError::Embedding(format!(
                            "API returned {}: {msg}", status
                        )));
                    }
                    let body: serde_json::Value = response
                        .body_mut()
                        .read_json()
                        .map_err(|e| HypatiaError::Embedding(format!(
                            "failed to parse API response: {e}"
                        )))?;
                    return Ok(body);
                }
                Err(e) => {
                    last_err = Some(format!("request failed: {e}"));
                    continue;
                }
            }
        }

        Err(HypatiaError::Embedding(format!(
            "all {} retries exhausted: {}",
            MAX_RETRIES,
            last_err.unwrap_or_else(|| "unknown error".into())
        )))
    }

    /// Parse a single embedding from the API response.
    fn parse_embedding(response: &serde_json::Value, index: usize) -> Result<Vec<f32>, HypatiaError> {
        let embedding = response
            .get("data")
            .and_then(|d| d.get(index))
            .and_then(|d| d.get("embedding"))
            .and_then(|e| e.as_array())
            .ok_or_else(|| {
                HypatiaError::Embedding("unexpected API response format".into())
            })?;

        let vector: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if vector.is_empty() {
            return Err(HypatiaError::Embedding(
                "API returned empty embedding".into(),
            ));
        }

        Ok(vector)
    }
}

impl EmbeddingProvider for RemoteApiProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError> {
        let response = self.request_with_retry(&serde_json::json!(text))?;
        Self::parse_embedding(&response, 0)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        std::env::var(&self.api_key_env).is_ok()
    }
}

// ── Null Provider (for when no embedding is configured) ────────────────

/// A provider that is always unavailable.
pub struct NullProvider;

impl EmbeddingProvider for NullProvider {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, HypatiaError> {
        Err(HypatiaError::ModelUnavailable(
            "no embedding provider configured".into(),
        ))
    }

    fn dimensions(&self) -> usize {
        0
    }

    fn is_available(&self) -> bool {
        false
    }
}

// ── Factory ───────────────────────────────────────────────────────────

/// Build the appropriate provider from config.
pub fn build_provider(config: &EmbeddingConfig) -> Box<dyn EmbeddingProvider> {
    match config.provider {
        ProviderKind::Local => {
            if config.local_files_exist() {
                Box::new(OnnxProvider::new(&config.local))
            } else {
                Box::new(NullProvider)
            }
        }
        ProviderKind::Remote => Box::new(RemoteApiProvider::new(&config.remote)),
    }
}

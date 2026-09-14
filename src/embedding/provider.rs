use std::cell::RefCell;
use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;

use super::config::{EmbeddingConfig, LocalConfig, PoolingStrategy, ProviderKind, RemoteConfig};
use crate::error::HypatiaError;

/// Trait for embedding providers (local ONNX or remote API).
pub trait EmbeddingProvider {
    /// Generate an embedding vector for the given text.
    fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError>;

    /// Embed many texts at once: one result per input, in order, so a bad input never
    /// costs the others their vectors.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>, HypatiaError>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// One fail-fast attempt at a batch, for automatic flushes: no retries, and no more
    /// requests than it takes to find the inputs that fail, so the command that triggered the
    /// flush never waits long on a struggling provider. `Err` means nothing got through.
    fn try_embed_batch(&self, texts: &[&str]) -> Result<BatchOutcome, BatchFailure> {
        Ok(BatchOutcome::from_results(self.embed_batch(texts)))
    }

    /// Vector dimensions of this provider.
    fn dimensions(&self) -> usize;

    /// Whether the provider is available for use.
    fn is_available(&self) -> bool;

    /// Drop whatever a loaded model holds, so a long-lived process keeps no model between
    /// requests. The next embedding loads it again. A no-op for providers that load nothing.
    fn release(&self) {}

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
    Unavailable {
        reason: String,
    },
    Pending {
        model_path: std::path::PathBuf,
        tokenizer_path: std::path::PathBuf,
    },
    Ready {
        session: Session,
        tokenizer: tokenizers::Tokenizer,
        // Kept so `release` can return to `Pending`.
        model_path: std::path::PathBuf,
        tokenizer_path: std::path::PathBuf,
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
                OnnxInner::Unavailable {
                    reason: "loading...".to_string(),
                },
            );

            match old {
                OnnxInner::Pending {
                    model_path,
                    tokenizer_path,
                } => match load_onnx_model(&model_path, &tokenizer_path) {
                    Ok((session, tokenizer)) => {
                        *inner = OnnxInner::Ready {
                            session,
                            tokenizer,
                            model_path,
                            tokenizer_path,
                        };
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
                },
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
    fn release(&self) {
        let mut inner = self.inner.borrow_mut();
        let pending = match &*inner {
            OnnxInner::Ready {
                model_path,
                tokenizer_path,
                ..
            } => Some(OnnxInner::Pending {
                model_path: model_path.clone(),
                tokenizer_path: tokenizer_path.clone(),
            }),
            _ => None,
        };
        if let Some(pending) = pending {
            // Dropping the session frees the model.
            *inner = pending;
        }
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, HypatiaError> {
        self.ensure_loaded()?;

        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            OnnxInner::Ready {
                session, tokenizer, ..
            } => run_onnx_inference(session, tokenizer, text, self.max_seq_length, self.pooling),
            _ => unreachable!("ensure_loaded should guarantee Ready state"),
        }
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>, HypatiaError>> {
        if let Err(e) = self.ensure_loaded() {
            return texts.iter().map(|_| Err(same_error(&e))).collect();
        }
        let mut inner = self.inner.borrow_mut();
        let OnnxInner::Ready {
            session, tokenizer, ..
        } = &mut *inner
        else {
            unreachable!("ensure_loaded should guarantee Ready state");
        };
        if !accepts_batches(session) {
            return texts
                .iter()
                .map(|t| {
                    run_onnx_inference(session, tokenizer, t, self.max_seq_length, self.pooling)
                })
                .collect();
        }
        let pad = pad_id(tokenizer);
        // Tokenize everything first: a tokenizer failure costs only its own text.
        let encoded: Vec<_> = texts
            .iter()
            .map(|t| encode(tokenizer, t, self.max_seq_length))
            .collect();
        let lens: Vec<usize> = encoded
            .iter()
            .map(|e| e.as_ref().map_or(0, |(ids, _)| ids.len()))
            .collect();
        // Sorted by length, each forward pass pads as little as possible.
        let mut order: Vec<usize> = (0..texts.len()).filter(|&i| encoded[i].is_ok()).collect();
        order.sort_by_key(|&i| lens[i]);
        let mut vectors: Vec<Option<Result<Vec<f32>, HypatiaError>>> =
            (0..texts.len()).map(|_| None).collect();
        for chunk in token_budget_chunks(&order, &lens) {
            let rows: Vec<(&[i64], &[i64])> = chunk
                .iter()
                .map(|&i| match &encoded[i] {
                    Ok((ids, mask)) => (ids.as_slice(), mask.as_slice()),
                    Err(_) => unreachable!("only encoded texts are batched"),
                })
                .collect();
            match run_onnx_batch(session, &rows, pad, self.pooling) {
                Ok(batch) => {
                    for (&i, vector) in chunk.iter().zip(batch) {
                        vectors[i] = Some(Ok(vector));
                    }
                }
                // Retry a failed pass row by row: that isolates a row that truly fails, and
                // survives exports whose batch size is fixed at something other than 1.
                Err(_) if rows.len() > 1 => {
                    for (&i, row) in chunk.iter().zip(&rows) {
                        vectors[i] = Some(
                            run_onnx_batch(session, std::slice::from_ref(row), pad, self.pooling)
                                .map(|mut v| v.remove(0)),
                        );
                    }
                }
                Err(e) => {
                    for &i in &chunk {
                        vectors[i] = Some(Err(same_error(&e)));
                    }
                }
            }
        }
        encoded
            .into_iter()
            .zip(vectors)
            .map(|(encoding, vector)| match (encoding, vector) {
                (_, Some(vector)) => vector,
                (Err(e), None) => Err(e),
                (Ok(_), None) => unreachable!("every encoded text belongs to a chunk"),
            })
            .collect()
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

/// Padded tokens per forward pass: bounds memory without starving short texts of batching.
const MAX_BATCH_TOKENS: usize = 8192;

/// Token ids and attention mask for one text, truncated to the model's window.
fn encode(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    max_seq_length: usize,
) -> Result<(Vec<i64>, Vec<i64>), HypatiaError> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| HypatiaError::Embedding(format!("tokenization failed: {e}")))?;
    let len = encoding.get_ids().len().min(max_seq_length);
    if len == 0 {
        // An all-padding row would pool to a zero vector.
        return Err(HypatiaError::Embedding("text produced no tokens".into()));
    }
    Ok((
        encoding.get_ids()[..len]
            .iter()
            .map(|&id| id as i64)
            .collect(),
        encoding.get_attention_mask()[..len]
            .iter()
            .map(|&m| m as i64)
            .collect(),
    ))
}

/// Run ONNX model inference on a single text input.
fn run_onnx_inference(
    session: &mut Session,
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    max_seq_length: usize,
    pooling: PoolingStrategy,
) -> Result<Vec<f32>, HypatiaError> {
    let (ids, mask) = encode(tokenizer, text, max_seq_length)?;
    // A batch of one has no padding, so the pad id is never used.
    Ok(run_onnx_batch(session, &[(&ids, &mask)], 0, pooling)?.remove(0))
}

/// One forward pass over right-padded rows. Padding is masked out of attention and pooling,
/// so each row gets the vector it would get alone.
fn run_onnx_batch(
    session: &mut Session,
    rows: &[(&[i64], &[i64])],
    pad_id: i64,
    pooling: PoolingStrategy,
) -> Result<Vec<Vec<f32>>, HypatiaError> {
    let width = rows.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
    let mut input_ids_array = Array2::from_elem((rows.len(), width), pad_id);
    let mut attention_mask_array = Array2::<i64>::zeros((rows.len(), width));
    for (r, (ids, mask)) in rows.iter().enumerate() {
        for (c, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
            input_ids_array[[r, c]] = id;
            attention_mask_array[[r, c]] = m;
        }
    }

    let input_ids_tensor = TensorRef::from_array_view(input_ids_array.view())
        .map_err(|e| HypatiaError::Embedding(format!("failed to create input_ids tensor: {e}")))?;

    let attention_mask_tensor =
        TensorRef::from_array_view(attention_mask_array.view()).map_err(|e| {
            HypatiaError::Embedding(format!("failed to create attention_mask tensor: {e}"))
        })?;

    let outputs = session
        .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
        .map_err(|e| HypatiaError::Embedding(format!("inference failed: {e}")))?;

    // Prefer sentence_embedding (index 1) if available, else token_embeddings (index 0)
    let idx = if outputs.len() > 1 { 1 } else { 0 };
    let output = outputs[idx]
        .try_extract_array::<f32>()
        .map_err(|e| HypatiaError::Embedding(format!("failed to extract output: {e}")))?;

    Ok((0..rows.len())
        .map(|r| {
            let mask = attention_mask_array.row(r);
            let mask = mask.as_slice().expect("row-major rows are contiguous");
            l2_normalize(&extract_embedding(&output, r, mask, idx == 0, pooling))
        })
        .collect())
}

/// Some exports pin the batch dimension to 1; those can only run one row at a time.
fn accepts_batches(session: &Session) -> bool {
    session.inputs().iter().all(|input| match input.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.first().is_none_or(|&d| d != 1),
        _ => true,
    })
}

/// The tokenizer's padding id. Only masked positions ever see it.
fn pad_id(tokenizer: &tokenizers::Tokenizer) -> i64 {
    tokenizer
        .get_padding()
        .map(|p| p.pad_id)
        .or_else(|| {
            ["<pad>", "[PAD]"]
                .iter()
                .find_map(|t| tokenizer.token_to_id(t))
        })
        .unwrap_or(0) as i64
}

/// Groups row indices, sorted by ascending length, so no forward pass exceeds
/// `MAX_BATCH_TOKENS` padded tokens. A single oversized text still gets its own pass.
fn token_budget_chunks(order: &[usize], lens: &[usize]) -> Vec<Vec<usize>> {
    let mut chunks = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    for &i in order {
        // Sorted ascending, so the newest row is the longest and sets the padded width.
        let width = lens[i].max(1);
        if !current.is_empty() && (current.len() + 1) * width > MAX_BATCH_TOKENS {
            chunks.push(std::mem::take(&mut current));
        }
        current.push(i);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Errors are not `Clone`; each input of a failed batch gets its own copy.
fn same_error(e: &HypatiaError) -> HypatiaError {
    match e {
        HypatiaError::ModelUnavailable(reason) => HypatiaError::ModelUnavailable(reason.clone()),
        HypatiaError::Embedding(message) => HypatiaError::Embedding(message.clone()),
        other => HypatiaError::Embedding(other.to_string()),
    }
}

/// Extract one row's embedding from (batched) model output.
/// `needs_pooling` is true when using token_embeddings (index 0), false for sentence_embedding (index 1).
/// `pooling` determines how to extract from 3D output; `attention_mask` is the row's padded
/// mask, so padding never contributes.
fn extract_embedding(
    hidden_states: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::IxDyn>,
    row: usize,
    attention_mask: &[i64],
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
                            result[j] += hidden_states[[row, i, j]];
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
                    result[j] = hidden_states[[row, 0, j]];
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
                    result[j] = hidden_states[[row, last_pos, j]];
                }
                result
            }
        }
    } else if shape.len() == 3 {
        // sentence_embedding output but 3D: take position 0 (CLS)
        let hidden_dim = shape[2];
        let mut result = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            result[j] = hidden_states[[row, 0, j]];
        }
        result
    } else if shape.len() == 2 {
        // Already pooled: [batch, hidden_dim]
        let hidden_dim = shape[1];
        let mut result = vec![0.0f32; hidden_dim];
        for j in 0..hidden_dim {
            result[j] = hidden_states[[row, j]];
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
/// An automatic flush runs inside another command, so it gets one short attempt.
const QUICK_TIMEOUT_SECS: u64 = 10;

/// Inputs per request: the backfill page size, well under OpenAI's limit.
const REMOTE_BATCH: usize = 128;
/// Largest response body read: 128 vectors of 4096 dimensions stay far below it.
const MAX_RESPONSE_BYTES: u64 = 64 * 1024 * 1024;

/// Server error bodies can be whole HTML pages: messages keep only their start.
fn excerpt(text: &str) -> String {
    const MAX_CHARS: usize = 200;
    let text = text.trim();
    match text.char_indices().nth(MAX_CHARS) {
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text.to_string(),
    }
}

/// Why nothing in an automatic batch got through, so the flush can decide what to do next.
#[derive(Debug, Clone, PartialEq)]
pub enum BatchFailure {
    /// The server turned the request down as sent (400, 413 and other 4xx), even with single
    /// inputs.
    Rejected(String),
    /// The key, model or endpoint is wrong (401, 403, 404, no key, a response that is not
    /// JSON); retrying will not help.
    Refused(String),
    /// Network trouble, rate limits or server errors: worth retrying later.
    Transient(String),
}

impl BatchFailure {
    pub fn message(&self) -> &str {
        match self {
            Self::Rejected(message) | Self::Refused(message) | Self::Transient(message) => message,
        }
    }
}

/// What an automatic batch produced.
#[derive(Debug)]
pub struct BatchOutcome {
    /// One result per input, in order.
    pub vectors: Vec<Result<Vec<f32>, HypatiaError>>,
    /// Inputs that failed on their own, while the provider handled others.
    pub refused_alone: Vec<usize>,
    /// Largest batch the server accepted after turning a larger one down as too large.
    pub accepted_size: Option<usize>,
}

impl BatchOutcome {
    /// Results with nothing learned about the server beyond them: an input failing beside
    /// others that succeeded failed on its own.
    pub fn from_results(vectors: Vec<Result<Vec<f32>, HypatiaError>>) -> Self {
        let refused_alone = if vectors.iter().any(Result::is_ok) {
            (0..vectors.len())
                .filter(|&i| vectors[i].is_err())
                .collect()
        } else {
            Vec::new()
        };
        Self {
            vectors,
            refused_alone,
            accepted_size: None,
        }
    }
}

/// A request that failed for good. `status` is set when the server rejected it; `transient`
/// when it failed only for reasons that may pass (rate limits, server or network errors).
struct RequestFailure {
    status: Option<u16>,
    transient: bool,
    message: String,
}

impl RequestFailure {
    /// What this failure says about every request like it.
    fn kind(&self) -> BatchFailure {
        let message = self.message.clone();
        match self.status {
            _ if self.transient => BatchFailure::Transient(message),
            None | Some(401 | 403 | 404) => BatchFailure::Refused(message),
            Some(_) => BatchFailure::Rejected(message),
        }
    }
}

/// Requests an automatic flush may send to find the inputs a server refuses. Refusals come
/// back fast, and isolating one input among 128 takes about 16 requests.
const QUICK_BUDGET: usize = 32;
/// Longest an automatic flush spends on the server, however many requests it has left.
const QUICK_DEADLINE_SECS: u64 = 30;

/// How one embedding call talks to the server, and what it has learned so far.
struct Attempt {
    timeout: std::time::Duration,
    retries: u32,
    /// Requests still allowed; `None` for no limit.
    budget: Option<usize>,
    /// When to stop sending; `None` for no limit.
    deadline: Option<std::time::Instant>,
    /// Whether the server has accepted any request in this call; see `embed_chunk`.
    proven: bool,
    /// Whether the server has turned a chunk down as too large (413) in this call.
    too_large: bool,
    /// Largest chunk accepted after a 413.
    accepted_size: Option<usize>,
    /// What the first failed request said; it explains a call where nothing got through.
    first_failure: Option<BatchFailure>,
}

impl Attempt {
    /// An explicit backfill: full retries, as many requests as it takes.
    fn thorough() -> Self {
        Self::new(
            std::time::Duration::from_secs(API_TIMEOUT_SECS),
            MAX_RETRIES,
            None,
            None,
        )
    }

    /// An automatic flush: a short timeout, no retries, a few requests and a few seconds at
    /// most.
    fn quick() -> Self {
        Self::new(
            std::time::Duration::from_secs(QUICK_TIMEOUT_SECS),
            0,
            Some(QUICK_BUDGET),
            Some(std::time::Duration::from_secs(QUICK_DEADLINE_SECS)),
        )
    }

    fn new(
        timeout: std::time::Duration,
        retries: u32,
        budget: Option<usize>,
        total: Option<std::time::Duration>,
    ) -> Self {
        Self {
            timeout,
            retries,
            budget,
            deadline: total.map(|total| std::time::Instant::now() + total),
            proven: false,
            too_large: false,
            accepted_size: None,
            first_failure: None,
        }
    }
}

/// An input left without a vector, and whether it failed on its own.
struct Failed {
    error: HypatiaError,
    alone: bool,
}

/// An input that failed along with its request.
fn failed(message: &str) -> Result<Vec<f32>, Failed> {
    Err(Failed {
        error: HypatiaError::Embedding(message.into()),
        alone: false,
    })
}

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

    fn api_key(&self) -> Result<String, RequestFailure> {
        std::env::var(&self.api_key_env).map_err(|_| RequestFailure {
            status: None,
            transient: false,
            message: format!("environment variable {} not set", self.api_key_env),
        })
    }

    /// Send an embedding request with timeout and retry.
    fn request_with_retry(
        &self,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value, RequestFailure> {
        self.request(
            input,
            std::time::Duration::from_secs(API_TIMEOUT_SECS),
            MAX_RETRIES,
        )
    }

    /// Send an embedding request, retrying rate limits, server errors and network trouble
    /// up to `retries` times.
    fn request(
        &self,
        input: &serde_json::Value,
        timeout: std::time::Duration,
        retries: u32,
    ) -> Result<serde_json::Value, RequestFailure> {
        let api_key = self.api_key()?;

        let mut request_body = serde_json::json!({
            "model": self.api_model,
            "input": input,
        });
        if self.dimensions > 0 {
            request_body["dimensions"] = serde_json::json!(self.dimensions);
        }

        let mut last_err = None;

        for attempt in 0..=retries {
            if attempt > 0 {
                let delay =
                    std::time::Duration::from_millis(RETRY_BASE_DELAY_MS * 2u64.pow(attempt - 1));
                eprintln!(
                    "    [remote-embed] retry {attempt}/{retries} after {}ms",
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
                    let status = response.status().as_u16();
                    if status >= 400 {
                        let msg =
                            excerpt(&response.body_mut().read_to_string().unwrap_or_default());
                        // Rate limits and server errors are transient; any other rejection is
                        // about the request itself and would fail again.
                        if status == 429 || status >= 500 {
                            last_err = Some(format!("API returned {status}: {msg}"));
                            continue;
                        }
                        return Err(RequestFailure {
                            status: Some(status),
                            transient: false,
                            message: format!("API returned {status}: {msg}"),
                        });
                    }
                    return response
                        .body_mut()
                        .with_config()
                        .limit(MAX_RESPONSE_BYTES)
                        .read_json()
                        .map_err(|e| RequestFailure {
                            status: None,
                            // A body that is not the expected JSON points at the wrong
                            // endpoint; a read that failed (timeout, reset) may pass later.
                            transient: !matches!(&e, ureq::Error::Json(json) if !json.is_io()),
                            message: format!("failed to parse API response: {e}"),
                        });
                }
                Err(e) => {
                    last_err = Some(format!("request failed: {e}"));
                    continue;
                }
            }
        }

        let last_err = last_err.unwrap_or_else(|| "unknown error".into());
        Err(RequestFailure {
            status: None,
            transient: true,
            message: if retries == 0 {
                last_err
            } else {
                format!("all {retries} retries exhausted: {last_err}")
            },
        })
    }

    /// One request for the whole chunk. 413 means too large: split in half and retry, down to
    /// single inputs. 400 (or 422) is ambiguous: it may refuse the request itself (say, an unsupported
    /// parameter) or just one input. Once the server has accepted any request in this call
    /// (`proven`), a 400 is about an input and the chunk is split; before that, two inputs are
    /// probed first, since either one might be the refused input.
    ///
    /// An input the server turns down alone, once it has accepted other requests, failed on
    /// its own. Returns the message of a failure every later request would repeat, so the
    /// caller can stop sending them.
    fn embed_chunk(
        &self,
        attempt: &mut Attempt,
        chunk: &[&str],
        results: &mut Vec<Result<Vec<f32>, Failed>>,
    ) -> Option<String> {
        let failure = match self.send(attempt, &serde_json::json!(chunk)) {
            Ok(response) => {
                if attempt.too_large {
                    attempt.accepted_size = attempt.accepted_size.max(Some(chunk.len()));
                }
                match Self::parse_batch(&response, chunk.len()) {
                    Ok(vectors) => results.extend(vectors.into_iter().map(Ok)),
                    Err(e) => results.extend(chunk.iter().map(|_| {
                        Err(Failed {
                            error: same_error(&e),
                            alone: false,
                        })
                    })),
                }
                return None;
            }
            Err(failure) => failure,
        };
        let splittable = chunk.len() > 1
            && match failure.status {
                Some(413) => {
                    attempt.too_large = true;
                    true
                }
                Some(400 | 422) => attempt.proven || self.probe(attempt, chunk),
                _ => false,
            };
        if splittable {
            let (left, right) = chunk.split_at(chunk.len() / 2);
            if let Some(fatal) = self.embed_chunk(attempt, left, results) {
                results.extend(right.iter().map(|_| failed(&fatal)));
                return Some(fatal);
            }
            return self.embed_chunk(attempt, right, results);
        }
        let alone = chunk.len() == 1
            && attempt.proven
            && matches!(failure.status, Some(status) if !matches!(status, 401 | 403 | 404));
        results.extend(chunk.iter().map(|_| {
            Err(Failed {
                error: HypatiaError::Embedding(failure.message.clone()),
                alone,
            })
        }));
        // A refused request (bad key, unknown model, rejected parameter) or an unreachable
        // server fails every later request too; a single refused input does not.
        let repeats = match failure.status {
            None | Some(401 | 403 | 404) => true,
            Some(400 | 422) => chunk.len() > 1 && !attempt.proven,
            _ => false,
        };
        repeats.then_some(failure.message)
    }

    /// Whether the server accepts a request at all, tried with the shortest input and then
    /// the longest: either might be the one it refuses.
    fn probe(&self, attempt: &mut Attempt, chunk: &[&str]) -> bool {
        let shortest = (0..chunk.len())
            .min_by_key(|&i| chunk[i].len())
            .expect("chunk is nonempty");
        let longest = (0..chunk.len())
            .max_by_key(|&i| chunk[i].len())
            .expect("chunk is nonempty");
        self.send(attempt, &serde_json::json!([chunk[shortest]]))
            .is_ok()
            || (longest != shortest
                && self
                    .send(attempt, &serde_json::json!([chunk[longest]]))
                    .is_ok())
    }

    /// One request under the attempt's timeout, retries, request budget and deadline.
    fn send(
        &self,
        attempt: &mut Attempt,
        input: &serde_json::Value,
    ) -> Result<serde_json::Value, RequestFailure> {
        let left = attempt
            .deadline
            .map(|deadline| deadline.saturating_duration_since(std::time::Instant::now()));
        let outcome = match &mut attempt.budget {
            _ if left.is_some_and(|left| left.is_zero()) => Err(RequestFailure {
                status: None,
                transient: true,
                message: "stopped after too long; `hypatia backfill` tries harder".into(),
            }),
            Some(0) => Err(RequestFailure {
                status: None,
                transient: true,
                message: "stopped after too many requests; `hypatia backfill` tries harder".into(),
            }),
            budget => {
                if let Some(requests) = budget {
                    *requests -= 1;
                }
                // The last request must not outlast the deadline either.
                let timeout = left.map_or(attempt.timeout, |left| attempt.timeout.min(left));
                self.request(input, timeout, attempt.retries)
            }
        };
        match &outcome {
            Ok(_) => attempt.proven = true,
            Err(failure) => {
                attempt.first_failure.get_or_insert_with(|| failure.kind());
            }
        }
        outcome
    }

    /// Embeds `texts` in chunks of `size` under `attempt`, stopping at a failure every later
    /// request would repeat.
    fn embed_chunks(
        &self,
        attempt: &mut Attempt,
        texts: &[&str],
        size: usize,
    ) -> Vec<Result<Vec<f32>, Failed>> {
        let mut results = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(size.max(1)) {
            if let Some(fatal) = self.embed_chunk(attempt, chunk, &mut results) {
                // Every later request would fail the same way: fail their inputs unsent.
                results.extend(texts[results.len()..].iter().map(|_| failed(&fatal)));
                break;
            }
        }
        results
    }

    /// Vectors in input order. The spec gives each item an `index`; servers that omit it
    /// are taken positionally.
    fn parse_batch(
        response: &serde_json::Value,
        expected: usize,
    ) -> Result<Vec<Vec<f32>>, HypatiaError> {
        let data = response
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| HypatiaError::Embedding("unexpected API response format".into()))?;
        if data.len() != expected {
            return Err(HypatiaError::Embedding(format!(
                "API returned {} embeddings for {expected} inputs",
                data.len()
            )));
        }
        let mut vectors: Vec<Option<Vec<f32>>> = vec![None; expected];
        for (position, item) in data.iter().enumerate() {
            let index = item
                .get("index")
                .and_then(|i| i.as_u64())
                .map_or(position, |i| i as usize);
            let slot = vectors.get_mut(index).ok_or_else(|| {
                HypatiaError::Embedding(format!("API returned out-of-range index {index}"))
            })?;
            if slot.is_some() {
                return Err(HypatiaError::Embedding(format!(
                    "API returned index {index} twice"
                )));
            }
            *slot = Some(Self::parse_vector(item)?);
        }
        Ok(vectors
            .into_iter()
            .map(|v| v.expect("counts match and no index repeats"))
            .collect())
    }

    /// Parse one item of the response's `data` array.
    fn parse_vector(item: &serde_json::Value) -> Result<Vec<f32>, HypatiaError> {
        let embedding = item
            .get("embedding")
            .and_then(|e| e.as_array())
            .ok_or_else(|| HypatiaError::Embedding("unexpected API response format".into()))?;

        // A value that is not a number must not quietly shorten the vector.
        let vector: Vec<f32> = embedding
            .iter()
            .map(|v| v.as_f64().map(|f| f as f32))
            .collect::<Option<_>>()
            .ok_or_else(|| {
                HypatiaError::Embedding("API returned a non-numeric embedding value".into())
            })?;

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
        let response = self
            .request_with_retry(&serde_json::json!(text))
            .map_err(|f| HypatiaError::Embedding(f.message))?;
        Ok(Self::parse_batch(&response, 1)?.remove(0))
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<Result<Vec<f32>, HypatiaError>> {
        self.embed_chunks(&mut Attempt::thorough(), texts, REMOTE_BATCH)
            .into_iter()
            .map(|result| result.map_err(|failed| failed.error))
            .collect()
    }

    fn try_embed_batch(&self, texts: &[&str]) -> Result<BatchOutcome, BatchFailure> {
        let mut attempt = Attempt::quick();
        let results = self.embed_chunks(&mut attempt, texts, texts.len());
        if !attempt.proven {
            // Nothing got through, so the first failure explains the whole batch.
            return Err(attempt
                .first_failure
                .unwrap_or_else(|| BatchFailure::Transient("no request was sent".into())));
        }
        let refused_alone = (0..results.len())
            .filter(|&i| matches!(&results[i], Err(Failed { alone: true, .. })))
            .collect();
        Ok(BatchOutcome {
            vectors: results
                .into_iter()
                .map(|result| result.map_err(|failed| failed.error))
                .collect(),
            refused_alone,
            accepted_size: attempt.accepted_size,
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn is_available(&self) -> bool {
        std::env::var(&self.api_key_env).is_ok()
    }
}

// ── Null Provider (for when no embedding is configured) ────────────────

/// A provider that is always unavailable, and says why.
pub struct NullProvider {
    reason: String,
}

impl NullProvider {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl EmbeddingProvider for NullProvider {
    fn embed(&self, _text: &str) -> Result<Vec<f32>, HypatiaError> {
        Err(HypatiaError::ModelUnavailable(self.reason.clone()))
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
                let reason = config.local_unavailable.clone().unwrap_or_else(|| {
                    let missing: Vec<String> =
                        [&config.local.model_path, &config.local.tokenizer_path]
                            .into_iter()
                            .filter(|p| !p.exists())
                            .map(|p| p.display().to_string())
                            .collect();
                    format!("embedding model files not found: {}", missing.join(", "))
                });
                Box::new(NullProvider::new(reason))
            }
        }
        ProviderKind::Remote => Box::new(RemoteApiProvider::new(&config.remote)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn padded_rows_pool_like_rows_alone() {
        // Row 0 has three real tokens; row 1 has one, followed by junk padding values.
        let batch =
            ndarray::Array3::from_shape_fn((2, 3, 2), |(r, i, j)| (r * 10 + i * 3 + j) as f32 + 1.);
        let masks: [&[i64]; 2] = [&[1, 1, 1], &[1, 0, 0]];
        for pooling in [
            PoolingStrategy::Mean,
            PoolingStrategy::Cls,
            PoolingStrategy::LastToken,
        ] {
            for (row, mask) in masks.iter().enumerate() {
                let real = mask.iter().filter(|&&m| m == 1).count();
                let alone = batch
                    .slice(ndarray::s![row..row + 1, ..real, ..])
                    .to_owned()
                    .into_dyn();
                assert_eq!(
                    extract_embedding(&batch.view().into_dyn(), row, mask, true, pooling),
                    extract_embedding(&alone.view(), 0, &mask[..real], true, pooling),
                    "{pooling:?} row {row}"
                );
            }
        }
    }

    /// A minimal HTTP server replying through `respond`. Each request body is sent on the
    /// returned channel before the reply, so once the client returns every request it made
    /// is already there: tests count requests without waiting, and cannot hang.
    fn serve(
        respond: impl Fn(&Value) -> (u16, Value) + Send + 'static,
    ) -> (String, std::sync::mpsc::Receiver<Value>) {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
        let (seen, requests) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                let body = read_request(&mut stream);
                let (status, reply) = respond(&body);
                if seen.send(body).is_err() {
                    return;
                }
                let reply = reply.to_string();
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        (url, requests)
    }

    /// Reads one request and returns its JSON body.
    fn read_request(stream: &mut std::net::TcpStream) -> Value {
        use std::io::{BufRead, Read};
        let mut reader = std::io::BufReader::new(stream);
        let mut length = 0;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap();
            }
        }
        let mut body = vec![0; length];
        reader.read_exact(&mut body).unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    /// A server answering every request with `raw`, a whole HTTP response sent as is.
    fn serve_raw(raw: &'static str) -> String {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { return };
                read_request(&mut stream);
                let _ = stream.write_all(raw.as_bytes());
            }
        });
        url
    }

    /// Input counts of the requests the server has seen, in order.
    fn batch_sizes(requests: &std::sync::mpsc::Receiver<Value>) -> Vec<usize> {
        requests
            .try_iter()
            .map(|body| body["input"].as_array().map_or(1, Vec::len))
            .collect()
    }

    fn remote(url: &str) -> RemoteApiProvider {
        RemoteApiProvider::new(&RemoteConfig {
            api_url: url.into(),
            // Any variable set on every platform: the mock server ignores the key.
            api_key_env: "PATH".into(),
            api_model: "test-model".into(),
            dimensions: 2,
        })
    }

    /// An OpenAI-style reply for these inputs, listed in reverse to exercise `index`.
    /// Each vector's first component is its input's length.
    fn embeddings(inputs: &Value) -> Value {
        let data: Vec<Value> = inputs
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .rev()
            .map(|(i, text)| {
                json!({"index": i, "embedding": [text.as_str().unwrap().len() as f32, 1.0]})
            })
            .collect();
        json!({ "data": data })
    }

    #[test]
    fn remote_batch_is_one_request_answered_in_input_order() {
        let (url, requests) = serve(|body| (200, embeddings(&body["input"])));
        let lengths: Vec<f32> = remote(&url)
            .embed_batch(&["a", "bbb", "cc"])
            .into_iter()
            .map(|v| v.unwrap()[0])
            .collect();
        assert_eq!(lengths, [1.0, 3.0, 2.0]);
        assert_eq!(batch_sizes(&requests), [3]);
    }

    #[test]
    fn batches_the_server_rejects_as_too_large_are_halved() {
        // Only single inputs fit, and "bad" is rejected on its own merits.
        let (url, requests) = serve(|body| {
            let inputs = body["input"].as_array().unwrap();
            if inputs.len() > 1 {
                (413, json!({"error": "too many inputs"}))
            } else if inputs[0] == "bad" {
                (400, json!({"error": "bad input"}))
            } else {
                (200, embeddings(&body["input"]))
            }
        });
        let got = remote(&url).embed_batch(&["a", "bb", "bad", "cccc"]);
        assert_eq!(
            got.iter().map(Result::is_ok).collect::<Vec<_>>(),
            [true, true, false, true]
        );
        assert_eq!(got[3].as_ref().unwrap()[0], 4.0);
        assert_eq!(batch_sizes(&requests), [4, 2, 1, 1, 2, 1, 1]);
    }

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("text {i}")).collect()
    }

    #[test]
    fn a_refused_request_is_probed_once_and_not_split() {
        // Every request is refused the same way (say, an unsupported parameter).
        let (url, requests) = serve(|_| (400, json!({"error": "unsupported parameter"})));
        let owned = texts(130);
        let inputs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let got = remote(&url).embed_batch(&inputs);
        assert_eq!(got.len(), 130);
        assert!(got.iter().all(Result::is_err));
        // The first chunk and two probes; the second chunk is never sent.
        assert_eq!(batch_sizes(&requests), [128, 1, 1]);
    }

    #[test]
    fn a_refused_input_is_isolated_even_when_it_is_the_shortest() {
        // Any request containing "b" is refused, and "b" is the shortest input.
        let (url, requests) = serve(|body| {
            if body["input"].as_array().unwrap().iter().any(|t| t == "b") {
                (400, json!({"error": "input rejected"}))
            } else {
                (200, embeddings(&body["input"]))
            }
        });
        let got = remote(&url).embed_batch(&["xxxx", "b", "yyyy", "zzzz"]);
        assert_eq!(
            got.iter().map(Result::is_ok).collect::<Vec<_>>(),
            [true, false, true, true]
        );
        // The chunk, both probes (the longest passes), then ordinary splitting.
        assert_eq!(batch_sizes(&requests), [4, 1, 1, 2, 1, 1, 2]);
    }

    #[test]
    fn server_errors_are_retried_but_rejections_are_not() {
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = attempts.clone();
        let (url, requests) = serve(move |body| {
            if counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                (503, json!({}))
            } else {
                (200, embeddings(&body["input"]))
            }
        });
        assert!(remote(&url).embed_batch(&["a"])[0].is_ok());
        assert_eq!(batch_sizes(&requests), [1, 1]);

        // A bad key is final: no retry, and later chunks are not sent.
        let (url, requests) = serve(|_| (401, json!({"error": "bad key"})));
        let owned = texts(130);
        let inputs: Vec<&str> = owned.iter().map(String::as_str).collect();
        let got = remote(&url).embed_batch(&inputs);
        assert_eq!(got.len(), 130);
        assert!(got.iter().all(Result::is_err));
        assert_eq!(batch_sizes(&requests), [128]);
    }

    #[test]
    fn a_quick_attempt_names_why_nothing_got_through() {
        let kind = |status: u16| {
            let (url, requests) = serve(move |_| (status, json!({"error": "no"})));
            let outcome = remote(&url).try_embed_batch(&["a", "b"]).map(|_| ());
            let kind = match outcome.unwrap_err() {
                BatchFailure::Rejected(_) => "rejected",
                BatchFailure::Refused(_) => "refused",
                BatchFailure::Transient(_) => "transient",
            };
            (kind, batch_sizes(&requests))
        };
        for (status, expected, requests) in [
            // Each input is tried alone: either might be the one refused.
            (400, "rejected", vec![2, 1, 1]),
            (413, "rejected", vec![2, 1, 1]),
            (422, "rejected", vec![2, 1, 1]),
            // Nothing else is tried again.
            (401, "refused", vec![2]),
            (403, "refused", vec![2]),
            (404, "refused", vec![2]),
            (429, "transient", vec![2]),
            (503, "transient", vec![2]),
        ] {
            assert_eq!(kind(status), (expected, requests), "{status}");
        }

        let (url, _requests) = serve(|body| (200, embeddings(&body["input"])));
        let vectors = remote(&url).try_embed_batch(&["a", "bbb"]).unwrap().vectors;
        assert_eq!(vectors[1].as_ref().unwrap()[0], 3.0);

        // A missing key is refused before anything is sent; an unreachable server is transient.
        let mut unset = remote(&url);
        unset.api_key_env = "HYPATIA_TEST_KEY_THAT_IS_NEVER_SET".into();
        assert!(matches!(
            unset.try_embed_batch(&["a"]),
            Err(BatchFailure::Refused(_))
        ));
        // Tests only listen on ephemeral ports, so none can take port 1 from under this one.
        assert!(matches!(
            remote("http://127.0.0.1:1/v1/embeddings").try_embed_batch(&["a"]),
            Err(BatchFailure::Transient(_))
        ));

        // A body that is not JSON points at the wrong endpoint; one cut short may pass later.
        let html = serve_raw(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 13\r\nConnection: close\r\n\r\n<html></html>",
        );
        assert!(matches!(
            remote(&html).try_embed_batch(&["a"]),
            Err(BatchFailure::Refused(_))
        ));
        let cut = serve_raw(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{\"data\": [",
        );
        assert!(matches!(
            remote(&cut).try_embed_batch(&["a"]),
            Err(BatchFailure::Transient(_))
        ));
    }

    #[test]
    fn a_quick_attempt_isolates_the_inputs_a_server_refuses() {
        let (url, requests) = serve(|body| {
            if body["input"].as_array().unwrap().iter().any(|t| t == "bad") {
                (400, json!({"error": "input too long"}))
            } else {
                (200, embeddings(&body["input"]))
            }
        });
        let outcome = remote(&url)
            .try_embed_batch(&["xxxx", "bad", "yy", "zzz"])
            .unwrap();
        assert_eq!(
            outcome
                .vectors
                .iter()
                .map(Result::is_ok)
                .collect::<Vec<_>>(),
            [true, false, true, true]
        );
        assert_eq!(
            (outcome.refused_alone, outcome.accepted_size),
            (vec![1], None)
        );
        // The batch, a probe with the shortest input, then ordinary splitting.
        assert_eq!(batch_sizes(&requests), [4, 1, 2, 1, 1, 2]);
    }

    #[test]
    fn a_quick_attempt_learns_the_batch_size_a_server_accepts() {
        let (url, _requests) = serve(|body| {
            if body["input"].as_array().unwrap().len() > 2 {
                (413, json!({"error": "too many inputs"}))
            } else {
                (200, embeddings(&body["input"]))
            }
        });
        let outcome = remote(&url)
            .try_embed_batch(&["a", "b", "c", "d", "e"])
            .unwrap();
        assert!(outcome.vectors.iter().all(Result::is_ok));
        assert_eq!(
            (outcome.refused_alone, outcome.accepted_size),
            (vec![], Some(2))
        );
    }

    #[test]
    fn a_quick_attempt_gives_up_after_a_few_requests() {
        let (url, requests) = serve(|_| (413, json!({"error": "too large"})));
        let owned = texts(128);
        let inputs: Vec<&str> = owned.iter().map(String::as_str).collect();
        assert!(matches!(
            remote(&url).try_embed_batch(&inputs),
            Err(BatchFailure::Rejected(_))
        ));
        assert_eq!(batch_sizes(&requests).len(), QUICK_BUDGET);
    }

    #[test]
    fn a_quick_attempt_stops_at_its_deadline() {
        let (url, requests) = serve(|body| (200, embeddings(&body["input"])));
        let mut attempt = Attempt::new(
            std::time::Duration::from_secs(10),
            0,
            None,
            Some(std::time::Duration::ZERO),
        );
        assert!(remote(&url).send(&mut attempt, &json!(["a"])).is_err());
        assert!(batch_sizes(&requests).is_empty());
        assert!(matches!(
            attempt.first_failure,
            Some(BatchFailure::Transient(_))
        ));
    }

    #[test]
    fn non_numeric_embedding_values_are_an_error() {
        let parse = |item: Value| RemoteApiProvider::parse_vector(&item);
        assert!(parse(json!({"embedding": [0.5, null, 0.25]})).is_err());
        assert!(parse(json!({"embedding": [0.5, "1"]})).is_err());
        assert_eq!(parse(json!({"embedding": [0.5, 1]})).unwrap(), [0.5, 1.0]);
    }

    #[test]
    fn long_server_messages_are_cut_short() {
        let page = format!("<html>{}</html>", "é".repeat(1000));
        let short = excerpt(&page);
        assert_eq!(short.chars().count(), 201);
        assert!(short.ends_with('…'));
        assert_eq!(excerpt("  bad key \n"), "bad key");
    }
}

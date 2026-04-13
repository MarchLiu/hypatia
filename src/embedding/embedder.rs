// Legacy re-exports — the OnnxProvider in provider.rs supersedes Embedder.
// Kept for backward compatibility with any external references.

pub use super::provider::OnnxProvider as Embedder;

#[cfg(test)]
mod model_tests {
    use super::*;
    use crate::embedding::EmbeddingProvider;
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

    fn make_provider() -> Embedder {
        let config = crate::embedding::config::LocalConfig {
            model_path: shelf_dir().join("embedding_model.onnx"),
            tokenizer_path: shelf_dir().join("tokenizer.json"),
            dimensions: 1024,
            max_seq_length: 8192,
            pooling: crate::embedding::config::PoolingStrategy::Mean,
        };
        Embedder::new(&config)
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
        let session = ort::session::Session::builder()
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
        let embedder = make_provider();
        assert!(embedder.is_available());

        let vector = embedder.embed("Hello, world! 你好世界").expect("embed should succeed");
        assert!(!vector.is_empty(), "embedding should not be empty");

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "L2 norm should be ~1.0, got {norm}");
    }

    #[test]
    fn embedder_semantic_similarity() {
        if !model_available() { return; }
        let embedder = make_provider();

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

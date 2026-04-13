pub mod config;
pub mod embedder;
pub mod provider;

pub use config::EmbeddingConfig;
pub use config::PoolingStrategy;
pub use provider::{EmbeddingProvider, OnnxProvider, RemoteApiProvider, build_provider};

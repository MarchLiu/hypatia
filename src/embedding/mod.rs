pub mod config;
pub mod embedder;
pub mod install;
pub mod provider;

pub use config::EmbeddingConfig;
pub use config::PoolingStrategy;
pub use provider::{
    BatchFailure, BatchOutcome, EmbeddingProvider, OnnxProvider, RemoteApiProvider, build_provider,
};

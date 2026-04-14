pub mod cli;
pub mod embedding;
pub mod engine;
pub mod error;
pub mod lab;
pub mod model;
pub mod service;
pub mod storage;
pub mod text;

pub use error::{HypatiaError, Result};
pub use lab::Lab;

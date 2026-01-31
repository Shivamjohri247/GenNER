//! Core error types for GenNER
use std::path::PathBuf;

/// Result type alias for GenNER
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Device for computation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Device {
    Cpu,
    Gpu(u32),
    Metal,
}

impl Default for Device {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Data type for tensors
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
}

/// Core error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Model loading error: {0}")]
    ModelLoading(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Generation error: {0}")]
    Generation(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("LoRA error: {0}")]
    LoRA(String),

    #[error("Entity parsing error: {0}")]
    EntityParsing(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Retrieval error: {0}")]
    Retrieval(String),

    #[error("Context error: {0}")]
    Context(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl From<bincode::error::EncodeError> for Error {
    fn from(err: bincode::error::EncodeError) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for Error {
    fn from(err: bincode::error::DecodeError) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::ModelLoading("Model not found".to_string());
        assert_eq!(err.to_string(), "Model loading error: Model not found");
    }
}

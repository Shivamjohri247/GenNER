//! NER pipeline module

pub mod entity;
pub mod prompt;
pub mod parser;
pub mod verification;
pub mod pipeline;

pub use entity::*;
pub use prompt::*;
pub use parser::*;
pub use verification::*;
pub use pipeline::*;

use crate::error::{Device, DType};

/// Retrieval strategy for demonstrations
#[derive(Clone, Debug, PartialEq)]
pub enum RetrievalStrategy {
    /// Random selection from training pool
    Random,

    /// kNN based on sentence embedding similarity
    SentenceKNN,

    /// kNN based on entity embedding similarity
    EntityKNN,

    /// Hybrid sentence + entity similarity
    Hybrid { sentence_weight: f32, entity_weight: f32 },
}

impl Default for RetrievalStrategy {
    fn default() -> Self {
        Self::Random
    }
}

impl RetrievalStrategy {
    /// Create hybrid strategy
    pub fn hybrid(sentence_weight: f32, entity_weight: f32) -> Self {
        Self::Hybrid {
            sentence_weight,
            entity_weight,
        }
    }
}

/// NER pipeline configuration
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    /// Entity prefix marker (default: "@@")
    pub entity_prefix: String,

    /// Entity suffix marker (default: "##")
    pub entity_suffix: String,

    /// Number of demonstrations to include in prompt
    pub num_demonstrations: usize,

    /// Retrieval strategy for demonstrations
    pub retrieval_strategy: RetrievalStrategy,

    /// Whether to enable self-verification
    pub verification_enabled: bool,

    /// Verification threshold (0-1)
    pub verification_threshold: f32,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Device for computation
    pub device: Device,

    /// Data type for computation
    pub dtype: DType,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            entity_prefix: "@@".to_string(),
            entity_suffix: "##".to_string(),
            num_demonstrations: 4,
            retrieval_strategy: RetrievalStrategy::default(),
            verification_enabled: true,
            verification_threshold: 0.5,
            max_seq_len: 2048,
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }
}

impl PipelineConfig {
    /// Create default config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set entity markers
    pub fn with_markers(mut self, prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        self.entity_prefix = prefix.into();
        self.entity_suffix = suffix.into();
        self
    }

    /// Set number of demonstrations
    pub fn with_demonstrations(mut self, num: usize) -> Self {
        self.num_demonstrations = num;
        self
    }

    /// Set retrieval strategy
    pub fn with_retrieval(mut self, strategy: RetrievalStrategy) -> Self {
        self.retrieval_strategy = strategy;
        self
    }

    /// Enable/disable verification
    pub fn with_verification(mut self, enabled: bool) -> Self {
        self.verification_enabled = enabled;
        self
    }

    /// Set verification threshold
    pub fn with_verification_threshold(mut self, threshold: f32) -> Self {
        self.verification_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set max sequence length
    pub fn with_max_seq_len(mut self, max_len: usize) -> Self {
        self.max_seq_len = max_len;
        self
    }
}

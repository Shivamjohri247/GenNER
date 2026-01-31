//! Inference engine

use genner_core::error::Result;

/// Inference engine for running models
pub struct InferenceEngine {
    /// Batch size for inference
    batch_size: usize,
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set the batch size
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::new(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_new() {
        let engine = InferenceEngine::new(16);
        assert_eq!(engine.batch_size(), 16);
    }

    #[test]
    fn test_engine_set_batch_size() {
        let mut engine = InferenceEngine::new(8);
        engine.set_batch_size(16);
        assert_eq!(engine.batch_size(), 16);
    }
}

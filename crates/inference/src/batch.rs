//! Batch inference

/// Batch configuration
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,

    /// Timeout for batch accumulation (in milliseconds)
    pub timeout_ms: u64,

    /// Whether to enable dynamic batching
    pub dynamic_batching: bool,
}

impl BatchConfig {
    /// Create a new batch config
    pub fn new(max_batch_size: usize, timeout_ms: u64) -> Self {
        Self {
            max_batch_size,
            timeout_ms,
            dynamic_batching: false,
        }
    }

    /// Enable dynamic batching
    pub fn with_dynamic_batching(mut self) -> Self {
        self.dynamic_batching = true;
        self
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new(8, 100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_new() {
        let config = BatchConfig::new(16, 200);
        assert_eq!(config.max_batch_size, 16);
        assert_eq!(config.timeout_ms, 200);
    }

    #[test]
    fn test_batch_config_dynamic() {
        let config = BatchConfig::new(8, 100).with_dynamic_batching();
        assert!(config.dynamic_batching);
    }
}

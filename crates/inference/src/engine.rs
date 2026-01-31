//! Inference engine for running NER models

use crate::cache::KVCache;
use crate::generator::{Generator, TokenCallback};
use genner_core::error::Result;
use std::time::Instant;

/// Inference engine for running models with caching and streaming
pub struct InferenceEngine {
    /// Batch size for inference
    batch_size: usize,

    /// KV-cache for efficient inference
    cache: KVCache,

    /// Text generator
    generator: Generator,

    /// Whether to enable prefix caching
    enable_prefix_cache: bool,

    /// Statistics
    stats: EngineStats,
}

/// Inference engine statistics
#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    /// Total number of tokens generated
    pub total_tokens: usize,

    /// Total number of requests processed
    pub total_requests: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Total time spent generating (milliseconds)
    pub total_time_ms: u64,

    /// Average tokens per second
    pub tokens_per_second: f32,
}

impl InferenceEngine {
    /// Create a new inference engine
    ///
    /// # Arguments
    /// * `batch_size` - Batch size for inference
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `max_cache_tokens` - Maximum tokens to cache
    pub fn new(
        batch_size: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
        max_cache_tokens: usize,
    ) -> Self {
        Self {
            batch_size,
            cache: KVCache::new(num_layers, num_heads, head_dim, max_cache_tokens),
            generator: Generator::default(),
            enable_prefix_cache: true,
            stats: EngineStats::default(),
        }
    }

    /// Create with default model config (32 layers, 32 heads, 128 dim, 2048 cache)
    pub fn with_defaults(batch_size: usize) -> Self {
        Self::new(batch_size, 32, 32, 128, 2048)
    }

    /// Get the batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set the batch size
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    /// Get the KV cache
    pub fn cache(&self) -> &KVCache {
        &self.cache
    }

    /// Get mutable reference to the KV cache
    pub fn cache_mut(&mut self) -> &mut KVCache {
        &mut self.cache
    }

    /// Get the generator
    pub fn generator(&self) -> &Generator {
        &self.generator
    }

    /// Get mutable reference to the generator
    pub fn generator_mut(&mut self) -> &mut Generator {
        &mut self.generator
    }

    /// Get whether prefix caching is enabled
    pub fn prefix_cache_enabled(&self) -> bool {
        self.enable_prefix_cache
    }

    /// Enable or disable prefix caching
    pub fn set_prefix_cache(&mut self, enabled: bool) {
        self.enable_prefix_cache = enabled;
    }

    /// Get engine statistics
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = EngineStats::default();
    }

    /// Generate text with streaming callback
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    /// * `callback` - Callback for each generated token
    ///
    /// # Returns
    /// Number of tokens generated
    pub fn generate_stream(
        &mut self,
        input_ids: &[u32],
        callback: &mut dyn TokenCallback,
    ) -> Result<usize> {
        let start_time = Instant::now();
        self.stats.total_requests += 1;

        let mut generated_count = 0;
        let mut token_counts = vec![0usize; 100000]; // Simple token count tracking

        // Prefill phase - process input tokens
        let cache_len = self.cache.current_len();
        if cache_len > 0 && self.enable_prefix_cache {
            // Check for cache hit
            // In real implementation, would check if prefix matches
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;
        }

        // Generation loop (placeholder - would use actual model)
        for _ in 0..self.generator.max_tokens() {
            // Simulate token generation
            let dummy_logits = vec![0.1; 100000];
            let token = self.generator.sample_token(dummy_logits, Some(&token_counts))?;

            // Track token count for penalties
            if token < token_counts.len() as u32 {
                token_counts[token as usize] += 1;
            }

            // Call callback
            let should_continue = callback.on_token(token, "");
            if !should_continue {
                break;
            }

            // Check stop sequences
            let dummy_text = "";
            if self.generator.should_stop(dummy_text) {
                break;
            }

            generated_count += 1;
            self.stats.total_tokens += 1;
        }

        callback.on_complete();

        let elapsed = start_time.elapsed().as_millis() as u64 as u64;
        self.stats.total_time_ms += elapsed;

        if elapsed > 0 {
            self.stats.tokens_per_second = (self.stats.total_tokens as f32 * 1000.0) / elapsed as f32;
        }

        Ok(generated_count)
    }

    /// Generate text without streaming
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    /// Tuple of (generated tokens, generation time in ms)
    pub fn generate(&mut self, input_ids: &[u32]) -> Result<(Vec<u32>, u64)> {
        let start_time = Instant::now();

        let mut callback = crate::generator::CollectingCallback::default();

        self.generate_stream(input_ids, &mut callback)?;

        let elapsed = start_time.elapsed().as_millis() as u64;

        Ok((callback.tokens, elapsed))
    }

    /// Batch generation (process multiple inputs)
    ///
    /// # Arguments
    /// * `batch_inputs` - Vector of input token IDs
    ///
    /// # Returns
    /// Vector of generated token sequences
    pub fn generate_batch(&mut self, batch_inputs: Vec<Vec<u32>>) -> Result<Vec<Vec<u32>>> {
        let start_time = Instant::now();

        let mut results = Vec::with_capacity(batch_inputs.len());

        for input_ids in batch_inputs {
            let (tokens, _) = self.generate(&input_ids)?;
            results.push(tokens);
        }

        let elapsed = start_time.elapsed().as_millis() as u64;
        self.stats.total_time_ms += elapsed;

        Ok(results)
    }

    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self) -> usize {
        // Cache size + approximate model size
        self.cache.size_bytes() + (self.batch_size * 1024 * 1024) // 1MB per sequence slot
    }

    /// Reset the engine state (clear cache, etc.)
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

impl Default for InferenceEngine {
    fn default() -> Self {
        Self::with_defaults(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_new() {
        let engine = InferenceEngine::new(16, 4, 8, 64, 1024);
        assert_eq!(engine.batch_size(), 16);
        assert_eq!(engine.cache.max_tokens(), 1024);
    }

    #[test]
    fn test_engine_with_defaults() {
        let engine = InferenceEngine::with_defaults(16);
        assert_eq!(engine.batch_size(), 16);
        assert_eq!(engine.cache.max_tokens(), 2048);
    }

    #[test]
    fn test_engine_set_batch_size() {
        let mut engine = InferenceEngine::with_defaults(8);
        engine.set_batch_size(16);
        assert_eq!(engine.batch_size(), 16);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = InferenceEngine::with_defaults(8);
        engine.cache_mut().set_current_len(100);
        engine.reset();
        assert_eq!(engine.cache.current_len(), 0);
    }

    #[test]
    fn test_engine_stats() {
        let engine = InferenceEngine::with_defaults(8);
        assert_eq!(engine.stats().total_tokens, 0);
        assert_eq!(engine.stats().total_requests, 0);
    }

    #[test]
    fn test_engine_reset_stats() {
        let mut engine = InferenceEngine::with_defaults(8);
        engine.stats.total_tokens = 1000;
        engine.stats.total_requests = 10;
        engine.reset_stats();
        assert_eq!(engine.stats().total_tokens, 0);
        assert_eq!(engine.stats().total_requests, 0);
    }

    #[test]
    fn test_engine_estimate_memory() {
        let engine = InferenceEngine::new(16, 4, 8, 64, 1024);
        let memory = engine.estimate_memory_usage();
        assert!(memory > 0);
        // Should be at least the cache size
        assert!(memory >= engine.cache.size_bytes());
    }

    #[test]
    fn test_engine_prefix_cache() {
        let mut engine = InferenceEngine::with_defaults(8);
        assert!(engine.prefix_cache_enabled());

        engine.set_prefix_cache(false);
        assert!(!engine.prefix_cache_enabled());
    }

    #[test]
    fn test_engine_generator_access() {
        let mut engine = InferenceEngine::with_defaults(8);

        // Test generator config through engine
        engine.generator_mut().set_temperature(0.8);
        assert_eq!(engine.generator().temperature(), 0.8);

        engine.generator_mut().set_max_tokens(1024);
        assert_eq!(engine.generator().max_tokens(), 1024);
    }

    #[test]
    fn test_engine_generate() {
        let mut engine = InferenceEngine::with_defaults(8);
        let input_ids = vec![1, 2, 3];

        let result = engine.generate(&input_ids);
        assert!(result.is_ok());

        let (tokens, elapsed) = result.unwrap();
        assert_eq!(tokens.len(), engine.generator().max_tokens());
        assert!(elapsed > 0);
    }

    #[test]
    fn test_engine_generate_batch() {
        let mut engine = InferenceEngine::with_defaults(8);
        let batch_inputs = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];

        let result = engine.generate_batch(batch_inputs);
        assert!(result.is_ok());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 2);
    }
}

//! KV-cache management for efficient transformer inference

use crate::cache::layer_cache::LayerCache;
use std::collections::HashMap;

/// Cache for storing key-value pairs from transformer attention
#[derive(Clone, Debug)]
pub struct KVCache {
    /// Maximum number of tokens to cache per layer
    max_tokens: usize,

    /// Number of layers
    num_layers: usize,

    /// Number of attention heads
    num_heads: usize,

    /// Head dimension
    head_dim: usize,

    /// Layer caches (layer index -> cache)
    layers: Vec<LayerCache>,

    /// Current sequence length
    current_len: usize,

    /// Cache statistics
    stats: CacheStats,
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: usize,

    /// Number of cache misses
    pub misses: usize,

    /// Total tokens processed
    pub total_tokens: usize,
}

impl KVCache {
    /// Create a new KV-cache
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension of each attention head
    /// * `max_tokens` - Maximum tokens to cache (default: context window size)
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_tokens: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerCache::new(num_heads, head_dim, max_tokens))
            .collect();

        Self {
            max_tokens,
            num_layers,
            num_heads,
            head_dim,
            layers,
            current_len: 0,
            stats: CacheStats::default(),
        }
    }

    /// Create with default values (32 layers, 32 heads, 128 dim, 2048 max tokens)
    pub fn default_config() -> Self {
        Self::new(32, 32, 128, 2048)
    }

    /// Get the maximum cache size
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Get the current cache size (tokens cached)
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Set the current cache size (for testing)
    pub fn set_current_len(&mut self, len: usize) {
        self.current_len = len;
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
        self.current_len = 0;
    }

    /// Reset cache state (keep allocation, clear data)
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.current_len = 0;
    }

    /// Update the cache with new key-value pairs
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index
    /// * `keys` - Key tensor (flattened)
    /// * `values` - Value tensor (flattened)
    /// * `num_tokens` - Number of new tokens being added
    pub fn update(&mut self, layer_idx: usize, keys: Vec<f32>, values: Vec<f32>, num_tokens: usize) -> Result<(), String> {
        if layer_idx >= self.num_layers {
            return Err(format!("Layer index {} out of bounds (num_layers={})", layer_idx, self.num_layers));
        }

        if self.current_len + num_tokens > self.max_tokens {
            return Err(format!("Cache overflow: {} + {} > {}", self.current_len, num_tokens, self.max_tokens));
        }

        self.layers[layer_idx].update(keys, values, self.current_len, num_tokens)?;
        self.stats.total_tokens += num_tokens;

        // Only increment current_len after updating all layers for this position
        // (caller should manage this by calling update for all layers then increment)
        Ok(())
    }

    /// Increment the current sequence length after updating all layers
    pub fn increment_len(&mut self, num_tokens: usize) {
        self.current_len += num_tokens;
    }

    /// Get cached keys for a layer
    pub fn get_keys(&self, layer_idx: usize) -> Option<&[f32]> {
        if layer_idx >= self.num_layers {
            return None;
        }
        self.layers[layer_idx].get_keys(self.current_len)
    }

    /// Get cached values for a layer
    pub fn get_values(&self, layer_idx: usize) -> Option<&[f32]> {
        if layer_idx >= self.num_layers {
            return None;
        }
        self.layers[layer_idx].get_values(self.current_len)
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get the cache size in bytes
    pub fn size_bytes(&self) -> usize {
        // Each layer stores 2 * num_heads * head_dim * max_tokens f32 values (keys + values)
        let per_layer = 2 * self.num_heads * self.head_dim * self.max_tokens;
        self.num_layers * per_layer * std::mem::size_of::<f32>()
    }

    /// Create a prefix cache from the first N tokens
    pub fn create_prefix_cache(&self, prefix_len: usize) -> Self {
        let mut new_cache = Self::new(self.num_layers, self.num_heads, self.head_dim, self.max_tokens);
        new_cache.current_len = prefix_len;

        for (i, layer) in self.layers.iter().enumerate() {
            new_cache.layers[i] = layer.slice(0, prefix_len);
        }

        new_cache
    }

    /// Merge another cache into this one (for multi-turn conversations)
    pub fn merge(&mut self, other: &KVCache) -> Result<(), String> {
        if self.num_layers != other.num_layers
            || self.num_heads != other.num_heads
            || self.head_dim != other.head_dim
        {
            return Err("Cache dimensions don't match".to_string());
        }

        if self.current_len + other.current_len > self.max_tokens {
            return Err("Combined cache exceeds max tokens".to_string());
        }

        for i in 0..self.num_layers {
            self.layers[i].append(&other.layers[i])?;
        }

        self.current_len += other.current_len;
        Ok(())
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Per-layer cache storage
mod layer_cache {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct LayerCache {
        /// Cached keys (flattened: [seq_len, num_heads, head_dim])
        keys: Vec<f32>,

        /// Cached values (flattened: [seq_len, num_heads, head_dim])
        values: Vec<f32>,

        /// Number of attention heads
        num_heads: usize,

        /// Head dimension
        head_dim: usize,

        /// Maximum sequence length
        max_len: usize,

        /// Current sequence length
        current_len: usize,
    }

    impl LayerCache {
        pub fn new(num_heads: usize, head_dim: usize, max_len: usize) -> Self {
            let total_size = num_heads * head_dim * max_len;
            Self {
                keys: vec![0.0; total_size],
                values: vec![0.0; total_size],
                num_heads,
                head_dim,
                max_len,
                current_len: 0,
            }
        }

        pub fn clear(&mut self) {
            self.keys.fill(0.0);
            self.values.fill(0.0);
            self.current_len = 0;
        }

        pub fn reset(&mut self) {
            self.current_len = 0;
        }

        pub fn update(&mut self, keys: Vec<f32>, values: Vec<f32>, offset: usize, num_tokens: usize) -> Result<(), String> {
            let expected_size = self.num_heads * self.head_dim * num_tokens;
            if keys.len() != expected_size || values.len() != expected_size {
                return Err(format!("Expected {} elements, got keys={}, values={}",
                    expected_size, keys.len(), values.len()));
            }

            if offset + num_tokens > self.max_len {
                return Err(format!("Offset {} + tokens {} exceeds max_len {}", offset, num_tokens, self.max_len));
            }

            let base_offset = offset * self.num_heads * self.head_dim;
            for (i, (&k, &v)) in keys.iter().zip(values.iter()).enumerate() {
                self.keys[base_offset + i] = k;
                self.values[base_offset + i] = v;
            }

            self.current_len = self.current_len.max(offset + num_tokens);
            Ok(())
        }

        pub fn get_keys(&self, len: usize) -> Option<&[f32]> {
            let total = len * self.num_heads * self.head_dim;
            self.keys.get(0..total)
        }

        pub fn get_values(&self, len: usize) -> Option<&[f32]> {
            let total = len * self.num_heads * self.head_dim;
            self.values.get(0..total)
        }

        pub fn slice(&self, start: usize, end: usize) -> Self {
            let mut new = Self::new(self.num_heads, self.head_dim, self.max_len);
            new.current_len = end - start;

            let start_idx = start * self.num_heads * self.head_dim;
            let end_idx = end * self.num_heads * self.head_dim;

            new.keys[..new.current_len * self.num_heads * self.head_dim]
                .copy_from_slice(&self.keys[start_idx..end_idx]);
            new.values[..new.current_len * self.num_heads * self.head_dim]
                .copy_from_slice(&self.values[start_idx..end_idx]);

            new
        }

        pub fn append(&mut self, other: &LayerCache) -> Result<(), String> {
            if self.num_heads != other.num_heads || self.head_dim != other.head_dim {
                return Err("Layer cache dimensions don't match".to_string());
            }

            let offset = self.current_len;
            let new_len = self.current_len + other.current_len;

            if new_len > self.max_len {
                return Err("Appended cache exceeds max length".to_string());
            }

            let start_idx = offset * self.num_heads * self.head_dim;
            let end_idx = new_len * self.num_heads * self.head_dim;

            self.keys[start_idx..end_idx]
                .copy_from_slice(&other.keys[..other.current_len * self.num_heads * self.head_dim]);
            self.values[start_idx..end_idx]
                .copy_from_slice(&other.values[..other.current_len * self.num_heads * self.head_dim]);

            self.current_len = new_len;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4, 8, 64, 1024);
        assert_eq!(cache.num_layers, 4);
        assert_eq!(cache.num_heads, 8);
        assert_eq!(cache.head_dim, 64);
        assert_eq!(cache.max_tokens, 1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KVCache::default_config();
        // Simulate some cache usage
        cache.current_len = 100;
        cache.clear();
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_kv_cache_stats() {
        let cache = KVCache::default_config();
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().total_tokens, 0);
    }

    #[test]
    fn test_kv_cache_size_bytes() {
        let cache = KVCache::new(4, 8, 64, 1024);
        // Each layer: 2 * 8 * 64 * 1024 * 4 bytes = 4MB per layer
        // Total: 4 * 4MB = 16MB
        let expected = 4 * 2 * 8 * 64 * 1024 * 4;
        assert_eq!(cache.size_bytes(), expected);
    }

    #[test]
    fn test_kv_cache_create_prefix() {
        let mut cache = KVCache::new(2, 4, 32, 100);
        cache.current_len = 50;

        let prefix = cache.create_prefix_cache(20);
        assert_eq!(prefix.current_len(), 20);
        assert_eq!(prefix.num_layers, 2);
        assert_eq!(prefix.num_heads, 4);
    }

    #[test]
    fn test_kv_cache_merge() {
        let mut cache1 = KVCache::new(2, 4, 32, 100);
        cache1.current_len = 10;

        let mut cache2 = KVCache::new(2, 4, 32, 100);
        cache2.current_len = 20;

        // Merging should work when dimensions match
        let result = cache1.merge(&cache2);
        assert!(result.is_ok());
        assert_eq!(cache1.current_len(), 30);
    }

    #[test]
    fn test_kv_cache_merge_dimension_mismatch() {
        let mut cache1 = KVCache::new(2, 4, 32, 100);
        let cache2 = KVCache::new(2, 8, 32, 100); // Different num_heads

        let result = cache1.merge(&cache2);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_merge_overflow() {
        let mut cache1 = KVCache::new(2, 4, 32, 50);
        cache1.current_len = 40;

        let mut cache2 = KVCache::new(2, 4, 32, 50);
        cache2.current_len = 20;

        let result = cache1.merge(&cache2);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_cache_increment_len() {
        let mut cache = KVCache::new(2, 4, 32, 100);
        assert_eq!(cache.current_len(), 0);

        cache.increment_len(10);
        assert_eq!(cache.current_len(), 10);

        cache.increment_len(5);
        assert_eq!(cache.current_len(), 15);
    }

    #[test]
    fn test_kv_cache_get_keys_empty() {
        let cache = KVCache::new(2, 4, 32, 100);
        assert!(cache.get_keys(0).is_some()); // Should return empty slice
    }

    #[test]
    fn test_kv_cache_get_keys_out_of_bounds() {
        let cache = KVCache::new(2, 4, 32, 100);
        assert!(cache.get_keys(10).is_none()); // Layer index out of bounds
    }
}

//! KV-cache management

/// KV-Cache for efficient inference
#[derive(Clone, Debug)]
pub struct KVCache {
    /// Maximum cache size
    max_size: usize,

    /// Current cache size
    current_size: usize,
}

impl KVCache {
    /// Create a new KV-cache
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            current_size: 0,
        }
    }

    /// Get the maximum cache size
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get the current cache size
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.current_size = 0;
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_new() {
        let cache = KVCache::new(2048);
        assert_eq!(cache.max_size(), 2048);
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = KVCache::new(1024);
        cache.current_size = 512;
        cache.clear();
        assert_eq!(cache.current_size(), 0);
    }
}

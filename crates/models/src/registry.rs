//! Model registry for managing available models

use std::collections::HashMap;

/// Model registry for registering and retrieving model implementations
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
}

/// Information about a registered model
#[derive(Clone, Debug)]
pub struct ModelInfo {
    /// Model name
    pub name: String,

    /// Model path or identifier
    pub path: String,

    /// Model size in parameters
    pub size_in_params: usize,

    /// Supported device types
    pub supported_devices: Vec<String>,
}

impl ModelRegistry {
    /// Create a new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            models: HashMap::new(),
        };

        // Register built-in models
        registry.register_built_in_models();
        registry
    }

    /// Register built-in models
    fn register_built_in_models(&mut self) {
        // Qwen models
        self.models.insert(
            "qwen2-0.5b".to_string(),
            ModelInfo {
                name: "Qwen2 0.5B".to_string(),
                path: "Qwen/Qwen2-0.5B".to_string(),
                size_in_params: 500_000_000,
                supported_devices: vec!["cpu".to_string(), "cuda".to_string()],
            },
        );

        self.models.insert(
            "qwen2-1.5b".to_string(),
            ModelInfo {
                name: "Qwen2 1.5B".to_string(),
                path: "Qwen/Qwen2-1.5B".to_string(),
                size_in_params: 1_500_000_000,
                supported_devices: vec!["cpu".to_string(), "cuda".to_string()],
            },
        );

        // Gemma models
        self.models.insert(
            "gemma2-2b".to_string(),
            ModelInfo {
                name: "Gemma2 2B".to_string(),
                path: "google/gemma-2-2b-it".to_string(),
                size_in_params: 2_000_000_000,
                supported_devices: vec!["cpu".to_string(), "cuda".to_string()],
            },
        );

        // Phi models
        self.models.insert(
            "phi3-mini".to_string(),
            ModelInfo {
                name: "Phi-3 Mini".to_string(),
                path: "microsoft/Phi-3-mini-4k-instruct".to_string(),
                size_in_params: 3_800_000_000,
                supported_devices: vec!["cpu".to_string(), "cuda".to_string()],
            },
        );
    }

    /// Get model info by name
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// List all registered models
    pub fn list(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// Register a custom model
    pub fn register(&mut self, info: ModelInfo) {
        self.models.insert(info.name.clone(), info);
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_new() {
        let registry = ModelRegistry::new();
        assert!(registry.get("qwen2-0.5b").is_some());
    }

    #[test]
    fn test_registry_list() {
        let registry = ModelRegistry::new();
        let models = registry.list();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_registry_register_custom() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelInfo {
            name: "custom-model".to_string(),
            path: "path/to/model".to_string(),
            size_in_params: 1_000_000_000,
            supported_devices: vec!["cpu".to_string()],
        });

        assert!(registry.get("custom-model").is_some());
    }
}

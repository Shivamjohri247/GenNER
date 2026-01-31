//! Base model implementation

use genner_core::error::{Device, DType, Result};
use genner_core::traits::model::ModelConfig;

/// Base model configuration helper
pub struct BaseModelConfig;

impl BaseModelConfig {
    /// Create a CPU config
    pub fn cpu(model_path: impl Into<String>) -> ModelConfig {
        ModelConfig::new(model_path).with_device(Device::Cpu)
    }

    /// Create a GPU config
    pub fn gpu(model_path: impl Into<String>, device_id: u32) -> ModelConfig {
        ModelConfig::new(model_path).with_device(Device::Gpu(device_id))
    }

    /// Create a Metal config for Apple Silicon
    pub fn metal(model_path: impl Into<String>) -> ModelConfig {
        ModelConfig::new(model_path).with_device(Device::Metal)
    }

    /// Create a config with F16 precision
    pub fn f16(model_path: impl Into<String>) -> ModelConfig {
        ModelConfig::new(model_path).with_dtype(DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_model_config_cpu() {
        let config = BaseModelConfig::cpu("test-model");
        assert_eq!(config.model_path, "test-model");
        assert_eq!(config.device, Device::Cpu);
    }

    #[test]
    fn test_base_model_config_f16() {
        let config = BaseModelConfig::f16("test-model");
        assert_eq!(config.dtype, DType::F16);
    }
}

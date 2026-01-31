//! Base Candle model implementation

use crate::tokenizer::HFTokenizerWrapper;
use candle_core::{Device as CandleDevice, DType as CandleDType, Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use genner_core::error::{Device as GenDevice, DType as GenDType, Error, Result};
use genner_core::traits::model::{GenerationOptions, ModelBackend, ModelConfig, QuantizationConfig};
use genner_core::traits::tokenizer::TokenizerTrait;
use std::path::Path;

/// Convert a CandleResult to a GenNER Result
pub trait ToGennerResult<T> {
    fn genner_result(self) -> Result<T>;
}

impl<T> ToGennerResult<T> for CandleResult<T> {
    fn genner_result(self) -> Result<T> {
        self.map_err(|e| Error::Generation(e.to_string()))
    }
}

/// Convert GenNER device to Candle device
pub fn to_candle_device(device: &GenDevice) -> CandleDevice {
    match device {
        GenDevice::Cpu => CandleDevice::Cpu,
        GenDevice::Gpu(id) => {
            CandleDevice::new_cuda(*id as usize).unwrap_or(CandleDevice::Cpu)
        }
        GenDevice::Metal => {
            CandleDevice::new_metal(0).unwrap_or(CandleDevice::Cpu)
        }
    }
}

/// Convert GenNER dtype to Candle dtype
pub fn to_candle_dtype(dtype: &GenDType) -> CandleDType {
    match dtype {
        GenDType::F32 => CandleDType::F32,
        GenDType::F16 => CandleDType::F16,
        GenDType::BF16 => CandleDType::BF16,
    }
}

/// Base Candle model configuration
#[derive(Clone, Debug)]
pub struct CandleModelConfig {
    /// Model path (local or Hugging Face Hub)
    pub model_path: String,

    /// Device for computation
    pub device: CandleDevice,

    /// Data type for tensors
    pub dtype: CandleDType,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Whether to use KV cache
    pub use_cache: bool,

    /// Quantization config
    pub quantization: Option<QuantizationConfig>,
}

impl CandleModelConfig {
    /// Create from ModelConfig
    pub fn from_model_config(config: &ModelConfig) -> Self {
        Self {
            model_path: config.model_path.clone(),
            device: to_candle_device(&config.device),
            dtype: to_candle_dtype(&config.dtype),
            max_seq_len: config.max_seq_len,
            use_cache: config.use_cache,
            quantization: config.quantization.clone(),
        }
    }
}

/// Base trait for Candle-based models
pub trait CandleModel: Send + Sync {
    /// Forward pass through the model
    fn forward(&self, input_ids: &Tensor, input_positions: &[usize]) -> CandleResult<Tensor>;

    /// Get the tokenizer
    fn tokenizer(&self) -> &HFTokenizerWrapper;

    /// Get the config
    fn config(&self) -> &CandleModelConfig;

    /// Get the vocab size
    fn vocab_size(&self) -> usize {
        self.tokenizer().vocab_size()
    }

    /// Get max sequence length
    fn max_seq_len(&self) -> usize {
        self.config().max_seq_len
    }
}

/// Mock Candle model for testing (useful before actual models are implemented)
#[derive(Debug, Clone)]
pub struct MockCandleModel {
    config: CandleModelConfig,
    tokenizer: HFTokenizerWrapper,
}

impl MockCandleModel {
    /// Create a new mock model
    pub fn new(config: CandleModelConfig) -> Result<Self> {
        // For mock, we'll create a simple tokenizer from pre-trained
        let tokenizer = HFTokenizerWrapper::from_pretrained(&config.model_path)
            .or_else(|_| HFTokenizerWrapper::from_pretrained("gpt2"))?;

        Ok(Self {
            config,
            tokenizer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_conversion() {
        assert!(matches!(to_candle_device(&GenDevice::Cpu), CandleDevice::Cpu));
        // Metal device may fall back to CPU if Metal is not available
        let metal_device = to_candle_device(&GenDevice::Metal);
        assert!(matches!(metal_device, CandleDevice::Metal(_) | CandleDevice::Cpu));
    }

    #[test]
    fn test_dtype_conversion() {
        assert!(matches!(to_candle_dtype(&GenDType::F32), CandleDType::F32));
        assert!(matches!(to_candle_dtype(&GenDType::F16), CandleDType::F16));
        assert!(matches!(to_candle_dtype(&GenDType::BF16), CandleDType::BF16));
    }

    #[test]
    fn test_candle_model_config() {
        let config = ModelConfig::new("test-model")
            .with_device(GenDevice::Cpu)
            .with_dtype(GenDType::F16);

        let candle_config = CandleModelConfig::from_model_config(&config);
        assert_eq!(candle_config.model_path, "test-model");
        assert!(matches!(candle_config.device, CandleDevice::Cpu));
        assert!(matches!(candle_config.dtype, CandleDType::F16));
    }
}

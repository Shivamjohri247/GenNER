//! Core model backend trait for all SLM implementations

use crate::error::{Device, DType, Error, Result};
use crate::traits::tokenizer::TokenizerTrait;
use crate::training::lora::LoRAAdapter;
use serde::{Deserialize, Serialize};

/// Model loading configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to model weights/config
    pub model_path: String,

    /// Device for computation
    pub device: Device,

    /// Data type for tensors
    pub dtype: DType,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Whether to use KV cache
    pub use_cache: bool,

    /// Quantization configuration
    pub quantization: Option<QuantizationConfig>,

    /// Trust remote code (for HuggingFace models)
    pub trust_remote_code: bool,

    /// Temperature for generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p for nucleus sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k for sampling
    #[serde(default = "default_top_k")]
    pub top_k: u32,
}

fn default_temperature() -> f32 {
    0.0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_top_k() -> u32 {
    1
}

impl ModelConfig {
    /// Create default config for a model path
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            device: Device::Cpu,
            dtype: DType::F32,
            max_seq_len: 2048,
            use_cache: true,
            quantization: None,
            trust_remote_code: false,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
        }
    }

    /// Set device for computation
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Set data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Set max sequence length
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Enable KV cache
    pub fn with_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = use_cache;
        self
    }

    /// Set quantization
    pub fn with_quantization(mut self, config: QuantizationConfig) -> Self {
        self.quantization = Some(config);
        self
    }

    /// Set generation parameters
    pub fn with_generation(mut self, temperature: f32, top_p: f32, top_k: u32) -> Self {
        self.temperature = temperature;
        self.top_p = top_p;
        self.top_k = top_k;
        self
    }

    /// Create Qwen2 0.5B CPU config
    pub fn qwen2_0_5b_cpu() -> Self {
        Self::new("Qwen/Qwen2-0.5B").with_device(Device::Cpu)
    }

    /// Create Qwen2 0.5B GPU config
    pub fn qwen2_0_5b_gpu(device_id: u32) -> Self {
        Self::new("Qwen/Qwen2-0.5B").with_device(Device::Gpu(device_id))
    }

    /// Create Qwen2 1.5B CPU config
    pub fn qwen2_1_5b_cpu() -> Self {
        Self::new("Qwen/Qwen2-1.5B").with_device(Device::Cpu)
    }

    /// Create Gemma2 2B CPU config
    pub fn gemma2_2b_cpu() -> Self {
        Self::new("google/gemma-2-2b-it").with_device(Device::Cpu)
    }

    /// Create Phi-3 CPU config
    pub fn phi3_mini_cpu() -> Self {
        Self::new("microsoft/Phi-3-mini-4k-instruct").with_device(Device::Cpu)
    }
}

/// Quantization configuration
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationConfig {
    /// 4-bit quantization with group size
    Q4 { group_size: usize },
    /// 8-bit quantization
    Q8,
}

impl QuantizationConfig {
    /// Create Q4 quantization config
    pub fn q4(group_size: usize) -> Self {
        Self::Q4 { group_size }
    }

    /// Create Q8 quantization config
    pub fn q8() -> Self {
        Self::Q8
    }
}

/// Generation options for text generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationOptions {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Temperature for sampling (0 = greedy)
    pub temperature: f32,

    /// Nucleus sampling threshold
    pub top_p: f32,

    /// Top-k sampling
    pub top_k: u32,

    /// Stop sequences
    pub stop_sequences: Vec<String>,

    /// Whether to echo the input
    pub echo: bool,

    /// Presence penalty
    pub presence_penalty: f32,

    /// Frequency penalty
    pub frequency_penalty: f32,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            stop_sequences: vec!["##".to_string()],  // Stop after entity marker
            echo: false,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl GenerationOptions {
    /// Create default generation options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0 = greedy)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set stop sequences
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = stop_sequences;
        self
    }

    /// Add a stop sequence
    pub fn add_stop_sequence(mut self, stop: impl Into<String>) -> Self {
        self.stop_sequences.push(stop.into());
        self
    }
}

/// Core model backend trait for all SLM implementations
///
/// This trait defines the interface that all language model implementations
/// must provide to work with the GenNER system.
pub trait ModelBackend: Send + Sync + 'static {
    /// Model-specific configuration
    type Config: Clone + Send + Sync;

    /// Tokenizer type
    type Tokenizer: TokenizerTrait;

    /// Load model from path
    fn load(config: ModelConfig) -> Result<Self>
    where
        Self: Sized;

    /// Get tokenizer reference
    fn tokenizer(&self) -> &Self::Tokenizer;

    /// Get mutable tokenizer reference
    fn tokenizer_mut(&mut self) -> &mut Self::Tokenizer;

    /// Generate text
    fn generate(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> Result<Vec<u32>> {
        let _ = (input_ids, max_tokens, temperature, top_p, top_k);
        Err(Error::Generation(
            "Generate not implemented for this model".to_string(),
        ))
    }

    /// Generate with options
    fn generate_with_options(
        &self,
        input_ids: &[u32],
        options: &GenerationOptions,
    ) -> Result<Vec<u32>> {
        self.generate(
            input_ids,
            options.max_tokens,
            options.temperature,
            options.top_p,
            options.top_k,
        )
    }

    /// Generate with streaming callback
    fn generate_stream(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let tokens = self.generate(input_ids, max_tokens, temperature, 1.0, 1)?;
        for token in &tokens {
            if !callback(*token) {
                break;
            }
        }
        Ok(tokens)
    }

    /// Get hidden states for embeddings
    fn hidden_states(
        &self,
        input_ids: &[u32],
        layer: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let _ = (input_ids, layer);
        Err(Error::Generation(
            "Hidden states not implemented for this model".to_string(),
        ))
    }

    /// Apply LoRA adapter
    fn apply_adapter(&mut self, adapter: &LoRAAdapter) -> Result<()> {
        let _ = adapter;
        Err(Error::LoRA(
            "Adapter application not implemented for this model".to_string(),
        ))
    }

    /// Remove adapter
    fn remove_adapter(&mut self) -> Result<()> {
        Err(Error::LoRA(
            "Adapter removal not implemented for this model".to_string(),
        ))
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get max sequence length
    fn max_seq_len(&self) -> usize;

    /// Get the model name/path
    fn model_name(&self) -> &str;

    /// Get the device being used
    fn device(&self) -> Device;

    /// Clone the model (expensive, use sparingly)
    fn clone_model(&self) -> Result<Box<dyn ModelBackend<Config = Self::Config, Tokenizer = Self::Tokenizer>>>
    where
        Self: Sized,
    {
        Err(Error::ModelLoading(
            "Model cloning not implemented".to_string(),
        ))
    }
}

/// Dynamic model backend for trait objects
pub trait DynModelBackend: Send + Sync {
    /// Get tokenizer reference as trait object
    fn tokenizer_dyn(&self) -> &dyn TokenizerTrait;

    /// Generate text
    fn generate_dyn(
        &self,
        input_ids: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: u32,
    ) -> Result<Vec<u32>>;

    /// Get vocabulary size
    fn vocab_size_dyn(&self) -> usize;

    /// Get max sequence length
    fn max_seq_len_dyn(&self) -> usize;

    /// Get the model name/path
    fn model_name_dyn(&self) -> &str;

    /// Get the device being used
    fn device_dyn(&self) -> Device;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_new() {
        let config = ModelConfig::new("test-model");
        assert_eq!(config.model_path, "test-model");
        assert_eq!(config.device, Device::Cpu);
        assert_eq!(config.dtype, DType::F32);
    }

    #[test]
    fn test_model_config_builder() {
        let config = ModelConfig::new("test-model")
            .with_device(Device::Gpu(0))
            .with_dtype(DType::F16)
            .with_max_seq_len(4096);

        assert_eq!(config.device, Device::Gpu(0));
        assert_eq!(config.dtype, DType::F16);
        assert_eq!(config.max_seq_len, 4096);
    }

    #[test]
    fn test_generation_options_default() {
        let opts = GenerationOptions::default();
        assert_eq!(opts.max_tokens, 512);
        assert_eq!(opts.temperature, 0.0);
        assert_eq!(opts.top_p, 1.0);
        assert_eq!(opts.top_k, 1);
    }

    #[test]
    fn test_generation_options_builder() {
        let opts = GenerationOptions::new()
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_top_p(0.9)
            .add_stop_sequence("##");

        assert_eq!(opts.max_tokens, 1024);
        assert_eq!(opts.temperature, 0.7);
        assert_eq!(opts.top_p, 0.9);
        assert!(opts.stop_sequences.contains(&"##".to_string()));
    }

    #[test]
    fn test_quantization_config() {
        let q4 = QuantizationConfig::q4(64);
        assert!(matches!(q4, QuantizationConfig::Q4 { group_size: 64 }));

        let q8 = QuantizationConfig::q8();
        assert!(matches!(q8, QuantizationConfig::Q8));
    }
}

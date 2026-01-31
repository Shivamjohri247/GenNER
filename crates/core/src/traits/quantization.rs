//! Quantization trait

use crate::error::{Device, DType, Error, Result};

/// Quantization trait
///
/// This trait defines the interface for quantizing models
/// to reduce memory usage and increase inference speed.
pub trait Quantize: Send + Sync {
    /// Quantize the model to 4-bit
    fn quantize_q4(&mut self, group_size: usize) -> Result<()>;

    /// Quantize the model to 8-bit
    fn quantize_q8(&mut self) -> Result<()>;

    /// Dequantize the model back to original precision
    fn dequantize(&mut self) -> Result<()>;

    /// Check if model is quantized
    fn is_quantized(&self) -> bool;

    /// Get current quantization type
    fn quantization_type(&self) -> Option<QuantizationType>;
}

/// Type of quantization applied
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantizationType {
    /// No quantization (FP32/FP16)
    None,

    /// 4-bit quantization (GPTQ/AWQ style)
    Q4 { group_size: usize },

    /// 8-bit dynamic quantization
    Q8,
}

impl QuantizationType {
    /// Get the bit width of this quantization type
    pub fn bit_width(&self) -> usize {
        match self {
            Self::None => 32,
            Self::Q4 { .. } => 4,
            Self::Q8 => 8,
        }
    }

    /// Calculate size reduction factor
    pub fn compression_factor(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Q4 { .. } => 8.0,  // 32/4 = 8x
            Self::Q8 => 4.0,          // 32/8 = 4x
        }
    }
}

/// Quantized tensor representation
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// Type of quantization
    pub quant_type: QuantizationType,

    /// Original shape
    pub shape: Vec<usize>,

    /// Quantized data (packed)
    pub data: Vec<u8>,

    /// Scale factors (per-channel or per-group)
    pub scale: Vec<f32>,

    /// Zero points (optional)
    pub zero_point: Option<Vec<i8>>,
}

impl QuantizedTensor {
    /// Create a new Q4 quantized tensor
    pub fn q4(
        shape: Vec<usize>,
        data: Vec<u8>,
        scale: Vec<f32>,
    ) -> Self {
        Self {
            quant_type: QuantizationType::Q4 { group_size: 64 },
            shape,
            data,
            scale,
            zero_point: None,
        }
    }

    /// Create a new Q8 quantized tensor
    pub fn q8(
        shape: Vec<usize>,
        data: Vec<u8>,
        scale: Vec<f32>,
        zero_point: Vec<i8>,
    ) -> Self {
        Self {
            quant_type: QuantizationType::Q8,
            shape,
            data,
            scale,
            zero_point: Some(zero_point),
        }
    }

    /// Get the total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of scale factors
    pub fn num_scales(&self) -> usize {
        self.scale.len()
    }
}

/// Quantization configuration
#[derive(Clone, Debug)]
pub struct QuantizationConfig {
    /// Type of quantization
    pub quant_type: QuantizationType,

    /// Device for quantized computation
    pub device: Device,

    /// Whether to use fused operations
    pub fused_ops: bool,

    /// Whether to cache dequantized weights
    pub cache_dequantized: bool,
}

impl QuantizationConfig {
    /// Create Q4 quantization config
    pub fn q4(group_size: usize) -> Self {
        Self {
            quant_type: QuantizationType::Q4 { group_size },
            device: Device::Cpu,
            fused_ops: true,
            cache_dequantized: false,
        }
    }

    /// Create Q8 quantization config
    pub fn q8() -> Self {
        Self {
            quant_type: QuantizationType::Q8,
            device: Device::Cpu,
            fused_ops: true,
            cache_dequantized: false,
        }
    }

    /// Set device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Enable/disable fused operations
    pub fn with_fused_ops(mut self, fused: bool) -> Self {
        self.fused_ops = fused;
        self
    }

    /// Enable/disable caching dequantized weights
    pub fn with_cache(mut self, cache: bool) -> Self {
        self.cache_dequantized = cache;
        self
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantizationType::None,
            device: Device::Cpu,
            fused_ops: true,
            cache_dequantized: false,
        }
    }
}

/// Simple quantizer for testing
#[derive(Debug, Clone)]
pub struct DummyQuantizer {
    quantized: bool,
    quant_type: Option<QuantizationType>,
}

impl DummyQuantizer {
    /// Create a new dummy quantizer
    pub fn new() -> Self {
        Self {
            quantized: false,
            quant_type: None,
        }
    }
}

impl Default for DummyQuantizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Quantize for DummyQuantizer {
    fn quantize_q4(&mut self, group_size: usize) -> Result<()> {
        self.quantized = true;
        self.quant_type = Some(QuantizationType::Q4 { group_size });
        Ok(())
    }

    fn quantize_q8(&mut self) -> Result<()> {
        self.quantized = true;
        self.quant_type = Some(QuantizationType::Q8);
        Ok(())
    }

    fn dequantize(&mut self) -> Result<()> {
        self.quantized = false;
        self.quant_type = None;
        Ok(())
    }

    fn is_quantized(&self) -> bool {
        self.quantized
    }

    fn quantization_type(&self) -> Option<QuantizationType> {
        self.quant_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_type() {
        let q4 = QuantizationType::Q4 { group_size: 64 };
        assert_eq!(q4.bit_width(), 4);
        assert_eq!(q4.compression_factor(), 8.0);

        let q8 = QuantizationType::Q8;
        assert_eq!(q8.bit_width(), 8);
        assert_eq!(q8.compression_factor(), 4.0);
    }

    #[test]
    fn test_quantized_tensor() {
        let tensor = QuantizedTensor::q4(vec![2, 2], vec![0, 1, 2, 3], vec![1.0, 1.0]);
        assert_eq!(tensor.num_elements(), 4);
        assert_eq!(tensor.num_scales(), 2);
    }

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::q4(64).with_device(Device::Gpu(0));
        assert!(matches!(config.quant_type, QuantizationType::Q4 { group_size: 64 }));
        assert_eq!(config.device, Device::Gpu(0));
    }

    #[test]
    fn test_dummy_quantizer() {
        let mut quantizer = DummyQuantizer::new();
        assert!(!quantizer.is_quantized());

        quantizer.quantize_q4(64).unwrap();
        assert!(quantizer.is_quantized());
        assert_eq!(
            quantizer.quantization_type(),
            Some(QuantizationType::Q4 { group_size: 64 })
        );

        quantizer.dequantize().unwrap();
        assert!(!quantizer.is_quantized());
    }
}

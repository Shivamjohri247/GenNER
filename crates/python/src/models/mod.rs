//! Python bindings for models

use pyo3::prelude::*;
use pyo3::types::PyDict;
use genner_core::error::Device;
use genner_core::traits::model::ModelConfig;

/// Model wrapper for Python
#[pyclass(name = "Model")]
pub struct PyModel {
    _config: ModelConfig,
}

#[pymethods]
impl PyModel {
    /// Load a model from a path or HuggingFace hub
    #[new]
    #[pyo3(signature = (model_path, **kwargs))]
    pub fn new(model_path: String, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut config = ModelConfig::new(model_path);

        if let Some(kwargs) = kwargs {
            if let Some(device) = kwargs.get_item("device")? {
                let device_str: String = device.extract()?;
                config = config.with_device(parse_device(&device_str));
            }
            if let Some(dtype) = kwargs.get_item("dtype")? {
                let dtype_str: String = dtype.extract()?;
                config = config.with_dtype(parse_dtype(&dtype_str));
            }
            if let Some(max_seq_len) = kwargs.get_item("max_seq_len")? {
                let max_seq_len_value: usize = max_seq_len.extract()?;
                config = config.with_max_seq_len(max_seq_len_value);
            }
        }

        Ok(PyModel { _config: config })
    }

    /// Get the model name
    #[getter]
    pub fn model_name(&self) -> String {
        "genner-model".to_string()
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        100000 // Placeholder
    }

    /// Get the max sequence length
    pub fn max_seq_len(&self) -> usize {
        2048 // Placeholder
    }
}

/// Parse device string
fn parse_device(s: &str) -> Device {
    match s.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        s if s.starts_with("gpu") || s.starts_with("cuda") => {
            let num = s.trim_start_matches("gpu")
                .trim_start_matches("cuda")
                .trim_start_matches(':')
                .parse::<u32>()
                .unwrap_or(0);
            Device::Gpu(num)
        }
        s if s.starts_with("metal") || s.starts_with("mps") => Device::Metal,
        _ => Device::Cpu,
    }
}

/// Parse dtype string
fn parse_dtype(s: &str) -> genner_core::error::DType {
    match s.to_lowercase().as_str() {
        "f32" | "float32" | "float" => genner_core::error::DType::F32,
        "f16" | "float16" | "half" => genner_core::error::DType::F16,
        "bf16" | "bfloat16" => genner_core::error::DType::BF16,
        _ => genner_core::error::DType::F32,
    }
}

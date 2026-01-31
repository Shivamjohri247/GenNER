//! Python bindings for training

use pyo3::prelude::*;
use genner_core::training::lora::LoRAConfig;
use genner_core::training::trainer::TrainingConfig;

/// LoRA Configuration
#[pyclass(name = "LoRAConfig")]
#[derive(Clone)]
pub struct PyLoRAConfig {
    inner: LoRAConfig,
}

#[pymethods]
impl PyLoRAConfig {
    #[new]
    #[pyo3(signature = (rank=16, alpha=32.0, dropout=0.05))]
    pub fn new(rank: usize, alpha: f32, dropout: f32) -> Self {
        Self {
            inner: LoRAConfig::new(rank, alpha).with_dropout(dropout),
        }
    }

    #[getter]
    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    #[getter]
    pub fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    pub fn dropout(&self) -> f32 {
        self.inner.dropout
    }
}

/// Training Configuration
#[pyclass(name = "TrainingConfig")]
pub struct PyTrainingConfig {
    inner: TrainingConfig,
}

#[pymethods]
impl PyTrainingConfig {
    #[new]
    #[pyo3(signature = (
        learning_rate=5e-5,
        batch_size=8,
        num_epochs=3,
        lora=None
    ))]
    pub fn new(
        learning_rate: f32,
        batch_size: usize,
        num_epochs: usize,
        lora: Option<PyLoRAConfig>,
    ) -> Self {
        let mut config = TrainingConfig::new()
            .with_learning_rate(learning_rate)
            .with_batch_size(batch_size)
            .with_num_epochs(num_epochs);

        if let Some(lora) = lora {
            config = config.with_lora(lora.inner);
        }

        Self { inner: config }
    }

    #[getter]
    pub fn learning_rate(&self) -> f32 {
        self.inner.learning_rate
    }

    #[getter]
    pub fn batch_size(&self) -> usize {
        self.inner.batch_size
    }

    #[getter]
    pub fn num_epochs(&self) -> usize {
        self.inner.num_epochs
    }
}

/// Trainer
#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    _config: TrainingConfig,
}

#[pymethods]
impl PyTrainer {
    #[new]
    pub fn new(config: &PyTrainingConfig) -> Self {
        Self {
            _config: config.inner.clone(),
        }
    }

    /// Train on a dataset
    pub fn train(&self, train_data: String, val_data: Option<String>) -> PyResult<String> {
        // Placeholder - in real implementation, this would:
        // 1. Load the dataset
        // 2. Run training loop
        // 3. Save adapter
        Ok("trained_adapter".to_string())
    }
}

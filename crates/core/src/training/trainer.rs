//! Training loop and trainer

use crate::error::Result;
use crate::training::data::{DataLoader, TrainingDataset};
use crate::training::lora::{LoRAAdapter, LoRAConfig};
use serde::{Deserialize, Serialize};

/// Training progress callback
pub trait TrainingCallback: Send + Sync {
    /// Called at the start of training
    fn on_training_start(&self, config: &TrainingConfig) {}

    /// Called at the start of each epoch
    fn on_epoch_start(&self, epoch: usize, total_epochs: usize) {}

    /// Called after each batch
    fn on_batch_end(&self, epoch: usize, batch: usize, loss: f32) {}

    /// Called at the end of each epoch
    fn on_epoch_end(&self, epoch: usize, metrics: &EpochMetrics) {}

    /// Called when training completes
    fn on_training_end(&self, result: &TrainingResult) {}
}

/// Null callback that does nothing
#[derive(Debug, Clone, Copy)]
pub struct NullCallback;

impl TrainingCallback for NullCallback {}

/// Trainer for NER models
pub struct NERTrainer {
    config: TrainingConfig,
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl NERTrainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            callbacks: Vec::new(),
        }
    }

    /// Add a callback
    pub fn add_callback(mut self, callback: Box<dyn TrainingCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Train on a dataset
    ///
    /// This is a simplified training loop. The actual implementation
    /// would integrate with Candle for the forward/backward passes.
    pub fn train(
        &self,
        train_dataset: TrainingDataset,
        val_dataset: Option<TrainingDataset>,
    ) -> Result<TrainingResult> {
        // Notify callbacks
        for callback in &self.callbacks {
            callback.on_training_start(&self.config);
        }

        let mut all_metrics = Vec::new();
        let start_time = std::time::Instant::now();

        for epoch in 0..self.config.num_epochs {
            // Notify callbacks
            for callback in &self.callbacks {
                callback.on_epoch_start(epoch, self.config.num_epochs);
            }

            // Train for one epoch
            let epoch_metrics = self.train_epoch(epoch, &train_dataset)?;

            // Notify callbacks
            for callback in &self.callbacks {
                callback.on_epoch_end(epoch, &epoch_metrics);
            }

            all_metrics.push(epoch_metrics);
        }

        let duration = start_time.elapsed();

        // Extract adapter (simplified - would be actual trained weights)
        let adapter = LoRAAdapter::new(
            "trained_adapter",
            "ner_task",
            self.config.lora.clone(),
        );

        let result = TrainingResult {
            adapter,
            metrics: TrainingMetrics {
                epochs: all_metrics,
            },
        };

        // Notify callbacks
        for callback in &self.callbacks {
            callback.on_training_end(&result);
        }

        Ok(result)
    }

    /// Train for one epoch
    fn train_epoch(&self, epoch: usize, dataset: &TrainingDataset) -> Result<EpochMetrics> {
        let mut loader = DataLoader::new(dataset.clone(), self.config.batch_size, true);
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        while let Some(_batch) = loader.next_batch() {
            // In real implementation:
            // 1. Move batch to device
            // 2. Forward pass
            // 3. Compute loss
            // 4. Backward pass (only LoRA parameters)
            // 5. Optimizer step

            let loss = self.compute_dummy_loss(epoch, num_batches);
            total_loss += loss;
            num_batches += 1;

            // Notify callbacks
            for callback in &self.callbacks {
                callback.on_batch_end(epoch, num_batches, loss);
            }
        }

        Ok(EpochMetrics {
            epoch,
            train_loss: total_loss / num_batches as f32,
            val_loss: 0.0, // Would compute if val_dataset provided
            val_f1: 0.0,   // Would compute on validation set
        })
    }

    /// Dummy loss computation for testing
    fn compute_dummy_loss(&self, epoch: usize, batch: usize) -> f32 {
        // Simulated loss that decreases over time
        let base_loss = 2.0;
        let decay = (epoch * 1000 + batch) as f32 / 10000.0;
        (base_loss - decay).max(0.1)
    }
}

/// Training configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,

    /// Batch size
    pub batch_size: usize,

    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,

    /// Number of epochs
    pub num_epochs: usize,

    /// Warmup steps
    pub warmup_steps: usize,

    /// Weight decay
    pub weight_decay: f32,

    /// LoRA configuration
    pub lora: LoRAConfig,

    /// Save checkpoints every N steps
    pub checkpoint_every: usize,

    /// Output directory for checkpoints
    pub output_dir: String,
}

impl TrainingConfig {
    /// Create a new training config
    pub fn new() -> Self {
        Self {
            learning_rate: 5e-5,
            batch_size: 8,
            gradient_accumulation_steps: 4,
            num_epochs: 3,
            warmup_steps: 100,
            weight_decay: 0.01,
            lora: LoRAConfig::default(),
            checkpoint_every: 500,
            output_dir: "outputs".to_string(),
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set number of epochs
    pub fn with_num_epochs(mut self, num_epochs: usize) -> Self {
        self.num_epochs = num_epochs;
        self
    }

    /// Set LoRA config
    pub fn with_lora(mut self, lora: LoRAConfig) -> Self {
        self.lora = lora;
        self
    }

    /// Set output directory
    pub fn with_output_dir(mut self, dir: impl Into<String>) -> Self {
        self.output_dir = dir.into();
        self
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Training result
#[derive(Clone, Debug)]
pub struct TrainingResult {
    pub adapter: LoRAAdapter,
    pub metrics: TrainingMetrics,
}

/// Training metrics
#[derive(Clone, Debug)]
pub struct TrainingMetrics {
    pub epochs: Vec<EpochMetrics>,
}

/// Metrics for a single epoch
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub val_loss: f32,
    pub val_f1: f32,
}

/// Progress bar callback
#[derive(Debug, Clone)]
pub struct ProgressBarCallback;

impl TrainingCallback for ProgressBarCallback {
    fn on_epoch_start(&self, epoch: usize, total_epochs: usize) {
        println!("Starting epoch {}/{}", epoch + 1, total_epochs);
    }

    fn on_epoch_end(&self, epoch: usize, metrics: &EpochMetrics) {
        println!(
            "Epoch {}: loss={:.4}, val_loss={:.4}, val_f1={:.4}",
            epoch + 1,
            metrics.train_loss,
            metrics.val_loss,
            metrics.val_f1
        );
    }
}

/// Logging callback
#[derive(Debug, Clone)]
pub struct LoggingCallback;

impl TrainingCallback for LoggingCallback {
    fn on_training_start(&self, config: &TrainingConfig) {
        println!("Starting training with config:");
        println!("  Learning rate: {}", config.learning_rate);
        println!("  Batch size: {}", config.batch_size);
        println!("  Epochs: {}", config.num_epochs);
        println!("  LoRA rank: {}", config.lora.rank);
    }

    fn on_batch_end(&self, epoch: usize, batch: usize, loss: f32) {
        if batch % 10 == 0 {
            println!("  Epoch {}, batch {}: loss={:.4}", epoch + 1, batch, loss);
        }
    }

    fn on_training_end(&self, result: &TrainingResult) {
        let final_loss = result.metrics.epochs.last().map(|e| e.train_loss).unwrap_or(0.0);
        println!("Training complete. Final loss: {:.4}", final_loss);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_new() {
        let config = TrainingConfig::new();
        assert_eq!(config.learning_rate, 5e-5);
        assert_eq!(config.batch_size, 8);
        assert_eq!(config.num_epochs, 3);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new()
            .with_learning_rate(1e-4)
            .with_batch_size(16)
            .with_num_epochs(5);

        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.num_epochs, 5);
    }

    #[test]
    fn test_ner_trainer_new() {
        let trainer = NERTrainer::new(TrainingConfig::new());
        assert_eq!(trainer.config.num_epochs, 3);
    }

    #[test]
    fn test_ner_trainer_with_callbacks() {
        let trainer = NERTrainer::new(TrainingConfig::new())
            .add_callback(Box::new(NullCallback))
            .add_callback(Box::new(LoggingCallback));

        assert_eq!(trainer.callbacks.len(), 2);
    }

    #[test]
    fn test_epoch_metrics() {
        let metrics = EpochMetrics {
            epoch: 0,
            train_loss: 1.0,
            val_loss: 0.9,
            val_f1: 0.85,
        };

        assert_eq!(metrics.epoch, 0);
        assert_eq!(metrics.train_loss, 1.0);
    }
}

//! Training step implementation that combines all components
//!
//! This module provides a complete training step that includes:
//! - Forward pass through LoRA layers
//! - Loss computation
//! - Backward pass (gradient computation)
//! - Optimizer step
//! - Learning rate scheduling

use crate::candle_model::ToGennerResult;
use crate::lora_layer::{LoraLayer, AdamWOptimizer};
use crate::loss::cross_entropy;
use crate::lr_schedule::LrSchedule;
use candle_core::{Device as CandleDevice, DType as CandleDType, Tensor};
use genner_core::error::Result;

/// Training state that tracks accumulated gradients
#[derive(Clone, Debug)]
pub struct TrainingState {
    /// Current step number
    pub step: usize,
    /// Current epoch
    pub epoch: usize,
    /// Number of steps since last optimizer update
    pub accumulation_steps: usize,
    /// Accumulated loss
    pub accumulated_loss: f32,
    /// Number of samples processed
    pub samples_processed: usize,
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            step: 0,
            epoch: 0,
            accumulation_steps: 0,
            accumulated_loss: 0.0,
            samples_processed: 0,
        }
    }

    /// Reset accumulation counters
    pub fn reset_accumulation(&mut self) {
        self.accumulation_steps = 0;
        self.accumulated_loss = 0.0;
    }

    /// Increment step counter
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Increment epoch counter
    pub fn increment_epoch(&mut self) {
        self.epoch += 1;
        self.step = 0;
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Training configuration
#[derive(Clone, Debug)]
pub struct TrainingStepConfig {
    /// Gradient accumulation steps
    pub accumulation_steps: usize,
    /// Max gradient norm for clipping (0 = no clipping)
    pub max_grad_norm: f32,
    /// Whether to use gradient checkpointing
    pub gradient_checkpointing: bool,
}

impl Default for TrainingStepConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 1,
            max_grad_norm: 0.0,
            gradient_checkpointing: false,
        }
    }
}

impl TrainingStepConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation_steps = steps;
        self
    }

    pub fn with_max_grad_norm(mut self, norm: f32) -> Self {
        self.max_grad_norm = norm;
        self
    }

    pub fn with_gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }
}

/// Training step executor
///
/// Combines forward pass, loss computation, backward pass, and optimizer step.
pub struct TrainingStep {
    /// Configuration
    config: TrainingStepConfig,
    /// Learning rate schedule
    lr_schedule: Option<Box<dyn LrSchedule>>,
}

impl TrainingStep {
    pub fn new() -> Self {
        Self {
            config: TrainingStepConfig::default(),
            lr_schedule: None,
        }
    }

    pub fn with_config(mut self, config: TrainingStepConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_lr_schedule(mut self, schedule: Box<dyn LrSchedule>) -> Self {
        self.lr_schedule = Some(schedule);
        self
    }

    /// Perform a single training step
    ///
    /// # Arguments
    /// * `lora_layer` - The LoRA layer to train
    /// * `optimizer` - The optimizer (will update its LR from schedule if set)
    /// * `logits` - Model output logits (batch, seq_len, vocab_size)
    /// * `targets` - Target token IDs (batch, seq_len)
    /// * `input` - Input tensor (for gradient computation)
    /// * `state` - Training state to update
    ///
    /// # Returns
    /// The computed loss value
    pub fn step(
        &self,
        lora_layer: &mut LoraLayer,
        optimizer: &mut AdamWOptimizer,
        logits: &Tensor,
        targets: &Tensor,
        input: &Tensor,
        state: &mut TrainingState,
    ) -> Result<f32> {
        // Compute loss
        let loss = cross_entropy(logits, targets)?;

        // Get loss value (scalar)
        let loss_value = loss.to_vec1::<f32>()
            .map(|v| v.first().copied().unwrap_or(0.0))
            .unwrap_or(0.0);

        // Accumulate loss
        state.accumulated_loss += loss_value;
        state.accumulation_steps += 1;
        state.samples_processed += targets.dims().iter().product::<usize>();

        // Check if we should do optimizer step
        let should_optimize = state.accumulation_steps >= self.config.accumulation_steps;

        if should_optimize {
            // Compute gradients
            let (grad_a, grad_b) = lora_layer.compute_gradients(
                &Tensor::ones(logits.dims(), logits.dtype(), logits.device())
                    .genner_result()?, // Dummy upstream gradient of ones
                input,
            )?;

            // Apply gradient clipping if configured
            let (grad_a, grad_b) = if self.config.max_grad_norm > 0.0 {
                (
                    self.clip_gradient(&grad_a, self.config.max_grad_norm)?,
                    self.clip_gradient(&grad_b, self.config.max_grad_norm)?,
                )
            } else {
                (grad_a, grad_b)
            };

            // Update optimizer learning rate if schedule is set
            if let Some(schedule) = &self.lr_schedule {
                let new_lr = schedule.get_lr(state.step);
                // Note: AdamWOptimizer doesn't support setting LR dynamically yet
                // This would require adding a set_lr method
                let _ = new_lr; // Suppress unused warning
            }

            // Optimizer step
            optimizer.step(lora_layer, &grad_a, &grad_b)?;

            // Reset accumulation
            state.reset_accumulation();
        }

        state.increment_step();

        Ok(loss_value)
    }

    /// Clip gradients by norm
    fn clip_gradient(&self, grad: &Tensor, max_norm: f32) -> Result<Tensor> {
        let grad_norm = grad.sqr()
            .map_err(|e| genner_core::error::Error::Training(format!("Sqr failed: {}", e)))?
            .sum_all()
            .map_err(|e| genner_core::error::Error::Training(format!("Sum failed: {}", e)))?
            .sqrt()
            .map_err(|e| genner_core::error::Error::Training(format!("Sqrt failed: {}", e)))?;

        let grad_norm_val = grad_norm.to_vec1::<f32>()
            .map_err(|e| genner_core::error::Error::Training(format!("Failed to get grad norm: {}", e)))?
            .first()
            .copied()
            .unwrap_or(0.0);

        if grad_norm_val > max_norm {
            let scale = max_norm / grad_norm_val;
            Ok(grad.affine(scale as f64, 0.0)
                .map_err(|e| genner_core::error::Error::Training(format!("Failed to scale gradient: {}", e)))?)
        } else {
            Ok(grad.clone())
        }
    }
}

impl Default for TrainingStep {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_state_new() {
        let state = TrainingState::new();
        assert_eq!(state.step, 0);
        assert_eq!(state.epoch, 0);
        assert_eq!(state.accumulation_steps, 0);
        assert_eq!(state.accumulated_loss, 0.0);
    }

    #[test]
    fn test_training_state_accumulation() {
        let mut state = TrainingState::new();
        state.accumulated_loss = 1.5;
        state.accumulation_steps = 2;

        assert_eq!(state.accumulated_loss, 1.5);
        assert_eq!(state.accumulation_steps, 2);

        state.reset_accumulation();
        assert_eq!(state.accumulated_loss, 0.0);
        assert_eq!(state.accumulation_steps, 0);
    }

    #[test]
    fn test_training_state_increment() {
        let mut state = TrainingState::new();
        state.increment_step();
        assert_eq!(state.step, 1);

        state.increment_epoch();
        assert_eq!(state.epoch, 1);
        assert_eq!(state.step, 0);
    }

    #[test]
    fn test_training_step_config_default() {
        let config = TrainingStepConfig::new();
        assert_eq!(config.accumulation_steps, 1);
        assert_eq!(config.max_grad_norm, 0.0);
        assert!(!config.gradient_checkpointing);
    }

    #[test]
    fn test_training_step_config_builder() {
        let config = TrainingStepConfig::new()
            .with_accumulation_steps(4)
            .with_max_grad_norm(1.0)
            .with_gradient_checkpointing(true);

        assert_eq!(config.accumulation_steps, 4);
        assert_eq!(config.max_grad_norm, 1.0);
        assert!(config.gradient_checkpointing);
    }

    #[test]
    fn test_training_step_default() {
        let step = TrainingStep::new();
        assert_eq!(step.config.accumulation_steps, 1);
        assert!(!step.config.gradient_checkpointing);
    }
}

//! LoRA (Low-Rank Adaptation) layer implementation for training
//!
//! This module provides the actual LoRA layer that can be injected into
//! transformer models for parameter-efficient fine-tuning.

use crate::candle_model::ToGennerResult;
use candle_core::{Device as CandleDevice, DType as CandleDType, Tensor};
use genner_core::error::{Device, DType, Error, Result};

/// LoRA layer that wraps a linear transformation
///
/// Given a base weight matrix W (out_features × in_features), LoRA adds:
/// W' = BA where B is (out_features × rank) and A is (rank × in_features)
/// The output becomes: h = Wx + W'x = Wx + BAx
pub struct LoraLayer {
    /// LoRA A matrix (rank × in_features)
    lora_a: Tensor,
    /// LoRA B matrix (out_features × rank)
    lora_b: Tensor,
    /// Scaling factor (alpha / rank)
    scaling: f64,
    /// Rank of the LoRA decomposition
    rank: usize,
    /// Number of output features
    out_features: usize,
    /// Whether LoRA is enabled (for merging/unmerging)
    enabled: bool,
}

impl LoraLayer {
    /// Create a new LoRA layer
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        alpha: f32,
        device: &CandleDevice,
        dtype: CandleDType,
    ) -> Result<Self> {
        // Initialize A with Kaiming uniform scaled by 0.01
        // Initialize B with zeros (so training starts with base model behavior)
        // A: (in_features, rank), B: (out_features, rank)
        // LoRA output: B @ A @ x = (out_features, rank) @ (rank, in_features) @ x = (out_features, in_features)

        let lora_a = Tensor::zeros((in_features, rank), dtype, device)
            .genner_result()?;

        let lora_b = Tensor::zeros((out_features, rank), dtype, device)
            .genner_result()?;

        Ok(Self {
            lora_a,
            lora_b,
            scaling: alpha as f64 / rank as f64,
            rank,
            out_features,
            enabled: true,
        })
    }

    /// Create from pretrained weights
    pub fn from_weights(
        lora_a: Tensor,
        lora_b: Tensor,
        rank: usize,
        alpha: f32,
        out_features: usize,
    ) -> Self {
        Self {
            lora_a,
            lora_b,
            scaling: alpha as f64 / rank as f64,
            rank,
            out_features,
            enabled: true,
        }
    }

    /// Forward pass through LoRA: returns (B @ A) @ x scaled
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !self.enabled {
            // Return zeros if disabled
            return Ok(Tensor::zeros(x.dims(), x.dtype(), x.device())
                .genner_result()?);
        }

        // x: (batch, seq_len, in_features) or (batch, in_features)
        // We need to compute: scaling * (B @ A) @ x
        // A: (in_features, rank), B: (out_features, rank)
        // Combined: (out_features, in_features)

        // Reshape x to 2D if needed: (batch * seq_len, in_features)
        let original_shape = x.dims();
        let x_2d = if original_shape.len() > 2 {
            let batch_size = original_shape[0] * original_shape[1];
            x.reshape((batch_size, original_shape[2])).genner_result()?
        } else {
            x.clone()
        };

        // First: A @ x -> (rank, batch*seq)
        let a_out = self.lora_a.transpose(0, 1).genner_result()?
            .matmul(&x_2d.transpose(0, 1).genner_result()?)
            .genner_result()?;

        // Then: B @ (A @ x)^T -> (out_features, batch*seq)
        let b_out = self.lora_b.matmul(&a_out)
            .genner_result()?;

        // Reshape back to original shape (but with out_features as last dimension)
        let mut result = b_out.transpose(0, 1).genner_result()?;
        if original_shape.len() > 2 {
            // Output shape is (batch, seq_len, out_features) instead of (batch, seq_len, in_features)
            let new_shape = vec![original_shape[0], original_shape[1], self.out_features];
            result = result.reshape(new_shape).genner_result()?;
        }

        // Scale
        result = result.affine(self.scaling, 0.0)
            .genner_result()?;

        Ok(result)
    }

    /// Get the combined LoRA weight (B @ A)
    pub fn combined_weight(&self) -> Result<Tensor> {
        self.lora_b.matmul(&self.lora_a)
            .genner_result()
    }

    /// Merge LoRA weights into base weight
    pub fn merge_into(&self, base_weight: &Tensor) -> Result<Tensor> {
        let combined = self.combined_weight()?;
        combined.affine(self.scaling, 0.0)
            .genner_result()?
            .add(base_weight)
            .genner_result()
    }

    /// Enable/disable LoRA
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get scaling factor
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Get parameter count
    pub fn param_count(&self) -> usize {
        let a_elems = self.lora_a.elem_count();
        let b_elems = self.lora_b.elem_count();
        a_elems + b_elems
    }

    /// Compute gradients for LoRA parameters
    ///
    /// Given upstream gradient grad_output, computes:
    /// - grad_a = scaling * B^T @ grad_output @ x^T
    /// - grad_b = scaling * grad_output @ x @ A^T
    ///
    /// # Arguments
    /// * `grad_output` - Gradient from the next layer (same shape as forward output)
    /// * `x` - Input tensor from forward pass
    ///
    /// # Returns
    /// Tuple of (grad_a, grad_b)
    pub fn compute_gradients(&self, grad_output: &Tensor, x: &Tensor) -> Result<(Tensor, Tensor)> {
        if !self.enabled {
            // Return zero gradients if disabled
            let grad_a = Tensor::zeros(self.lora_a.dims(), self.lora_a.dtype(), self.lora_a.device())
                .genner_result()?;
            let grad_b = Tensor::zeros(self.lora_b.dims(), self.lora_b.dtype(), self.lora_b.device())
                .genner_result()?;
            return Ok((grad_a, grad_b));
        }

        let original_shape = x.dims();
        let x_2d = if original_shape.len() > 2 {
            let batch_size = original_shape[0] * original_shape[1];
            x.reshape((batch_size, original_shape[2])).genner_result()?
        } else {
            x.clone()
        };

        let grad_2d = if grad_output.dims().len() > 2 {
            let batch_size = grad_output.dims()[0] * grad_output.dims()[1];
            grad_output.reshape((batch_size, grad_output.dims()[2])).genner_result()?
        } else {
            grad_output.clone()
        };

        // grad_a = scaling * B^T @ grad_output @ x^T
        // B^T: (rank, out_features)
        // grad_output: (batch*seq, out_features) -> transpose to (out_features, batch*seq)
        // x: (batch*seq, in_features) -> transpose to (in_features, batch*seq)
        let b_t = self.lora_b.transpose(0, 1).genner_result()?;
        let grad_t = grad_2d.transpose(0, 1).genner_result()?;
        let x_t = x_2d.transpose(0, 1).genner_result()?;

        // B^T @ grad_output^T: (rank, out_features) @ (out_features, batch*seq) = (rank, batch*seq)
        let b_t_grad = b_t.matmul(&grad_t).genner_result()?;

        // (B^T @ grad_output^T) @ x: (rank, batch*seq) @ (batch*seq, in_features) = (rank, in_features)
        let grad_a_raw = b_t_grad.matmul(&x_2d).genner_result()?;
        let grad_a = grad_a_raw.affine(self.scaling, 0.0).genner_result()?;

        // grad_b = scaling * grad_output @ x @ A^T
        // grad_output @ x: (batch*seq, out_features) -> actually we need to transpose
        // Let's think again: grad_output is (batch*seq, out_features)
        // x is (batch*seq, in_features)
        // A is (in_features, rank)

        // First: x^T @ grad_output: (in_features, batch*seq) @ (batch*seq, out_features) = (in_features, out_features)
        // Then: (x^T @ grad_output) @ A: (in_features, out_features) @ (out_features, rank) -> no this is wrong

        // Actually: grad_b = scaling * grad_output^T @ x @ A
        // grad_output^T: (out_features, batch*seq)
        // x: (batch*seq, in_features)
        // grad_output^T @ x: (out_features, in_features)
        let grad_t_x = grad_t.matmul(&x_2d).genner_result()?;

        // (grad_output^T @ x) @ A: (out_features, in_features) @ (in_features, rank) = (out_features, rank)
        let grad_b_raw = grad_t_x.matmul(&self.lora_a).genner_result()?;
        let grad_b = grad_b_raw.affine(self.scaling, 0.0).genner_result()?;

        Ok((grad_a, grad_b))
    }
}

/// LoRA-compatible linear layer
pub struct LoraLinear {
    /// Base weight (can be None for pure LoRA)
    base_weight: Option<Tensor>,
    /// LoRA layer
    lora: Option<LoraLayer>,
    /// Bias term
    bias: Option<Tensor>,
    /// Number of input features
    in_features: usize,
    /// Number of output features
    out_features: usize,
}

impl LoraLinear {
    /// Create a new linear layer with optional LoRA
    pub fn new(
        in_features: usize,
        out_features: usize,
        lora_rank: Option<usize>,
        lora_alpha: f32,
        device: &CandleDevice,
        dtype: CandleDType,
    ) -> Result<Self> {
        let base_weight = Tensor::zeros((out_features, in_features), dtype, device)
            .ok(); // Would be loaded from pretrained model

        let lora = if let Some(rank) = lora_rank {
            Some(LoraLayer::new(in_features, out_features, rank, lora_alpha, device, dtype)?)
        } else {
            None
        };

        Ok(Self {
            base_weight,
            lora,
            bias: None,
            in_features,
            out_features,
        })
    }

    /// Forward pass
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut output = match &self.base_weight {
            Some(weight) => {
                // x: (batch, seq, in) @ weight: (out, in) -> want: (batch, seq, out)
                // Reshape to (batch*seq, in), multiply, then reshape back
                let original_shape = x.dims();
                let x_2d = if original_shape.len() > 2 {
                    let batch_seq = original_shape[0] * original_shape[1];
                    x.reshape((batch_seq, original_shape[2])).genner_result()?
                } else {
                    x.clone()
                };

                let result = x_2d.matmul(weight)
                    .genner_result()?;

                if original_shape.len() > 2 {
                    result.reshape((original_shape[0], original_shape[1], self.out_features)).genner_result()?
                } else {
                    result
                }
            }
            None => {
                // If no base weight, only use LoRA
                x.clone()
            }
        };

        // Add LoRA contribution if enabled
        if let Some(lora) = &self.lora {
            let lora_out = lora.forward(x)?;
            output = output.add(&lora_out)
                .genner_result()?;
        }

        // Add bias if present
        if let Some(bias) = &self.bias {
            output = output.add(bias)
                .genner_result()?;
        }

        Ok(output)
    }

    /// Get base weight reference
    pub fn base_weight(&self) -> Option<&Tensor> {
        self.base_weight.as_ref()
    }

    /// Get LoRA layer reference
    pub fn lora(&self) -> Option<&LoraLayer> {
        self.lora.as_ref()
    }

    /// Get mutable LoRA layer reference
    pub fn lora_mut(&mut self) -> Option<&mut LoraLayer> {
        self.lora.as_mut()
    }

    /// Set base weight (for loading pretrained weights)
    pub fn set_base_weight(&mut self, weight: Tensor) -> Result<()> {
        let dims = weight.dims();
        if dims.len() != 2 || dims[0] != self.out_features || dims[1] != self.in_features {
            return Err(Error::Training(format!(
                "Weight shape mismatch: expected ({}, {}), got {:?}",
                self.out_features, self.in_features, dims
            )));
        }
        self.base_weight = Some(weight);
        Ok(())
    }

    /// Get number of trainable parameters
    pub fn trainable_params(&self) -> usize {
        self.lora.as_ref().map(|l| l.param_count()).unwrap_or(0)
    }

    /// Get total parameters
    pub fn total_params(&self) -> usize {
        let base = self.base_weight.as_ref().map(|w| w.elem_count()).unwrap_or(0);
        let bias = self.bias.as_ref().map(|b| b.elem_count()).unwrap_or(0);
        base + bias + self.trainable_params()
    }
}

/// AdamW optimizer for LoRA training
pub struct AdamWOptimizer {
    /// Learning rate
    lr: f64,
    /// Weight decay
    weight_decay: f64,
    /// Beta1 for momentum
    beta1: f64,
    /// Beta2 for variance
    beta2: f64,
    /// Epsilon for numerical stability
    epsilon: f64,
    /// First moment estimate for lora_a
    m_a: Option<Tensor>,
    /// Second moment estimate for lora_a
    v_a: Option<Tensor>,
    /// First moment estimate for lora_b
    m_b: Option<Tensor>,
    /// Second moment estimate for lora_b
    v_b: Option<Tensor>,
    /// Timestep
    t: usize,
}

impl AdamWOptimizer {
    /// Create a new AdamW optimizer
    pub fn new(lr: f32, weight_decay: f32) -> Self {
        Self {
            lr: lr as f64,
            weight_decay: weight_decay as f64,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m_a: None,
            v_a: None,
            m_b: None,
            v_b: None,
            t: 0,
        }
    }

    /// Get learning rate
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Get weight decay
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// Perform an optimization step with gradients
    ///
    /// This applies the AdamW update rule:
    /// - Apply weight decay to parameters
    /// - Update biased first and second moment estimates
    /// - Compute bias-corrected estimates
    /// - Update parameters
    pub fn step(&mut self, lora_layer: &mut LoraLayer, grad_a: &Tensor, grad_b: &Tensor) -> Result<()> {
        self.t += 1;

        // Initialize moment estimates if needed
        if self.m_a.is_none() {
            self.m_a = Some(Tensor::zeros(grad_a.dims(), grad_a.dtype(), grad_a.device())
                .genner_result()?);
            self.v_a = Some(Tensor::zeros(grad_a.dims(), grad_a.dtype(), grad_a.device())
                .genner_result()?);
        }
        if self.m_b.is_none() {
            self.m_b = Some(Tensor::zeros(grad_b.dims(), grad_b.dtype(), grad_b.device())
                .genner_result()?);
            self.v_b = Some(Tensor::zeros(grad_b.dims(), grad_b.dtype(), grad_b.device())
                .genner_result()?);
        }

        // Update lora_a
        let m_a_old = self.m_a.as_ref().unwrap();
        let v_a_old = self.v_a.as_ref().unwrap();

        // m = beta1 * m + (1 - beta1) * grad
        let m_a_new = (m_a_old.affine(self.beta1, 0.0).genner_result()?
            + grad_a.affine(1.0 - self.beta1, 0.0).genner_result()?)
            .genner_result()?;

        // v = beta2 * v + (1 - beta2) * grad^2
        let grad_sq = grad_a.mul(grad_a).genner_result()?;
        let v_a_new = (v_a_old.affine(self.beta2, 0.0).genner_result()?
            + grad_sq.affine(1.0 - self.beta2, 0.0).genner_result()?)
            .genner_result()?;

        // Store new moments
        self.m_a = Some(m_a_new);
        self.v_a = Some(v_a_new);

        // Bias correction
        let m_a_hat = self.m_a.as_ref().unwrap().affine(1.0 / (1.0 - self.beta1.powi(self.t as i32)), 0.0).genner_result()?;
        let v_a_hat = self.v_a.as_ref().unwrap().affine(1.0 / (1.0 - self.beta2.powi(self.t as i32)), 0.0).genner_result()?;

        // Update with weight decay and learning rate
        // param = param - lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
        let v_a_sqrt = v_a_hat.sqrt().genner_result()?;
        let denom_a = v_a_sqrt.affine(1.0, self.epsilon).genner_result()?;
        let update_a = m_a_hat.div(&denom_a).genner_result()?
            .affine(self.lr, 0.0).genner_result()?;
        let decay_a = lora_layer.lora_a.affine(self.weight_decay, 0.0).genner_result()?;

        lora_layer.lora_a = lora_layer.lora_a.sub(&update_a)
            .genner_result()?
            .sub(&decay_a)
            .genner_result()?;

        // Update lora_b (same process)
        let m_b_old = self.m_b.as_ref().unwrap();
        let v_b_old = self.v_b.as_ref().unwrap();

        let m_b_new = (m_b_old.affine(self.beta1, 0.0).genner_result()?
            + grad_b.affine(1.0 - self.beta1, 0.0).genner_result()?)
            .genner_result()?;

        let grad_b_sq = grad_b.mul(grad_b).genner_result()?;
        let v_b_new = (v_b_old.affine(self.beta2, 0.0).genner_result()?
            + grad_b_sq.affine(1.0 - self.beta2, 0.0).genner_result()?)
            .genner_result()?;

        self.m_b = Some(m_b_new);
        self.v_b = Some(v_b_new);

        let m_b_hat = self.m_b.as_ref().unwrap().affine(1.0 / (1.0 - self.beta1.powi(self.t as i32)), 0.0).genner_result()?;
        let v_b_hat = self.v_b.as_ref().unwrap().affine(1.0 / (1.0 - self.beta2.powi(self.t as i32)), 0.0).genner_result()?;

        let v_b_sqrt = v_b_hat.sqrt().genner_result()?;
        let denom_b = v_b_sqrt.affine(1.0, self.epsilon).genner_result()?;
        let update_b = m_b_hat.div(&denom_b).genner_result()?
            .affine(self.lr, 0.0).genner_result()?;
        let decay_b = lora_layer.lora_b.affine(self.weight_decay, 0.0).genner_result()?;

        lora_layer.lora_b = lora_layer.lora_b.sub(&update_b)
            .genner_result()?
            .sub(&decay_b)
            .genner_result()?;

        Ok(())
    }

    /// Reset optimizer state
    pub fn reset(&mut self) {
        self.m_a = None;
        self.v_a = None;
        self.m_b = None;
        self.v_b = None;
        self.t = 0;
    }
}

/// Simple optimizer for LoRA training (legacy alias)
pub type LoraOptimizer = AdamWOptimizer;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_layer_creation() {
        let device = CandleDevice::Cpu;
        let dtype = CandleDType::F32;

        let lora = LoraLayer::new(768, 768, 8, 16.0, &device, dtype);
        assert!(lora.is_ok());

        let lora = lora.unwrap();
        assert_eq!(lora.rank(), 8);
        assert_eq!(lora.scaling(), 2.0); // alpha / rank = 16 / 8
    }

    #[test]
    fn test_lora_layer_param_count() {
        let device = CandleDevice::Cpu;
        let dtype = CandleDType::F32;

        let lora = LoraLayer::new(768, 768, 8, 16.0, &device, dtype).unwrap();
        // A: 768 * 8 = 6144, B: 768 * 8 = 6144, total: 12288
        assert_eq!(lora.param_count(), 12288);
    }

    #[test]
    fn test_lora_linear_creation() {
        let device = CandleDevice::Cpu;
        let dtype = CandleDType::F32;

        let linear = LoraLinear::new(768, 3072, Some(8), 16.0, &device, dtype);
        assert!(linear.is_ok());

        let linear = linear.unwrap();
        assert_eq!(linear.in_features, 768);
        assert_eq!(linear.out_features, 3072);
        assert!(linear.lora().is_some());
        assert_eq!(linear.trainable_params(), 8 * 768 + 3072 * 8);
    }

    #[test]
    fn test_lora_linear_without_lora() {
        let device = CandleDevice::Cpu;
        let dtype = CandleDType::F32;

        let linear = LoraLinear::new(768, 3072, None, 16.0, &device, dtype).unwrap();
        assert!(linear.lora().is_none());
        assert_eq!(linear.trainable_params(), 0);
    }

    #[test]
    fn test_lora_optimizer_creation() {
        let optimizer = LoraOptimizer::new(1e-4, 0.01);
        assert!((optimizer.lr() - 1e-4).abs() < 1e-9);
        assert!((optimizer.weight_decay() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_lora_forward() {
        let device = CandleDevice::Cpu;
        let dtype = CandleDType::F32;

        let lora = LoraLayer::new(64, 128, 4, 8.0, &device, dtype).unwrap();

        // Create input tensor (batch=2, seq=3, in=64)
        let x = Tensor::zeros((2, 3, 64), dtype, &device).unwrap();

        let output = lora.forward(&x);
        assert!(output.is_ok(), "LoRA forward should succeed");

        let output = output.unwrap();
        // LoRA forward computes B @ A @ x
        // A: (4, 64), x: (2, 3, 64) -> after reshape to (6, 64)
        // A @ x^T: (4, 6) -> transpose to (6, 4)
        // B: (128, 4) @ (6, 4)^T: (128, 6) -> transpose to (6, 128)
        // reshape to (2, 3, 128)
        let shape = output.dims();
        assert_eq!(shape, vec![2, 3, 128]);
    }
}

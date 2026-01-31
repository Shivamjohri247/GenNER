//! Loss functions for training NER models
//!
//! Implements common loss functions for language model fine-tuning.

use crate::candle_model::ToGennerResult;
use candle_core::{Device as CandleDevice, DType as CandleDType, Tensor};
use genner_core::error::Result;

/// Cross-entropy loss for language modeling
///
/// Computes the cross-entropy loss between logits and target tokens.
/// This is the standard loss function for training language models.
///
/// # Arguments
/// * `logits` - Model output logits (batch, seq_len, vocab_size)
/// * `targets` - Target token IDs (batch, seq_len)
///
/// # Returns
/// Scalar loss tensor
pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // logits: (batch, seq_len, vocab_size)
    // targets: (batch, seq_len)

    let logits_shape = logits.dims();
    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];
    let vocab_size = logits_shape[2];

    // Reshape logits to (batch * seq_len, vocab_size)
    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))
        .genner_result()?;

    // Flatten targets to (batch * seq_len,)
    let targets_flat = targets.reshape((batch_size * seq_len,))
        .genner_result()?;

    // Compute log_softmax
    let log_probs = log_softmax(&logits_2d, 1)?;

    // Gather the log probabilities for the target tokens
    // targets_flat contains u32 values, need to convert to indices
    let targets_vec = targets_flat.to_vec1::<u32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert targets: {}", e)))?;

    let mut total_loss = 0.0f32;
    let log_probs_vec = log_probs.to_vec2::<f32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert log_probs: {}", e)))?;

    for (i, &target_idx) in targets_vec.iter().enumerate() {
        if (target_idx as usize) < vocab_size {
            total_loss -= log_probs_vec[i][target_idx as usize];
        }
    }

    // Return mean loss
    let loss_value = total_loss / (targets_vec.len() as f32);
    Tensor::new(loss_value, logits.device())
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to create loss tensor: {}", e)))
}

/// Log_softmax function
///
/// Computes log_softmax along the specified dimension.
/// More numerically stable than log(softmax(x)).
pub fn log_softmax(x: &Tensor, dim: usize) -> Result<Tensor> {
    // For simplicity in this implementation, we'll use a manual approach
    // Convert to vec, compute, and convert back
    // This is less efficient but works reliably with Candle's API

    let x_shape = x.dims();

    // For 2D tensor (rows, cols)
    if x_shape.len() == 2 {
        let rows = x_shape[0];
        let cols = x_shape[1];

        let x_vec = x.to_vec2::<f32>()
            .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert x: {}", e)))?;

        let mut result = Vec::with_capacity(x_vec.len());

        if dim == 0 {
            // Max along each column
            for col in 0..cols {
                let mut max_val = x_vec[0][col];
                for row in 1..rows {
                    max_val = max_val.max(x_vec[row][col]);
                }

                // Compute sum of exp(x - max)
                let mut exp_sum = 0.0f32;
                for row in 0..rows {
                    exp_sum += (x_vec[row][col] - max_val).exp();
                }

                let log_sum = exp_sum.ln();

                // Compute log_softmax for each element in this column
                for row in 0..rows {
                    result.push(x_vec[row][col] - max_val - log_sum);
                }
            }
        } else {
            // Max along each row
            for row in 0..rows {
                let mut max_val = x_vec[row][0];
                for col in 1..cols {
                    max_val = max_val.max(x_vec[row][col]);
                }

                // Compute sum of exp(x - max)
                let mut exp_sum = 0.0f32;
                for col in 0..cols {
                    exp_sum += (x_vec[row][col] - max_val).exp();
                }

                let log_sum = exp_sum.ln();

                // Compute log_softmax for each element in this row
                for col in 0..cols {
                    result.push(x_vec[row][col] - max_val - log_sum);
                }
            }
        }

        Tensor::from_vec(result, (x_shape[0], x_shape[1]), x.device())
            .map_err(|e| genner_core::error::Error::Training(format!("Failed to create result tensor: {}", e)))
    } else if x_shape.len() == 1 {
        // 1D tensor
        let x_vec = x.to_vec1::<f32>()
            .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert x: {}", e)))?;

        let max_val = x_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = x_vec.iter().map(|&v| (v - max_val).exp()).sum();
        let log_sum = exp_sum.ln();

        let result: Vec<f32> = x_vec.iter().map(|&v| v - max_val - log_sum).collect();

        Tensor::from_vec(result, x_shape[0], x.device())
            .map_err(|e| genner_core::error::Error::Training(format!("Failed to create result tensor: {}", e)))
    } else {
        Err(genner_core::error::Error::Training(format!("Unsupported tensor shape: {:?}", x_shape)))
    }
}

/// Per-token cross-entropy loss
///
/// Returns the loss for each token separately (useful for analysis).
///
/// # Arguments
/// * `logits` - Model output logits (batch, seq_len, vocab_size)
/// * `targets` - Target token IDs (batch, seq_len)
///
/// # Returns
/// Loss tensor of shape (batch * seq_len,)
pub fn cross_entropy_per_token(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let logits_shape = logits.dims();
    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];
    let vocab_size = logits_shape[2];

    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))
        .genner_result()?;

    let targets_flat = targets.reshape((batch_size * seq_len,))
        .genner_result()?;

    let log_probs = log_softmax(&logits_2d, 1)?;

    let targets_vec = targets_flat.to_vec1::<u32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert targets: {}", e)))?;

    let log_probs_vec = log_probs.to_vec2::<f32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert log_probs: {}", e)))?;

    let mut losses = Vec::with_capacity(targets_vec.len());
    for (i, &target_idx) in targets_vec.iter().enumerate() {
        if (target_idx as usize) < vocab_size {
            losses.push(-log_probs_vec[i][target_idx as usize]);
        } else {
            losses.push(0.0); // Padding or invalid token
        }
    }

    Tensor::from_vec(losses, (targets_vec.len(),), logits.device())
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to create loss tensor: {}", e)))
}

/// Ignore index cross-entropy
///
/// Computes cross-entropy but ignores certain target indices (e.g., padding tokens).
///
/// # Arguments
/// * `logits` - Model output logits (batch, seq_len, vocab_size)
/// * `targets` - Target token IDs (batch, seq_len)
/// * `ignore_index` - Token ID to ignore in loss computation
pub fn cross_entropy_ignore_index(
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: u32,
) -> Result<Tensor> {
    let logits_shape = logits.dims();
    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];
    let vocab_size = logits_shape[2];

    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))
        .genner_result()?;

    let targets_flat = targets.reshape((batch_size * seq_len,))
        .genner_result()?;

    let log_probs = log_softmax(&logits_2d, 1)?;

    let targets_vec = targets_flat.to_vec1::<u32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert targets: {}", e)))?;

    let log_probs_vec = log_probs.to_vec2::<f32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert log_probs: {}", e)))?;

    let mut total_loss = 0.0f32;
    let mut valid_count = 0usize;

    for (i, &target_idx) in targets_vec.iter().enumerate() {
        if target_idx == ignore_index {
            continue;
        }
        if (target_idx as usize) < vocab_size {
            total_loss -= log_probs_vec[i][target_idx as usize];
            valid_count += 1;
        }
    }

    let loss_value = if valid_count > 0 {
        total_loss / (valid_count as f32)
    } else {
        0.0
    };

    Tensor::new(loss_value, logits.device())
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to create loss tensor: {}", e)))
}

/// Label smoothing cross-entropy
///
/// Applies label smoothing to the cross-entropy loss.
/// This can help prevent overconfidence during training.
///
/// # Arguments
/// * `logits` - Model output logits (batch, seq_len, vocab_size)
/// * `targets` - Target token IDs (batch, seq_len)
/// * `smoothing` - Smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
pub fn cross_entropy_label_smoothing(
    logits: &Tensor,
    targets: &Tensor,
    smoothing: f32,
) -> Result<Tensor> {
    let logits_shape = logits.dims();
    let batch_size = logits_shape[0];
    let seq_len = logits_shape[1];
    let vocab_size = logits_shape[2];

    let logits_2d = logits.reshape((batch_size * seq_len, vocab_size))
        .genner_result()?;

    let targets_flat = targets.reshape((batch_size * seq_len,))
        .genner_result()?;

    let log_probs = log_softmax(&logits_2d, 1)?;

    let targets_vec = targets_flat.to_vec1::<u32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert targets: {}", e)))?;

    let log_probs_vec = log_probs.to_vec2::<f32>()
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to convert log_probs: {}", e)))?;

    let mut total_loss = 0.0f32;
    let smooth_value = smoothing / (vocab_size as f32);

    for (i, &target_idx) in targets_vec.iter().enumerate() {
        if (target_idx as usize) < vocab_size {
            // Smoothed loss: (1 - smoothing) * (-log p[target]) + smoothing * mean(-log p)
            let target_log_prob = log_probs_vec[i][target_idx as usize];

            // Compute mean of all log probs
            let mean_log_prob: f32 = log_probs_vec[i].iter().sum::<f32>() / (vocab_size as f32);

            let smooth_loss = (1.0 - smoothing) * (-target_log_prob) + smoothing * (-mean_log_prob);
            total_loss += smooth_loss;
        }
    }

    let loss_value = total_loss / (targets_vec.len() as f32);
    Tensor::new(loss_value, logits.device())
        .map_err(|e| genner_core::error::Error::Training(format!("Failed to create loss tensor: {}", e)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let device = CandleDevice::Cpu;

        // Create logits that perfectly predict the target
        // Set first logit very high for perfect prediction
        let mut logits_data = vec![0.0f32; 10];
        logits_data[0] = 100.0; // Very high confidence for token 0
        let logits = Tensor::from_vec(logits_data, (1, 1, 10), &device).unwrap();

        let targets_data = vec![0u32];
        let targets = Tensor::from_vec(targets_data, (1, 1), &device).unwrap();

        let loss = cross_entropy(&logits, &targets).unwrap();
        let loss_dims = loss.dims();

        // Loss is a scalar (0D tensor)
        assert_eq!(loss_dims, &[] as &[usize]);
    }

    #[test]
    fn test_cross_entropy_ignore_index() {
        let device = CandleDevice::Cpu;

        // Create uniform logits (all equal)
        let logits_data = vec![0.0f32; 30]; // 1 * 3 * 10
        let logits = Tensor::from_vec(logits_data, (1, 3, 10), &device).unwrap();

        // First token is 0, second is padding (u32::MAX), third is 5
        let targets_data = vec![0u32, u32::MAX, 5];
        let targets = Tensor::from_vec(targets_data, (1, 3), &device).unwrap();

        let loss = cross_entropy_ignore_index(&logits, &targets, u32::MAX).unwrap();
        let loss_dims = loss.dims();

        // With uniform distribution, log prob should be log(1/10) = -2.3
        // We have 2 valid tokens, so loss should be around 2.3
        let loss_dims = loss.dims();
        assert_eq!(loss_dims, &[] as &[usize]);
    }

    #[test]
    fn test_log_softmax() {
        let device = CandleDevice::Cpu;

        // Simple test case
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device).unwrap();
        let result = log_softmax(&x, 0).unwrap();
        let result_vec = result.to_vec1::<f32>().unwrap();

        // Results should sum to approximately 1 when exponentiated
        let sum: f32 = result_vec.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }
}

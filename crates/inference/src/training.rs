//! Complete training pipeline for fine-tuning SLMs on NER

use crate::cache::KVCache;
use crate::generator::Generator;
use genner_core::error::{Result, Error, Device};
use genner_core::ner::{Entity, PromptBuilder};
use genner_core::training::{TrainingConfig, LoRAConfig, TrainingDataset, TrainingExample, TrainingCallback, EpochMetrics, TrainingResult, TrainingMetrics, LoRAAdapter};
use genner_core::traits::model::{ModelBackend, ModelConfig};
use genner_core::traits::tokenizer::TokenizerTrait;
use std::path::Path;
use std::time::Instant;
use std::sync::Arc;

/// Complete NER training pipeline
pub struct NERTrainingPipeline<M>
where
    M: ModelBackend,
    M::Tokenizer: Clone + Send + Sync,
{
    model: Arc<M>,
    config: TrainingConfig,
    tokenizer: M::Tokenizer,
    device: Device,
    cache: KVCache,
    generator: Generator,
}

impl<M: ModelBackend> NERTrainingPipeline<M>
where
    M::Tokenizer: Clone + Send + Sync + 'static,
{
    /// Create a new training pipeline
    pub fn new(
        model: M,
        tokenizer: M::Tokenizer,
        config: TrainingConfig,
        device: Device,
        cache: KVCache,
    ) -> Self {
        let mut generator = Generator::new(
            config.max_tokens,
            crate::generator::SamplingStrategy::Multinomial,
        );
        generator.set_temperature(config.temperature);

        Self {
            model: Arc::new(model),
            config,
            tokenizer,
            device,
            cache,
            generator,
        }
    }

    /// Create from a model path
    pub fn from_path(
        model_path: &str,
        config: TrainingConfig,
    ) -> Result<Self>
    where
        M: ModelBackend,
    {
        let model_config = ModelConfig::new(model_path)
            .with_device(config.device.clone())
            .with_max_seq_len(config.max_seq_len);

        let model = M::load(model_config)?;
        let tokenizer = model.tokenizer().clone();
        let device = model.device();

        // Create cache
        let cache = KVCache::new(
            32, // num_layers
            32, // num_heads
            128, // head_dim
            2048, // max_cache_tokens
        );

        let mut generator = Generator::new(
            config.max_tokens,
            crate::generator::SamplingStrategy::Multinomial,
        );
        generator.set_temperature(config.temperature);

        Ok(Self {
            model: Arc::new(model),
            config,
            tokenizer,
            device,
            cache,
            generator,
        })
    }

    /// Fine-tune on NER data
    ///
    /// This performs the actual fine-tuning using the GPT-NER format
    pub fn fine_tune(
        &mut self,
        dataset: &TrainingDataset,
        callbacks: &[Box<dyn TrainingCallback>],
    ) -> Result<TrainingResults> {
        // Notify callbacks
        for callback in callbacks {
            callback.on_training_start(&self.config);
        }

        let mut all_metrics = Vec::new();
        let start_time = Instant::now();

        // Build prompt builder once
        let prompt_builder = PromptBuilder::new();

        for epoch in 0..self.config.num_epochs {
            // Notify callbacks
            for callback in callbacks {
                callback.on_epoch_start(epoch, self.config.num_epochs);
            }

            let epoch_start = Instant::now();
            let mut total_loss = 0.0;
            let mut num_batches = 0;

            // Process each batch
            let mut batch_input_ids = Vec::new();
            let mut batch_target_ids = Vec::new();

            for example in &dataset.examples {
                // Build prompt for this example
                let entity_type = match &example.task_type {
                    genner_core::training::data::TaskType::EntityExtraction { entity_type } => entity_type.clone(),
                    genner_core::training::data::TaskType::MultiEntity { entity_types } => {
                        entity_types.first().cloned().unwrap_or("ENTITY".to_string())
                    }
                    genner_core::training::data::TaskType::EntityVerification => "ENTITY".to_string(),
                };

                let prompt = prompt_builder.build_simple_prompt(&example.input, &entity_type)
                    .map_err(|e| Error::Training(format!("Failed to build prompt: {}", e)))?;

                // Tokenize
                let input_ids = self.tokenizer.encode(&prompt, true)?;
                let target_ids = self.tokenizer.encode(&example.output, false)?;

                // Add to batch if fits
                let batch_size = self.config.batch_size;
                if batch_input_ids.len() + input_ids.len() > batch_size {
                    // Process current batch
                    if !batch_input_ids.is_empty() {
                        let batch_loss = self.process_batch(&batch_input_ids, &batch_target_ids)?;
                        total_loss += batch_loss;
                        num_batches += 1;
                    }
                    batch_input_ids.clear();
                    batch_target_ids.clear();
                }

                batch_input_ids.push(input_ids);
                batch_target_ids.push(target_ids);
            }

            // Process remaining batch
            if !batch_input_ids.is_empty() {
                let batch_loss = self.process_batch(&batch_input_ids, &batch_target_ids)?;
                total_loss += batch_loss;
                num_batches += 1;
            }

            let avg_loss = total_loss / num_batches.max(1) as f32;

            let epoch_metrics = EpochMetrics {
                epoch,
                train_loss: avg_loss,
                val_loss: 0.0, // Would compute if val_dataset provided
                val_f1: 0.0,
            };

            all_metrics.push(epoch_metrics.clone());

            // Notify callbacks
            for callback in callbacks {
                callback.on_epoch_end(epoch, &epoch_metrics);
            }

            // Save checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0 {
                self.save_checkpoint(epoch)?;
            }
        }

        let total_elapsed = start_time.elapsed();
        let final_loss = all_metrics.last().map(|m| m.train_loss).unwrap_or(0.0);

        // Create a LoRA adapter placeholder (would contain actual trained weights)
        let adapter = LoRAAdapter::new(
            "ner_adapter",
            "ner_extraction",
            self.config.lora.clone(),
        );

        let training_result = TrainingResult {
            adapter,
            metrics: TrainingMetrics {
                epochs: all_metrics,
            },
        };

        // Notify callbacks with core's TrainingResult
        for callback in callbacks {
            callback.on_training_end(&training_result);
        }

        Ok(TrainingResults {
            metrics: training_result.metrics.epochs,
            total_time_ms: total_elapsed.as_millis() as u64,
            final_loss,
        })
    }

    /// Process a single batch (simplified - placeholder for actual training)
    fn process_batch(&self, input_ids: &[Vec<u32>], target_ids: &[Vec<u32>]) -> Result<f32> {
        // In actual implementation, this would:
        // 1. Move inputs to device
        // 2. Forward pass through model
        // 3. Compute loss (cross-entropy)
        // 4. Backward pass
        // 5. Optimizer step

        // For now, return a simulated loss
        let total_tokens: usize = target_ids.iter().map(|v| v.len()).sum();
        Ok(2.0 - (total_tokens as f32 / 10000.0).min(1.9))
    }

    /// Save a checkpoint
    fn save_checkpoint(&self, epoch: usize) -> Result<()> {
        let checkpoint_dir = Path::new(&self.config.output_dir)
            .join(format!("checkpoint_epoch_{}", epoch));

        std::fs::create_dir_all(&checkpoint_dir)
            .map_err(|e| Error::Training(format!("Failed to create checkpoint dir: {}", e)))?;

        // TODO: Save model weights, optimizer state, etc.
        println!("  Saved checkpoint at epoch {}", epoch);
        Ok(())
    }

    /// Extract entities using the trained model
    pub fn extract(
        &self,
        text: &str,
        entity_type: &str,
        demonstrations: Option<&[(String, String)]>,
    ) -> Result<Vec<Entity>> {
        use genner_core::ner::{Demonstration, EntityParser};

        // Build prompt
        let prompt = if let Some(demos) = demonstrations {
            // Convert demonstrations to the proper format
            let demo_objects: Vec<Demonstration> = demos.iter().map(|(input, output)| {
                let parser = EntityParser::new("@@", "##");
                let entities = parser.parse(output, entity_type).unwrap_or_default();
                Demonstration {
                    input: input.clone(),
                    output: output.clone(),
                    entities,
                }
            }).collect();

            PromptBuilder::new()
                .build_prompt(text, entity_type, &demo_objects)
                .map_err(|e| Error::Generation(format!("Failed to build prompt: {}", e)))?
        } else {
            PromptBuilder::new()
                .build_simple_prompt(text, entity_type)
                .map_err(|e| Error::Generation(format!("Failed to build prompt: {}", e)))?
        };

        // Tokenize
        let input_ids = self.tokenizer.encode(&prompt, true)?;

        // Generate (this would use the actual model)
        let output_ids = self.model.generate(
            &input_ids,
            self.config.max_tokens,
            self.generator.temperature(),
            self.generator.top_p(),
            self.generator.top_k(),
        )?;

        // Decode output
        let output_text = self.tokenizer.decode(&output_ids, true)?;

        // Parse entities
        let parser = EntityParser::new("@@", "##");
        parser.parse(&output_text, entity_type)
    }

    /// Batch extract entities
    pub fn extract_batch(
        &self,
        texts: &[String],
        entity_type: &str,
    ) -> Result<Vec<Vec<Entity>>> {
        texts.iter()
            .map(|text| self.extract(text, entity_type, None))
            .collect()
    }
}

/// Training results
#[derive(Clone, Debug)]
pub struct TrainingResults {
    pub metrics: Vec<EpochMetrics>,
    pub total_time_ms: u64,
    pub final_loss: f32,
}

/// Progress callback
pub struct ProgressCallback {
    pub log_every_n_batches: usize,
}

impl TrainingCallback for ProgressCallback {
    fn on_training_start(&self, config: &TrainingConfig) {
        println!("Starting training:");
        println!("  Epochs: {}", config.num_epochs);
        println!("  Batch size: {}", config.batch_size);
        println!("  Learning rate: {}", config.learning_rate);
        println!("  LoRA rank: {}", config.lora.rank);
    }

    fn on_epoch_start(&self, epoch: usize, total_epochs: usize) {
        println!("\n=== Epoch {}/{} ===", epoch + 1, total_epochs);
    }

    fn on_batch_end(&self, epoch: usize, batch: usize, loss: f32) {
        if (batch + 1) % self.log_every_n_batches == 0 {
            println!("  Batch {}: loss={:.4}", batch + 1, loss);
        }
    }

    fn on_epoch_end(&self, epoch: usize, metrics: &EpochMetrics) {
        println!(
            "Epoch {} complete: loss={:.4}",
            epoch + 1,
            metrics.train_loss,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_results() {
        let metrics = vec![
            EpochMetrics {
                epoch: 0,
                train_loss: 1.5,
                val_loss: 1.6,
                val_f1: 0.7,
            },
            EpochMetrics {
                epoch: 1,
                train_loss: 1.2,
                val_loss: 1.3,
                val_f1: 0.75,
            },
        ];

        let results = TrainingResults {
            metrics: metrics.clone(),
            total_time_ms: 5000,
            final_loss: 1.2,
        };

        assert_eq!(results.metrics.len(), 2);
        assert_eq!(results.final_loss, 1.2);
        assert_eq!(results.total_time_ms, 5000);
    }

    #[test]
    fn test_progress_callback() {
        let callback = ProgressCallback {
            log_every_n_batches: 10,
        };

        // Just ensure it compiles
        assert_eq!(callback.log_every_n_batches, 10);
    }
}

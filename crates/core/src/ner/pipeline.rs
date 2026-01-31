//! NER pipeline orchestrator

use crate::error::{Device, DType, Result};
use crate::ner::entity::NERTask;
use crate::ner::prompt::Demonstration;
use crate::ner::{
    Entity, EntityParser, PipelineConfig,
    PromptBuilder, SelfVerifier,
};
use crate::traits::model::{GenerationOptions, ModelBackend};
use crate::traits::tokenizer::TokenizerTrait;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Main NER pipeline
pub struct NERPipeline<M: ModelBackend> {
    model: M,
    config: PipelineConfig,
    prompt_builder: PromptBuilder,
    entity_parser: EntityParser,
    verifier: SelfVerifier,
    demonstration_pool: HashMap<String, Vec<Demonstration>>,
}

impl<M: ModelBackend> NERPipeline<M> {
    /// Create a new NER pipeline
    pub fn new(model: M) -> Self {
        let config = PipelineConfig::default();
        Self::with_config(model, config)
    }

    /// Create with custom config
    pub fn with_config(model: M, config: PipelineConfig) -> Self {
        let prompt_builder = PromptBuilder::new()
            .with_markers(&config.entity_prefix, &config.entity_suffix)
            .with_demonstrations(config.num_demonstrations);

        let entity_parser = EntityParser::new(&config.entity_prefix, &config.entity_suffix);

        Self {
            model,
            config,
            prompt_builder,
            entity_parser,
            verifier: SelfVerifier::new(),
            demonstration_pool: HashMap::new(),
        }
    }

    /// Add demonstrations to the pool for an entity type
    pub fn add_demonstrations(&mut self, entity_type: &str, demonstrations: Vec<Demonstration>) {
        self.demonstration_pool
            .insert(entity_type.to_string(), demonstrations);
    }

    /// Retrieve demonstrations for an entity type
    fn retrieve_demonstrations(&self, entity_type: &str) -> Vec<Demonstration> {
        match self.config.retrieval_strategy {
            crate::ner::RetrievalStrategy::Random => {
                if let Some(demos) = self.demonstration_pool.get(entity_type) {
                    let mut rng = rand::thread_rng();
                    let mut indices: Vec<usize> = (0..demos.len()).collect();
                    indices.shuffle(&mut rng);
                    indices
                        .into_iter()
                        .take(self.config.num_demonstrations)
                        .filter_map(|i| demos.get(i).cloned())
                        .collect()
                } else {
                    Vec::new()
                }
            }
            _ => {
                // For other strategies, just return available demos
                // (kNN would be implemented with EmbeddingStore)
                self.demonstration_pool
                    .get(entity_type)
                    .map(|demos| {
                        demos
                            .iter()
                            .take(self.config.num_demonstrations)
                            .cloned()
                            .collect()
                    })
                    .unwrap_or_default()
            }
        }
    }

    /// Extract entities from text for a single entity type
    pub fn extract(&self, text: &str, entity_type: &str) -> Result<ExtractionResult> {
        let start_time = std::time::Instant::now();

        // 1. Retrieve demonstrations
        let demonstrations = self.retrieve_demonstrations(entity_type);

        // 2. Build prompt
        let prompt = self
            .prompt_builder
            .build_prompt(text, entity_type, &demonstrations)?;

        // 3. Generate marked output
        let input_ids = self.model.tokenizer().encode(&prompt, true)?;
        let options = GenerationOptions::new()
            .with_max_tokens(self.config.max_seq_len)
            .with_temperature(0.0)
            .with_stop_sequences(vec![self.config.entity_suffix.clone()]);

        let generation_start = std::time::Instant::now();
        let output_ids = self.model.generate_with_options(&input_ids, &options)?;
        let generation_time = generation_start.elapsed();

        let raw_output = self.model.tokenizer().decode(&output_ids, true)?;

        // 4. Parse entities from marked text
        let entities = self.entity_parser.parse(&raw_output, entity_type)?;

        // 5. Verify entities (if enabled)
        let verified_entities = if self.config.verification_enabled {
            let verification_start = std::time::Instant::now();
            let verified = self
                .verifier
                .filter_entities(&self.model, text, &entities, self.config.verification_threshold)?;
            let verification_time = verification_start.elapsed();

            let mut metadata = ExtractionMetadata::default();
            metadata.inference_time_ms = generation_time.as_millis() as u64;
            metadata.verification_time_ms = verification_time.as_millis() as u64;
            metadata.total_time_ms = start_time.elapsed().as_millis() as u64;

            ExtractionResult {
                text: text.to_string(),
                raw_output: raw_output.clone(),
                entities: entities.clone(),
                verified_entities: verified,
                metadata,
            }
        } else {
            let mut metadata = ExtractionMetadata::default();
            metadata.inference_time_ms = generation_time.as_millis() as u64;
            metadata.total_time_ms = start_time.elapsed().as_millis() as u64;

            ExtractionResult {
                text: text.to_string(),
                raw_output: raw_output.clone(),
                entities: entities.clone(),
                verified_entities: entities.clone(),
                metadata,
            }
        };

        Ok(verified_entities)
    }

    /// Extract all entity types
    pub fn extract_all(&self, text: &str, entity_types: &[String]) -> Result<Vec<Entity>> {
        let mut all_entities = Vec::new();
        for entity_type in entity_types {
            let result = self.extract(text, entity_type)?;
            all_entities.extend(result.verified_entities);
        }
        Ok(all_entities)
    }

    /// Extract with custom demonstrations
    pub fn extract_with_demonstrations(
        &self,
        text: &str,
        entity_type: &str,
        demonstrations: &[Demonstration],
    ) -> Result<ExtractionResult> {
        let start_time = std::time::Instant::now();

        // Build prompt
        let prompt = self
            .prompt_builder
            .build_prompt(text, entity_type, demonstrations)?;

        // Generate
        let input_ids = self.model.tokenizer().encode(&prompt, true)?;
        let options = GenerationOptions::new()
            .with_max_tokens(self.config.max_seq_len)
            .with_temperature(0.0)
            .with_stop_sequences(vec![self.config.entity_suffix.clone()]);

        let output_ids = self.model.generate_with_options(&input_ids, &options)?;
        let raw_output = self.model.tokenizer().decode(&output_ids, true)?;

        // Parse
        let entities = self.entity_parser.parse(&raw_output, entity_type)?;

        Ok(ExtractionResult {
            text: text.to_string(),
            raw_output,
            entities: entities.clone(),
            verified_entities: entities,
            metadata: ExtractionMetadata {
                total_time_ms: start_time.elapsed().as_millis() as u64,
                ..Default::default()
            },
        })
    }

    /// Batch extraction
    pub fn extract_batch(&self, texts: &[String], entity_type: &str) -> Result<Vec<ExtractionResult>> {
        texts
            .iter()
            .map(|text| self.extract(text, entity_type))
            .collect()
    }

    /// Get reference to the model
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Get mutable reference to the model
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Get the config
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }

    /// Update config
    pub fn set_config(&mut self, config: PipelineConfig) {
        self.config = config;
    }
}

/// Extraction result
#[derive(Clone, Debug)]
pub struct ExtractionResult {
    /// Original input text
    pub text: String,

    /// Raw model output (with markers)
    pub raw_output: String,

    /// Parsed entities (before verification)
    pub entities: Vec<Entity>,

    /// Verified entities (after verification)
    pub verified_entities: Vec<Entity>,

    /// Extraction metadata
    pub metadata: ExtractionMetadata,
}

/// Extraction metadata
#[derive(Clone, Debug, Default)]
pub struct ExtractionMetadata {
    /// Total inference time in milliseconds
    pub total_time_ms: u64,

    /// Generation time in milliseconds
    pub inference_time_ms: u64,

    /// Verification time in milliseconds
    pub verification_time_ms: u64,

    /// Retrieval time in milliseconds
    pub retrieval_time_ms: u64,

    /// Number of input tokens
    pub input_tokens: usize,

    /// Number of generated tokens
    pub output_tokens: usize,

    /// Whether caching was used
    pub cached: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::tokenizer::WhitespaceTokenizer;

    // Mock model for testing
    struct MockModel;

    impl ModelBackend for MockModel {
        type Config = ();
        type Tokenizer = WhitespaceTokenizer;

        fn load(_config: crate::traits::model::ModelConfig) -> Result<Self>
        where
            Self: Sized,
        {
            Ok(Self)
        }

        fn tokenizer(&self) -> &Self::Tokenizer {
            static TOKENIZER: WhitespaceTokenizer = WhitespaceTokenizer;
            &TOKENIZER
        }

        fn tokenizer_mut(&mut self) -> &mut Self::Tokenizer {
            static mut TOKENIZER: WhitespaceTokenizer = WhitespaceTokenizer;
            unsafe { &mut TOKENIZER }
        }

        fn generate(
            &self,
            _input_ids: &[u32],
            _max_tokens: usize,
            _temperature: f32,
            _top_p: f32,
            _top_k: u32,
        ) -> Result<Vec<u32>> {
            // Return token IDs for "John"
            Ok(vec![123, 456])
        }

        fn vocab_size(&self) -> usize {
            100000
        }

        fn max_seq_len(&self) -> usize {
            2048
        }

        fn model_name(&self) -> &str {
            "mock"
        }

        fn device(&self) -> Device {
            Device::Cpu
        }
    }

    #[test]
    fn test_pipeline_new() {
        let model = MockModel;
        let pipeline = NERPipeline::new(model);
        assert_eq!(pipeline.config().num_demonstrations, 4);
    }

    #[test]
    fn test_pipeline_with_config() {
        let config = PipelineConfig::new()
            .with_demonstrations(8);

        let model = MockModel;
        let pipeline = NERPipeline::with_config(model, config);
        assert_eq!(pipeline.config().num_demonstrations, 8);
    }

    #[test]
    fn test_add_demonstrations() {
        let model = MockModel;
        let mut pipeline = NERPipeline::new(model);

        let demos = vec![Demonstration::new(
            "John left",
            "@@John## left",
            vec![Entity::new("John", "PER", 0, 4)],
        )];

        pipeline.add_demonstrations("PER", demos);
        // Demonstrations are added to internal pool
    }

    #[test]
    fn test_extraction_metadata_default() {
        let metadata = ExtractionMetadata::default();
        assert_eq!(metadata.total_time_ms, 0);
        assert_eq!(metadata.inference_time_ms, 0);
    }
}

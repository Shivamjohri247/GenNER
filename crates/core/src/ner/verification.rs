//! Self-verification for filtering hallucinated entities

use crate::error::Result;
use crate::ner::Entity;
use crate::traits::model::{GenerationOptions, ModelBackend};
use crate::traits::tokenizer::TokenizerTrait;

/// Self-verifier for entity validation
pub struct SelfVerifier;

impl SelfVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self
    }

    /// Verify a single entity
    ///
    /// Returns a confidence score (0-1) indicating how likely
    /// the entity is correct.
    pub fn verify_entity<M: ModelBackend>(
        &self,
        model: &M,
        context: &str,
        entity: &Entity,
    ) -> Result<f32> {
        // Build verification prompt
        let prompt = format!(
            "Given the sentence: '{}'\nIs '{}' a {} entity? Answer yes or no.",
            context, entity.text, entity.label
        );

        // Tokenize and generate
        let input_ids = model.tokenizer().encode(&prompt, true)?;
        let options = GenerationOptions::new()
            .with_max_tokens(10)
            .with_temperature(0.0)
            .with_stop_sequences(vec![".".to_string()]);

        let output_ids = model.generate_with_options(&input_ids, &options)?;
        let output = model.tokenizer().decode(&output_ids, true)?;

        // Parse response
        let output_lower = output.to_lowercase();
        if output_lower.contains("yes") {
            Ok(1.0)
        } else if output_lower.contains("no") {
            Ok(0.0)
        } else {
            // Uncertain response
            Ok(0.5)
        }
    }

    /// Verify multiple entities
    ///
    /// Returns confidence scores for each entity.
    pub fn verify_entities<M: ModelBackend>(
        &self,
        model: &M,
        context: &str,
        entities: &[Entity],
    ) -> Result<Vec<f32>> {
        entities
            .iter()
            .map(|entity| self.verify_entity(model, context, entity))
            .collect()
    }

    /// Verify and filter entities
    ///
    /// Returns only entities with confidence above the threshold.
    pub fn filter_entities<M: ModelBackend>(
        &self,
        model: &M,
        context: &str,
        entities: &[Entity],
        threshold: f32,
    ) -> Result<Vec<Entity>> {
        let scores = self.verify_entities(model, context, entities)?;

        let filtered: Vec<Entity> = entities
            .iter()
            .zip(scores.iter())
            .filter(|(_, &score)| score >= threshold)
            .map(|(entity, &score)| entity.clone().with_confidence_value(score))
            .collect();

        Ok(filtered)
    }

    /// Batch verify entities from multiple contexts
    pub fn verify_batch<M: ModelBackend>(
        &self,
        model: &M,
        contexts: &[(String, Vec<Entity>)],
    ) -> Result<Vec<Vec<f32>>> {
        contexts
            .iter()
            .map(|(context, entities)| self.verify_entities(model, context, entities))
            .collect()
    }
}

impl Default for SelfVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Verification result
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// The entity being verified
    pub entity: Entity,

    /// Confidence score (0-1)
    pub confidence: f32,

    /// Whether the entity passed verification
    pub verified: bool,

    /// Raw model output
    pub raw_output: String,
}

impl VerificationResult {
    /// Create a new verification result
    pub fn new(entity: Entity, confidence: f32, raw_output: String) -> Self {
        let verified = confidence >= 0.5;
        Self {
            entity: entity.with_confidence_value(confidence),
            confidence,
            verified,
            raw_output,
        }
    }

    /// Create a passed verification
    pub fn passed(entity: Entity, raw_output: String) -> Self {
        Self::new(entity, 1.0, raw_output)
    }

    /// Create a failed verification
    pub fn failed(entity: Entity, raw_output: String) -> Self {
        Self::new(entity, 0.0, raw_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::tokenizer::WhitespaceTokenizer;

    // Mock model for testing
    struct MockModel {
        always_yes: bool,
    }

    impl MockModel {
        fn new(always_yes: bool) -> Self {
            Self { always_yes }
        }
    }

    impl ModelBackend for MockModel {
        type Config = ();
        type Tokenizer = WhitespaceTokenizer;

        fn load(_config: crate::traits::model::ModelConfig) -> Result<Self>
        where
            Self: Sized,
        {
            Ok(Self { always_yes: true })
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
            // Return token IDs for "yes" or "no"
            if self.always_yes {
                Ok(vec![Self::hash_token("yes")])
            } else {
                Ok(vec![Self::hash_token("no")])
            }
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

        fn device(&self) -> crate::error::Device {
            crate::error::Device::Cpu
        }
    }

    impl MockModel {
        fn hash_token(s: &str) -> u32 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            (h.finish() % 100000) as u32
        }
    }

    #[test]
    fn test_verifier_new() {
        let verifier = SelfVerifier::new();
        // Just check it creates successfully
        assert!(true);
    }

    #[test]
    fn test_verification_result_new() {
        let entity = Entity::new("John", "PER", 0, 4);
        let result = VerificationResult::new(entity.clone(), 0.8, "yes".to_string());
        assert!(!result.verified); // 0.8 >= 0.5, so verified
        assert!(result.verified);
    }

    #[test]
    fn test_verification_result_passed() {
        let entity = Entity::new("John", "PER", 0, 4);
        let result = VerificationResult::passed(entity.clone(), "yes".to_string());
        assert!(result.verified);
        assert_eq!(result.confidence, 1.0);
    }

    #[test]
    fn test_verification_result_failed() {
        let entity = Entity::new("John", "PER", 0, 4);
        let result = VerificationResult::failed(entity.clone(), "no".to_string());
        assert!(!result.verified);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_verify_with_mock_model() {
        let verifier = SelfVerifier::new();
        let model = MockModel::new(true);
        let entity = Entity::new("John", "PER", 0, 4);

        let confidence = verifier
            .verify_entity(&model, "John is here", &entity)
            .unwrap();

        assert_eq!(confidence, 1.0);
    }

    #[test]
    fn test_filter_entities_with_threshold() {
        let verifier = SelfVerifier::new();
        let model = MockModel::new(true);

        let entities = vec![
            Entity::new("John", "PER", 0, 4),
            Entity::new("Paris", "LOC", 8, 13),
        ];

        let filtered = verifier
            .filter_entities(&model, "John in Paris", &entities, 0.5)
            .unwrap();

        assert_eq!(filtered.len(), 2);
    }
}

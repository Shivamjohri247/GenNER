//! LoRA (Low-Rank Adaptation) implementation

use bincode::{Encode, Decode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LoRA configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Rank of the LoRA matrices
    pub rank: usize,

    /// Alpha scaling factor
    pub alpha: f32,

    /// Dropout probability
    pub dropout: f32,

    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,

    /// Bias training strategy
    pub bias: LoRABias,

    /// Scaling: alpha / rank
    #[serde(skip)]
    pub scaling: f32,
}

impl LoRAConfig {
    /// Create a new LoRA config
    pub fn new(rank: usize, alpha: f32) -> Self {
        let scaling = alpha / rank as f32;
        Self {
            rank,
            alpha,
            dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
                "k_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            bias: LoRABias::None,
            scaling,
        }
    }

    /// Create with default rank 16
    pub fn default_rank() -> Self {
        Self::new(16, 32.0)
    }

    /// Set rank
    pub fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self.scaling = self.alpha / rank as f32;
        self
    }

    /// Set alpha
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self.scaling = alpha / self.rank as f32;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout.clamp(0.0, 1.0);
        self
    }

    /// Set target modules
    pub fn with_target_modules(mut self, modules: Vec<String>) -> Self {
        self.target_modules = modules;
        self
    }

    /// Set bias strategy
    pub fn with_bias(mut self, bias: LoRABias) -> Self {
        self.bias = bias;
        self
    }

    /// Get the effective scaling factor
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self::default_rank()
    }
}

/// Bias training strategy for LoRA
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoRABias {
    /// Don't train any bias
    None,

    /// Train all biases
    All,

    /// Train only LoRA biases
    LoraOnly,
}

/// Trained LoRA adapter
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoRAAdapter {
    /// Adapter name
    pub name: String,

    /// Task name
    pub task_name: String,

    /// LoRA configuration
    pub config: LoRAConfig,

    /// LoRA A matrices (module name -> weights)
    pub lora_a: HashMap<String, Vec<f32>>,

    /// LoRA B matrices (module name -> weights)
    pub lora_b: HashMap<String, Vec<f32>>,

    /// Adapter metadata
    pub metadata: AdapterMetadata,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter
    pub fn new(
        name: impl Into<String>,
        task_name: impl Into<String>,
        config: LoRAConfig,
    ) -> Self {
        Self {
            name: name.into(),
            task_name: task_name.into(),
            config,
            lora_a: HashMap::new(),
            lora_b: HashMap::new(),
            metadata: AdapterMetadata::default(),
        }
    }

    /// Add a LoRA matrix pair for a module
    pub fn add_module(&mut self, module_name: impl Into<String>, a: Vec<f32>, b: Vec<f32>) {
        let name = module_name.into();
        self.lora_a.insert(name.clone(), a);
        self.lora_b.insert(name, b);
    }

    /// Get the number of parameters in this adapter
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        for (a, b) in self.lora_a.values().zip(self.lora_b.values()) {
            count += a.len() + b.len();
        }
        count
    }

    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        self.num_parameters() * std::mem::size_of::<f32>()
    }

    /// Validate the adapter
    pub fn validate(&self) -> Result<(), String> {
        // Check that A and B matrices exist for the same modules
        for module in self.lora_a.keys() {
            if !self.lora_b.contains_key(module) {
                return Err(format!("Missing B matrix for module: {}", module));
            }
        }

        // Check dimensions match
        for (module, a) in &self.lora_a {
            if let Some(b) = self.lora_b.get(module) {
                // For LoRA, A is (in_features, rank) and B is (rank, out_features)
                // or A is (rank, in_features) and B is (out_features, rank)
                // We just check that the rank dimension matches
                let a_size = a.len();
                let b_size = b.len();

                if a_size == 0 || b_size == 0 {
                    return Err(format!("Empty matrix for module: {}", module));
                }

                // Verify that one dimension equals the configured rank
                let rank = self.config.rank;
                if a_size % rank != 0 && b_size % rank != 0 {
                    return Err(format!(
                        "Matrix dimensions don't match rank {} for module: {}",
                        rank, module
                    ));
                }
            }
        }

        Ok(())
    }

    /// Merge with another adapter (simple average)
    pub fn merge(&self, other: &LoRAAdapter, weight: f32) -> Result<LoRAAdapter, String> {
        if self.config.rank != other.config.rank {
            return Err("Cannot merge adapters with different ranks".to_string());
        }

        let mut merged = LoRAAdapter {
            name: format!("{}_merged_{}", self.name, other.name),
            task_name: format!("{}_{}", self.task_name, other.task_name),
            config: self.config.clone(),
            lora_a: HashMap::new(),
            lora_b: HashMap::new(),
            metadata: AdapterMetadata {
                merged_from: Some(vec![self.name.clone(), other.name.clone()]),
                ..Default::default()
            },
        };

        // Merge A matrices
        for (module, a) in &self.lora_a {
            if let Some(other_a) = other.lora_a.get(module) {
                let merged_a: Vec<f32> = a.iter()
                    .zip(other_a.iter())
                    .map(|(x, y)| x * (1.0 - weight) + y * weight)
                    .collect();
                merged.lora_a.insert(module.clone(), merged_a);
            }
        }

        // Merge B matrices
        for (module, b) in &self.lora_b {
            if let Some(other_b) = other.lora_b.get(module) {
                let merged_b: Vec<f32> = b.iter()
                    .zip(other_b.iter())
                    .map(|(x, y)| x * (1.0 - weight) + y * weight)
                    .collect();
                merged.lora_b.insert(module.clone(), merged_b);
            }
        }

        Ok(merged)
    }
}

/// Wrapper for chrono DateTime that implements bincode Encode/Decode
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SerializableDateTime {
    pub inner: chrono::DateTime<chrono::Utc>,
}

impl From<chrono::DateTime<chrono::Utc>> for SerializableDateTime {
    fn from(dt: chrono::DateTime<chrono::Utc>) -> Self {
        Self { inner: dt }
    }
}

impl From<SerializableDateTime> for chrono::DateTime<chrono::Utc> {
    fn from(dt: SerializableDateTime) -> Self {
        dt.inner
    }
}

impl bincode::Encode for SerializableDateTime {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        self.inner.timestamp().encode(encoder)
    }
}

impl<'de> bincode::BorrowDecode<'de, ()> for SerializableDateTime {
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let timestamp = i64::borrow_decode(decoder)?;
        Ok(Self {
            inner: chrono::DateTime::from_timestamp(timestamp, 0).unwrap_or(chrono::Utc::now()),
        })
    }
}

/// Adapter metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterMetadata {
    /// Creation timestamp
    pub created_at: SerializableDateTime,

    /// Base model used for training
    pub base_model: String,

    /// Number of training samples
    pub training_samples: usize,

    /// Number of epochs trained
    pub epochs: u32,

    /// Final loss value
    pub final_loss: f32,

    /// Entity types this adapter handles
    pub entity_types: Vec<String>,

    /// If this is a merged adapter, which adapters were merged
    pub merged_from: Option<Vec<String>>,

    /// Training time in seconds
    pub training_duration_secs: f64,
}

impl Default for AdapterMetadata {
    fn default() -> Self {
        Self {
            created_at: SerializableDateTime::from(chrono::Utc::now()),
            base_model: String::new(),
            training_samples: 0,
            epochs: 0,
            final_loss: 0.0,
            entity_types: Vec::new(),
            merged_from: None,
            training_duration_secs: 0.0,
        }
    }
}

/// Fusion strategy for multi-adapter inference
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Use only the adapter matching the entity type
    Switch,

    /// Linear combination of adapter weights
    Linear,

    /// TIES-Merging style composition
    Ties,
}

/// Adapter composition for multi-task learning
#[derive(Clone, Debug)]
pub struct AdapterComposition {
    /// Adapters in the composition
    pub adapters: Vec<LoRAAdapter>,

    /// Fusion strategy
    pub strategy: FusionStrategy,

    /// Weights for linear fusion
    pub weights: Vec<f32>,
}

impl AdapterComposition {
    /// Create a new composition
    pub fn new(adapters: Vec<LoRAAdapter>, strategy: FusionStrategy) -> Self {
        let weights = vec![1.0 / adapters.len() as f32; adapters.len()];
        Self {
            adapters,
            strategy,
            weights,
        }
    }

    /// Create with linear fusion and custom weights
    pub fn linear_with_weights(adapters: Vec<LoRAAdapter>, weights: Vec<f32>) -> Self {
        assert_eq!(
            adapters.len(),
            weights.len(),
            "Adapters and weights must have same length"
        );
        Self {
            adapters,
            strategy: FusionStrategy::Linear,
            weights,
        }
    }

    /// Get adapter by name
    pub fn get_adapter(&self, name: &str) -> Option<&LoRAAdapter> {
        self.adapters.iter().find(|a| a.name == name)
    }

    /// Get all entity types
    pub fn entity_types(&self) -> Vec<String> {
        let mut types = Vec::new();
        for adapter in &self.adapters {
            for ty in &adapter.metadata.entity_types {
                if !types.contains(ty) {
                    types.push(ty.clone());
                }
            }
        }
        types
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_new() {
        let config = LoRAConfig::new(8, 16.0);
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.scaling, 2.0);
    }

    #[test]
    fn test_lora_config_builder() {
        let config = LoRAConfig::default()
            .with_rank(32)
            .with_alpha(64.0)
            .with_dropout(0.1);

        assert_eq!(config.rank, 32);
        assert_eq!(config.alpha, 64.0);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_lora_adapter_new() {
        let adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
        assert_eq!(adapter.name, "test");
        assert_eq!(adapter.task_name, "ner_task");
    }

    #[test]
    fn test_lora_adapter_add_module() {
        let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
        adapter.add_module("q_proj", vec![0.0; 100], vec![0.0; 200]);

        assert!(adapter.lora_a.contains_key("q_proj"));
        assert!(adapter.lora_b.contains_key("q_proj"));
    }

    #[test]
    fn test_lora_adapter_validate() {
        let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::new(16, 32.0));
        adapter.add_module("q_proj", vec![0.0; 1600], vec![0.0; 3200]); // (100, 16) and (16, 200)

        assert!(adapter.validate().is_ok());
    }

    #[test]
    fn test_lora_adapter_validate_missing_b() {
        let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
        adapter.lora_a.insert("q_proj".to_string(), vec![0.0; 100]);

        assert!(adapter.validate().is_err());
    }

    #[test]
    fn test_adapter_metadata_default() {
        let metadata = AdapterMetadata::default();
        assert_eq!(metadata.training_samples, 0);
        assert_eq!(metadata.epochs, 0);
    }

    #[test]
    fn test_adapter_composition_new() {
        let adapters = vec![
            LoRAAdapter::new("a1", "task1", LoRAConfig::default()),
            LoRAAdapter::new("a2", "task2", LoRAConfig::default()),
        ];

        let composition = AdapterComposition::new(adapters.clone(), FusionStrategy::Switch);
        assert_eq!(composition.adapters.len(), 2);
        assert_eq!(composition.weights, vec![0.5, 0.5]);
    }

    #[test]
    fn test_adapter_merge() {
        let mut a1 = LoRAAdapter::new("a1", "task1", LoRAConfig::new(8, 16.0));
        a1.add_module("q_proj", vec![1.0; 80], vec![2.0; 80]);

        let mut a2 = LoRAAdapter::new("a2", "task2", LoRAConfig::new(8, 16.0));
        a2.add_module("q_proj", vec![3.0; 80], vec![5.0; 80]);

        let merged = a1.merge(&a2, 0.5).unwrap();
        assert!(merged.name.contains("merged"));
        assert_eq!(merged.lora_a["q_proj"].len(), 80);
    }

    #[test]
    fn test_adapter_merge_different_rank() {
        let a1 = LoRAAdapter::new("a1", "task1", LoRAConfig::new(8, 16.0));
        let a2 = LoRAAdapter::new("a2", "task2", LoRAConfig::new(16, 32.0));

        let result = a1.merge(&a2, 0.5);
        assert!(result.is_err());
    }
}

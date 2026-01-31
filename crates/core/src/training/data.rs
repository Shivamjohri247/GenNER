//! Data loading and batching for training

use crate::error::Result;
use crate::ner::{Entity, EntityType};
use crate::training::lora::SerializableDateTime;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Training example
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Example ID
    pub id: String,

    /// Task type
    pub task_type: TaskType,

    /// Input prompt
    pub input: String,

    /// Target output
    pub output: String,

    /// Entities in this example
    pub entities: Vec<Entity>,

    /// Metadata
    pub metadata: ExampleMetadata,
}

/// Task type
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Entity extraction for a specific type
    EntityExtraction { entity_type: String },

    /// Entity verification
    EntityVerification,

    /// Multi-type extraction
    MultiEntity { entity_types: Vec<String> },
}

impl TaskType {
    /// Get the entity types for this task
    pub fn entity_types(&self) -> Vec<String> {
        match self {
            Self::EntityExtraction { entity_type } => vec![entity_type.clone()],
            Self::EntityVerification => vec![],
            Self::MultiEntity { entity_types } => entity_types.clone(),
        }
    }
}

/// Example metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExampleMetadata {
    /// Original text (before marking)
    pub original_text: String,

    /// Number of tokens in input
    pub input_tokens: usize,

    /// Number of tokens in output
    pub output_tokens: usize,

    /// Source dataset
    pub source: String,

    /// Sample weight (for weighted sampling)
    pub weight: f32,
}

impl Default for ExampleMetadata {
    fn default() -> Self {
        Self {
            original_text: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            source: String::new(),
            weight: 1.0,
        }
    }
}

/// Training dataset
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Dataset metadata
    pub metadata: DatasetMetadata,

    /// Training examples
    pub examples: Vec<TrainingExample>,

    /// Entity type definitions
    pub entity_types: Vec<EntityType>,
}

/// Dataset metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name
    pub name: String,

    /// Dataset version
    pub version: String,

    /// Description
    pub description: String,

    /// Creation date
    pub created_at: SerializableDateTime,

    /// Total samples
    pub total_samples: usize,

    /// Average tokens per sample
    pub avg_tokens_per_sample: f32,

    /// Max sequence length
    pub max_seq_length: usize,
}

impl Default for DatasetMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            version: "1.0".to_string(),
            description: String::new(),
            created_at: SerializableDateTime::from(chrono::Utc::now()),
            total_samples: 0,
            avg_tokens_per_sample: 0.0,
            max_seq_length: 2048,
        }
    }
}

impl TrainingDataset {
    /// Create a new empty dataset
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            metadata: DatasetMetadata {
                name: name.into(),
                ..Default::default()
            },
            examples: Vec::new(),
            entity_types: Vec::new(),
        }
    }

    /// Add an example to the dataset
    pub fn add_example(&mut self, example: TrainingExample) {
        self.examples.push(example);
        self.metadata.total_samples = self.examples.len();
    }

    /// Load from JSON file
    pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read_to_string(path.as_ref())?;
        let dataset: TrainingDataset = serde_json::from_str(&data)?;
        Ok(dataset)
    }

    /// Save to JSON file
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path.as_ref(), data)?;
        Ok(())
    }

    /// Load from binary format
    pub fn load_binary(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let dataset: TrainingDataset = bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;
        Ok(dataset)
    }

    /// Save to binary format
    pub fn save_binary(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = bincode::serde::encode_to_vec(self, bincode::config::standard())?;
        std::fs::write(path.as_ref(), data)?;
        Ok(())
    }

    /// Split into train and validation sets
    pub fn split(&self, val_ratio: f32) -> (TrainingDataset, TrainingDataset) {
        let val_size = (self.examples.len() as f32 * val_ratio).ceil() as usize;
        let val_size = val_size.min(self.examples.len() - 1);

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.examples.len()).collect();
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = (rand::random::<u64>() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let val_indices: std::collections::HashSet<_> = indices[..val_size].iter().cloned().collect();

        let mut train = TrainingDataset::new(format!("{}_train", self.metadata.name));
        train.metadata = self.metadata.clone();
        train.entity_types = self.entity_types.clone();

        let mut val = TrainingDataset::new(format!("{}_val", self.metadata.name));
        val.metadata = self.metadata.clone();
        val.metadata.total_samples = 0;
        val.entity_types = self.entity_types.clone();

        for (i, example) in self.examples.iter().cloned().enumerate() {
            if val_indices.contains(&i) {
                val.add_example(example);
            } else {
                train.add_example(example);
            }
        }

        (train, val)
    }

    /// Get examples by entity type
    pub fn examples_by_entity_type(&self, entity_type: &str) -> Vec<&TrainingExample> {
        self.examples
            .iter()
            .filter(|ex| {
                ex.task_type
                    .entity_types()
                    .contains(&entity_type.to_string())
            })
            .collect()
    }

    /// Shuffle the dataset
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.examples.shuffle(&mut rng);
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }
}

/// Data batch
#[derive(Clone, Debug)]
pub struct DataBatch {
    /// Input token IDs
    pub input_ids: Vec<Vec<u32>>,

    /// Target token IDs
    pub target_ids: Vec<Vec<u32>>,

    /// Attention masks
    pub attention_masks: Vec<Vec<u8>>,

    /// Example IDs
    pub example_ids: Vec<String>,
}

impl DataBatch {
    /// Create a new batch
    pub fn new(batch_size: usize) -> Self {
        Self {
            input_ids: Vec::with_capacity(batch_size),
            target_ids: Vec::with_capacity(batch_size),
            attention_masks: Vec::with_capacity(batch_size),
            example_ids: Vec::with_capacity(batch_size),
        }
    }

    /// Add a sample to the batch
    pub fn add(&mut self, input_ids: Vec<u32>, target_ids: Vec<u32>, example_id: String) {
        let seq_len = input_ids.len().max(target_ids.len());
        let mut attention_mask = vec![1u8; seq_len];

        // Pad to same length
        let mut padded_input = input_ids.clone();
        let mut padded_target = target_ids.clone();

        while padded_input.len() < seq_len {
            padded_input.push(0); // Pad token
            attention_mask[input_ids.len()] = 0;
        }

        while padded_target.len() < seq_len {
            padded_target.push(0);
        }

        self.input_ids.push(padded_input);
        self.target_ids.push(padded_target);
        self.attention_masks.push(attention_mask);
        self.example_ids.push(example_id);
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

/// Data loader for creating batches
#[derive(Clone, Debug)]
pub struct DataLoader {
    /// Dataset to load from
    dataset: TrainingDataset,

    /// Batch size
    batch_size: usize,

    /// Shuffle each epoch
    shuffle: bool,

    /// Current position
    position: usize,

    /// Current epoch
    epoch: usize,
}

impl DataLoader {
    /// Create a new data loader
    pub fn new(dataset: TrainingDataset, batch_size: usize, shuffle: bool) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            position: 0,
            epoch: 0,
        }
    }

    /// Get the next batch
    pub fn next_batch(&mut self) -> Option<DataBatch> {
        if self.position >= self.dataset.examples.len() {
            return None;
        }

        let end = (self.position + self.batch_size).min(self.dataset.examples.len());
        let batch_examples = &self.dataset.examples[self.position..end];

        self.position = end;

        let mut batch = DataBatch::new(batch_examples.len());
        for example in batch_examples {
            // Note: In real implementation, we'd tokenize here
            // For now, just use placeholder token IDs
            batch.add(
                vec![0], // Placeholder: would be actual token IDs
                vec![0],
                example.id.clone(),
            );
        }

        Some(batch)
    }

    /// Reset for next epoch
    pub fn reset(&mut self) {
        self.position = 0;
        self.epoch += 1;

        if self.shuffle {
            self.dataset.shuffle();
        }
    }

    /// Get total number of batches
    pub fn num_batches(&self) -> usize {
        (self.dataset.examples.len() + self.batch_size - 1) / self.batch_size
    }

    /// Get current epoch
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Check if iteration is complete
    pub fn is_finished(&self) -> bool {
        self.position >= self.dataset.examples.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_new() {
        let dataset = TrainingDataset::new("test");
        assert_eq!(dataset.metadata.name, "test");
        assert!(dataset.is_empty());
    }

    #[test]
    fn test_dataset_add_example() {
        let mut dataset = TrainingDataset::new("test");
        dataset.add_example(TrainingExample {
            id: "1".to_string(),
            task_type: TaskType::EntityExtraction {
                entity_type: "PER".to_string(),
            },
            input: "test".to_string(),
            output: "test".to_string(),
            entities: vec![],
            metadata: ExampleMetadata::default(),
        });

        assert_eq!(dataset.len(), 1);
    }

    #[test]
    fn test_dataset_split() {
        let mut dataset = TrainingDataset::new("test");
        for i in 0..100 {
            dataset.add_example(TrainingExample {
                id: i.to_string(),
                task_type: TaskType::EntityExtraction {
                    entity_type: "PER".to_string(),
                },
                input: format!("input {}", i),
                output: format!("output {}", i),
                entities: vec![],
                metadata: ExampleMetadata::default(),
            });
        }

        let (train, val) = dataset.split(0.2);
        assert!(train.len() > val.len());
        assert_eq!(train.len() + val.len(), 100);
    }

    #[test]
    fn test_task_type_entity_types() {
        let task = TaskType::EntityExtraction {
            entity_type: "PER".to_string(),
        };
        assert_eq!(task.entity_types(), vec!["PER"]);

        let task = TaskType::MultiEntity {
            entity_types: vec!["PER".to_string(), "LOC".to_string()],
        };
        assert_eq!(task.entity_types(), vec!["PER", "LOC"]);
    }

    #[test]
    fn test_data_batch_new() {
        let batch = DataBatch::new(10);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_data_batch_add() {
        let mut batch = DataBatch::new(2);
        batch.add(vec![1, 2, 3], vec![4, 5], "test".to_string());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.example_ids[0], "test");
    }

    #[test]
    fn test_data_loader_new() {
        let dataset = TrainingDataset::new("test");
        let loader = DataLoader::new(dataset, 10, true);
        assert_eq!(loader.num_batches(), 0);
        assert!(loader.is_finished());
    }
}

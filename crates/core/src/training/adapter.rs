//! Adapter storage and management

use crate::error::Result;
use crate::training::lora::{LoRAAdapter, LoRAConfig, AdapterMetadata, SerializableDateTime};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Adapter storage for managing trained LoRA adapters
#[derive(Clone, Debug)]
pub struct AdapterStore {
    /// Storage directory
    storage_dir: PathBuf,

    /// In-memory cache of loaded adapters
    cache: HashMap<String, LoRAAdapter>,

    /// Adapter index
    index: AdapterIndex,
}

impl AdapterStore {
    /// Create a new adapter store
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        let storage_dir = storage_dir.into();
        Self {
            storage_dir,
            cache: HashMap::new(),
            index: AdapterIndex::new(),
        }
    }

    /// Initialize the store (create directory if needed)
    pub fn initialize(&mut self) -> Result<()> {
        std::fs::create_dir_all(&self.storage_dir)?;
        self.load_index()?;
        Ok(())
    }

    /// Save an adapter to the store
    pub fn save(&mut self, adapter: &LoRAAdapter) -> Result<PathBuf> {
        let adapter_path = self.adapter_path(&adapter.name);

        // Ensure directory exists
        if let Some(parent) = adapter_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Serialize adapter
        let data = bincode::serde::encode_to_vec(adapter, bincode::config::standard())?;

        // Write to file
        std::fs::write(&adapter_path, data)?;

        // Update index
        self.index.add_adapter(
            adapter.name.clone(),
            adapter.task_name.clone(),
            adapter.metadata.entity_types.clone(),
        );

        // Save index
        self.save_index()?;

        Ok(adapter_path)
    }

    /// Load an adapter from the store
    pub fn load(&mut self, name: &str) -> Result<LoRAAdapter> {
        // Check cache first
        if let Some(adapter) = self.cache.get(name) {
            return Ok(adapter.clone());
        }

        let adapter_path = self.adapter_path(name);

        if !adapter_path.exists() {
            return Err(crate::error::Error::NotFound(format!(
                "Adapter '{}' not found at {:?}",
                name, adapter_path
            )));
        }

        let data = std::fs::read(&adapter_path)?;
        let adapter: LoRAAdapter = bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;

        // Cache it
        self.cache.insert(name.to_string(), adapter.clone());

        Ok(adapter)
    }

    /// Delete an adapter from the store
    pub fn delete(&mut self, name: &str) -> Result<()> {
        let adapter_path = self.adapter_path(name);

        if adapter_path.exists() {
            std::fs::remove_file(&adapter_path)?;
        }

        self.cache.remove(name);
        self.index.remove_adapter(name);
        self.save_index()?;

        Ok(())
    }

    /// List all adapters in the store
    pub fn list(&self) -> Vec<String> {
        self.index.adapters().to_vec()
    }

    /// List adapters by task
    pub fn list_by_task(&self, task_name: &str) -> Vec<String> {
        self.index
            .adapters_by_task(task_name)
            .to_vec()
    }

    /// List adapters by entity type
    pub fn list_by_entity_type(&self, entity_type: &str) -> Vec<String> {
        self.index
            .adapters_by_entity_type(entity_type)
            .to_vec()
    }

    /// Get adapter metadata without loading the full adapter
    pub fn get_metadata(&self, name: &str) -> Option<AdapterMetadata> {
        self.index.get_metadata(name)
    }

    /// Check if an adapter exists
    pub fn contains(&self, name: &str) -> bool {
        self.cache.contains_key(name) || self.adapter_path(name).exists()
    }

    /// Clear the adapter cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the path for an adapter
    fn adapter_path(&self, name: &str) -> PathBuf {
        let safe_name = name.replace('/', "__").replace('\\', "__");
        self.storage_dir.join(format!("{}.adapter", safe_name))
    }

    /// Load the adapter index
    fn load_index(&mut self) -> Result<()> {
        let index_path = self.index_path();
        if index_path.exists() {
            let data = std::fs::read(&index_path)?;
            self.index = bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;
        }
        Ok(())
    }

    /// Save the adapter index
    fn save_index(&self) -> Result<()> {
        let index_path = self.index_path();
        let data = bincode::serde::encode_to_vec(&self.index, bincode::config::standard())?;
        std::fs::write(&index_path, data)?;
        Ok(())
    }

    /// Get the index file path
    fn index_path(&self) -> PathBuf {
        self.storage_dir.join("index.bin")
    }
}

/// Index for fast adapter lookup
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdapterIndex {
    /// All adapters
    adapters: HashMap<String, AdapterMetadata>,

    /// Task name -> adapter names
    by_task: HashMap<String, Vec<String>>,

    /// Entity type -> adapter names
    by_entity_type: HashMap<String, Vec<String>>,
}

impl AdapterIndex {
    /// Create a new index
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            by_task: HashMap::new(),
            by_entity_type: HashMap::new(),
        }
    }

    /// Add an adapter to the index
    pub fn add_adapter(
        &mut self,
        name: String,
        task_name: String,
        entity_types: Vec<String>,
    ) {
        // Create metadata
        let metadata = AdapterMetadata {
            created_at: SerializableDateTime::from(chrono::Utc::now()),
            base_model: String::new(),
            training_samples: 0,
            epochs: 0,
            final_loss: 0.0,
            entity_types: entity_types.clone(),
            merged_from: None,
            training_duration_secs: 0.0,
        };

        // Add to main index
        self.adapters.insert(name.clone(), metadata);

        // Index by task
        self.by_task
            .entry(task_name)
            .or_insert_with(Vec::new)
            .push(name.clone());

        // Index by entity type
        for entity_type in entity_types {
            self.by_entity_type
                .entry(entity_type)
                .or_insert_with(Vec::new)
                .push(name.clone());
        }
    }

    /// Remove an adapter from the index
    pub fn remove_adapter(&mut self, name: &str) {
        if let Some(metadata) = self.adapters.remove(name) {
            // Remove from task index
            for (_, adapters) in self.by_task.iter_mut() {
                adapters.retain(|n| n != name);
            }

            // Remove from entity type index
            for (_, adapters) in self.by_entity_type.iter_mut() {
                adapters.retain(|n| n != name);
            }
        }
    }

    /// Get all adapter names
    pub fn adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Get adapters by task
    pub fn adapters_by_task(&self, task_name: &str) -> Vec<String> {
        self.by_task
            .get(task_name)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get adapters by entity type
    pub fn adapters_by_entity_type(&self, entity_type: &str) -> Vec<String> {
        self.by_entity_type
            .get(entity_type)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Get metadata for an adapter
    pub fn get_metadata(&self, name: &str) -> Option<AdapterMetadata> {
        self.adapters.get(name).cloned()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }

    /// Get number of adapters
    pub fn len(&self) -> usize {
        self.adapters.len()
    }
}

impl Default for AdapterIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporary storage for adapters during training
#[derive(Clone, Debug)]
pub struct RehearsalBuffer {
    /// Buffer storage
    buffer: Vec<TrainingSample>,

    /// Maximum buffer size
    max_size: usize,

    /// Sampling strategy
    strategy: RehearsalStrategy,
}

/// Sampling strategy for rehearsal buffer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RehearsalStrategy {
    /// Random sampling
    Random,

    /// Reservoir sampling (maintain uniform distribution)
    Reservoir,

    /// Prioritize recent samples
    Recent,
}

impl RehearsalBuffer {
    /// Create a new rehearsal buffer
    pub fn new(max_size: usize, strategy: RehearsalStrategy) -> Self {
        Self {
            buffer: Vec::new(),
            max_size,
            strategy,
        }
    }

    /// Add samples to the buffer
    pub fn add(&mut self, samples: Vec<TrainingSample>) {
        match self.strategy {
            RehearsalStrategy::Random => {
                for sample in samples {
                    if self.buffer.len() >= self.max_size {
                        // Randomly replace
                        let idx = rand::random::<usize>() % self.buffer.len();
                        self.buffer[idx] = sample;
                    } else {
                        self.buffer.push(sample);
                    }
                }
            }
            RehearsalStrategy::Reservoir => {
                for (i, sample) in samples.into_iter().enumerate() {
                    let total = self.buffer.len();
                    if total < self.max_size {
                        self.buffer.push(sample);
                    } else {
                        // Reservoir sampling
                        let j = rand::random::<usize>() % (total + i + 1);
                        if j < self.max_size {
                            self.buffer[j] = sample;
                        }
                    }
                }
            }
            RehearsalStrategy::Recent => {
                for sample in samples {
                    if self.buffer.len() >= self.max_size {
                        self.buffer.remove(0);
                    }
                    self.buffer.push(sample);
                }
            }
        }
    }

    /// Sample from the buffer
    pub fn sample(&self, n: usize) -> Vec<TrainingSample> {
        if self.buffer.is_empty() {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..self.buffer.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        indices
            .into_iter()
            .take(n)
            .filter_map(|i| self.buffer.get(i).cloned())
            .collect()
    }

    /// Get the current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Training sample for rehearsal
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Sample ID
    pub id: String,

    /// Input text
    pub input: String,

    /// Target output
    pub output: String,

    /// Task/entity type
    pub task: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_index_new() {
        let index = AdapterIndex::new();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_adapter_index_add() {
        let mut index = AdapterIndex::new();
        index.add_adapter(
            "test_adapter".to_string(),
            "test_task".to_string(),
            vec!["PER".to_string(), "LOC".to_string()],
        );

        assert_eq!(index.len(), 1);
        assert!(index.adapters().contains(&"test_adapter".to_string()));
    }

    #[test]
    fn test_rehearsal_buffer_new() {
        let buffer = RehearsalBuffer::new(100, RehearsalStrategy::Random);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_rehearsal_buffer_add() {
        let mut buffer = RehearsalBuffer::new(2, RehearsalStrategy::Recent);
        buffer.add(vec![
            TrainingSample {
                id: "1".to_string(),
                input: "test1".to_string(),
                output: "output1".to_string(),
                task: "PER".to_string(),
            },
            TrainingSample {
                id: "2".to_string(),
                input: "test2".to_string(),
                output: "output2".to_string(),
                task: "PER".to_string(),
            },
        ]);

        assert_eq!(buffer.len(), 2);
    }
}

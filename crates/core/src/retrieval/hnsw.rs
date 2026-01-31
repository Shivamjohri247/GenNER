//! HNSW-based embedding store for efficient kNN retrieval

use crate::error::Result;
use crate::traits::embedding::{EmbeddingModel, EmbeddingStore, EntityEmbedding, SentenceEmbedding};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// HNSW-based embedding store for fast approximate nearest neighbor search
pub struct HNSWEmbeddingStore {
    /// Sentence embeddings indexed with HNSW
    sentences: HNSWIndex<SentenceEmbedding>,

    /// Entity embeddings indexed with HNSW
    entities: HNSWIndex<EntityEmbedding>,

    /// Embedding dimension
    dimension: usize,

    /// M parameter for HNSW (number of bidirectional links)
    m: usize,

    /// ef_construction parameter for HNSW
    ef_construction: usize,
}

/// HNSW index wrapper
#[derive(Clone, Serialize, Deserialize)]
struct HNSWIndex<T> {
    /// Items in the index
    items: Vec<T>,

    /// Vectors for each item
    vectors: Vec<Vec<f32>>,
}

impl<T> HNSWIndex<T> {
    /// Create a new empty index
    fn new(dimension: usize, m: usize) -> Self {
        Self {
            items: Vec::new(),
            vectors: Vec::new(),
        }
    }

    /// Add an item to the index
    fn add(&mut self, item: T, vector: Vec<f32>) {
        self.items.push(item);
        self.vectors.push(vector);
    }

    /// Build HNSW index (simplified - uses hnsw crate if available)
    fn build_index(&mut self) -> Result<()> {
        // For now, we'll use linear scan since HNSW integration requires more setup
        // In production, this would use the hnsw crate to build the actual index
        Ok(())
    }

    /// Find k nearest neighbors using linear scan
    /// TODO: Replace with actual HNSW search
    fn find_knn_linear(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<_> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| {
                let sim = cosine_similarity(query, vec);
                (idx, sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.into_iter().take(k).collect()
    }
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

impl HNSWEmbeddingStore {
    /// Create a new HNSW embedding store
    pub fn new(dimension: usize) -> Self {
        Self {
            sentences: HNSWIndex::new(dimension, 16),
            entities: HNSWIndex::new(dimension, 16),
            dimension,
            m: 16,
            ef_construction: 200,
        }
    }

    /// Create with HNSW parameters
    pub fn with_params(dimension: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            sentences: HNSWIndex::new(dimension, m),
            entities: HNSWIndex::new(dimension, m),
            dimension,
            m,
            ef_construction,
        }
    }

    /// Add a sentence embedding
    pub fn add_sentence(&mut self, embedding: SentenceEmbedding) -> Result<()> {
        if !self.is_valid_vector(&embedding.vector) {
            return Err(crate::error::Error::Validation(
                "Invalid embedding dimension".to_string(),
            ));
        }

        let vector = embedding.vector.clone();
        self.sentences.add(embedding, vector);
        Ok(())
    }

    /// Add multiple sentence embeddings
    pub fn add_sentences(&mut self, embeddings: Vec<SentenceEmbedding>) -> Result<()> {
        for emb in embeddings {
            self.add_sentence(emb)?;
        }
        Ok(())
    }

    /// Add an entity embedding
    pub fn add_entity(&mut self, embedding: EntityEmbedding) -> Result<()> {
        if !self.is_valid_vector(&embedding.vector) {
            return Err(crate::error::Error::Validation(
                "Invalid embedding dimension".to_string(),
            ));
        }

        let vector = embedding.vector.clone();
        self.entities.add(embedding, vector);
        Ok(())
    }

    /// Find k nearest sentence neighbors
    pub fn find_knn(&self, query: &[f32], k: usize) -> Vec<(SentenceEmbedding, f32)> {
        let indices = self.sentences.find_knn_linear(query, k);
        indices
            .into_iter()
            .filter_map(|(idx, sim)| {
                self.sentences.items.get(idx).map(|item| (item.clone(), sim))
            })
            .collect()
    }

    /// Find k nearest entity neighbors
    pub fn find_knn_entities(
        &self,
        query: &[f32],
        k: usize,
    ) -> Vec<(EntityEmbedding, f32)> {
        let indices = self.entities.find_knn_linear(query, k);
        indices
            .into_iter()
            .filter_map(|(idx, sim)| {
                self.entities.items.get(idx).map(|item| (item.clone(), sim))
            })
            .collect()
    }

    /// Find k nearest neighbors by text query
    pub fn find_knn_by_text(
        &self,
        model: &impl EmbeddingModel,
        query: &str,
        k: usize,
    ) -> Result<Vec<(SentenceEmbedding, f32)>> {
        let query_vec = model.embed_sentence(query)?;
        Ok(self.find_knn(&query_vec, k))
    }

    /// Build the HNSW indices for fast search
    pub fn build_index(&mut self) -> Result<()> {
        self.sentences.build_index()?;
        self.entities.build_index()?;
        Ok(())
    }

    /// Save the store to disk
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serde::encode_to_vec(
            &(self.sentences.clone(), self.entities.clone()),
            bincode::config::standard(),
        )?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load the store from disk
    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        let (sentences, entities): (HNSWIndex<SentenceEmbedding>, HNSWIndex<EntityEmbedding>) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;
        self.sentences = sentences;
        self.entities = entities;
        Ok(())
    }

    /// Get number of sentences
    pub fn len_sentences(&self) -> usize {
        self.sentences.items.len()
    }

    /// Get number of entities
    pub fn len_entities(&self) -> usize {
        self.entities.items.len()
    }

    /// Get total size
    pub fn len(&self) -> usize {
        self.len_sentences() + self.len_entities()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all embeddings
    pub fn clear(&mut self) -> Result<()> {
        self.sentences = HNSWIndex::new(self.dimension, self.m);
        self.entities = HNSWIndex::new(self.dimension, self.m);
        Ok(())
    }

    /// Check if a vector is valid
    fn is_valid_vector(&self, vec: &[f32]) -> bool {
        vec.len() == self.dimension && vec.iter().all(|f| f.is_finite())
    }
}

impl EmbeddingStore for HNSWEmbeddingStore {
    fn add_sentence(&mut self, embedding: SentenceEmbedding) -> Result<()> {
        self.add_sentence(embedding)
    }

    fn add_sentences(&mut self, embeddings: Vec<SentenceEmbedding>) -> Result<()> {
        self.add_sentences(embeddings)
    }

    fn find_knn(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(SentenceEmbedding, f32)>> {
        // Use MeanPoolEmbedding for encoding the query
        let model = crate::traits::embedding::MeanPoolEmbedding::new(self.dimension);
        let query_vec = model.embed_sentence(query)?;
        Ok(self.find_knn(&query_vec, k))
    }

    fn find_knn_entities(
        &self,
        query: &str,
        entity_type: Option<&str>,
        k: usize,
    ) -> Result<Vec<(EntityEmbedding, f32)>> {
        let model = crate::traits::embedding::MeanPoolEmbedding::new(self.dimension);
        let query_vec = model.embed_sentence(query)?;

        let results = self.entities.find_knn_linear(&query_vec, k);

        let filtered: Vec<_> = if let Some(et) = entity_type {
            results
                .into_iter()
                .filter_map(|(idx, sim)| {
                    self.entities.items.get(idx).filter(|emb| emb.entity_type == et)
                        .map(|emb| (emb.clone(), sim))
                })
                .collect()
        } else {
            results
                .into_iter()
                .filter_map(|(idx, sim)| {
                    self.entities.items.get(idx).map(|emb| (emb.clone(), sim))
                })
                .collect()
        };

        // Sort by similarity
        let mut sorted = filtered;
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(sorted.into_iter().take(k).collect())
    }

    fn build_index(&mut self) -> Result<()> {
        self.build_index()
    }

    fn save(&self, path: &str) -> Result<()> {
        self.save(path)
    }

    fn load(&mut self, path: &str) -> Result<()> {
        self.load(path)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn clear(&mut self) -> Result<()> {
        self.clear()
    }
}

impl Default for HNSWEmbeddingStore {
    fn default() -> Self {
        Self::new(768)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_store_new() {
        let store = HNSWEmbeddingStore::new(128);
        assert!(store.is_empty());
        assert_eq!(store.dimension, 128);
    }

    #[test]
    fn test_hnsw_store_add_sentence() {
        let mut store = HNSWEmbeddingStore::new(128);

        let emb = SentenceEmbedding {
            text: "test sentence".to_string(),
            vector: vec![0.0; 128],
            entities: vec![],
            entity_types: vec![],
        };

        store.add_sentence(emb).unwrap();
        assert_eq!(store.len_sentences(), 1);
    }

    #[test]
    fn test_hnsw_store_knn() {
        let mut store = HNSWEmbeddingStore::new(128);

        let emb1 = SentenceEmbedding {
            text: "apple pie".to_string(),
            vector: {
                let mut v = vec![0.0; 128];
                v[0] = 1.0;
                v
            },
            entities: vec![],
            entity_types: vec![],
        };

        let emb2 = SentenceEmbedding {
            text: "banana bread".to_string(),
            vector: {
                let mut v = vec![0.0; 128];
                v[1] = 1.0;
                v
            },
            entities: vec![],
            entity_types: vec![],
        };

        store.add_sentence(emb1).unwrap();
        store.add_sentence(emb2).unwrap();

        let query = {
            let mut v = vec![0.0; 128];
            v[0] = 1.0;
            v
        };

        let results = store.find_knn(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.text, "apple pie"); // Should match
    }

    #[test]
    fn test_hnsw_store_save_load() {
        let mut store = HNSWEmbeddingStore::new(128);

        let emb = SentenceEmbedding {
            text: "test".to_string(),
            vector: vec![0.0; 128],
            entities: vec![],
            entity_types: vec![],
        };

        store.add_sentence(emb.clone()).unwrap();

        // Save to temp file
        let temp_path = "/tmp/test_hnsw_store.bin";
        store.save(temp_path).unwrap();

        // Load into new store
        let mut store2 = HNSWEmbeddingStore::new(128);
        store2.load(temp_path).unwrap();

        assert_eq!(store2.len_sentences(), 1);
        assert_eq!(store2.sentences.items[0].text, "test");

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_hnsw_store_entity_embeddings() {
        let mut store = HNSWEmbeddingStore::new(128);

        let emb = EntityEmbedding {
            entity_text: "Apple".to_string(),
            entity_type: "ORG".to_string(),
            vector: {
                let mut v = vec![0.0; 128];
                v[0] = 1.0;
                v
            },
            context: "Apple Inc.".to_string(),
            span: (0, 5),
        };

        store.add_entity(emb).unwrap();
        assert_eq!(store.len_entities(), 1);
    }
}

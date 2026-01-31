//! Embedding traits for kNN retrieval

use crate::error::Result;
use crate::ner::Entity;
use serde::{Deserialize, Serialize};

/// Sentence embedding with metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SentenceEmbedding {
    /// Original text
    pub text: String,

    /// Embedding vector
    pub vector: Vec<f32>,

    /// Entities found in the sentence
    pub entities: Vec<Entity>,

    /// Entity types present
    pub entity_types: Vec<String>,
}

/// Entity-level embedding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EntityEmbedding {
    /// Entity text
    pub entity_text: String,

    /// Entity type
    pub entity_type: String,

    /// Embedding vector
    pub vector: Vec<f32>,

    /// Context sentence
    pub context: String,

    /// Span in context (start, end)
    pub span: (usize, usize),
}

/// Embedding model trait
pub trait EmbeddingModel: Send + Sync {
    /// Get single sentence embedding
    fn embed_sentence(&self, text: &str) -> Result<Vec<f32>>;

    /// Get multiple sentence embeddings
    fn embed_sentences(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_sentence(t)).collect()
    }

    /// Get entity embedding (entity + context)
    fn embed_entity(&self, text: &str, span: (usize, usize)) -> Result<Vec<f32>> {
        let _ = span;
        self.embed_sentence(text)
    }

    /// Get embedding dimension
    fn embedding_dim(&self) -> usize;

    /// Check if vector is valid
    fn is_valid_vector(&self, vec: &[f32]) -> bool {
        vec.len() == self.embedding_dim() && vec.iter().all(|f| f.is_finite())
    }

    /// Normalize vector (L2)
    fn normalize(&self, mut vec: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
        vec
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(crate::error::Error::Validation(format!(
                "Vector dimension mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm_a * norm_b))
        }
    }

    /// Compute Euclidean distance
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(crate::error::Error::Validation(format!(
                "Vector dimension mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        Ok(sum_sq.sqrt())
    }
}

/// Embedding store trait for kNN retrieval
pub trait EmbeddingStore: Send + Sync {
    /// Add a sentence embedding
    fn add_sentence(&mut self, embedding: SentenceEmbedding) -> Result<()>;

    /// Add multiple sentence embeddings
    fn add_sentences(&mut self, embeddings: Vec<SentenceEmbedding>) -> Result<()> {
        for emb in embeddings {
            self.add_sentence(emb)?;
        }
        Ok(())
    }

    /// Find k nearest neighbors by sentence
    fn find_knn(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(SentenceEmbedding, f32)>>;

    /// Find k nearest neighbors by entity
    fn find_knn_entities(
        &self,
        query: &str,
        entity_type: Option<&str>,
        k: usize,
    ) -> Result<Vec<(EntityEmbedding, f32)>>;

    /// Build index for efficient retrieval
    fn build_index(&mut self) -> Result<()> {
        // Default: no-op (linear scan)
        Ok(())
    }

    /// Save index to disk
    fn save(&self, path: &str) -> Result<()>;

    /// Load index from disk
    fn load(&mut self, path: &str) -> Result<()>;

    /// Get number of sentences in store
    fn len(&self) -> usize;

    /// Check if store is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all embeddings
    fn clear(&mut self) -> Result<()>;
}

/// Simple mean-pooled embedding model
#[derive(Debug, Clone)]
pub struct MeanPoolEmbedding {
    /// Embedding dimension
    dim: usize,
}

impl MeanPoolEmbedding {
    /// Create a new mean-pooled embedding model
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Create with default dimension (768)
    pub fn default_dim() -> Self {
        Self::new(768)
    }
}

impl EmbeddingModel for MeanPoolEmbedding {
    fn embed_sentence(&self, text: &str) -> Result<Vec<f32>> {
        // Simple character-based hashing for testing
        let mut result = vec![0.0; self.dim];
        let chars: Vec<char> = text.chars().collect();

        for (i, &c) in chars.iter().enumerate() {
            let idx = (c as usize) % self.dim;
            let position_weight = 1.0 / (1.0 + i as f32);
            result[idx] += position_weight;
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in result.iter_mut() {
                *v /= norm;
            }
        }

        Ok(result)
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// In-memory embedding store using linear scan
#[derive(Debug, Default)]
pub struct MemoryEmbeddingStore {
    /// Stored sentence embeddings
    sentences: Vec<SentenceEmbedding>,

    /// Stored entity embeddings
    entities: Vec<EntityEmbedding>,
}

impl MemoryEmbeddingStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            sentences: Vec::with_capacity(capacity),
            entities: Vec::new(),
        }
    }
}

impl EmbeddingStore for MemoryEmbeddingStore {
    fn add_sentence(&mut self, embedding: SentenceEmbedding) -> Result<()> {
        self.sentences.push(embedding);
        Ok(())
    }

    fn find_knn(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(SentenceEmbedding, f32)>> {
        let model = MeanPoolEmbedding::default_dim();
        let query_vec = model.embed_sentence(query)?;

        let mut similarities: Vec<_> = self
            .sentences
            .iter()
            .map(|emb| {
                let sim = model.cosine_similarity(&query_vec, &emb.vector).unwrap_or(0.0);
                (emb.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities.into_iter().take(k).collect())
    }

    fn find_knn_entities(
        &self,
        query: &str,
        entity_type: Option<&str>,
        k: usize,
    ) -> Result<Vec<(EntityEmbedding, f32)>> {
        let model = MeanPoolEmbedding::default_dim();
        let query_vec = model.embed_sentence(query)?;

        let filtered: Vec<_> = if let Some(et) = entity_type {
            self.entities.iter().filter(|e| e.entity_type == et).collect()
        } else {
            self.entities.iter().collect()
        };

        let mut similarities: Vec<(EntityEmbedding, f32)> = filtered
            .into_iter()
            .map(|emb| {
                let sim = model.cosine_similarity(&query_vec, &emb.vector).unwrap_or(0.0);
                (emb.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(similarities.into_iter().take(k).collect())
    }

    fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serde::encode_to_vec(&self.sentences, bincode::config::standard())?;
        std::fs::write(path, data)?;
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        let sentences: Vec<SentenceEmbedding> =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;
        self.sentences = sentences;
        Ok(())
    }

    fn len(&self) -> usize {
        self.sentences.len()
    }

    fn clear(&mut self) -> Result<()> {
        self.sentences.clear();
        self.entities.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_pool_embedding() {
        let model = MeanPoolEmbedding::default_dim();
        let vec = model.embed_sentence("hello world").unwrap();
        assert_eq!(vec.len(), 768);
        assert!(model.is_valid_vector(&vec));
    }

    #[test]
    fn test_cosine_similarity() {
        let model = MeanPoolEmbedding::default_dim();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert!((model.cosine_similarity(&a, &b).unwrap() - 1.0).abs() < 0.001);
        assert!((model.cosine_similarity(&a, &c).unwrap() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_store() {
        let mut store = MemoryEmbeddingStore::new();

        let emb1 = SentenceEmbedding {
            text: "hello world".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            entities: vec![],
            entity_types: vec![],
        };

        store.add_sentence(emb1.clone()).unwrap();

        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_knn_search() {
        let mut store = MemoryEmbeddingStore::new();

        let emb1 = SentenceEmbedding {
            text: "apple in new york".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            entities: vec![],
            entity_types: vec![],
        };

        let emb2 = SentenceEmbedding {
            text: "banana in paris".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            entities: vec![],
            entity_types: vec![],
        };

        store.add_sentences(vec![emb1, emb2]).unwrap();

        let results = store.find_knn("new york", 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

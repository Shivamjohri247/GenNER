//! Embedding store for kNN retrieval

use crate::error::Result;
use crate::ner::{Demonstration, Entity};
use crate::traits::embedding::{EmbeddingModel, EmbeddingStore, SentenceEmbedding};
use std::collections::HashMap;
use rand::seq::SliceRandom;

/// Demonstration store organized by entity type
#[derive(Clone, Debug)]
pub struct DemonstrationStore {
    /// Demonstrations by entity type
    by_type: HashMap<String, Vec<Demonstration>>,

    /// All demonstrations
    all: Vec<Demonstration>,
}

impl DemonstrationStore {
    /// Create a new demonstration store
    pub fn new() -> Self {
        Self {
            by_type: HashMap::new(),
            all: Vec::new(),
        }
    }

    /// Add a demonstration for an entity type
    pub fn add(&mut self, entity_type: &str, demo: Demonstration) {
        self.by_type
            .entry(entity_type.to_string())
            .or_insert_with(Vec::new)
            .push(demo.clone());
        self.all.push(demo);
    }

    /// Add multiple demonstrations
    pub fn add_all(&mut self, entity_type: &str, demos: Vec<Demonstration>) {
        for demo in demos {
            self.add(entity_type, demo);
        }
    }

    /// Get demonstrations for an entity type
    pub fn get(&self, entity_type: &str) -> Vec<Demonstration> {
        self.by_type
            .get(entity_type)
            .map(|v| v.clone())
            .unwrap_or_default()
    }

    /// Sample random demonstrations for an entity type
    pub fn sample(&self, entity_type: &str, n: usize) -> Vec<Demonstration> {
        let demos = self.get(entity_type);
        if demos.len() <= n {
            return demos;
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..demos.len()).collect();
        indices.shuffle(&mut rng);

        indices.into_iter().take(n).filter_map(|i| demos.get(i).cloned()).collect()
    }

    /// Get all demonstrations
    pub fn all(&self) -> &[Demonstration] {
        &self.all
    }

    /// Get all entity types
    pub fn entity_types(&self) -> Vec<String> {
        self.by_type.keys().cloned().collect()
    }

    /// Check if store is empty
    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// Get number of demonstrations
    pub fn len(&self) -> usize {
        self.all.len()
    }

    /// Clear all demonstrations
    pub fn clear(&mut self) {
        self.by_type.clear();
        self.all.clear();
    }
}

impl Default for DemonstrationStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Indexed demonstration store with embeddings
pub struct IndexedDemonstrationStore<E: EmbeddingModel> {
    /// Base demonstration store
    store: DemonstrationStore,

    /// Embedding model
    embedding_model: E,

    /// Embedded demonstrations by type
    embedded: HashMap<String, Vec<SentenceEmbedding>>,
}

impl<E: EmbeddingModel> IndexedDemonstrationStore<E> {
    /// Create a new indexed store
    pub fn new(embedding_model: E) -> Self {
        Self {
            store: DemonstrationStore::new(),
            embedding_model: embedding_model,
            embedded: HashMap::new(),
        }
    }

    /// Add a demonstration with embedding
    pub fn add(&mut self, entity_type: &str, demo: Demonstration) -> Result<()> {
        // Add to base store
        self.store.add(entity_type, demo.clone());

        // Generate embedding
        let embedding = self.embedding_model.embed_sentence(&demo.input)?;

        let sent_emb = SentenceEmbedding {
            text: demo.input.clone(),
            vector: embedding,
            entities: demo.entities.clone(),
            entity_types: vec![entity_type.to_string()],
        };

        self.embedded
            .entry(entity_type.to_string())
            .or_insert_with(Vec::new)
            .push(sent_emb);

        Ok(())
    }

    /// Find k nearest neighbors for a query
    pub fn find_knn(
        &self,
        entity_type: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<(Demonstration, f32)>> {
        let query_emb = self.embedding_model.embed_sentence(query)?;

        if let Some(embeddings) = self.embedded.get(entity_type) {
            let mut similarities: Vec<_> = embeddings
                .iter()
                .map(|emb| {
                    let sim = self
                        .embedding_model
                        .cosine_similarity(&query_emb, &emb.vector)
                        .unwrap_or(0.0);
                    (emb, sim)
                })
                .collect();

            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let demos = self.store.get(entity_type);
            let mut result = Vec::new();

            for (emb, sim) in similarities.into_iter().take(k) {
                if let Some(demo) = demos.iter().find(|d| d.input == emb.text) {
                    result.push((demo.clone(), sim));
                }
            }

            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    /// Get the base store
    pub fn store(&self) -> &DemonstrationStore {
        &self.store
    }

    /// Get mutable reference to base store
    pub fn store_mut(&mut self) -> &mut DemonstrationStore {
        &mut self.store
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::embedding::MeanPoolEmbedding;
    use crate::ner::Entity;

    #[test]
    fn test_demonstration_store_new() {
        let store = DemonstrationStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_demonstration_store_add() {
        let mut store = DemonstrationStore::new();

        let demo = Demonstration::new(
            "John left",
            "@@John## left",
            vec![Entity::new("John", "PER", 0, 4)],
        );

        store.add("PER", demo);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("PER").len(), 1);
    }

    #[test]
    fn test_demonstration_store_add_all() {
        let mut store = DemonstrationStore::new();

        let demos = vec![
            Demonstration::new("John left", "@@John## left", vec![]),
            Demonstration::new("Mary arrived", "@@Mary## arrived", vec![]),
        ];

        store.add_all("PER", demos);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_demonstration_store_sample() {
        let mut store = DemonstrationStore::new();

        for i in 0..10 {
            store.add(
                "PER",
                Demonstration::new(
                    format!("text {}", i),
                    format!("marked {}", i),
                    vec![],
                ),
            );
        }

        let sampled = store.sample("PER", 5);
        assert_eq!(sampled.len(), 5);
    }

    #[test]
    fn test_demonstration_store_entity_types() {
        let mut store = DemonstrationStore::new();

        store.add("PER", Demonstration::new("John", "@@John##", vec![]));
        store.add("LOC", Demonstration::new("Paris", "@@Paris##", vec![]));

        let types = store.entity_types();
        assert_eq!(types.len(), 2);
        assert!(types.contains(&"PER".to_string()));
        assert!(types.contains(&"LOC".to_string()));
    }

    #[test]
    fn test_indexed_store() {
        let embedding_model = MeanPoolEmbedding::default_dim();
        let mut store = IndexedDemonstrationStore::new(embedding_model);

        let demo = Demonstration::new(
            "John left",
            "@@John## left",
            vec![Entity::new("John", "PER", 0, 4)],
        );

        store.add("PER", demo).unwrap();
        assert_eq!(store.store().len(), 1);
    }

    #[test]
    fn test_indexed_store_find_knn() {
        let embedding_model = MeanPoolEmbedding::default_dim();
        let mut store = IndexedDemonstrationStore::new(embedding_model);

        store
            .add(
                "PER",
                Demonstration::new("John left", "@@John## left", vec![]),
            )
            .unwrap();
        store
            .add(
                "PER",
                Demonstration::new("Mary arrived", "@@Mary## arrived", vec![]),
            )
            .unwrap();

        let results = store.find_knn("PER", "John went home", 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

//! kNN retrieval strategies

use crate::error::Result;
use crate::ner::Demonstration;
use crate::traits::embedding::EmbeddingModel;
use crate::retrieval::store::DemonstrationStore;

/// kNN retrieval strategy
pub enum KNNStrategy {
    /// Random sampling
    Random,

    /// Sentence-level similarity
    SentenceSimilarity,

    /// Entity-level similarity
    EntitySimilarity,

    /// Hybrid (sentence + entity)
    Hybrid { sentence_weight: f32, entity_weight: f32 },
}

/// kNN retriever
pub struct KNNRetriever<E: EmbeddingModel> {
    embedding_model: E,
    strategy: KNNStrategy,
}

impl<E: EmbeddingModel> KNNRetriever<E> {
    /// Create a new kNN retriever
    pub fn new(embedding_model: E, strategy: KNNStrategy) -> Self {
        Self {
            embedding_model: embedding_model,
            strategy,
        }
    }

    /// Retrieve k demonstrations for a query
    pub fn retrieve(
        &self,
        store: &DemonstrationStore,
        entity_type: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<Demonstration>> {
        match self.strategy {
            KNNStrategy::Random => {
                let demos = store.get(entity_type);
                if demos.len() <= k {
                    return Ok(demos);
                }

                let mut indices: Vec<usize> = (0..demos.len()).collect();
                for i in (1..k).rev() {
                    let j = rand::random::<usize>() % (i + 1);
                    indices.swap(i, j);
                }

                Ok(indices
                    .into_iter()
                    .take(k)
                    .filter_map(|i| demos.get(i).cloned())
                    .collect())
            }
            KNNStrategy::SentenceSimilarity => {
                let query_emb = self.embedding_model.embed_sentence(query)?;

                let demos = store.get(entity_type);
                let mut scored: Vec<_> = demos
                    .iter()
                    .map(|demo| {
                        let score = self
                            .embedding_model
                            .embed_sentence(&demo.input)
                            .ok()
                            .and_then(|emb| {
                                self.embedding_model.cosine_similarity(&query_emb, &emb).ok()
                            })
                            .unwrap_or(0.0);
                        (demo.clone(), score)
                    })
                    .collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored.into_iter().take(k).map(|(d, _)| d).collect())
            }
            KNNStrategy::EntitySimilarity => {
                // For entity similarity, we'd need entity embeddings
                // For now, fall back to sentence similarity
                let query_emb = self.embedding_model.embed_sentence(query)?;

                let demos = store.get(entity_type);
                let mut scored: Vec<_> = demos
                    .iter()
                    .map(|demo| {
                        let score = self
                            .embedding_model
                            .embed_sentence(&demo.input)
                            .ok()
                            .and_then(|emb| {
                                self.embedding_model.cosine_similarity(&query_emb, &emb).ok()
                            })
                            .unwrap_or(0.0);
                        (demo.clone(), score)
                    })
                    .collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored.into_iter().take(k).map(|(d, _)| d).collect())
            }
            KNNStrategy::Hybrid {
                sentence_weight,
                entity_weight,
            } => {
                let query_emb = self.embedding_model.embed_sentence(query)?;

                let demos = store.get(entity_type);
                let mut scored: Vec<_> = demos
                    .iter()
                    .map(|demo| {
                        let sentence_score = self
                            .embedding_model
                            .embed_sentence(&demo.input)
                            .ok()
                            .and_then(|emb| {
                                self.embedding_model.cosine_similarity(&query_emb, &emb).ok()
                            })
                            .unwrap_or(0.0);

                        // Entity score would be computed from entity embeddings
                        let entity_score = if demo.entities.is_empty() {
                            0.0
                        } else {
                            // Simple heuristic: check if entities are mentioned
                            demo.entities.iter().any(|e| query.contains(&e.text)) as u8 as f32
                        };

                        let combined = sentence_weight * sentence_score + entity_weight * entity_score;
                        (demo.clone(), combined)
                    })
                    .collect();

                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored.into_iter().take(k).map(|(d, _)| d).collect())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ner::Entity;
    use crate::traits::embedding::MeanPoolEmbedding;

    #[test]
    fn test_knn_retriever_random() {
        let embedding_model = MeanPoolEmbedding::default_dim();
        let retriever = KNNRetriever::new(embedding_model, KNNStrategy::Random);

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

        let results = retriever.retrieve(&store, "PER", "query", 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_knn_retriever_sentence_similarity() {
        let embedding_model = MeanPoolEmbedding::default_dim();
        let retriever = KNNRetriever::new(embedding_model, KNNStrategy::SentenceSimilarity);

        let mut store = DemonstrationStore::new();
        store.add(
            "PER",
            Demonstration::new("John went home", "@@John## went home", vec![]),
        );
        store.add(
            "PER",
            Demonstration::new("Mary left early", "@@Mary## left early", vec![]),
        );

        let results = retriever.retrieve(&store, "PER", "John is here", 2).unwrap();
        assert_eq!(results.len(), 2);
    }
}

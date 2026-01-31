//! Retrieval strategies for demonstrations

use crate::error::Result;
use crate::ner::Demonstration;
use crate::traits::embedding::EmbeddingModel;

/// Retrieval strategy trait
pub trait RetrievalStrategy: Send + Sync {
    /// Retrieve k demonstrations for the given query
    fn retrieve(
        &self,
        demonstrations: &[Demonstration],
        query: &str,
        k: usize,
    ) -> Result<Vec<Demonstration>>;
}

/// Random retrieval strategy
#[derive(Debug, Clone, Copy)]
pub struct RandomRetrieval;

impl RetrievalStrategy for RandomRetrieval {
    fn retrieve(
        &self,
        demonstrations: &[Demonstration],
        _query: &str,
        k: usize,
    ) -> Result<Vec<Demonstration>> {
        if demonstrations.len() <= k {
            return Ok(demonstrations.to_vec());
        }

        let mut indices: Vec<usize> = (0..demonstrations.len()).collect();
        // Fisher-Yates shuffle for first k
        for i in (1..k).rev() {
            let j = rand::random::<usize>() % (i + 1);
            indices.swap(i, j);
        }

        Ok(indices
            .into_iter()
            .take(k)
            .filter_map(|i| demonstrations.get(i).cloned())
            .collect())
    }
}

/// Sequential retrieval (take first k)
#[derive(Debug, Clone, Copy)]
pub struct SequentialRetrieval;

impl RetrievalStrategy for SequentialRetrieval {
    fn retrieve(
        &self,
        demonstrations: &[Demonstration],
        _query: &str,
        k: usize,
    ) -> Result<Vec<Demonstration>> {
        Ok(demonstrations
            .iter()
            .take(k)
            .cloned()
            .collect())
    }
}

/// Embedding-based retrieval
pub struct EmbeddingRetrieval<E: EmbeddingModel> {
    embedding_model: E,
}

impl<E: EmbeddingModel> EmbeddingRetrieval<E> {
    pub fn new(embedding_model: E) -> Self {
        Self { embedding_model }
    }
}

impl<E: EmbeddingModel> RetrievalStrategy for EmbeddingRetrieval<E> {
    fn retrieve(
        &self,
        demonstrations: &[Demonstration],
        query: &str,
        k: usize,
    ) -> Result<Vec<Demonstration>> {
        let query_emb = self.embedding_model.embed_sentence(query)?;

        let mut scored: Vec<_> = demonstrations
            .iter()
            .map(|demo| {
                let score = self
                    .embedding_model
                    .embed_sentence(&demo.input)
                    .ok()
                    .and_then(|emb| self.embedding_model.cosine_similarity(&query_emb, &emb).ok())
                    .unwrap_or(0.0);
                (demo.clone(), score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(demo, _)| demo)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::embedding::MeanPoolEmbedding;

    fn create_demo(text: &str) -> Demonstration {
        Demonstration::new(text, text, vec![])
    }

    #[test]
    fn test_random_retrieval() {
        let strategy = RandomRetrieval;
        let demos = (0..10).map(|i| create_demo(&format!("text {}", i))).collect::<Vec<_>>();

        let results = strategy.retrieve(&demos, "query", 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_sequential_retrieval() {
        let strategy = SequentialRetrieval;
        let demos = (0..10).map(|i| create_demo(&format!("text {}", i))).collect::<Vec<_>>();

        let results = strategy.retrieve(&demos, "query", 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].input, "text 0");
    }
}

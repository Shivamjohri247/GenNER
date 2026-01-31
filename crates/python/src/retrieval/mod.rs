//! Python bindings for kNN retrieval

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use crate::convert::entity_to_dict;
use genner_core::retrieval::HNSWEmbeddingStore;
use genner_core::traits::embedding::{SentenceEmbedding, EntityEmbedding};

/// kNN Retriever for few-shot demonstration retrieval
#[pyclass(name = "Retriever")]
pub struct PyRetriever {
    store: HNSWEmbeddingStore,
    dimension: usize,
}

#[pymethods]
impl PyRetriever {
    /// Create a new retriever
    ///
    /// Args:
    ///     dimension: Embedding dimension (default: 768)
    #[new]
    #[pyo3(signature = (dimension=768))]
    pub fn new(dimension: usize) -> Self {
        Self {
            store: HNSWEmbeddingStore::new(dimension),
            dimension,
        }
    }

    /// Add a sentence with its embedding
    ///
    /// Args:
    ///     text: The sentence text
    ///     embedding: The embedding vector as list of floats
    ///     entities: Optional list of entities in the sentence
    pub fn add_sentence(
        &mut self,
        py: Python,
        text: String,
        embedding: &Bound<'_, PyList>,
        entities: Option<&Bound<'_, PyList>>,
    ) -> PyResult<()> {
        let vector: Vec<f32> = embedding.iter()
            .map(|v| v.extract::<f32>())
            .collect::<Result<Vec<_>, _>>()?;

        let mut rust_entities = Vec::new();
        let mut entity_types_set = std::collections::HashSet::new();

        if let Some(entities_list) = entities {
            for item in entities_list.iter() {
                let dict = item.downcast::<PyDict>()?;
                let entity_text: String = dict.get_item("text")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'text'"))?
                    .extract()?;
                let label: String = dict.get_item("label")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'label'"))?
                    .extract()?;
                let start: usize = dict.get_item("start")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'start'"))?
                    .extract()?;
                let end: usize = dict.get_item("end")?
                    .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'end'"))?
                    .extract()?;

                entity_types_set.insert(label.clone());
                use genner_core::ner::Entity;
                rust_entities.push(Entity::new(entity_text, label, start, end));
            }
        }

        let entity_types: Vec<String> = entity_types_set.into_iter().collect();

        let embedding = SentenceEmbedding {
            text,
            vector,
            entities: rust_entities,
            entity_types,
        };

        self.store.add_sentence(embedding)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Add multiple sentences at once
    ///
    /// Args:
    ///     items: List of dicts with keys: text, embedding, entities (optional)
    pub fn add_sentences_batch(&mut self, py: Python, items: &Bound<'_, PyList>) -> PyResult<usize> {
        let mut embeddings = Vec::new();

        for item in items.iter() {
            let dict = item.downcast::<PyDict>()?;
            let text: String = dict.get_item("text")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'text'"))?
                .extract()?;

            let embedding_value = dict.get_item("embedding")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'embedding'"))?;
            let embedding_list = embedding_value.downcast::<PyList>()?;

            let vector: Vec<f32> = embedding_list.iter()
                .map(|v| v.extract::<f32>())
                .collect::<Result<Vec<_>, _>>()?;

            let mut rust_entities = Vec::new();
            let mut entity_types_set = std::collections::HashSet::new();

            if let Some(entities) = dict.get_item("entities")? {
                let entities_list: &Bound<'_, PyList> = entities.downcast::<PyList>()?;
                for entity_item in entities_list.iter() {
                    let entity_dict = entity_item.downcast::<PyDict>()?;
                    let entity_text: String = entity_dict.get_item("text")?
                        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'text'"))?
                        .extract()?;
                    let label: String = entity_dict.get_item("label")?
                        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'label'"))?
                        .extract()?;
                    let start: usize = entity_dict.get_item("start")?
                        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'start'"))?
                        .extract()?;
                    let end: usize = entity_dict.get_item("end")?
                        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'end'"))?
                        .extract()?;

                    entity_types_set.insert(label.clone());
                    use genner_core::ner::Entity;
                    rust_entities.push(Entity::new(entity_text, label, start, end));
                }
            }

            let entity_types: Vec<String> = entity_types_set.into_iter().collect();

            embeddings.push(SentenceEmbedding {
                text,
                vector,
                entities: rust_entities,
                entity_types,
            });
        }

        self.store.add_sentences(embeddings)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(items.len())
    }

    /// Find k nearest neighbors by text query
    ///
    /// Args:
    ///     k: Number of neighbors to return
    ///
    /// Returns:
    ///     List of dicts with keys: text, similarity, entities
    pub fn find_knn(
        &self,
        py: Python,
        k: usize,
    ) -> PyResult<PyObject> {
        // For now, use a simple mock embedding
        // In production, this would use an actual embedding model
        let query_vec = vec![0.0f32; self.dimension];

        let results = self.store.find_knn(&query_vec, k);

        let list = PyList::empty(py);
        for (embedding, similarity) in results {
            let dict = PyDict::new(py);
            dict.set_item("text", embedding.text)?;
            dict.set_item("similarity", similarity)?;

            let entities_list = PyList::empty(py);
            for entity in &embedding.entities {
                let entity_dict = entity_to_dict(py, entity)?;
                entities_list.append(entity_dict)?;
            }
            dict.set_item("entities", entities_list)?;

            list.append(dict)?;
        }

        Ok(list.into())
    }

    /// Find k nearest entity neighbors
    ///
    /// Args:
    ///     k: Number of neighbors to return
    ///     entity_type: Optional entity type filter
    ///
    /// Returns:
    ///     List of entity dicts with similarity scores
    #[pyo3(signature = (k, entity_type=None))]
    pub fn find_knn_entities(
        &self,
        py: Python,
        k: usize,
        entity_type: Option<String>,
    ) -> PyResult<PyObject> {
        let query_vec = vec![0.0f32; self.dimension];

        let results = self.store.find_knn_entities(&query_vec, k);

        let list = PyList::empty(py);
        for (embedding, similarity) in results {
            // Filter by entity type if specified
            if let Some(ref et) = entity_type {
                if embedding.entity_type != *et {
                    continue;
                }
            }

            let dict = PyDict::new(py);
            dict.set_item("entity_text", embedding.entity_text)?;
            dict.set_item("entity_type", embedding.entity_type)?;
            dict.set_item("similarity", similarity)?;
            dict.set_item("context", embedding.context)?;
            dict.set_item("span", (embedding.span.0, embedding.span.1))?;

            list.append(dict)?;
        }

        Ok(list.into())
    }

    /// Build the HNSW index for efficient search
    ///
    /// This should be called after adding all embeddings.
    pub fn build_index(&mut self) -> PyResult<()> {
        self.store.build_index()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Save the retrieval index to disk
    ///
    /// Args:
    ///     path: Path to save the index
    pub fn save(&self, path: String) -> PyResult<()> {
        self.store.save(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Load a retrieval index from disk
    ///
    /// Args:
    ///     path: Path to load the index from
    pub fn load(&mut self, path: String) -> PyResult<()> {
        self.store.load(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Get the number of sentences in the store
    fn len_sentences(&self) -> usize {
        self.store.len_sentences()
    }

    /// Get the number of entities in the store
    fn len_entities(&self) -> usize {
        self.store.len_entities()
    }

    /// Get the total size
    fn __len__(&self) -> usize {
        self.store.len()
    }

    /// Check if the store is empty
    fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clear all embeddings
    pub fn clear(&mut self) -> PyResult<()> {
        self.store.clear()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retriever_new() {
        let retriever = PyRetriever::new(128);
        assert!(retriever.is_empty());
        assert_eq!(retriever.__len__(), 0);
    }

    #[test]
    fn test_retriever_dimension() {
        let retriever = PyRetriever::new(256);
        assert_eq!(retriever.len_sentences(), 0);
        assert_eq!(retriever.len_entities(), 0);
        assert!(retriever.is_empty());
    }

    #[test]
    fn test_retriever_clear() {
        let mut retriever = PyRetriever::new(128);
        // Clear should work even when empty
        assert!(retriever.clear().is_ok());
        assert!(retriever.is_empty());
    }
}

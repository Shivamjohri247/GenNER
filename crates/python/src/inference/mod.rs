//! Python bindings for inference

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use crate::convert::{entities_to_list, entity_to_dict};
use genner_core::ner::Entity;

/// Inference Runner for NER extraction
#[pyclass(name = "InferenceRunner")]
pub struct PyInferenceRunner {
    batch_size: usize,
    max_tokens: usize,
    temperature: f32,
    use_cache: bool,
}

#[pymethods]
impl PyInferenceRunner {
    /// Create a new inference runner
    ///
    /// Args:
    ///     batch_size: Batch size for inference (default: 8)
    ///     max_tokens: Maximum tokens to generate (default: 512)
    ///     temperature: Sampling temperature (default: 0.0)
    ///     use_cache: Whether to use KV-cache (default: true)
    #[new]
    #[pyo3(signature = (batch_size=8, max_tokens=512, temperature=0.0, use_cache=true))]
    pub fn new(
        batch_size: usize,
        max_tokens: usize,
        temperature: f32,
        use_cache: bool,
    ) -> Self {
        Self {
            batch_size,
            max_tokens,
            temperature: temperature.clamp(0.0, 2.0),
            use_cache,
        }
    }

    /// Get the batch size
    #[getter]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Set the batch size
    #[setter]
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }

    /// Get max tokens
    #[getter]
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Set max tokens
    #[setter]
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }

    /// Get temperature
    #[getter]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Set temperature
    #[setter]
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.clamp(0.0, 2.0);
    }

    /// Get whether cache is enabled
    #[getter]
    pub fn use_cache(&self) -> bool {
        self.use_cache
    }

    /// Set whether to use cache
    #[setter]
    pub fn set_use_cache(&mut self, use_cache: bool) {
        self.use_cache = use_cache;
    }

    /// Extract entities from text (placeholder for actual model inference)
    ///
    /// Args:
    ///     text: Input text
    ///     entity_type: Entity type to extract (e.g., "PER", "ORG", "LOC")
    ///
    /// Returns:
    ///     List of entity dicts with keys: text, label, start, end, confidence
    ///
    /// Example:
    ///     >>> runner = InferenceRunner()
    ///     >>> entities = runner.extract("Apple Inc. was founded by Steve Jobs.", "ORG")
    ///     >>> # Would return: [{"text": "Apple Inc.", "label": "ORG", "start": 0, "end": 10, "confidence": 0.95}]
    pub fn extract(&self, py: Python, text: String, entity_type: String) -> PyResult<PyObject> {
        // Placeholder - in production, this would:
        // 1. Build the prompt using PromptBuilder
        // 2. Tokenize the input
        // 3. Run model inference
        // 4. Parse the output
        // 5. Verify entities

        // For now, return empty list
        let list = PyList::empty(py);
        Ok(list.into())
    }

    /// Extract entities from multiple texts
    ///
    /// Args:
    ///     texts: List of input texts
    ///     entity_type: Entity type to extract
    ///
    /// Returns:
    ///     List of lists of entity dicts
    pub fn extract_batch(
        &self,
        py: Python,
        texts: Vec<String>,
        entity_type: String,
    ) -> PyResult<PyObject> {
        let result_list = PyList::empty(py);

        // Placeholder - process each text
        for _ in texts {
            let list = PyList::empty(py);
            result_list.append(list)?;
        }

        Ok(result_list.into())
    }

    /// Extract all entity types from text
    ///
    /// Args:
    ///     text: Input text
    ///     entity_types: List of entity types to extract
    ///
    /// Returns:
    ///     Dict mapping entity type to list of entities
    pub fn extract_all(
        &self,
        py: Python,
        text: String,
        entity_types: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let result_dict = PyDict::new(py);

        for et in entity_types.iter() {
            let entity_type: String = et.extract()?;
            let entities = self.extract(py, text.clone(), entity_type.clone())?;
            result_dict.set_item(entity_type, entities)?;
        }

        Ok(result_dict.into())
    }

    /// Extract entities from marked text
    ///
    /// This is useful when you have model output with entity markers.
    ///
    /// Args:
    ///     marked_text: Text with entity markers (e.g., "@@John## went to @@Paris##")
    ///     entity_types: List of entity types to parse
    ///
    /// Returns:
    ///     Dict mapping entity type to list of entities
    pub fn parse_marked_output(
        &self,
        py: Python,
        marked_text: String,
        entity_types: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        use genner_core::ner::EntityParser;

        let result_dict = PyDict::new(py);
        let parser = EntityParser::new("@@", "##");

        for et in entity_types.iter() {
            let entity_type: String = et.extract()?;
            match parser.parse(&marked_text, &entity_type) {
                Ok(entities) => {
                    let entities_list = entities_to_list(py, &entities)?;
                    result_dict.set_item(entity_type, entities_list)?;
                }
                Err(_) => {
                    // If parsing fails, set empty list
                    result_dict.set_item(entity_type, PyList::empty(py))?;
                }
            }
        }

        Ok(result_dict.into())
    }

    /// Get statistics about the inference runner
    ///
    /// Returns:
    ///     Dict with current configuration
    pub fn stats(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("batch_size", self.batch_size)?;
        dict.set_item("max_tokens", self.max_tokens)?;
        dict.set_item("temperature", self.temperature)?;
        dict.set_item("use_cache", self.use_cache)?;
        dict.set_item("cache_size_mb", 0)?; // Would be actual cache size
        Ok(dict.into())
    }

    /// Reset the inference state (clear cache, etc.)
    pub fn reset(&mut self) {
        // Reset any cached state
    }
}

impl Default for PyInferenceRunner {
    fn default() -> Self {
        Self::new(8, 512, 0.0, true)
    }
}

/// Create a simple prompt for NER extraction
#[pyfunction]
#[pyo3(signature = (text, entity_type, demonstrations=None))]
pub fn build_ner_prompt(
    py: Python,
    text: String,
    entity_type: String,
    demonstrations: Option<&Bound<'_, PyList>>,
) -> PyResult<String> {
    use genner_core::ner::PromptBuilder;

    let builder = PromptBuilder::new();

    // Collect demonstrations if provided
    let mut demos = Vec::new();
    if let Some(demo_list) = demonstrations {
        for item in demo_list.iter() {
            let pair: &Bound<'_, PyList> = item.downcast::<PyList>()
                .map_err(|_| pyo3::exceptions::PyTypeError::new_err(
                    "Demonstrations must be list of (input, output) tuples"
                ))?;

            if pair.len() != 2 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Each demonstration must be a (input, output) pair"
                ));
            }

            let input: String = pair.get_item(0)?.extract()?;
            let output: String = pair.get_item(1)?.extract()?;

            // Parse entities from output for the demonstration
            let parser = genner_core::ner::EntityParser::new("@@", "##");
            let entities = match parser.parse(&output, &entity_type) {
                Ok(e) => e,
                Err(_) => Vec::new(),
            };

            demos.push(genner_core::ner::Demonstration {
                input,
                output,
                entities,
            });
        }
    }

    builder.build_prompt(&text, &entity_type, &demos)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_runner_new() {
        let runner = PyInferenceRunner::new(16, 256, 0.5, true);
        assert_eq!(runner.batch_size(), 16);
        assert_eq!(runner.max_tokens(), 256);
        assert_eq!(runner.temperature(), 0.5);
        assert!(runner.use_cache());
    }

    #[test]
    fn test_inference_runner_setters() {
        let mut runner = PyInferenceRunner::default();
        runner.set_batch_size(32);
        runner.set_max_tokens(1024);
        runner.set_temperature(0.8);
        runner.set_use_cache(false);

        assert_eq!(runner.batch_size(), 32);
        assert_eq!(runner.max_tokens(), 1024);
        assert_eq!(runner.temperature(), 0.8);
        assert!(!runner.use_cache());
    }

    #[test]
    fn test_inference_runner_temperature_clamp() {
        let mut runner = PyInferenceRunner::default();
        runner.set_temperature(5.0); // Should be clamped to 2.0
        assert_eq!(runner.temperature(), 2.0);
    }

    #[test]
    fn test_inference_runner_max_tokens() {
        let runner = PyInferenceRunner::new(8, 1024, 0.5, true);
        assert_eq!(runner.max_tokens(), 1024);

        let mut runner = PyInferenceRunner::default();
        runner.set_max_tokens(2048);
        assert_eq!(runner.max_tokens(), 2048);
    }

    #[test]
    fn test_inference_runner_use_cache() {
        let runner = PyInferenceRunner::new(8, 512, 0.0, false);
        assert!(!runner.use_cache());

        let mut runner = PyInferenceRunner::default();
        runner.set_use_cache(false);
        assert!(!runner.use_cache());
    }

    #[test]
    fn test_inference_runner_reset() {
        let mut runner = PyInferenceRunner::default();
        runner.set_temperature(1.5);
        runner.reset();
        // Reset should clear any cached state (temp remains unchanged)
        assert_eq!(runner.temperature(), 1.5);
    }
}

//! Python bindings for NER extraction

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};
use crate::convert::{entities_to_list, entity_to_dict};
use genner_core::ner::{Entity, EntityParser, PromptBuilder, Demonstration};

/// NER Extractor
///
/// Provides entity marking, parsing, and text manipulation utilities
/// for the GPT-NER format (@@entity##).
#[pyclass(name = "Extractor")]
pub struct PyExtractor {
    entity_prefix: String,
    entity_suffix: String,
}

#[pymethods]
impl PyExtractor {
    /// Create a new extractor
    ///
    /// Args:
    ///     prefix: Entity marker prefix (default: "@@")
    ///     suffix: Entity marker suffix (default: "##")
    #[new]
    #[pyo3(signature = (prefix="@@", suffix="##"))]
    pub fn new(prefix: &str, suffix: &str) -> Self {
        Self {
            entity_prefix: prefix.to_string(),
            entity_suffix: suffix.to_string(),
        }
    }

    /// Mark entities in text using the entity markers
    ///
    /// Args:
    ///     text: The input text to mark entities in
    ///     entities: List of entity dicts with keys: text, label, start, end
    ///
    /// Returns:
    ///     Text with entities marked (e.g., "@@John## went to @@Paris##")
    ///
    /// Example:
    ///     >>> extractor = Extractor()
    ///     >>> entities = [{"text": "John", "label": "PER", "start": 0, "end": 4}]
    ///     >>> extractor.mark_entities("John went home", entities)
    ///     "@@John## went home"
    pub fn mark_entities(&self, py: Python, text: String, entities: &Bound<'_, PyList>) -> PyResult<String> {
        let mut rust_entities = Vec::new();
        for item in entities.iter() {
            let dict = item.downcast::<PyDict>()?;
            let text_item = dict.get_item("text")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Missing 'text' key")
            })?;
            let entity_text: String = text_item.extract()?;
            let label_item = dict.get_item("label")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Missing 'label' key")
            })?;
            let label: String = label_item.extract()?;
            let start_item = dict.get_item("start")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Missing 'start' key")
            })?;
            let start: usize = start_item.extract()?;
            let end_item = dict.get_item("end")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Missing 'end' key")
            })?;
            let end: usize = end_item.extract()?;

            // Validate entity matches text span
            if start < text.len() && end <= text.len() && start < end {
                let actual_text = &text[start..end];
                if actual_text == entity_text || entity_text.is_empty() {
                    rust_entities.push(Entity::new(actual_text, label, start, end));
                } else {
                    rust_entities.push(Entity::new(entity_text, label, start, end));
                }
            }
        }

        // Use PromptBuilder for marking
        let builder = PromptBuilder::new().with_markers(&self.entity_prefix, &self.entity_suffix);
        Ok(builder.mark_entities(&text, &rust_entities))
    }

    /// Parse entities from marked text
    ///
    /// Args:
    ///     marked_text: Text with entity markers (e.g., "@@John## went home")
    ///     label: The entity type label to assign (e.g., "PER")
    ///
    /// Returns:
    ///     List of entity dicts with keys: text, label, start, end, confidence
    ///
    /// Example:
    ///     >>> extractor = Extractor()
    ///     >>> extractor.parse_entities("@@John## went to @@Paris##", "PER")
    ///     [{"text": "John", "label": "PER", "start": 0, "end": 4, "confidence": 1.0}]
    pub fn parse_entities(&self, py: Python, marked_text: String, label: String) -> PyResult<PyObject> {
        let parser = EntityParser::new(&self.entity_prefix, &self.entity_suffix);
        match parser.parse(&marked_text, &label) {
            Ok(entities) => entities_to_list(py, &entities),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    /// Parse multiple entity types from marked text
    ///
    /// Args:
    ///     marked_text: Text with entity markers
    ///     labels: List of entity type labels to parse
    ///
    /// Returns:
    ///     Dict mapping label to list of entities
    ///
    /// Example:
    ///     >>> extractor = Extractor()
    ///     >>> result = extractor.parse_entities_multi("@@John## works at @@Google##", ["PER", "ORG"])
    ///     >>> result["PER"]
    ///     [{"text": "John", "label": "PER", ...}]
    pub fn parse_entities_multi(
        &self,
        py: Python,
        marked_text: String,
        labels: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let result_dict = PyDict::new(py);
        let parser = EntityParser::new(&self.entity_prefix, &self.entity_suffix);

        for label_item in labels.iter() {
            let label: String = label_item.extract()?;
            match parser.parse(&marked_text, &label) {
                Ok(entities) => {
                    let entities_list = entities_to_list(py, &entities)?;
                    result_dict.set_item(label, entities_list)?;
                }
                Err(_) => {
                    // If parsing fails for this label, set empty list
                    result_dict.set_item(label, PyList::empty(py))?;
                }
            }
        }

        Ok(result_dict.into())
    }

    /// Unmark text (remove entity markers)
    ///
    /// Args:
    ///     text: Marked text
    ///
    /// Returns:
    ///     Text with entity markers removed
    ///
    /// Example:
    ///     >>> extractor = Extractor()
    ///     >>> extractor.unmark("@@John## went home")
    ///     "John went home"
    pub fn unmark(&self, text: &str) -> String {
        text.replace(&self.entity_prefix, "").replace(&self.entity_suffix, "")
    }

    /// Build a prompt with demonstrations for NER
    ///
    /// Args:
    ///     entity_type: The entity type to extract (e.g., "PER")
    ///     demonstrations: List of (input, output) tuples
    ///     input_text: The input text to process
    ///
    /// Returns:
    ///     Formatted prompt string
    ///
    /// Example:
    ///     >>> extractor = Extractor()
    ///     >>> demos = [("John arrived", "@@John## arrived")]
    ///     >>> extractor.build_prompt("PER", demos, "Mary left")
    ///     "I am an excellent linguist...\\n\\nInput: John arrived\\nOutput: @@John## arrived\\n\\nInput: Mary left\\nOutput:"
    pub fn build_prompt(
        &self,
        py: Python,
        entity_type: String,
        demonstrations: &Bound<'_, PyList>,
        input_text: String,
    ) -> PyResult<String> {
        let builder = PromptBuilder::new()
            .with_markers(&self.entity_prefix, &self.entity_suffix);

        let mut rust_demos = Vec::new();
        for item in demonstrations.iter() {
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
            let parser = EntityParser::new(&self.entity_prefix, &self.entity_suffix);
            let entities = match parser.parse(&output, &entity_type) {
                Ok(e) => e,
                Err(_) => Vec::new(),
            };

            rust_demos.push(Demonstration {
                input,
                output,
                entities,
            });
        }

        builder.build_prompt(&input_text, &entity_type, &rust_demos)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Build a simple prompt without demonstrations
    ///
    /// Args:
    ///     entity_type: The entity type to extract
    ///     input_text: The input text to process
    ///
    /// Returns:
    ///     Formatted prompt string
    pub fn build_simple_prompt(
        &self,
        entity_type: String,
        input_text: String,
    ) -> PyResult<String> {
        let builder = PromptBuilder::new()
            .with_markers(&self.entity_prefix, &self.entity_suffix);

        builder.build_simple_prompt(&input_text, &entity_type)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get the entity prefix
    #[getter]
    pub fn prefix(&self) -> String {
        self.entity_prefix.clone()
    }

    /// Get the entity suffix
    #[getter]
    pub fn suffix(&self) -> String {
        self.entity_suffix.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_new() {
        let extractor = PyExtractor::new("@@", "##");
        assert_eq!(extractor.prefix(), "@@");
        assert_eq!(extractor.suffix(), "##");
    }

    #[test]
    fn test_unmark() {
        let extractor = PyExtractor::new("@@", "##");
        assert_eq!(extractor.unmark("@@John## went home"), "John went home");
    }

    #[test]
    fn test_extractor_custom_markers() {
        let extractor = PyExtractor::new("[[", "]]");
        assert_eq!(extractor.prefix(), "[[");
        assert_eq!(extractor.suffix(), "]]");
        assert_eq!(extractor.unmark("[[John]] went home"), "John went home");
    }

    #[test]
    fn test_extractor_build_simple_prompt() {
        let extractor = PyExtractor::new("@@", "##");
        let result = extractor.build_simple_prompt("PER".to_string(), "John went home".to_string()).unwrap();
        assert!(result.contains("PER"));
        assert!(result.contains("John went home"));
        assert!(result.contains("Input:"));
        assert!(result.contains("Output:"));
    }
}

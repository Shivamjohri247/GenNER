//! Python bindings for NER extraction

use pyo3::prelude::*;
use pyo3::types::PyList;
use crate::convert::{entities_to_list, entity_to_dict};

/// NER Extractor
#[pyclass(name = "Extractor")]
pub struct PyExtractor {
    entity_prefix: String,
    entity_suffix: String,
}

#[pymethods]
impl PyExtractor {
    /// Create a new extractor
    #[new]
    #[pyo3(signature = (prefix="@@@@", suffix="##"))]
    pub fn new(prefix: &str, suffix: &str) -> Self {
        Self {
            entity_prefix: prefix.to_string(),
            entity_suffix: suffix.to_string(),
        }
    }

    /// Mark entities in text
    pub fn mark_entities(&self, py: Python, text: String, entities: &Bound<'_, PyList>) -> PyResult<String> {
        use genner_core::ner::Entity;

        let mut rust_entities = Vec::new();
        for item in entities.iter() {
            let dict = item.downcast::<pyo3::types::PyDict>()?;
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
            rust_entities.push(Entity::new(entity_text, label, start, end));
        }

        // Simple marking implementation
        let mut result = text.clone();
        for entity in rust_entities.iter().rev() {
            if entity.start < result.len() && entity.end <= result.len() {
                let marked = format!("{}{}{}{}", self.entity_prefix, &result[entity.start..entity.end], self.entity_suffix, &result[entity.end..]);
                result = format!("{}{}", &result[..entity.start], marked);
            }
        }

        Ok(result)
    }

    /// Parse entities from marked text
    pub fn parse_entities(&self, py: Python, marked_text: String, label: String) -> PyResult<PyObject> {
        use genner_core::ner::EntityParser;

        let parser = EntityParser::new(&self.entity_prefix, &self.entity_suffix);
        match parser.parse(&marked_text, &label) {
            Ok(entities) => entities_to_list(py, &entities),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    /// Unmark text (remove entity markers)
    pub fn unmark(&self, text: String) -> String {
        text.replace(&self.entity_prefix, "").replace(&self.entity_suffix, "")
    }
}

//! Type conversions between Rust and Python

use pyo3::{prelude::*, types::PyList};
use genner_core::ner::Entity;

/// Convert a Rust entity to a Python dict
pub fn entity_to_dict(py: Python, entity: &Entity) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("text", &entity.text)?;
    dict.set_item("label", &entity.label)?;
    dict.set_item("start", entity.start)?;
    dict.set_item("end", entity.end)?;
    dict.set_item("confidence", entity.confidence)?;
    Ok(dict.into())
}

/// Convert Rust entities to a Python list of dicts
pub fn entities_to_list(py: Python, entities: &[Entity]) -> PyResult<PyObject> {
    let list = pyo3::types::PyList::empty(py);
    for entity in entities {
        let dict = entity_to_dict(py, entity)?;
        list.append(dict)?;
    }
    Ok(list.into())
}

/// Parse entity types from Python
pub fn parse_entity_types<'a>(
    py: Python<'a>,
    value: &Bound<'a, PyAny>,
) -> PyResult<Vec<String>> {
    if let Ok(s) = value.extract::<String>() {
        Ok(vec![s])
    } else if let Ok(list) = value.downcast::<PyList>() {
        list.iter()
            .map(|item| item.extract::<String>())
            .collect()
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "entity_types must be a string or list of strings",
        ))
    }
}

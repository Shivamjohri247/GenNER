//! Python bindings for inference

use pyo3::prelude::*;

/// Inference Runner
#[pyclass(name = "InferenceRunner")]
pub struct PyInferenceRunner {
    batch_size: usize,
}

#[pymethods]
impl PyInferenceRunner {
    #[new]
    #[pyo3(signature = (batch_size=8))]
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Run inference on a single text
    pub fn extract(&self, py: Python, text: String, entity_type: String) -> PyResult<PyObject> {
        // Placeholder - would call actual model
        let list = pyo3::types::PyList::empty_bound(py);
        Ok(list.into())
    }

    /// Run inference on multiple texts
    pub fn extract_batch(&self, py: Python, texts: Vec<String>, entity_type: String) -> PyResult<PyObject> {
        // Placeholder - would process in batches
        let list = pyo3::types::PyList::empty_bound(py);
        Ok(list.into())
    }

    #[getter]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    #[setter]
    pub fn set_batch_size(&mut self, size: usize) {
        self.batch_size = size;
    }
}

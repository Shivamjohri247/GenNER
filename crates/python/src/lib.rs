//! PyO3 Python bindings for GenNER

use pyo3::prelude::*;

mod error;
mod convert;
pub mod models;
pub mod ner;
pub mod training;
pub mod inference;

use error::PythonError;

/// GenNER: Generic Named Entity Recognition with SLM Fine-tuning
#[pymodule]
fn genner_python(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<models::PyModel>()?;
    m.add_class::<ner::PyExtractor>()?;
    m.add_class::<training::PyTrainer>()?;
    m.add_class::<training::PyLoRAConfig>()?;
    m.add_class::<training::PyTrainingConfig>()?;
    m.add_class::<inference::PyInferenceRunner>()?;

    m.add("GennerError", _py.get_type::<PythonError>())?;

    Ok(())
}

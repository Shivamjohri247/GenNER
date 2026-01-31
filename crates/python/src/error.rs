//! Python error types

use pyo3::{create_exception, exceptions::PyRuntimeError, PyErr};

create_exception! {
    genner_python,  // module name
    PythonError,     // exception name
    PyRuntimeError   // base class
}

/// Convert a Rust error to a Python exception
pub fn to_py_err(err: genner_core::Error) -> PyErr {
    PythonError::new_err(err.to_string())
}

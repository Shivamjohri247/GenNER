//! GenNER Core Library
//!
//! This library provides the core functionality for the GenNER NER system,
//! including trait definitions, error handling, and core data structures.

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod error;
pub mod traits;

pub mod ner;
pub mod retrieval;
pub mod training;

pub use error::{Error, Result};
pub use ner::*;
pub use traits::*;
pub use training::*;

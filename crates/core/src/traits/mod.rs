//! Core trait definitions for GenNER

pub mod model;
pub mod tokenizer;
pub mod embedding;
pub mod quantization;

pub use model::*;
pub use tokenizer::*;
pub use embedding::*;
pub use quantization::*;

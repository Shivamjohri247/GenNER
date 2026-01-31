//! Model implementations for GenNER

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod registry;
pub mod base;
pub mod tokenizer;
pub mod candle_model;
pub mod qwen;
pub mod llama;
pub mod lora_layer;
pub mod training_optimizations;
pub mod loss;
pub mod lr_schedule;
pub mod training_step;

// Re-exports
pub use registry::*;
pub use tokenizer::HFTokenizerWrapper;
pub use candle_model::{CandleModel, CandleModelConfig};
pub use qwen::Qwen2;
pub use llama::Llama;
pub use lora_layer::{LoraLayer, LoraLinear, LoraOptimizer};
pub use training_optimizations::{
    PackedBatch,
    CheckpointConfig,
    CheckpointedActivations,
    fused_lora_forward,
    AdamW8bit,
    TrainingOptimizations,
};
pub use loss::{
    cross_entropy,
    cross_entropy_per_token,
    cross_entropy_ignore_index,
    cross_entropy_label_smoothing,
    log_softmax,
};
pub use lr_schedule::{
    LrSchedule,
    ConstantLr,
    WarmupLr,
    CosineAnnealingLr,
    LinearDecayLr,
    ExponentialDecayLr,
    InvSqrtLr,
    PolynomialDecayLr,
    StepLr,
    LrScheduleBuilder,
};
pub use training_step::{
    TrainingState,
    TrainingStepConfig,
    TrainingStep,
};

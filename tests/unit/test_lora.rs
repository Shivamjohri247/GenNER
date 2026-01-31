//! Unit tests for LoRA configuration

use genner_core::training::lora::{LoRAConfig, LoRAAdapter, AdapterMetadata, FusionStrategy};

#[test]
fn test_lora_config_new() {
    let config = LoRAConfig::new(8, 16.0);
    assert_eq!(config.rank, 8);
    assert_eq!(config.alpha, 16.0);
    assert_eq!(config.scaling, 2.0);
}

#[test]
fn test_lora_config_builder() {
    let config = LoRAConfig::default()
        .with_rank(32)
        .with_alpha(64.0)
        .with_dropout(0.1);

    assert_eq!(config.rank, 32);
    assert_eq!(config.alpha, 64.0);
    assert_eq!(config.dropout, 0.1);
}

#[test]
fn test_lora_adapter_new() {
    let adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
    assert_eq!(adapter.name, "test");
    assert_eq!(adapter.task_name, "ner_task");
}

#[test]
fn test_lora_adapter_add_module() {
    let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
    adapter.add_module("q_proj", vec![0.0; 100], vec![0.0; 200]);

    assert!(adapter.lora_a.contains_key("q_proj"));
    assert!(adapter.lora_b.contains_key("q_proj"));
}

#[test]
fn test_lora_adapter_validate() {
    let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::new(16, 32.0));
    adapter.add_module("q_proj", vec![0.0; 1600], vec![0.0; 3200]);

    assert!(adapter.validate().is_ok());
}

#[test]
fn test_lora_adapter_validate_missing_b() {
    let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
    adapter.lora_a.insert("q_proj".to_string(), vec![0.0; 100]);

    assert!(adapter.validate().is_err());
}

#[test]
fn test_adapter_metadata_default() {
    let metadata = AdapterMetadata::default();
    assert_eq!(metadata.training_samples, 0);
    assert_eq!(metadata.epochs, 0);
}

#[test]
fn test_adapter_num_parameters() {
    let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
    adapter.add_module("q_proj", vec![0.0; 100], vec![0.0; 200]);

    assert_eq!(adapter.num_parameters(), 300);
}

#[test]
fn test_adapter_size_bytes() {
    let mut adapter = LoRAAdapter::new("test", "ner_task", LoRAConfig::default());
    adapter.add_module("q_proj", vec![0.0; 100], vec![0.0; 200]);

    assert_eq!(adapter.size_bytes(), 300 * 4); // 300 f32 values * 4 bytes
}

//! Unit tests for prompt building

use genner_core::ner::{PromptBuilder, Demonstration, Entity};

#[test]
fn test_prompt_builder_default() {
    let builder = PromptBuilder::new();
    assert_eq!(builder.entity_prefix, "@@");
    assert_eq!(builder.entity_suffix, "##");
}

#[test]
fn test_build_task_description() {
    let builder = PromptBuilder::new();
    let desc = builder.build_task_description("PER");
    assert!(desc.contains("PER"));
    assert!(desc.contains("linguist"));
}

#[test]
fn test_build_simple_prompt() {
    let builder = PromptBuilder::new();
    let prompt = builder.build_simple_prompt("John went home", "PER").unwrap();
    assert!(prompt.contains("John went home"));
    assert!(prompt.contains("Output:"));
}

#[test]
fn test_mark_entities() {
    let builder = PromptBuilder::new();
    let entities = vec![
        Entity::new("John", "PER", 0, 4),
        Entity::new("Paris", "LOC", 12, 17),
    ];
    let marked = builder.mark_entities("John went to Paris", &entities);
    assert_eq!(marked, "@@John## went to @@Paris##");
}

#[test]
fn test_parse_marked_text() {
    let builder = PromptBuilder::new();
    let text = "@@John## went to @@Paris##";
    let entities = builder.parse_marked_text(text);
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].0, "John");
    assert_eq!(entities[1].0, "Paris");
}

#[test]
fn test_demonstration_format() {
    let demo = Demonstration::new(
        "John went home",
        "@@John## went home",
        vec![Entity::new("John", "PER", 0, 4)],
    );
    let formatted = demo.format();
    assert!(formatted.contains("Input:"));
    assert!(formatted.contains("Output:"));
}

#[test]
fn test_demonstration_from_entities() {
    let entities = vec![
        Entity::new("John", "PER", 0, 4),
        Entity::new("Paris", "LOC", 12, 17),
    ];
    let demo = Demonstration::from_entities("John went to Paris", entities);
    assert!(demo.output.contains("@@John##"));
    assert!(demo.output.contains("@@Paris##"));
}

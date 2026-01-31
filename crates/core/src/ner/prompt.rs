//! Prompt construction for GPT-NER

use crate::error::Result;
use crate::ner::Entity;
use serde::{Deserialize, Serialize};

/// Demonstration example for few-shot prompting
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Demonstration {
    /// Input sentence
    pub input: String,

    /// Output with marked entities
    pub output: String,

    /// Entities in this example
    pub entities: Vec<Entity>,
}

impl Demonstration {
    /// Create a new demonstration
    pub fn new(
        input: impl Into<String>,
        output: impl Into<String>,
        entities: Vec<Entity>,
    ) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            entities,
        }
    }

    /// Create from unmarked text and entities
    pub fn from_entities(text: impl Into<String>, entities: Vec<Entity>) -> Self {
        let text_str = text.into();
        let output = Self::mark_entities_in_text(&text_str, &entities, "@@", "##");
        Self {
            input: text_str.clone(),
            output,
            entities,
        }
    }

    /// Mark entities in text
    fn mark_entities_in_text(text: &str, entities: &[Entity], prefix: &str, suffix: &str) -> String {
        if entities.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();
        let mut sorted_entities = entities.to_vec();
        sorted_entities.sort_by_key(|e| std::cmp::Reverse(e.start));

        for entity in &sorted_entities {
            // When processing right to left, positions in result are still valid
            // because we haven't modified the left side yet
            if entity.start < result.len() && entity.end <= result.len() {
                let entity_text = &result[entity.start..entity.end];
                let marked = format!("{}{}{}", prefix, entity_text, suffix);
                result = format!("{}{}{}", &result[..entity.start], marked, &result[entity.end..]);
            }
        }

        result
    }

    /// Format as string for prompt
    pub fn format(&self) -> String {
        format!("Input: {}\nOutput: {}", self.input, self.output)
    }
}

/// Prompt builder for GPT-NER
#[derive(Clone, Debug)]
pub struct PromptBuilder {
    /// Entity prefix marker
    entity_prefix: String,

    /// Entity suffix marker
    entity_suffix: String,

    /// Task description template
    task_template: String,

    /// Number of demonstrations to include
    num_demonstrations: usize,
}

impl PromptBuilder {
    /// Create a new prompt builder
    pub fn new() -> Self {
        Self {
            entity_prefix: "@@".to_string(),
            entity_suffix: "##".to_string(),
            task_template: "I am an excellent linguist. The task is to label {entity_type} entities in the given sentence. Below are some examples.".to_string(),
            num_demonstrations: 4,
        }
    }

    /// Set entity markers
    pub fn with_markers(mut self, prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        self.entity_prefix = prefix.into();
        self.entity_suffix = suffix.into();
        self
    }

    /// Set task template
    pub fn with_task_template(mut self, template: impl Into<String>) -> Self {
        self.task_template = template.into();
        self
    }

    /// Set number of demonstrations
    pub fn with_demonstrations(mut self, num: usize) -> Self {
        self.num_demonstrations = num;
        self
    }

    /// Build task description for entity type
    pub fn build_task_description(&self, entity_type: &str) -> String {
        self.task_template.replace("{entity_type}", entity_type)
    }

    /// Build prompt with demonstrations
    pub fn build_prompt(
        &self,
        input: &str,
        entity_type: &str,
        demonstrations: &[Demonstration],
    ) -> Result<String> {
        let mut prompt = String::new();

        // Add task description
        prompt.push_str(&self.build_task_description(entity_type));
        prompt.push_str("\n\n");

        // Add demonstrations
        for demo in demonstrations.iter().take(self.num_demonstrations) {
            prompt.push_str(&demo.format());
            prompt.push_str("\n\n");
        }

        // Add input
        prompt.push_str("Input: ");
        prompt.push_str(input);
        prompt.push_str("\nOutput:");

        Ok(prompt)
    }

    /// Build prompt without demonstrations
    pub fn build_simple_prompt(&self, input: &str, entity_type: &str) -> Result<String> {
        let mut prompt = String::new();

        prompt.push_str(&self.build_task_description(entity_type));
        prompt.push_str("\n\n");
        prompt.push_str("Input: ");
        prompt.push_str(input);
        prompt.push_str("\nOutput:");

        Ok(prompt)
    }

    /// Build verification prompt
    pub fn build_verification_prompt(
        &self,
        context: &str,
        entity: &str,
        entity_type: &str,
    ) -> Result<String> {
        Ok(format!(
            "Given the sentence: '{}'\nIs '{}' a {} entity? Answer yes or no.",
            context, entity, entity_type
        ))
    }

    /// Mark entities in text
    pub fn mark_entities(&self, text: &str, entities: &[Entity]) -> String {
        Demonstration::mark_entities_in_text(text, entities, &self.entity_prefix, &self.entity_suffix)
    }

    /// Parse marked entities from text
    pub fn parse_marked_text(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut entities = Vec::new();
        let prefix = &self.entity_prefix;
        let suffix = &self.entity_suffix;

        let mut pos = 0;
        while pos < text.len() {
            if let Some(start) = text[pos..].find(prefix) {
                let actual_start = pos + start;
                let after_prefix = actual_start + prefix.len();

                if let Some(end_idx) = text[after_prefix..].find(suffix) {
                    let actual_end = after_prefix + end_idx;
                    let entity_text = &text[after_prefix..actual_end];

                    // Calculate original position by skipping prefix
                    let original_start = actual_start;
                    let original_end = actual_end;

                    entities.push((entity_text.to_string(), original_start, original_end));
                    pos = actual_end + suffix.len();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        entities
    }
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demonstration_new() {
        let demo = Demonstration::new(
            "John went home",
            "@@John## went home",
            vec![Entity::new("John", "PER", 0, 4)],
        );
        assert_eq!(demo.input, "John went home");
        assert_eq!(demo.output, "@@John## went home");
    }

    #[test]
    fn test_demonstration_from_entities() {
        let demo = Demonstration::from_entities(
            "John went to Paris",
            vec![
                Entity::new("John", "PER", 0, 4),
                Entity::new("Paris", "LOC", 13, 18),  // Paris is at position 13-18
            ],
        );
        assert_eq!(demo.input, "John went to Paris");
        assert!(demo.output.contains("@@John##"));
        assert!(demo.output.contains("@@Paris##"));
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
    fn test_prompt_builder_new() {
        let builder = PromptBuilder::new();
        assert_eq!(builder.entity_prefix, "@@");
        assert_eq!(builder.entity_suffix, "##");
    }

    #[test]
    fn test_prompt_builder_with_markers() {
        let builder = PromptBuilder::new().with_markers("<<", ">>");
        assert_eq!(builder.entity_prefix, "<<");
        assert_eq!(builder.entity_suffix, ">>");
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
    fn test_build_prompt_with_demonstrations() {
        let builder = PromptBuilder::new().with_demonstrations(2);
        let demos = vec![
            Demonstration::new("Mary left", "@@Mary## left", vec![]),
            Demonstration::new("Bob arrived", "@@Bob## arrived", vec![]),
        ];
        let prompt = builder.build_prompt("John went home", "PER", &demos).unwrap();
        assert!(prompt.contains("Mary left"));
        assert!(prompt.contains("Bob arrived"));
        assert!(prompt.contains("John went home"));
    }

    #[test]
    fn test_mark_entities() {
        let builder = PromptBuilder::new();
        let entities = vec![
            Entity::new("John", "PER", 0, 4),
            Entity::new("Paris", "LOC", 13, 18),  // Paris is at position 13-18
        ];
        let marked = builder.mark_entities("John went to Paris", &entities);
        assert_eq!(marked, "@@John## went to @@Paris##");
    }

    #[test]
    fn test_mark_entities_empty() {
        let builder = PromptBuilder::new();
        let marked = builder.mark_entities("No entities here", &[]);
        assert_eq!(marked, "No entities here");
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
    fn test_build_verification_prompt() {
        let builder = PromptBuilder::new();
        let prompt = builder
            .build_verification_prompt("John is here", "John", "PER")
            .unwrap();
        assert!(prompt.contains("John is here"));
        assert!(prompt.contains("PER"));
        assert!(prompt.contains("yes or no"));
    }
}

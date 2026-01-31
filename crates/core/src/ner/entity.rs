//! Entity types and structures

use serde::{Deserialize, Serialize};

/// A named entity
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    /// The entity text
    pub text: String,

    /// Entity type label
    pub label: String,

    /// Start position in original text
    pub start: usize,

    /// End position in original text
    pub end: usize,

    /// Confidence score (0-1)
    pub confidence: f32,
}

impl Entity {
    /// Create a new entity
    pub fn new(text: impl Into<String>, label: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            label: label.into(),
            start,
            end,
            confidence: 1.0,
        }
    }

    /// Create with confidence
    pub fn with_confidence(
        text: impl Into<String>,
        label: impl Into<String>,
        start: usize,
        end: usize,
        confidence: f32,
    ) -> Self {
        Self {
            text: text.into(),
            label: label.into(),
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Get the span as a tuple
    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }

    /// Get the length of the entity text
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if entity is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end || self.text.is_empty()
    }

    /// Validate entity against text
    pub fn validate(&self, text: &str) -> bool {
        if self.start >= self.end || self.end > text.len() {
            return false;
        }
        text[self.start..self.end] == self.text
    }

    /// Create a copy with new confidence
    pub fn with_confidence_value(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Entity type definition
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityType {
    /// Entity type name (e.g., "PER", "LOC")
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Example entities
    pub examples: Vec<String>,
}

impl EntityType {
    /// Create a new entity type
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        examples: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            examples,
        }
    }

    /// Create with empty examples
    pub fn simple(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            examples: Vec::new(),
        }
    }
}

/// NER task specification
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NERTask {
    /// Task name
    pub name: String,

    /// Entity types to extract
    pub entity_types: Vec<EntityType>,

    /// Task description for prompting
    pub description: String,
}

impl NERTask {
    /// Create a new NER task
    pub fn new(
        name: impl Into<String>,
        entity_types: Vec<EntityType>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            entity_types,
            description: description.into(),
        }
    }

    /// Get entity type names
    pub fn entity_type_names(&self) -> Vec<String> {
        self.entity_types.iter().map(|e| e.name.clone()).collect()
    }

    /// Find entity type by name
    pub fn find_entity_type(&self, name: &str) -> Option<&EntityType> {
        self.entity_types.iter().find(|e| e.name == name)
    }
}

/// Standard CoNLL-2003 entity types
pub fn conll2003_entity_types() -> Vec<EntityType> {
    vec![
        EntityType::new(
            "PER",
            "Named persons including fictional characters",
            vec!["John Smith".to_string(), "Mary".to_string(), "Dr. Johnson".to_string()],
        ),
        EntityType::new(
            "LOC",
            "Named locations including cities, countries, regions",
            vec!["New York".to_string(), "France".to_string(), "California".to_string()],
        ),
        EntityType::new(
            "ORG",
            "Organizations, companies, institutions",
            vec!["Google".to_string(), "United Nations".to_string(), "MIT".to_string()],
        ),
        EntityType::new(
            "MISC",
            "Miscellaneous entities including events, nationalities, products",
            vec!["World Cup".to_string(), "American".to_string(), "iPhone".to_string()],
        ),
    ]
}

/// Standard biomedical entity types
pub fn biomedical_entity_types() -> Vec<EntityType> {
    vec![
        EntityType::new(
            "DISEASE",
            "Diseases, disorders, symptoms",
            vec!["diabetes".to_string(), "Alzheimer's".to_string(), "COVID-19".to_string()],
        ),
        EntityType::new(
            "DRUG",
            "Drugs, medications, treatments",
            vec!["aspirin".to_string(), "insulin".to_string(), "Paxlovid".to_string()],
        ),
        EntityType::new(
            "GENE",
            "Genes, proteins",
            vec!["TP53".to_string(), "BRCA1".to_string(), "hemoglobin".to_string()],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_new() {
        let entity = Entity::new("John", "PER", 0, 4);
        assert_eq!(entity.text, "John");
        assert_eq!(entity.label, "PER");
        assert_eq!(entity.start, 0);
        assert_eq!(entity.end, 4);
        assert_eq!(entity.confidence, 1.0);
    }

    #[test]
    fn test_entity_validate() {
        let entity = Entity::new("John", "PER", 0, 4);
        assert!(entity.validate("John went home"));

        let invalid = Entity::new("John", "PER", 0, 10);
        assert!(!invalid.validate("John"));
    }

    #[test]
    fn test_entity_confidence_clamp() {
        let entity = Entity::with_confidence("John", "PER", 0, 4, 1.5);
        assert_eq!(entity.confidence, 1.0);

        let entity2 = Entity::with_confidence("John", "PER", 0, 4, -0.5);
        assert_eq!(entity2.confidence, 0.0);
    }

    #[test]
    fn test_entity_type() {
        let et = EntityType::new(
            "PER",
            "Person entity",
            vec!["John".to_string(), "Mary".to_string()],
        );
        assert_eq!(et.name, "PER");
        assert_eq!(et.examples.len(), 2);
    }

    #[test]
    fn test_ner_task() {
        let task = NERTask::new(
            "conll2003",
            conll2003_entity_types(),
            "Extract named entities",
        );
        assert_eq!(task.name, "conll2003");
        assert_eq!(task.entity_type_names(), vec!["PER", "LOC", "ORG", "MISC"]);
    }

    #[test]
    fn test_conll2003_entity_types() {
        let types = conll2003_entity_types();
        assert_eq!(types.len(), 4);
        assert_eq!(types[0].name, "PER");
        assert_eq!(types[1].name, "LOC");
    }

    #[test]
    fn test_biomedical_entity_types() {
        let types = biomedical_entity_types();
        assert_eq!(types.len(), 3);
        assert_eq!(types[0].name, "DISEASE");
    }
}

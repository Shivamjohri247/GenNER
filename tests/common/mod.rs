//! Common test utilities

use genner_core::ner::{Entity, EntityType, Demonstration};

/// Create a test entity
pub fn test_entity() -> Entity {
    Entity::new("John", "PER", 0, 4)
}

/// Create test entities
pub fn test_entities() -> Vec<Entity> {
    vec![
        Entity::new("John", "PER", 0, 4),
        Entity::new("Paris", "LOC", 14, 19),
    ]
}

/// Create a test demonstration
pub fn test_demonstration() -> Demonstration {
    Demonstration::new(
        "John went to Paris",
        "@@John## went to @@Paris##",
        test_entities(),
    )
}

/// Create standard entity types
pub fn standard_entity_types() -> Vec<EntityType> {
    vec![
        EntityType::new("PER", "Person", vec!["John".to_string(), "Mary".to_string()]),
        EntityType::new("LOC", "Location", vec!["Paris".to_string(), "London".to_string()]),
        EntityType::new("ORG", "Organization", vec!["Google".to_string()]),
    ]
}

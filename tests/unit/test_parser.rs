//! Unit tests for entity parsing

use genner_core::ner::EntityParser;

#[test]
fn test_parser_new() {
    let parser = EntityParser::new("@@", "##");
    assert_eq!(parser.prefix, "@@");
    assert_eq!(parser.suffix, "##");
}

#[test]
fn test_parse_single_entity() {
    let parser = EntityParser::default_markers();
    let entities = parser.parse("@@John## went home", "PER").unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "John");
    assert_eq!(entities[0].label, "PER");
    assert_eq!(entities[0].start, 0);
    assert_eq!(entities[0].end, 4);
}

#[test]
fn test_parse_multiple_entities() {
    let parser = EntityParser::default_markers();
    let entities = parser.parse("@@John## went to @@Paris##", "MIXED").unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(entities[0].text, "John");
    assert_eq!(entities[1].text, "Paris");
}

#[test]
fn test_parse_no_entities() {
    let parser = EntityParser::default_markers();
    let entities = parser.parse("Just plain text", "PER").unwrap();
    assert_eq!(entities.len(), 0);
}

#[test]
fn test_parse_unmatched_markers() {
    let parser = EntityParser::default_markers();
    let result = parser.parse("@@John went home", "PER");
    assert!(result.is_err());
}

#[test]
fn test_unmark_text() {
    let parser = EntityParser::default_markers();
    let unmarked = parser.unmark("@@John## went to @@Paris##");
    assert_eq!(unmarked, "John went to Paris");
}

#[test]
fn test_validate_markers() {
    let parser = EntityParser::default_markers();
    assert!(parser.validate_markers("@@John## went home"));
    assert!(!parser.validate_markers("@@John went home"));
}

#[test]
fn test_count_entities() {
    let parser = EntityParser::default_markers();
    assert_eq!(parser.count_entities("@@John## went to @@Paris##"), 2);
    assert_eq!(parser.count_entities("No entities"), 0);
}

#[test]
fn test_custom_markers() {
    let parser = EntityParser::new("<<", ">>");
    let entities = parser.parse("<<John>> went home", "PER").unwrap();
    assert_eq!(entities.len(), 1);
    assert_eq!(entities[0].text, "John");
}

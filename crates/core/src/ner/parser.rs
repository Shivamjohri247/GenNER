//! Entity parser for extracting entities from marked text

use crate::error::{Error, Result};
use crate::ner::Entity;

/// Entity parser for extracting entities from GPT-NER marked text
#[derive(Clone, Debug)]
pub struct EntityParser {
    /// Entity prefix marker
    prefix: String,

    /// Entity suffix marker
    suffix: String,
}

impl EntityParser {
    /// Create a new entity parser
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }
    }

    /// Create with default markers (@@ and ##)
    pub fn default_markers() -> Self {
        Self::new("@@", "##")
    }

    /// Parse entities from marked text
    ///
    /// Given text like "@@John## went to @@Paris##", extract entities
    /// and their positions in the unmarked text.
    pub fn parse(&self, marked_text: &str, label: impl Into<String>) -> Result<Vec<Entity>> {
        let label = label.into();
        let mut entities = Vec::new();
        let prefix = &self.prefix;
        let suffix = &self.suffix;

        // First, find all marked entities
        // We store: (start_of_prefix, end_of_entity_text, entity_text)
        let mut marked_spans = Vec::new();
        let mut pos = 0;

        while pos < marked_text.len() {
            if let Some(start) = marked_text[pos..].find(prefix) {
                let actual_start = pos + start;
                let after_prefix = actual_start + prefix.len();

                if let Some(end_idx) = marked_text[after_prefix..].find(suffix) {
                    let actual_end = after_prefix + end_idx;
                    let entity_text = marked_text[after_prefix..actual_end].to_string();
                    let end_of_marker = actual_end + suffix.len();

                    marked_spans.push((actual_start, actual_end, entity_text));
                    pos = end_of_marker;
                } else {
                    // Mismatched markers
                    return Err(Error::EntityParsing(
                        "Unmatched entity prefix marker".to_string(),
                    ));
                }
            } else {
                break;
            }
        }

        // Calculate positions in unmarked text
        // The offset tracks how many characters we've removed (markers) up to this point
        let mut offset = 0isize;
        for (start, end, text) in &marked_spans {
            // In unmarked text, the entity starts at the position where prefix starts, minus offset
            let unmarked_start = (*start as isize + offset) as usize;
            // In unmarked text, the entity ends at the position where entity text ends, minus offset
            // But we also need to account for the prefix length before this entity
            let unmarked_end = (*end as isize + offset - self.prefix.len() as isize) as usize;

            entities.push(Entity::new(
                text.clone(),
                label.clone(),
                unmarked_start,
                unmarked_end,
            ));

            // Adjust offset for next entity (remove marker lengths)
            offset -= (self.prefix.len() + self.suffix.len()) as isize;
        }

        Ok(entities)
    }

    /// Parse entities with multiple labels from marked text
    ///
    /// This parses text where different entity types might be marked differently.
    /// For now, we use a single label for all entities.
    pub fn parse_with_label(&self, marked_text: &str, label: &str) -> Result<Vec<Entity>> {
        self.parse(marked_text, label)
    }

    /// Parse all entities from marked text, detecting positions
    ///
    /// This version attempts to map the marked entities back to the original
    /// text positions more accurately.
    pub fn parse_from_original(
        &self,
        original: &str,
        marked: &str,
        label: impl Into<String>,
    ) -> Result<Vec<Entity>> {
        let label = label.into();
        let mut entities = Vec::new();

        let prefix = &self.prefix;
        let suffix = &self.suffix;

        let mut pos = 0;

        while pos < marked.len() {
            // Find next marker
            if let Some(marker_start) = marked[pos..].find(prefix) {
                let actual_start = pos + marker_start;
                let after_prefix = actual_start + prefix.len();

                // Calculate position in original text before this marker
                // Count characters before marker start
                let marked_before = &marked[..actual_start];
                // Find the entity in original from the current position
                if let Some(marker_end) = marked[after_prefix..].find(suffix) {
                    let actual_end = after_prefix + marker_end;
                    let entity_text = marked[after_prefix..actual_end].to_string();
                    let end_of_marker = actual_end + suffix.len();

                    // Find entity in original text by searching forward
                    // from where we've already processed
                    let processed_len = marked.len() - marked[actual_start..].len();
                    let orig_processed = self.find_original_position(original, marked, processed_len);

                    if let Some(entity_start) = original[orig_processed..].find(&entity_text) {
                        let start = orig_processed + entity_start;
                        let end = start + entity_text.len();

                        entities.push(Entity::new(entity_text, label.clone(), start, end));
                    }

                    pos = end_of_marker;
                } else {
                    return Err(Error::EntityParsing(
                        "Unmatched entity prefix marker".to_string(),
                    ));
                }
            } else {
                break;
            }
        }

        Ok(entities)
    }

    /// Helper to find position in original text corresponding to marked text position
    fn find_original_position(&self, original: &str, marked: &str, marked_pos: usize) -> usize {
        // Simple approach: unmark the text and count
        let unmarked_up_to = self.unmark(&marked[..marked_pos.min(marked.len())]);
        // Find how many chars of original match
        let mut result = 0;
        for (i, c) in unmarked_up_to.chars().enumerate() {
            if original.chars().nth(i) == Some(c) {
                result = i + 1;
            } else {
                break;
            }
        }
        original.chars().take(result).map(|c| c.len_utf8()).sum()
    }

    /// Unmark text - remove entity markers
    pub fn unmark(&self, marked_text: &str) -> String {
        let mut result = marked_text.to_string();
        let prefix = &self.prefix;
        let suffix = &self.suffix;

        // Remove all occurrences
        while let Some(start) = result.find(prefix) {
            let after_prefix = start + prefix.len();
            if let Some(end) = result[after_prefix..].find(suffix) {
                let actual_end = after_prefix + end + suffix.len();
                result = format!("{}{}{}", &result[..start], &result[after_prefix..after_prefix + end], &result[actual_end..]);
            } else {
                break;
            }
        }

        result
    }

    /// Validate marker matching in text
    pub fn validate_markers(&self, text: &str) -> bool {
        let prefix_count = text.matches(&self.prefix).count();
        let suffix_count = text.matches(&self.suffix).count();
        prefix_count == suffix_count
    }

    /// Count entities in marked text
    pub fn count_entities(&self, marked_text: &str) -> usize {
        marked_text.matches(&self.prefix).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(!parser.validate_markers("John## went home"));
    }

    #[test]
    fn test_count_entities() {
        let parser = EntityParser::default_markers();
        assert_eq!(parser.count_entities("@@John## went to @@Paris##"), 2);
        assert_eq!(parser.count_entities("No entities"), 0);
    }

    #[test]
    fn test_parse_from_original() {
        let parser = EntityParser::default_markers();
        let original = "John went to Paris";
        let marked = "@@John## went to @@Paris##";

        let entities = parser.parse_from_original(original, marked, "MIXED").unwrap();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].text, "John");
        assert_eq!(entities[1].text, "Paris");
        assert!(entities[0].validate(original));
        assert!(entities[1].validate(original));
    }

    #[test]
    fn test_custom_markers() {
        let parser = EntityParser::new("<<", ">>");
        let entities = parser.parse("<<John>> went home", "PER").unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "John");
    }

    #[test]
    fn test_overlapping_positions() {
        let parser = EntityParser::default_markers();
        let entities = parser.parse("@@John Smith## is here", "PER").unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "John Smith");
        assert_eq!(entities[0].end - entities[0].start, 10); // "John Smith".len()
    }
}

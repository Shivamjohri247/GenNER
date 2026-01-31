//! Text utilities

/// Normalize whitespace in text
pub fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Truncate text to max length, trying to break at word boundary
pub fn truncate_smart(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }

    // Try to find last space before max_len
    if let Some(last_space) = text[..max_len].rfind(' ') {
        text[..last_space].to_string() + "..."
    } else {
        text[..max_len.saturating_sub(3)].to_string() + "..."
    }
}

/// Count tokens roughly (1 token ≈ 4 characters)
pub fn estimate_tokens(text: &str) -> usize {
    (text.len() / 4).max(1)
}

/// Escape special regex characters
pub fn escape_regex(text: &str) -> String {
    let special_chars = ['\\', '.', '+', '*', '?', '^', '$', '(', ')', '[', ']', '{', '}', '|'];
    let mut result = String::with_capacity(text.len() * 2);

    for c in text.chars() {
        if special_chars.contains(&c) {
            result.push('\\');
        }
        result.push(c);
    }

    result
}

/// Check if text is likely English
pub fn is_likely_english(text: &str) -> bool {
    let ascii_chars = text.chars().filter(|c| c.is_ascii()).count() as f64;
    let total_chars = text.chars().count() as f64;

    if total_chars == 0.0 {
        return true;
    }

    let ascii_ratio = ascii_chars / total_chars;
    ascii_ratio > 0.7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("hello   world"), "hello world");
        assert_eq!(normalize_whitespace("  a  b  c  "), "a b c");
    }

    #[test]
    fn test_truncate_smart() {
        assert_eq!(truncate_smart("hello world", 20), "hello world");
        assert_eq!(truncate_smart("hello world", 8), "hello...");
        assert_eq!(truncate_smart("helloworld", 8), "hello...");
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello"), 1);
        // "hello world test" = 17 chars, 17 / 4 = 4
        assert_eq!(estimate_tokens("hello world test"), 4);
    }

    #[test]
    fn test_escape_regex() {
        assert_eq!(escape_regex("a.b"), "a\\.b");
        assert_eq!(escape_regex("test+"), "test\\+");
    }

    #[test]
    fn test_is_likely_english() {
        assert!(is_likely_english("hello world"));
        assert!(is_likely_english(""));
        assert!(!is_likely_english("你好世界"));
    }
}

//! Tokenizer trait

use crate::error::{Error, Result};

/// Tokenizer trait
///
/// This trait defines the interface that tokenizers must provide
/// to work with the GenNER system.
pub trait TokenizerTrait: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String>;

    /// Get token count for text
    fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.encode(text, false)?.len())
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the token for a given ID
    fn id_to_token(&self, id: u32) -> Option<String>;

    /// Get the ID for a given token
    fn token_to_id(&self, token: &str) -> Option<u32>;

    /// Check if text is within max sequence length
    fn fits_within(&self, text: &str, max_len: usize, add_special_tokens: bool) -> bool {
        match self.token_count(text) {
            Ok(count) => {
                let special = if add_special_tokens { 2 } else { 0 };
                count + special <= max_len
            }
            Err(_) => false,
        }
    }

    /// Truncate text to fit within max sequence length
    fn truncate_to_fit(&self, text: &str, max_len: usize, add_special_tokens: bool) -> String {
        let special = if add_special_tokens { 2 } else { 0 };
        let available = max_len.saturating_sub(special);

        // Binary search for the truncation point
        let mut left = 0;
        let mut right = text.len();

        while left < right {
            let mid = (left + right + 1) / 2;
            let candidate = &text[..mid];
            if let Ok(count) = self.token_count(candidate) {
                if count <= available {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            } else {
                right = mid - 1;
            }
        }

        text[..left].to_string()
    }

    /// Encode batch of texts
    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Decode batch of token IDs
    fn decode_batch(&self, batch: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        batch
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }
}

/// Simple whitespace tokenizer for testing
#[derive(Debug, Clone)]
pub struct WhitespaceTokenizer;

impl TokenizerTrait for WhitespaceTokenizer {
    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<u32>> {
        // Simple hash-based encoding for testing
        Ok(text.split_whitespace().map(|t| Self::hash_token(t)).collect())
    }

    fn decode(&self, ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
        // For testing, just return the IDs as strings
        Ok(ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "))
    }

    fn vocab_size(&self) -> usize {
        100000
    }

    fn id_to_token(&self, _id: u32) -> Option<String> {
        None
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        Some(Self::hash_token(token))
    }
}

impl WhitespaceTokenizer {
    fn hash_token(token: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        (hasher.finish() % 100000) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer_encode() {
        let tokenizer = WhitespaceTokenizer;
        let ids = tokenizer.encode("hello world", false).unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_whitespace_tokenizer_decode() {
        let tokenizer = WhitespaceTokenizer;
        let text = tokenizer.decode(&[1, 2, 3], false).unwrap();
        assert_eq!(text, "1 2 3");
    }

    #[test]
    fn test_whitespace_tokenizer_count() {
        let tokenizer = WhitespaceTokenizer;
        let count = tokenizer.token_count("hello world test").unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_whitespace_tokenizer_batch() {
        let tokenizer = WhitespaceTokenizer;
        let batch = tokenizer.encode_batch(&["hello world", "foo bar"], false).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].len(), 2);
        assert_eq!(batch[1].len(), 2);
    }
}

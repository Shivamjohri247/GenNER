//! Tokenizer integration using the tokenizers crate

use genner_core::error::{Error, Result};
use genner_core::traits::tokenizer::TokenizerTrait;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer as HFTokenizer;

/// Hugging Face tokenizer wrapper
#[derive(Clone, Debug)]
pub struct HFTokenizerWrapper {
    inner: Arc<HFTokenizer>,
}

impl HFTokenizerWrapper {
    /// Create a new tokenizer from a Hugging Face model path
    pub fn from_pretrained(path: impl AsRef<str>) -> Result<Self> {
        let path = path.as_ref();

        // Use hf-hub to download the tokenizer if it's not a local path
        let tokenizer = if Path::new(path).exists() {
            // Load from local file
            let tokenizer_file = Path::new(path).join("tokenizer.json");
            HFTokenizer::from_file(tokenizer_file).map_err(|e| {
                Error::Tokenization(format!("Failed to load tokenizer from {}: {}", path, e))
            })?
        } else {
            // Download from HF Hub using hf-hub
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| Error::Tokenization(format!("Failed to create HF API: {}", e)))?;
            let api = api.model(path.to_string());
            let tokenizer_path = api.get("tokenizer.json")
                .map_err(|e| Error::Tokenization(format!("Failed to get tokenizer from HF Hub: {}", e)))?;
            HFTokenizer::from_file(tokenizer_path).map_err(|e| {
                Error::Tokenization(format!("Failed to load tokenizer: {}", e))
            })?
        };

        // Apply the padding token if not set
        let mut tokenizer = tokenizer;
        if tokenizer.get_padding().is_none() {
            tokenizer.with_padding(None);
        }

        Ok(Self {
            inner: Arc::new(tokenizer),
        })
    }

    /// Get the underlying tokenizer
    pub fn inner(&self) -> &HFTokenizer {
        &self.inner
    }
}

impl TokenizerTrait for HFTokenizerWrapper {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| Error::Tokenization(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = ids.to_vec();
        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| Error::Tokenization(format!("Decoding failed: {}", e)))
    }

    /// Get token count for text
    fn token_count(&self, text: &str) -> Result<usize> {
        Ok(self.inner.encode(text, false).map_err(|e| {
            Error::Tokenization(format!("Token counting failed: {}", e))
        })?.len())
    }

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Get the token for a given ID
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.decode(&[id], true).ok()
    }

    /// Get the ID for a given token
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.get_vocab(false).get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        // This test will be skipped if the tokenizer file doesn't exist
        let result = HFTokenizerWrapper::from_pretrained("nonexistent-model");
        assert!(result.is_err());
    }

    #[test]
    fn test_tokenizer_wrapper_send_sync() {
        // Ensure the tokenizer is Send + Sync for use in async contexts
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<HFTokenizerWrapper>();
    }
}

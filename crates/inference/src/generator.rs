//! Text generation

use genner_core::error::Result;

/// Text generator
pub struct Generator {
    /// Maximum tokens to generate
    max_tokens: usize,

    /// Temperature for sampling
    temperature: f32,
}

impl Generator {
    /// Create a new generator
    pub fn new(max_tokens: usize, temperature: f32) -> Self {
        Self {
            max_tokens,
            temperature: temperature.clamp(0.0, 2.0),
        }
    }

    /// Get max tokens
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Set max tokens
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
    }

    /// Get temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.clamp(0.0, 2.0);
    }
}

impl Default for Generator {
    fn default() -> Self {
        Self::new(512, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_new() {
        let gen = Generator::new(256, 0.5);
        assert_eq!(gen.max_tokens(), 256);
        assert_eq!(gen.temperature(), 0.5);
    }

    #[test]
    fn test_generator_temperature_clamp() {
        let mut gen = Generator::new(512, 0.0);
        gen.set_temperature(5.0); // Should be clamped to 2.0
        assert_eq!(gen.temperature(), 2.0);
    }
}

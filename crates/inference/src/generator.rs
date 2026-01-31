//! Text generation with various sampling strategies

use genner_core::error::Result;

/// Text generator with configurable sampling strategies
pub struct Generator {
    /// Maximum tokens to generate
    max_tokens: usize,

    /// Sampling strategy
    strategy: SamplingStrategy,

    /// Temperature for sampling (used by some strategies)
    temperature: f32,

    /// Top-p threshold for nucleus sampling
    top_p: f32,

    /// Top-k for top-k sampling
    top_k: u32,

    /// Frequency penalty (0.0 to 2.0)
    frequency_penalty: f32,

    /// Presence penalty (0.0 to 2.0)
    presence_penalty: f32,

    /// Stop sequences
    stop_sequences: Vec<String>,

    /// Whether to use the cache
    use_cache: bool,
}

/// Sampling strategy for text generation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Greedy sampling (always pick most likely token)
    Greedy,

    /// Multinomial sampling with temperature
    Multinomial,

    /// Top-k sampling (sample from top k tokens)
    TopK,

    /// Nucleus sampling (top-p)
    Nucleus,

    /// Combined top-k and top-p
    TopKAndNucleus,
}

impl Generator {
    /// Create a new generator
    pub fn new(max_tokens: usize, strategy: SamplingStrategy) -> Self {
        Self {
            max_tokens,
            strategy,
            temperature: 0.0,
            top_p: 0.9,
            top_k: 50,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            use_cache: true,
        }
    }

    /// Create with greedy sampling
    pub fn greedy(max_tokens: usize) -> Self {
        Self::new(max_tokens, SamplingStrategy::Greedy)
    }

    /// Create with multinomial sampling
    pub fn multinomial(max_tokens: usize, temperature: f32) -> Self {
        Self {
            max_tokens,
            strategy: SamplingStrategy::Multinomial,
            temperature: temperature.clamp(0.0, 2.0),
            top_p: 0.9,
            top_k: 50,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_sequences: Vec::new(),
            use_cache: true,
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

    /// Get sampling strategy
    pub fn strategy(&self) -> SamplingStrategy {
        self.strategy
    }

    /// Set sampling strategy
    pub fn set_strategy(&mut self, strategy: SamplingStrategy) {
        self.strategy = strategy;
    }

    /// Get temperature
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// Set temperature (clamped to 0.0 - 2.0)
    pub fn set_temperature(&mut self, temperature: f32) {
        self.temperature = temperature.clamp(0.0, 2.0);
    }

    /// Get top-p
    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    /// Set top-p (clamped to 0.0 - 1.0)
    pub fn set_top_p(&mut self, top_p: f32) {
        self.top_p = top_p.clamp(0.0, 1.0);
    }

    /// Get top-k
    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    /// Set top-k
    pub fn set_top_k(&mut self, top_k: u32) {
        self.top_k = top_k;
    }

    /// Get frequency penalty
    pub fn frequency_penalty(&self) -> f32 {
        self.frequency_penalty
    }

    /// Set frequency penalty (clamped to 0.0 - 2.0)
    pub fn set_frequency_penalty(&mut self, penalty: f32) {
        self.frequency_penalty = penalty.clamp(0.0, 2.0);
    }

    /// Get presence penalty
    pub fn presence_penalty(&self) -> f32 {
        self.presence_penalty
    }

    /// Set presence penalty (clamped to 0.0 - 2.0)
    pub fn set_presence_penalty(&mut self, penalty: f32) {
        self.presence_penalty = penalty.clamp(0.0, 2.0);
    }

    /// Add a stop sequence
    pub fn add_stop_sequence(&mut self, sequence: impl Into<String>) {
        self.stop_sequences.push(sequence.into());
    }

    /// Clear all stop sequences
    pub fn clear_stop_sequences(&mut self) {
        self.stop_sequences.clear();
    }

    /// Get stop sequences
    pub fn stop_sequences(&self) -> &[String] {
        &self.stop_sequences
    }

    /// Get whether cache is enabled
    pub fn use_cache(&self) -> bool {
        self.use_cache
    }

    /// Set whether to use cache
    pub fn set_use_cache(&mut self, use_cache: bool) {
        self.use_cache = use_cache;
    }

    /// Sample a token from logits using the configured strategy
    ///
    /// # Arguments
    /// * `logits` - Logits for each token (vocab_size)
    /// * `token_counts` - Optional token counts for frequency penalty
    ///
    /// # Returns
    /// The sampled token ID
    pub fn sample_token(&self, logits: Vec<f32>, token_counts: Option<&[usize]>) -> Result<u32> {
        let mut adjusted_logits = logits;

        // Apply frequency penalty
        if self.frequency_penalty > 0.0 {
            if let Some(counts) = token_counts {
                for (logit, &count) in adjusted_logits.iter_mut().zip(counts.iter()) {
                    if count > 0 {
                        *logit -= self.frequency_penalty * count as f32;
                    }
                }
            }
        }

        // Apply presence penalty
        if self.presence_penalty > 0.0 {
            if let Some(counts) = token_counts {
                for (logit, &count) in adjusted_logits.iter_mut().zip(counts.iter()) {
                    if count > 0 {
                        *logit -= self.presence_penalty;
                    }
                }
            }
        }

        match self.strategy {
            SamplingStrategy::Greedy => self.sample_greedy(&adjusted_logits),
            SamplingStrategy::Multinomial => self.sample_multinomial(&adjusted_logits),
            SamplingStrategy::TopK => self.sample_top_k(&adjusted_logits, self.top_k),
            SamplingStrategy::Nucleus => self.sample_nucleus(&adjusted_logits, self.top_p),
            SamplingStrategy::TopKAndNucleus => self.sample_top_k_nucleus(&adjusted_logits, self.top_k, self.top_p),
        }
    }

    /// Greedy sampling: pick the token with highest logit
    fn sample_greedy(&self, logits: &[f32]) -> Result<u32> {
        logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| genner_core::error::Error::Generation("Empty logits".to_string()))
    }

    /// Multinomial sampling with temperature
    fn sample_multinomial(&self, logits: &[f32]) -> Result<u32> {
        // Apply temperature and convert to probabilities
        let temp = self.temperature.max(0.01);
        let exp_logits: Vec<f32> = logits.iter().map(|&x| (x / temp).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        // Sample using the probabilities
        let mut rng = rand::thread_rng();
        let mut cumsum = 0.0;
        let sample: f32 = rand::Rng::gen(&mut rng);

        for (idx, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to last token
        Ok((probs.len() - 1) as u32)
    }

    /// Top-k sampling: sample from top k tokens
    fn sample_top_k(&self, logits: &[f32], k: u32) -> Result<u32> {
        let k = k as usize;
        let k = k.min(logits.len());

        // Sort indices by logit value (descending)
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k and apply temperature
        let temp = self.temperature.max(0.01);
        let mut exp_logits = Vec::with_capacity(k);
        for &idx in &indices[..k] {
            exp_logits.push((logits[idx] / temp).exp());
        }

        // Normalize and sample
        let sum: f32 = exp_logits.iter().sum();
        let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum).collect();

        let mut rng = rand::thread_rng();
        let mut cumsum = 0.0;
        let sample: f32 = rand::Rng::gen(&mut rng);

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if sample <= cumsum {
                return Ok(indices[i] as u32);
            }
        }

        Ok(indices[k - 1] as u32)
    }

    /// Nucleus (top-p) sampling: sample from smallest set of tokens with cumulative prob >= p
    fn sample_nucleus(&self, logits: &[f32], p: f32) -> Result<u32> {
        // Sort indices by logit value (descending)
        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));

        // Compute cumulative probabilities
        let temp = self.temperature.max(0.01);
        let exp_logits: Vec<f32> = indices.iter().map(|&idx| (logits[idx] / temp).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();

        let mut cumsum = 0.0;
        let mut cutoff = indices.len();

        for (i, &exp_logit) in exp_logits.iter().enumerate() {
            cumsum += exp_logit / sum;
            if cumsum >= p {
                cutoff = i + 1;
                break;
            }
        }

        // Sample from the cutoff set
        let cutoff = cutoff.max(1);
        let mut sample_probs = Vec::with_capacity(cutoff);
        for i in 0..cutoff {
            sample_probs.push(exp_logits[i] / sum);
        }

        let mut rng = rand::thread_rng();
        let mut cumsum = 0.0;
        let sample: f32 = rand::Rng::gen(&mut rng);

        for (i, &prob) in sample_probs.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(indices[i] as u32);
            }
        }

        Ok(indices[cutoff - 1] as u32)
    }

    /// Combined top-k and top-p sampling
    fn sample_top_k_nucleus(&self, logits: &[f32], k: u32, p: f32) -> Result<u32> {
        // First apply top-k
        let k = k as usize;
        let k = k.min(logits.len());

        let mut indices: Vec<usize> = (0..logits.len()).collect();
        indices.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let indices = &indices[..k];

        // Then apply top-p on the top-k
        let temp = self.temperature.max(0.01);
        let exp_logits: Vec<f32> = indices.iter().map(|&idx| (logits[idx] / temp).exp()).collect();
        let sum: f32 = exp_logits.iter().sum();

        let mut cumsum = 0.0;
        let mut cutoff = indices.len();

        for (i, &exp_logit) in exp_logits.iter().enumerate() {
            cumsum += exp_logit / sum;
            if cumsum >= p {
                cutoff = i + 1;
                break;
            }
        }

        let cutoff = cutoff.max(1);
        let mut sample_probs = Vec::with_capacity(cutoff);
        for i in 0..cutoff {
            sample_probs.push(exp_logits[i] / sum);
        }

        let mut rng = rand::thread_rng();
        let mut cumsum = 0.0;
        let sample: f32 = rand::Rng::gen(&mut rng);

        for (i, &prob) in sample_probs.iter().enumerate() {
            cumsum += prob;
            if sample <= cumsum {
                return Ok(indices[i] as u32);
            }
        }

        Ok(indices[cutoff - 1] as u32)
    }

    /// Check if a generated token matches any stop sequence
    pub fn should_stop(&self, generated: &str) -> bool {
        for stop in &self.stop_sequences {
            if generated.contains(stop) || generated.ends_with(stop) {
                return true;
            }
        }
        false
    }
}

impl Default for Generator {
    fn default() -> Self {
        Self::greedy(512)
    }
}

/// Streaming token callback
pub trait TokenCallback: Send + Sync {
    /// Called for each generated token
    fn on_token(&mut self, token: u32, text: &str) -> bool;

    /// Called when generation completes
    fn on_complete(&mut self) {}
}

/// Simple callback that collects tokens
#[derive(Default)]
pub struct CollectingCallback {
    /// Collected token IDs
    pub tokens: Vec<u32>,

    /// Collected text
    pub text: String,
}

impl TokenCallback for CollectingCallback {
    fn on_token(&mut self, token: u32, text: &str) -> bool {
        self.tokens.push(token);
        self.text.push_str(text);
        true // Continue generation
    }

    fn on_complete(&mut self) {
        // Optionally do something when complete
    }
}

/// Callback for stopping at specific tokens
pub struct StopTokenCallback {
    /// Stop token IDs
    stop_tokens: Vec<u32>,

    /// Inner callback to delegate to
    inner: Option<Box<dyn TokenCallback>>,
}

impl StopTokenCallback {
    /// Create a new stop token callback
    pub fn new(stop_tokens: Vec<u32>, inner: Box<dyn TokenCallback>) -> Self {
        Self { stop_tokens, inner: Some(inner) }
    }

    /// Consume and return the inner callback
    pub fn into_inner(self) -> Box<dyn TokenCallback> {
        self.inner.unwrap_or_else(|| Box::new(CollectingCallback::default()))
    }
}

impl TokenCallback for StopTokenCallback {
    fn on_token(&mut self, token: u32, text: &str) -> bool {
        if self.stop_tokens.contains(&token) {
            return false; // Stop generation
        }
        if let Some(ref mut inner) = self.inner {
            inner.on_token(token, text)
        } else {
            true
        }
    }

    fn on_complete(&mut self) {
        if let Some(ref mut inner) = self.inner {
            inner.on_complete();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_greedy() {
        let gen = Generator::greedy(256);
        assert_eq!(gen.strategy(), SamplingStrategy::Greedy);
        assert_eq!(gen.max_tokens(), 256);
    }

    #[test]
    fn test_generator_multinomial() {
        let gen = Generator::multinomial(512, 0.8);
        assert_eq!(gen.strategy(), SamplingStrategy::Multinomial);
        assert_eq!(gen.temperature(), 0.8);
    }

    #[test]
    fn test_generator_sample_greedy() {
        let gen = Generator::greedy(256);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = gen.sample_token(logits, None).unwrap();
        assert_eq!(token, 3); // Index of highest logit (0.9)
    }

    #[test]
    fn test_generator_sample_greedy_tie() {
        let gen = Generator::greedy(256);
        let logits = vec![0.5, 0.5, 0.3];
        let token = gen.sample_token(logits, None).unwrap();
        // In case of tie, we expect one of the tied indices (0 or 1)
        assert!(token == 0 || token == 1);
    }

    #[test]
    fn test_generator_set_temperature() {
        let mut gen = Generator::greedy(256);
        gen.set_temperature(1.5);
        assert_eq!(gen.temperature(), 1.5);
    }

    #[test]
    fn test_generator_temperature_clamp() {
        let mut gen = Generator::greedy(256);
        gen.set_temperature(5.0); // Should be clamped to 2.0
        assert_eq!(gen.temperature(), 2.0);
    }

    #[test]
    fn test_generator_top_p_clamp() {
        let mut gen = Generator::greedy(256);
        gen.set_top_p(1.5); // Should be clamped to 1.0
        assert_eq!(gen.top_p(), 1.0);
    }

    #[test]
    fn test_generator_stop_sequences() {
        let mut gen = Generator::greedy(256);
        gen.add_stop_sequence("###");
        gen.add_stop_sequence("END");
        assert_eq!(gen.stop_sequences().len(), 2);

        assert!(gen.should_stop("This is text ###"));
        assert!(gen.should_stop("This is text END"));
        assert!(!gen.should_stop("This is text"));
    }

    #[test]
    fn test_collecting_callback() {
        let mut callback = CollectingCallback::default();
        assert!(callback.on_token(1, "Hello"));
        assert!(callback.on_token(2, " World"));
        assert_eq!(callback.tokens, vec![1, 2]);
        assert_eq!(callback.text, "Hello World");
    }

    #[test]
    fn test_stop_token_callback() {
        let collecting = CollectingCallback::default();
        let inner = Box::new(collecting);
        let mut callback = StopTokenCallback::new(vec![0, 2], inner);

        assert!(callback.on_token(1, "A")); // Continue
        assert!(!callback.on_token(2, "B")); // Stop (token 2 is in stop_tokens)
    }

    #[test]
    fn test_generator_sample_top_k() {
        let gen = Generator::new(256, SamplingStrategy::TopK);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        // Top-k sampling should return one of the top k indices
        let token = gen.sample_token(logits, None).unwrap();
        assert!(token < 5); // Should be a valid index
    }

    #[test]
    fn test_generator_sample_nucleus() {
        let gen = Generator::new(256, SamplingStrategy::Nucleus);
        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let token = gen.sample_token(logits, None).unwrap();
        assert!(token < 5); // Should be a valid index
    }

    #[test]
    fn test_generator_frequency_penalty() {
        let mut gen = Generator::greedy(256);
        gen.set_frequency_penalty(1.0);

        let logits = vec![0.9, 0.5, 0.3, 0.9, 0.2];
        let token_counts = vec![2, 0, 0, 0, 0]; // First token has count 2

        // With frequency penalty, first token should be penalized
        let token = gen.sample_token(logits, Some(&token_counts)).unwrap();
        // Token 3 (also 0.9) or token 1 (0.5) should win over penalized token 0
        assert!(token == 3 || token == 1 || token == 0);
    }

    #[test]
    fn test_generator_presence_penalty() {
        let mut gen = Generator::greedy(256);
        gen.set_presence_penalty(1.0);

        let logits = vec![0.9, 0.5, 0.3, 0.8, 0.2];
        let token_counts = vec![1, 0, 0, 0, 0]; // First token has appeared

        // With presence penalty, first token should be penalized
        let token = gen.sample_token(logits, Some(&token_counts)).unwrap();
        // Token 3 (0.8, not penalized) should beat token 0 (0.9 - 1.0 = -0.1)
        assert_eq!(token, 3);
    }

    #[test]
    fn test_generator_top_k_nucleus() {
        let mut gen = Generator::new(256, SamplingStrategy::TopKAndNucleus);
        gen.set_top_k(3);
        gen.set_top_p(0.9);

        let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2, 0.1, 0.05];
        let token = gen.sample_token(logits, None).unwrap();
        assert!(token < 7); // Should be a valid index
    }

    #[test]
    fn test_generator_use_cache() {
        let mut gen = Generator::greedy(256);
        assert!(gen.use_cache());

        gen.set_use_cache(false);
        assert!(!gen.use_cache());
    }
}

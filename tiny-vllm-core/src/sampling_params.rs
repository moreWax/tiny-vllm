//! Sampling parameters used during token generation.
//!
//! This is a trimmed down variant of the Python `SamplingParams` class
//! providing only the fields required by the Rust implementation.

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_tokens: 16,
            ignore_eos: false,
        }
    }
}

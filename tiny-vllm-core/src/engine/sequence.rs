//! TokenSequence management for LLM inference.
//!
//! Tracks request status, tokens, and sampling parameters.
//! Designed for clarity, extensibility, and performance.

use std::sync::atomic::{AtomicU64, Ordering};
use crate::engine::sampling_params::SamplingParams;

/// Global atomic for unique sequence IDs.
static SEQUENCE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Possible states for a sequence during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Pending,
    Active,
    Finished,
    Interrupted,
    Failed,
}

/// Result from a finished sequence.
#[derive(Debug, Clone)]
pub struct TokenSequenceResult {
    pub seq_id: u64,
    pub text: String,
    pub all_token_ids: Vec<u32>,
    pub completion_token_ids: Vec<u32>,
    pub prompt_token_count: usize,
    pub completion_token_count: usize,
    pub state: SequenceState,
}

/// Main structure for a single inference request.
#[derive(Debug, Clone)]
pub struct TokenSequence {
    seq_id: u64,
    state: SequenceState,
    token_ids: Vec<u32>,
    last_token: u32,
    prompt_token_count: usize,
    cached_token_count: usize,
    block_table: Vec<usize>,
    sampling: SamplingParams,
    block_size: usize,
}

impl TokenSequence {
    /// Create a new token sequence from prompt tokens and parameters.
    pub fn new(prompt: Vec<u32>, sampling: SamplingParams, block_size: usize) -> Self {
        let seq_id = SEQUENCE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let prompt_token_count = prompt.len();
        let last_token = prompt.last().copied().unwrap_or(0);
        TokenSequence {
            seq_id,
            state: SequenceState::Pending,
            token_ids: prompt,
            last_token,
            prompt_token_count,
            cached_token_count: 0,
            block_table: Vec::new(),
            sampling,
            block_size,
        }
    }

    /// Returns the sequence id.
    pub fn id(&self) -> u64 {
        self.seq_id
    }

    /// Returns the current sequence state.
    pub fn state(&self) -> SequenceState {
        self.state
    }

    /// Returns the total number of tokens.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    /// True if the sequence has no tokens.
    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// Returns all tokens.
    pub fn all_tokens(&self) -> &[u32] {
        &self.token_ids
    }

    /// Returns all tokens as Vec (for API compatibility).
    pub fn all_token_vec(&self) -> Vec<u32> {
        self.token_ids.clone()
    }

    /// Returns prompt tokens as a slice.
    pub fn prompt_tokens(&self) -> &[u32] {
        &self.token_ids[..self.prompt_token_count]
    }

    /// Returns completion tokens as a new Vec.
    pub fn completion_tokens(&self) -> Vec<u32> {
        if self.prompt_token_count > self.token_ids.len() {
            Vec::new()
        } else {
            self.token_ids[self.prompt_token_count..].to_vec()
        }
    }

    /// Number of completion (generated) tokens.
    pub fn completion_token_count(&self) -> usize {
        self.token_ids.len().saturating_sub(self.prompt_token_count)
    }

    /// True if this sequence is finished or failed.
    pub fn is_finished(&self) -> bool {
        matches!(self.state, SequenceState::Finished | SequenceState::Failed)
    }

    /// True if this sequence can be scheduled.
    pub fn is_schedulable(&self) -> bool {
        matches!(self.state, SequenceState::Pending | SequenceState::Interrupted)
    }

    /// Appends a token to the sequence.
    pub fn push_token(&mut self, token: u32) {
        self.token_ids.push(token);
        self.last_token = token;
    }

    /// Returns the last token.
    pub fn last_token(&self) -> u32 {
        self.last_token
    }

    /// Number of blocks needed for current tokens.
    pub fn block_count(&self) -> usize {
        (self.len() + self.block_size - 1) / self.block_size
    }

    /// Number of cached blocks.
    pub fn cached_block_count(&self) -> usize {
        self.cached_token_count / self.block_size
    }

    /// Number of tokens in the last block.
    pub fn last_block_token_count(&self) -> usize {
        match (self.len(), self.block_size) {
            (0, _) => 0,
            (tk, bs) if tk % bs == 0 => bs,
            (tk, bs) => tk % bs,
        }
    }

    /// Get tokens for a particular block as a Vec.
    pub fn block_tokens(&self, block_idx: usize) -> Vec<u32> {
        let start = block_idx * self.block_size;
        let end = ((block_idx + 1) * self.block_size).min(self.len());
        if start >= end {
            Vec::new()
        } else {
            self.token_ids[start..end].to_vec()
        }
    }

    /// Should inference stop (due to max tokens or EOS)?
    pub fn should_stop(&self, eos_token: Option<u32>) -> bool {
        if self.completion_token_count() >= self.sampling.max_tokens {
            return true;
        }
        if !self.sampling.ignore_eos {
            if let Some(eos_id) = eos_token {
                if self.last_token == eos_id {
                    return true;
                }
            }
        }
        false
    }

    /// Mark as finished.
    pub fn mark_finished(&mut self) {
        self.state = SequenceState::Finished;
    }

    /// Mark as interrupted and clear caches.
    pub fn interrupt(&mut self) {
        self.state = SequenceState::Interrupted;
        self.block_table.clear();
        self.cached_token_count = 0;
    }

    /// Mark as active.
    pub fn activate(&mut self) {
        self.state = SequenceState::Active;
    }

    /// Expose sampling parameters.
    pub fn sampling_params(&self) -> &SamplingParams {
        &self.sampling
    }

    /// Set block table (for scheduler, etc).
    pub fn set_block_table(&mut self, table: Vec<usize>) {
        self.block_table = table;
    }

    /// Returns a reference to the block table.
    pub fn block_table(&self) -> &[usize] {
        &self.block_table
    }

    /// Set cached token count (for KV cache tracking).
    pub fn set_cached_token_count(&mut self, n: usize) {
        self.cached_token_count = n;
    }

    /// Build a result object.
    pub fn to_result(&self, text: String) -> TokenSequenceResult {
        TokenSequenceResult {
            seq_id: self.seq_id,
            text,
            all_token_ids: self.token_ids.clone(),
            completion_token_ids: self.completion_tokens(),
            prompt_token_count: self.prompt_token_count,
            completion_token_count: self.completion_token_count(),
            state: self.state,
        }
    }
}

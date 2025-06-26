//! Sequence utilities for inference
//!
//! This module contains a lightweight `Sequence` type used by the
//! `ModelRunner`. The structure closely mirrors the Python implementation
//! but only exposes the functionality required by the current Rust code.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::sampling_params::SamplingParams;

/// Status of a sequence inside the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Preempted,
    Finished,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// Representation of a single token sequence.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub seq_id: u64,
    pub status: SequenceStatus,
    token_ids: Vec<i64>,
    pub last_token: i64,
    pub num_tokens: usize,
    pub num_prompt_tokens: usize,
    pub num_cached_tokens: usize,
    pub block_table: Vec<usize>,
    pub sampling_params: SamplingParams,
}

impl Sequence {
    /// Create a new sequence from the provided tokens and sampling parameters.
    pub fn new(token_ids: Vec<i64>, sampling_params: SamplingParams) -> Self {
        let last_token = *token_ids.last().unwrap_or(&-1);
        let num_tokens = token_ids.len();
        let seq_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            seq_id,
            status: SequenceStatus::Waiting,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            sampling_params,
        }
    }

    /// Return the total number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.num_tokens
    }

    /// Return all token ids as a slice without additional allocation.
    pub fn all_token_ids(&self) -> &[i64] {
        &self.token_ids
    }

    /// Number of KV cache blocks used by this sequence.
    pub fn num_blocks(&self) -> usize {
        (self.len() + Self::block_size() - 1) / Self::block_size()
    }

    /// Block size in tokens.
    pub fn block_size() -> usize {
        256
    }

    /// Borrow a block of tokens without allocation.
    pub fn block_slice(&self, i: usize) -> &[i64] {
        let start = i * Self::block_size();
        let end = usize::min(start + Self::block_size(), self.num_tokens);
        &self.token_ids[start..end]
    }

    /// Append a new token to the sequence.
    pub fn append_token(&mut self, token_id: i64) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        self.num_tokens += 1;
    }

    pub fn num_completion_tokens(&self) -> usize {
        self.num_tokens - self.num_prompt_tokens
    }

    /// Determine whether the sequence should stop given optional eos id.
    pub fn should_stop(&self, eos: Option<i64>) -> bool {
        if !self.sampling_params.ignore_eos {
            if let Some(id) = eos {
                if self.last_token == id {
                    return true;
                }
            }
        }
        self.num_completion_tokens() >= self.sampling_params.max_tokens
    }

    /// Mark the sequence as finished.
    pub fn finish(&mut self) {
        self.status = SequenceStatus::Finished;
    }
}

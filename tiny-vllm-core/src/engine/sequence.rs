//! Sequence utilities for inference
//!
//! This module contains a lightweight `Sequence` type used by the
//! `ModelRunner`. The structure closely mirrors the Python implementation
//! but only exposes the functionality required by the current Rust code.

use crate::sampling_params::SamplingParams;

/// Representation of a single token sequence.
#[derive(Debug, Clone)]
pub struct Sequence {
    token_ids: Vec<i64>,
    /// ID of the last token in the sequence.
    pub last_token: i64,
    /// Mapping of blocks allocated for the KV cache.
    pub block_table: Vec<i32>,
    /// Parameters used for sampling new tokens.
    pub sampling_params: SamplingParams,
}

impl Sequence {
    /// Create a new sequence from the provided tokens and sampling parameters.
    pub fn new(token_ids: Vec<i64>, sampling_params: SamplingParams) -> Self {
        let last_token = *token_ids.last().unwrap_or(&-1);
        Self { token_ids, last_token, block_table: Vec::new(), sampling_params }
    }

    /// Return the total number of tokens in the sequence.
    pub fn len(&self) -> usize {
        self.token_ids.len()
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
}

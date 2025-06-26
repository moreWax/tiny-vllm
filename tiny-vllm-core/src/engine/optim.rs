//! Memory optimization utilities such as the block cache manager.
//!
//! This is a direct translation of the Python `block_manager` module used in
//! the original NanoVLLM project. The implementation manages a table of
//! reusable blocks to store key/value cache segments.

use xxhash_rust::xxh64::Xxh64;

/// Compute a rolling 64-bit hash over a block of token IDs.
///
/// When `prefix` is provided it is hashed first using little endian byte order
/// to match the Python implementation.
pub fn compute_hash(token_ids: &[i64], prefix: Option<u64>) -> u64 {
    let mut hasher = Xxh64::new(0);
    if let Some(p) = prefix {
        hasher.update(&p.to_le_bytes());
    }
    for id in token_ids {
        hasher.update(&id.to_le_bytes());
    }
    hasher.digest()
}

/// Minimal sequence representation storing token IDs and block allocation info.
#[derive(Debug)]
pub struct Sequence {
    pub token_ids: Vec<i64>,
    pub num_cached_tokens: usize,
    pub block_table: Vec<usize>,
    pub block_size: usize,
}

impl Sequence {
    pub fn new(token_ids: Vec<i64>, block_size: usize) -> Self {
        Self {
            token_ids,
            num_cached_tokens: 0,
            block_table: Vec::new(),
            block_size,
        }
    }

    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn num_blocks(&self) -> usize {
        (self.len() + self.block_size - 1) / self.block_size
    }

    pub fn block(&self, i: usize) -> Vec<i64> {
        let start = i * self.block_size;
        let end = usize::min(start + self.block_size, self.len());
        self.token_ids[start..end].to_vec()
    }

    /// Borrow a block without allocation.
    pub fn block_slice(&self, i: usize) -> &[i64] {
        let start = i * self.block_size;
        let end = usize::min(start + self.block_size, self.len());
        &self.token_ids[start..end]
    }

    /// Iterate over blocks as slices.
    pub fn blocks(&self) -> impl Iterator<Item = &[i64]> {
        self.token_ids.chunks(self.block_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::block_manager::BlockManager;

    #[test]
    fn test_compute_hash() {
        let rust_hash = compute_hash(&[1, 2, 3, 4], None);
        // Value computed from the reference Python implementation.
        assert_eq!(rust_hash, 8356527653647720045);
    }

    #[test]
    fn test_allocate_and_deallocate() {
        let mut seq = Sequence::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 4);
        let mut manager = BlockManager::new(4, 4);
        assert!(manager.can_allocate(&seq));
        manager.allocate(&mut seq).unwrap();
        assert_eq!(seq.block_table.len(), seq.num_blocks());
        assert_eq!(seq.num_cached_tokens, 0); // no cache on first allocation

        manager.deallocate(&mut seq);
        assert!(seq.block_table.is_empty());
        assert_eq!(manager.get_stats().free_blocks, 4);
    }
}

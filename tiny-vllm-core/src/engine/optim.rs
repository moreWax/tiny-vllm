//! Memory optimization utilities such as the block cache manager.
//!
//! This is a direct translation of the Python `block_manager` module used in
//! the original NanoVLLM project. The implementation manages a table of
//! reusable blocks to store key/value cache segments.

use std::collections::{HashMap, HashSet, VecDeque};

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

/// A single cache block.
#[derive(Debug)]
pub struct Block {
    pub block_id: usize,
    pub ref_count: usize,
    pub hash: i64,
    pub token_ids: Vec<i64>,
}

impl Block {
    pub fn new(block_id: usize) -> Self {
        Self { block_id, ref_count: 0, hash: -1, token_ids: Vec::new() }
    }

    pub fn update(&mut self, hash: i64, token_ids: Vec<i64>) {
        assert_ne!(hash, -1);
        self.hash = hash;
        self.token_ids = token_ids;
    }

    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids.clear();
    }
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
        Self { token_ids, num_cached_tokens: 0, block_table: Vec::new(), block_size }
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

/// Block manager tracking free and used cache blocks.
#[derive(Debug)]
pub struct BlockManager {
    block_size: usize,
    blocks: Vec<Block>,
    hash_to_block_id: HashMap<i64, usize>,
    free_block_ids: VecDeque<usize>,
    used_block_ids: HashSet<usize>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0);
        let blocks = (0..num_blocks).map(Block::new).collect::<Vec<_>>();
        let free_block_ids = (0..num_blocks).collect::<VecDeque<_>>();
        Self {
            block_size,
            blocks,
            hash_to_block_id: HashMap::new(),
            free_block_ids,
            used_block_ids: HashSet::new(),
        }
    }

    fn allocate_block(&mut self, block_id: usize) -> &mut Block {
        let block = &mut self.blocks[block_id];
        assert_eq!(block.ref_count, 0);
        block.reset();
        if let Some(pos) = self.free_block_ids.iter().position(|&b| b == block_id) {
            self.free_block_ids.remove(pos);
        }
        self.used_block_ids.insert(block_id);
        block
    }

    fn deallocate_block(&mut self, block_id: usize) {
        assert_eq!(self.blocks[block_id].ref_count, 0);
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
    }

    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_block_ids.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) {
        assert!(seq.block_table.is_empty());
        let mut h: Option<u64> = None;
        let mut cache_miss = false;
        for i in 0..seq.num_blocks() {
            let token_ids = seq.block(i);
            h = if token_ids.len() == self.block_size {
                Some(compute_hash(&token_ids, h))
            } else {
                None
            };
            let mut block_id = h
                .and_then(|k| self.hash_to_block_id.get(&(k as i64)).copied())
                .unwrap_or(usize::MAX);
            if block_id == usize::MAX || self.blocks[block_id].token_ids != token_ids {
                cache_miss = true;
            }
            let block: &mut Block = if cache_miss {
                block_id = *self.free_block_ids.front().expect("no free blocks");
                self.allocate_block(block_id)
            } else {
                seq.num_cached_tokens += self.block_size;
                if self.used_block_ids.contains(&block_id) {
                    let b = &mut self.blocks[block_id];
                    b.ref_count += 1;
                    b
                } else {
                    self.allocate_block(block_id)
                }
            };
            if let Some(hash) = h {
                block.update(hash as i64, token_ids.clone());
                self.hash_to_block_id.insert(hash as i64, block_id);
            }
            seq.block_table.push(block_id);
        }
    }

    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            let block = &mut self.blocks[block_id];
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        seq.block_table.clear();
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        let needed = if seq.len() % self.block_size == 1 { 1 } else { 0 };
        self.free_block_ids.len() >= needed
    }

    pub fn may_append(&mut self, seq: &mut Sequence) {
        let last_id = *seq.block_table.last().expect("empty block table");
        let seq_len = seq.len();
        if seq_len % self.block_size == 1 {
            let last_hash = self.blocks[last_id].hash;
            assert_ne!(last_hash, -1);
            let block_id = *self.free_block_ids.front().expect("no free blocks");
            self.allocate_block(block_id);
            seq.block_table.push(block_id);
        } else if seq_len % self.block_size == 0 {
            let prefix = if seq.block_table.len() > 1 {
                self.blocks[seq.block_table[seq.block_table.len() - 2]].hash as u64
            } else {
                u64::MAX
            };
            let token_ids = seq.block(seq.num_blocks() - 1);
            let h = compute_hash(&token_ids, if prefix == u64::MAX { None } else { Some(prefix) });
            let last_block = &mut self.blocks[last_id];
            assert_eq!(last_block.hash, -1);
            last_block.update(h as i64, token_ids);
            self.hash_to_block_id.insert(h as i64, last_block.block_id);
        } else {
            let last_block = &self.blocks[last_id];
            assert_eq!(last_block.hash, -1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        manager.allocate(&mut seq);
        assert_eq!(seq.block_table.len(), seq.num_blocks());
        assert_eq!(seq.num_cached_tokens, 0); // no cache on first allocation

        manager.deallocate(&mut seq);
        assert!(seq.block_table.is_empty());
        assert_eq!(manager.free_block_ids.len(), 4);
    }
}

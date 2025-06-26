//! Key/value cache block manager.
//!
//! This module maintains a pool of reusable blocks for the model KV cache.
//! Blocks are reference counted and deduplicated using a hash of their
//! contents so that shared prefixes across sequences only occupy memory once.

use std::collections::{HashMap, HashSet, VecDeque};

use anyhow::{anyhow, Result};

use crate::engine::optim::{compute_hash as hash_tokens, Sequence};

/// A single block of cached tokens.
#[derive(Debug, Clone)]
pub struct Block {
    pub block_id: usize,
    ref_count: usize,
    hash: Option<u64>,
    token_ids: Vec<i64>,
}

impl Block {
    fn new(block_id: usize) -> Self {
        Self { block_id, ref_count: 0, hash: None, token_ids: Vec::new() }
    }

    fn update(&mut self, hash: u64, token_ids: Vec<i64>) {
        self.hash = Some(hash);
        self.token_ids = token_ids;
    }

    fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = None;
        self.token_ids.clear();
    }

    fn is_free(&self) -> bool { self.ref_count == 0 }

    fn add_ref(&mut self) { self.ref_count += 1; }

    fn release(&mut self) {
        assert!(self.ref_count > 0, "release on zero ref block {}", self.block_id);
        self.ref_count -= 1;
    }
}

/// Memory manager for KV cache blocks.
#[derive(Debug)]
pub struct BlockManager {
    block_size: usize,
    blocks: Vec<Block>,
    hash_to_block: HashMap<u64, usize>,
    free_blocks: VecDeque<usize>,
    used_blocks: HashSet<usize>,
}

impl BlockManager {
    /// Create a new manager with the given number of blocks.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0 && block_size > 0);
        let blocks = (0..num_blocks).map(Block::new).collect();
        Self {
            block_size,
            blocks,
            hash_to_block: HashMap::new(),
            free_blocks: (0..num_blocks).collect(),
            used_blocks: HashSet::new(),
        }
    }

    fn allocate_block(&mut self, block_id: usize) -> &mut Block {
        let block = &mut self.blocks[block_id];
        assert!(block.is_free());
        block.reset();
        if let Some(pos) = self.free_blocks.iter().position(|&id| id == block_id) {
            self.free_blocks.remove(pos);
        }
        self.used_blocks.insert(block_id);
        block
    }

    fn deallocate_block(&mut self, block_id: usize) {
        let block = &mut self.blocks[block_id];
        assert!(block.is_free());
        self.used_blocks.remove(&block_id);
        self.free_blocks.push_back(block_id);
        if let Some(h) = block.hash.take() {
            self.hash_to_block.remove(&h);
        }
    }

    pub fn can_allocate(&self, seq: &Sequence) -> bool {
        self.free_blocks.len() >= seq.num_blocks()
    }

    pub fn allocate(&mut self, seq: &mut Sequence) -> Result<()> {
        if !seq.block_table.is_empty() {
            return Err(anyhow!("sequence already allocated"));
        }
        if !self.can_allocate(seq) {
            return Err(anyhow!("not enough free blocks"));
        }

        let mut prefix: Option<u64> = None;
        let mut seen_miss = false;
        for i in 0..seq.num_blocks() {
            let tokens = seq.block_slice(i);
            let cur_hash = (tokens.len() == self.block_size)
                .then(|| hash_tokens(tokens, prefix));

            let mut use_cache = false;
            let block_id = match (cur_hash, cur_hash.and_then(|h| self.hash_to_block.get(&h).copied())) {
                (Some(_h), Some(existing)) if !seen_miss && self.blocks[existing].token_ids == tokens => {
                    if self.used_blocks.contains(&existing) {
                        self.blocks[existing].add_ref();
                    }
                    use_cache = true;
                    existing
                }
                _ => {
                    seen_miss = true;
                    self.allocate_new_block(cur_hash, tokens)?
                }
            };

            if use_cache {
                seq.num_cached_tokens += self.block_size;
            }
            seq.block_table.push(block_id);
            prefix = cur_hash;
        }
        Ok(())
    }

    fn allocate_new_block(&mut self, hash: Option<u64>, token_ids: &[i64]) -> Result<usize> {
        let block_id = self
            .free_blocks
            .pop_front()
            .ok_or_else(|| anyhow!("no free blocks"))?;
        let block = self.allocate_block(block_id);
        if let Some(h) = hash {
            block.update(h, token_ids.to_vec());
            self.hash_to_block.insert(h, block_id);
        } else {
            block.token_ids.extend_from_slice(token_ids);
        }
        Ok(block_id)
    }

    pub fn deallocate(&mut self, seq: &mut Sequence) {
        while let Some(id) = seq.block_table.pop() {
            let block = &mut self.blocks[id];
            block.release();
            if block.is_free() {
                self.deallocate_block(id);
            }
        }
        seq.num_cached_tokens = 0;
    }

    pub fn can_append(&self, seq: &Sequence) -> bool {
        match seq.len() % self.block_size {

            SEQUENCE_REMAINDER_FOR_APPEND => !self.free_blocks.is_empty(),

            1 => !self.free_blocks.is_empty(),

            _ => true,
        }
    }

    pub fn may_append(&mut self, seq: &mut Sequence) -> Result<()> {
        if seq.block_table.is_empty() {
            return Err(anyhow!("sequence has no blocks"));
        }
        let last_idx = seq.block_table.len() - 1;
        let last_id = seq.block_table[last_idx];
        let prefix = if last_idx > 0 {
            self.blocks[seq.block_table[last_idx - 1]].hash
        } else {
            None
        };
        let last_block = &mut self.blocks[last_id];

        match seq.len() % self.block_size {
            BLOCK_ALIGNMENT_REMAINDER if last_block.hash.is_some() => {

            1 if last_block.hash.is_some() => {

                let new_id = self
                    .free_blocks
                    .pop_front()
                    .ok_or_else(|| anyhow!("no free blocks"))?;
                self.allocate_block(new_id);
                seq.block_table.push(new_id);
            }
            0 if last_block.hash.is_none() => {
                let tokens = seq.block_slice(seq.num_blocks() - 1);
                let h = hash_tokens(tokens, prefix);
                last_block.update(h, tokens.to_vec());
                self.hash_to_block.insert(h, last_id);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn get_stats(&self) -> BlockManagerStats {
        BlockManagerStats {
            total_blocks: self.blocks.len(),
            free_blocks: self.free_blocks.len(),
            used_blocks: self.used_blocks.len(),
            cached_blocks: self.hash_to_block.len(),
            block_size: self.block_size,
        }
    }

    pub fn get_block(&self, id: usize) -> Option<&Block> { self.blocks.get(id) }
    pub fn block_size(&self) -> usize { self.block_size }
    pub fn num_blocks(&self) -> usize { self.blocks.len() }
}

#[derive(Debug, Clone)]
pub struct BlockManagerStats {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub used_blocks: usize,
    pub cached_blocks: usize,
    pub block_size: usize,
}

impl BlockManagerStats {
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 { 0.0 } else { self.used_blocks as f64 / self.total_blocks as f64 * 100.0 }
    }
    pub fn cache_efficiency(&self) -> f64 {
        if self.used_blocks == 0 { 0.0 } else { self.cached_blocks as f64 / self.used_blocks as f64 * 100.0 }
    }
}


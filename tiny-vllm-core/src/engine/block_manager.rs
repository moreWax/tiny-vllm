//! Async, functional, and ergonomic arena for LLM KV-cache blocks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use anyhow::{Result, anyhow};
use tokio::sync::Mutex;
use futures::stream::{FuturesUnordered, StreamExt};

use crate::engine::optim::compute_hash as hash_tokens;
use crate::engine::sequence::Sequence;

/// Trait for reference-counted, hashable chunks.
#[async_trait::async_trait]
pub trait ArenaChunk: Send + Sync {
    fn is_vacant(&self) -> bool;
    fn get_refs(&self) -> usize;
    async fn add_ref(&mut self);
    async fn sub_ref(&mut self);
    fn get_tokens(&self) -> &[i64];
    fn set_tokens(&mut self, tokens: &[i64]);
    fn assign_hash(&mut self, hash: u64);
    fn get_hash(&self) -> Option<u64>;
    fn clear_hash(&mut self);
    fn index(&self) -> usize;
}

#[derive(Debug)]
pub struct AsyncChunk {
    idx: usize,
    refs: usize,
    hash: Option<u64>,
    tokens: Vec<i64>,
}

#[async_trait::async_trait]
impl ArenaChunk for AsyncChunk {
    fn is_vacant(&self) -> bool { self.refs == 0 }
    fn get_refs(&self) -> usize { self.refs }
    async fn add_ref(&mut self) { self.refs += 1; }
    async fn sub_ref(&mut self) {
        match self.refs {
            0 => panic!("Negative refs for chunk {}", self.idx),
            n => self.refs = n - 1,
        }
    }
    fn get_tokens(&self) -> &[i64] { &self.tokens }
    fn set_tokens(&mut self, toks: &[i64]) {
        self.tokens.clear();
        self.tokens.extend_from_slice(toks);
    }
    fn assign_hash(&mut self, h: u64) { self.hash = Some(h); }
    fn get_hash(&self) -> Option<u64> { self.hash }
    fn clear_hash(&mut self) { self.hash = None; }
    fn index(&self) -> usize { self.idx }
}

pub struct AsyncArena {
    chunks: Vec<Arc<Mutex<AsyncChunk>>>,
    hash_idx: Mutex<HashMap<u64, usize>>,
    spare: Mutex<VecDeque<usize>>,
    engaged: Mutex<HashSet<usize>>,
    unit: usize,
}

impl AsyncArena {
    pub fn new(cap: usize, unit: usize) -> Self {
        Self {
            chunks: (0..cap).map(|i| Arc::new(Mutex::new(AsyncChunk { idx: i, refs: 0, hash: None, tokens: Vec::new() }))).collect(),
            hash_idx: Mutex::new(HashMap::new()),
            spare: Mutex::new((0..cap).collect()),
            engaged: Mutex::new(HashSet::new()),
            unit,
        }
    }

    async fn activate_chunk(&self, idx: usize) -> Arc<Mutex<AsyncChunk>> {
        let chunk = self.chunks[idx].clone();
        let mut c = chunk.lock().await;
        match c.is_vacant() {
            true => {
                c.refs = 1;
                c.hash = None;
                c.tokens.clear();
                drop(c);
                let mut spare = self.spare.lock().await;
                if let Some(pos) = spare.iter().position(|&i| i == idx) {
                    spare.remove(pos);
                }
                drop(spare);
                self.engaged.lock().await.insert(idx);
                chunk
            }
            false => panic!("Chunk {} already active", idx),
        }
    }

    async fn retire_chunk(&self, idx: usize) {
        let chunk = self.chunks[idx].clone();
        let mut c = chunk.lock().await;
        match c.is_vacant() {
            true => {
                self.engaged.lock().await.remove(&idx);
                self.spare.lock().await.push_back(idx);
                c.hash.take().map(|h| self.hash_idx.lock().await.remove(&h));
            }
            false => panic!("Retire called on in-use chunk {}", idx),
        }
    }

    pub async fn can_reserve(&self, seq: &Sequence) -> bool {
        self.spare.lock().await.len() >= seq.num_blocks()
    }

    pub async fn reserve(&self, seq: &mut Sequence) -> Result<()> {
        match seq.block_table.is_empty() {
            false => return Err(anyhow!("Sequence already allocated")),
            true => (),
        }
        match self.can_reserve(seq).await {
            false => return Err(anyhow!("Arena exhausted")),
            true => (),
        }

        let mut prev_hash = None;
        let mut missed = false;

        // Use FuturesUnordered to parallelize chunk search/allocation if needed
        for idx in 0..seq.num_blocks() {
            let toks = seq.block_slice(idx);
            let is_full = toks.len() == self.unit;
            let hash = match is_full { true => Some(hash_tokens(toks, prev_hash)), false => None };

            let (reused, chunk_idx) = match hash.and_then(|h| self.hash_idx.lock().await.get(&h).copied()) {
                Some(existing) if !missed && self.chunks[existing].lock().await.tokens == toks => {
                    let chunk = self.chunks[existing].clone();
                    let mut guard = chunk.lock().await;
                    match self.engaged.lock().await.contains(&existing) {
                        true => guard.add_ref().await,
                        false => { drop(guard); self.activate_chunk(existing).await; }
                    }
                    (true, existing)
                }
                _ => {
                    missed = true;
                    let id = self.new_chunk(hash, toks).await?;
                    (false, id)
                }
            };

            match reused {
                true => seq.num_cached_tokens += self.unit,
                false => (),
            }
            seq.block_table.push(chunk_idx);
            prev_hash = hash;
        }
        Ok(())
    }

    async fn new_chunk(&self, hash: Option<u64>, toks: &[i64]) -> Result<usize> {
        match self.spare.lock().await.pop_front() {
            Some(idx) => {
                let chunk = self.activate_chunk(idx).await;
                let mut guard = chunk.lock().await;
                match hash {
                    Some(h) => {
                        guard.assign_hash(h);
                        guard.set_tokens(toks);
                        self.hash_idx.lock().await.insert(h, idx);
                    }
                    None => guard.set_tokens(toks),
                }
                Ok(idx)
            }
            None => Err(anyhow!("No chunks available")),
        }
    }

    pub async fn release(&self, seq: &mut Sequence) {
        let mut block_indices = std::mem::take(&mut seq.block_table);
        let futures: FuturesUnordered<_> = block_indices.drain(..)
            .map(|idx| async move {
                let chunk = self.chunks[idx].clone();
                let mut guard = chunk.lock().await;
                guard.sub_ref().await;
                match guard.is_vacant() {
                    true => self.retire_chunk(idx).await,
                    false => (),
                }
            }).collect();
        futures.collect::<Vec<()>>().await;
        seq.num_cached_tokens = 0;
    }

    pub async fn can_extend(&self, seq: &Sequence) -> bool {
        match seq.len() % self.unit {
            1 => !self.spare.lock().await.is_empty(),
            _ => true,
        }
    }

    pub async fn maybe_extend(&self, seq: &mut Sequence) -> Result<()> {
        match seq.block_table.is_empty() {
            true => return Err(anyhow!("No chunks allocated")),
            false => (),
        }
        let last = seq.block_table.len() - 1;
        let last_idx = seq.block_table[last];
        let prev_hash = match last { 0 => None, n => self.chunks[seq.block_table[n-1]].lock().await.hash };
        let mut last_chunk = self.chunks[last_idx].lock().await;

        match (seq.len() % self.unit, last_chunk.hash.is_some()) {
            (1, true) => {
                match self.spare.lock().await.pop_front() {
                    Some(idx) => {
                        drop(last_chunk);
                        self.activate_chunk(idx).await;
                        seq.block_table.push(idx);
                    }
                    None => return Err(anyhow!("No chunks available for extension")),
                }
            }
            (0, false) => {
                let toks = seq.block_slice(seq.num_blocks() - 1);
                let h = hash_tokens(toks, prev_hash);
                last_chunk.assign_hash(h);
                last_chunk.set_tokens(toks);
                self.hash_idx.lock().await.insert(h, last_idx);
            }
            _ => (),
        }
        Ok(())
    }

    pub async fn stats(&self) -> ArenaStats {
        ArenaStats {
            total: self.chunks.len(),
            available: self.spare.lock().await.len(),
            engaged: self.engaged.lock().await.len(),
            deduped: self.hash_idx.lock().await.len(),
            unit: self.unit,
        }
    }

    pub async fn get_chunk(&self, idx: usize) -> Option<Arc<Mutex<AsyncChunk>>> {
        self.chunks.get(idx).cloned()
    }
    pub fn unit_size(&self) -> usize { self.unit }
    pub fn chunk_count(&self) -> usize { self.chunks.len() }
}

#[derive(Debug, Clone)]
pub struct ArenaStats {
    pub total: usize,
    pub available: usize,
    pub engaged: usize,
    pub deduped: usize,
    pub unit: usize,
}

impl ArenaStats {
    pub fn utilization(&self) -> f64 {
        match self.total {
            0 => 0.0,
            n => (self.engaged as f64 / n as f64) * 100.0,
        }
    }
    pub fn dedup_efficiency(&self) -> f64 {
        match self.engaged {
            0 => 0.0,
            n => (self.deduped as f64 / n as f64) * 100.0,
        }
    }
}

//! Scheduler for managing sequence execution and batching
//!
//! This module implements a simplified variant of the Python scheduler used in
//! the reference implementation.  It keeps allocations to a minimum and avoids
//! heap indirection by using `VecDeque` for queue management.

use std::collections::VecDeque;

use anyhow::{anyhow, Result};

use crate::config::VllmConfig;
use crate::engine::block_manager::BlockManager;
use crate::engine::sequence::{Sequence, SequenceStatus};

/// Core scheduler responsible for batching sequences.
#[derive(Debug)]
pub struct Scheduler {
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    eos_token_id: i64,
    block_manager: BlockManager,
    waiting: VecDeque<Sequence>,
    running: VecDeque<Sequence>,
}

impl Scheduler {
    /// Create a new scheduler using the provided configuration.
    pub fn new(cfg: &VllmConfig) -> Self {
        let block_manager = BlockManager::new(
            cfg.num_kvcache_blocks.max(1) as usize,
            cfg.kvcache_block_size,
        );
        Self {
            max_num_seqs: cfg.max_num_seqs,
            max_num_batched_tokens: cfg.max_num_batched_tokens,
            eos_token_id: cfg.eos,
            block_manager,
            waiting: VecDeque::new(),
            running: VecDeque::new(),
        }
    }

    /// Whether there are no active sequences.
    pub fn is_finished(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }

    /// Add a sequence to the waiting queue.
    pub fn add(&mut self, seq: Sequence) {
        self.waiting.push_back(seq);
    }

    /// Schedule sequences for execution. Returns the batch and a flag
    /// indicating whether this is a prefill step.
    pub fn schedule(&mut self) -> Result<(Vec<Sequence>, bool)> {
        if let Some(batch) = self.try_prefill()? {
            return Ok((batch, true));
        }
        let batch = self.try_decode()?;
        Ok((batch, false))
    }

    fn try_prefill(&mut self) -> Result<Option<Vec<Sequence>>> {
        if self.waiting.is_empty() {
            return Ok(None);
        }
        let mut num_seqs = 0;
        let mut num_tokens = 0;
        let mut batch = Vec::new();
        while let Some(front) = self.waiting.front() {
            if num_seqs >= self.max_num_seqs {
                break;
            }
            let req_tokens = front.len() - front.num_cached_tokens;
            if num_tokens + req_tokens > self.max_num_batched_tokens {
                break;
            }
            if !self.block_manager.can_allocate(front) {
                break;
            }
            let mut seq = self.waiting.pop_front().expect("front checked");
            self.block_manager.allocate(&mut seq)?;
            seq.status = SequenceStatus::Running;
            num_seqs += 1;
            num_tokens += req_tokens;
            self.running.push_back(seq.clone());
            batch.push(seq);
        }
        Ok((!batch.is_empty()).then_some(batch))
    }

    fn try_decode(&mut self) -> Result<Vec<Sequence>> {
        let mut batch = Vec::new();
        let mut new_running = VecDeque::new();

        'outer: while let Some(mut seq) = self.running.pop_front() {
            if batch.len() >= self.max_num_seqs {
                new_running.push_back(seq);
                continue;
            }

            while !self.block_manager.can_append(&seq) {
                if let Some(victim) = self.running.pop_back() {
                    self.preempt(victim);
                } else if let Some(victim) = batch.pop() {
                    self.preempt(victim);
                } else {
                    self.preempt(seq);
                    continue 'outer;
                }
            }

            self.block_manager.may_append(&mut seq)?;
            batch.push(seq.clone());
            new_running.push_back(seq);
        }

        self.running = new_running;
        if batch.is_empty() {
            return Err(anyhow!("no sequences to decode"));
        }
        Ok(batch)
    }

    fn preempt(&mut self, mut seq: Sequence) {
        seq.status = SequenceStatus::Preempted;
        self.block_manager.deallocate(&mut seq);
        self.waiting.push_front(seq);
    }

    /// Update sequences after running the model.
    pub fn postprocess(&mut self, mut seqs: Vec<Sequence>, token_ids: Vec<i64>) -> Result<()> {
        if seqs.len() != token_ids.len() {
            return Err(anyhow!("length mismatch"));
        }
        for (mut seq, id) in seqs.into_iter().zip(token_ids) {
            seq.append_token(id);
            if seq.should_stop(Some(self.eos_token_id)) {
                seq.finish();
                self.block_manager.deallocate(&mut seq);
                self.remove_running(seq.seq_id);
            } else {
                self.update_running(seq);
            }
        }
        Ok(())
    }

    fn remove_running(&mut self, seq_id: u64) {
        self.running.retain(|s| s.seq_id != seq_id);
    }

    fn update_running(&mut self, updated: Sequence) {
        if let Some(pos) = self.running.iter().position(|s| s.seq_id == updated.seq_id) {
            self.running.remove(pos);
            self.running.insert(pos, updated);
        } else {
            self.running.push_back(updated);
        }
    }

    /// Percentage of used cache blocks (0.0 - 1.0).
    pub fn memory_pressure(&self) -> f64 {
        let stats = self.block_manager.get_stats();
        if stats.total_blocks == 0 {
            0.0
        } else {
            1.0 - (stats.free_blocks as f64 / stats.total_blocks as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_params::SamplingParams;

    fn cfg() -> VllmConfig {
        VllmConfig { max_num_batched_tokens: 32, max_num_seqs: 4, ..Default::default() }
    }

    #[test]
    fn test_add_and_finish() {
        let mut sch = Scheduler::new(&cfg());
        let mut sp = SamplingParams::default();
        sp.max_tokens = 1;
        let seq = Sequence::new(vec![1, 2, 3], sp);
        sch.add(seq);
        assert!(!sch.is_finished());
        let (batch, is_prefill) = sch.schedule().unwrap();
        assert!(is_prefill);
        let tokens = vec![4];
        sch.postprocess(batch, tokens).unwrap();
        assert!(sch.is_finished());
    }
}

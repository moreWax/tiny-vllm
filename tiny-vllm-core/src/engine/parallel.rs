//! Stubbed parallel execution utilities.
//!
//! This module provides minimal implementations of the parallel primitives
//! used throughout the engine. The real implementation will provide true
//! distributed execution backed by CUDA/NCCL. For now we simply maintain the
//! world size and rank in-process and perform no communication.

use std::sync::{Mutex, OnceLock};

#[derive(Debug, Default, Clone, Copy)]
struct ParallelState {
    world_size: usize,
    rank: usize,
    initialized: bool,
}

static STATE: OnceLock<Mutex<ParallelState>> = OnceLock::new();

fn state() -> &'static Mutex<ParallelState> {
    STATE.get_or_init(|| Mutex::new(ParallelState::default()))
}

/// Initialize the parallel runtime.
///
/// In this stub implementation we simply record the provided rank and
/// world size. The real implementation will set up inter-process
/// communication.
pub fn init_process_group(world_size: usize, rank: usize) {
    let mut s = state().lock().unwrap();
    s.world_size = world_size.max(1);
    s.rank = rank.min(world_size.saturating_sub(1));
    s.initialized = true;
}

/// Destroy the parallel runtime, resetting it to defaults.
pub fn destroy_process_group() {
    let mut s = state().lock().unwrap();
    *s = ParallelState::default();
}

/// Return the world size of the current process group.
pub fn get_world_size() -> usize {
    let s = state().lock().unwrap();
    if s.initialized { s.world_size } else { 1 }
}

/// Return the rank of the current process within the process group.
pub fn get_rank() -> usize {
    let s = state().lock().unwrap();
    if s.initialized { s.rank } else { 0 }
}

/// Synchronize all processes. In the stub this is a no-op.
pub fn barrier() {
    // no-op
}

/// Perform an all-reduce operation on `value` in-place.
///
/// The stub implementation leaves `value` unchanged.
pub fn all_reduce<T>(_value: &mut T) {
    // no-op
}

/// Gather `input` to `output` on the `root` process.
///
/// The stub simply clones `input` into `output` when called on the root.
pub fn gather<T: Clone>(input: &T, output: Option<&mut Vec<T>>, root: usize) {
    if get_rank() == root {
        if let Some(out) = output {
            out.push(input.clone());
        }
    }
}


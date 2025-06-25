//! Engine utilities for model execution.

use std::sync::OnceLock;

pub mod parallel;

pub mod optim;

pub mod session;
pub mod block_manager;
pub mod llm_engine;
pub mod model_runner;
pub mod scheduler;
pub mod sequence;

/// Core inference engine responsible for running models.
#[derive(Debug)]
pub struct Engine {
    num_threads: usize,
}

impl Engine {
    /// Create a new engine instance using the provided number of threads.
    pub fn new(num_threads: usize) -> Self {
        Self { num_threads: num_threads.max(1) }
    }

    /// Return the number of worker threads used by the engine.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

static GLOBAL_ENGINE: OnceLock<Engine> = OnceLock::new();

/// Initialize a global engine with the given number of threads.
pub fn init_global(num_threads: usize) {
    let _ = GLOBAL_ENGINE.set(Engine::new(num_threads));
}

/// Get a reference to the global engine, initializing it on first use with one thread.
pub fn global() -> &'static Engine {
    GLOBAL_ENGINE.get_or_init(|| Engine::new(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_basic() {
        let engine = Engine::new(4);
        assert_eq!(engine.num_threads(), 4);
    }
}
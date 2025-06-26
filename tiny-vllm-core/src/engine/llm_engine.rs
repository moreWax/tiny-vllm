//! High level inference engine.
//!
//! This module exposes a small synchronous inference engine built on top of
//! [`InferenceQueue`](crate::engine::parallel::InferenceQueue).  It mirrors the
//! behaviour of the original Python implementation while remaining fully
//! self contained.  The engine manages a model instance and a pool of worker
//! threads to execute requests concurrently.

use std::sync::Arc;

use crate::config::VllmConfig;
use crate::engine::parallel::InferenceQueue;
use crate::model::Model;

/// Engine state during runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineState {
    Running,
    Stopped,
}

impl Default for EngineState {
    fn default() -> Self {
        EngineState::Running
    }
}

/// Core engine used for batched text generation.
#[derive(Debug)]
pub struct LlmEngine {
    queue: InferenceQueue,
    config: VllmConfig,
    state: EngineState,
}

impl LlmEngine {
    /// Create a new engine instance from `config`.
    pub fn new(config: VllmConfig) -> Self {
        let workers = config.tensor_parallel_size.max(1);
        let model = Arc::new(Model::new(config.model.clone()));
        let queue = InferenceQueue::new(workers, model);
        Self {
            queue,
            config,
            state: EngineState::Running,
        }
    }

    /// Instantiate an engine using only the model identifier.
    pub fn from_model(model: String) -> Self {
        Self::new(VllmConfig {
            model,
            ..Default::default()
        })
    }

    /// Submit `prompts` for inference and wait for all results.
    pub fn generate<I, S>(&self, prompts: I) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        prompts
            .into_iter()
            .map(|p| self.queue.submit(p.into()))
            .map(|h| h.wait())
            .flatten()
            .collect()
    }

    /// Return a reference to the engine configuration.
    pub fn config(&self) -> &VllmConfig {
        &self.config
    }

    /// Whether the engine is still running.
    pub fn is_running(&self) -> bool {
        matches!(self.state, EngineState::Running)
    }

    /// Shut down the engine and stop accepting new requests.
    pub fn shutdown(&mut self) {
        self.state = EngineState::Stopped;
    }
}

/// Builder used to construct [`LlmEngine`] instances.
pub struct LlmEngineBuilder {
    config: VllmConfig,
}

impl LlmEngineBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: VllmConfig::default(),
        }
    }

    /// Set the model identifier.
    pub fn model<M: Into<String>>(mut self, model: M) -> Self {
        self.config.model = model.into();
        self
    }

    /// Set the maximum number of sequences processed in parallel.
    pub fn max_num_seqs(mut self, n: usize) -> Self {
        self.config.max_num_seqs = n;
        self
    }

    /// Set the maximum number of batched tokens per step.
    pub fn max_num_batched_tokens(mut self, n: usize) -> Self {
        self.config.max_num_batched_tokens = n;
        self
    }

    /// Set the tensor parallel world size.
    pub fn tensor_parallel_size(mut self, n: usize) -> Self {
        self.config.tensor_parallel_size = n;
        self
    }

    /// Build the engine from the current configuration.
    pub fn build(self) -> LlmEngine {
        LlmEngine::new(self.config)
    }
}

impl Default for LlmEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_generation() {
        let engine = LlmEngineBuilder::new()
            .model("demo".to_string())
            .tensor_parallel_size(2)
            .build();
        let prompts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut outputs = engine.generate(prompts);
        outputs.sort();
        assert_eq!(outputs.len(), 3);
        assert!(outputs.iter().all(|o| o.starts_with("demo:")));
    }

    #[test]
    fn test_engine_state() {
        let mut engine = LlmEngineBuilder::new().model("x").build();
        assert!(engine.is_running());
        engine.shutdown();
        assert!(!engine.is_running());
    }
}

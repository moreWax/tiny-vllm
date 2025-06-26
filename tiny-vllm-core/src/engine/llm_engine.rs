use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{BoxStream, FuturesUnordered, StreamExt};
use tokio::sync::{Mutex, mpsc};

use crate::config::VllmConfig;
use crate::engine::parallel::InferenceQueue;
use crate::model::Model;

// =============================
// Engine State
// =============================

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

// =============================
// Inference Result
// =============================

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub prompt_idx: usize,
    pub output: String,
}

// =============================
// Scheduler Trait
// =============================

#[async_trait]
pub trait Scheduler: Send + Sync {
    type Request: Send + 'static;
    type Output: Send + 'static;

    async fn schedule(&self, requests: Vec<Self::Request>) -> Vec<Self::Output>;
    fn schedule_stream(&self, requests: Vec<Self::Request>) -> BoxStream<'static, Self::Output>;
}

// =============================
// Scheduler Config Enum
// =============================

pub enum SchedulerConfig {
    Default,
    Priority,
    Deadline,
    DynamicBatch { batch_window: Duration },
}

// =============================
// Scheduler Implementations
// =============================

// -- DefaultScheduler --

pub struct DefaultScheduler;

#[async_trait]
impl Scheduler for DefaultScheduler {
    type Request = (usize, tokio::task::JoinHandle<String>);
    type Output = InferenceResult;

    async fn schedule(&self, requests: Vec<Self::Request>) -> Vec<Self::Output> {
        let mut results = futures::future::join_all(
            requests.into_iter().map(|(idx, handle)| async move {
                let output = handle.await.unwrap_or_else(|_| "".to_string());
                InferenceResult { prompt_idx: idx, output }
            })
        ).await;
        results.sort_by_key(|r| r.prompt_idx);
        results
    }

    fn schedule_stream(&self, requests: Vec<Self::Request>) -> BoxStream<'static, Self::Output> {
        let stream = FuturesUnordered::from_iter(requests.into_iter().map(|(idx, handle)| async move {
            let output = handle.await.unwrap_or_else(|_| "".to_string());
            InferenceResult { prompt_idx: idx, output }
        }));
        Box::pin(stream)
    }
}

// -- PriorityScheduler --

#[derive(Clone)]
pub struct PriorityRequest {
    pub priority: u32,
    pub idx: usize,
    pub handle: tokio::task::JoinHandle<String>,
}

pub struct PriorityScheduler;

#[async_trait]
impl Scheduler for PriorityScheduler {
    type Request = PriorityRequest;
    type Output = InferenceResult;

    async fn schedule(&self, mut requests: Vec<Self::Request>) -> Vec<Self::Output> {
        requests.sort_by_key(|r| std::cmp::Reverse(r.priority));
        let mut results = futures::future::join_all(
            requests.into_iter().map(|r| async move {
                let output = r.handle.await.unwrap_or_else(|_| "".to_string());
                InferenceResult { prompt_idx: r.idx, output }
            })
        ).await;
        results.sort_by_key(|r| r.prompt_idx);
        results
    }

    fn schedule_stream(&self, mut requests: Vec<Self::Request>) -> BoxStream<'static, Self::Output> {
        requests.sort_by_key(|r| std::cmp::Reverse(r.priority));
        let stream = FuturesUnordered::from_iter(requests.into_iter().map(|r| async move {
            let output = r.handle.await.unwrap_or_else(|_| "".to_string());
            InferenceResult { prompt_idx: r.idx, output }
        }));
        Box::pin(stream)
    }
}

// -- DeadlineScheduler --

#[derive(Clone)]
pub struct DeadlineRequest {
    pub deadline: Instant,
    pub idx: usize,
    pub handle: tokio::task::JoinHandle<String>,
}

pub struct DeadlineScheduler;

#[async_trait]
impl Scheduler for DeadlineScheduler {
    type Request = DeadlineRequest;
    type Output = InferenceResult;

    async fn schedule(&self, mut requests: Vec<Self::Request>) -> Vec<Self::Output> {
        requests.sort_by_key(|r| r.deadline);
        let mut results = futures::future::join_all(
            requests.into_iter().map(|r| async move {
                let output = r.handle.await.unwrap_or_else(|_| "".to_string());
                InferenceResult { prompt_idx: r.idx, output }
            })
        ).await;
        results.sort_by_key(|r| r.prompt_idx);
        results
    }

    fn schedule_stream(&self, mut requests: Vec<Self::Request>) -> BoxStream<'static, Self::Output> {
        requests.sort_by_key(|r| r.deadline);
        let stream = FuturesUnordered::from_iter(requests.into_iter().map(|r| async move {
            let output = r.handle.await.unwrap_or_else(|_| "".to_string());
            InferenceResult { prompt_idx: r.idx, output }
        }));
        Box::pin(stream)
    }
}

// -- DynamicBatchScheduler --
// Batches requests over a fixed window and processes them together.

pub struct DynamicBatchRequest {
    pub idx: usize,
    pub handle: tokio::task::JoinHandle<String>,
}

pub struct DynamicBatchScheduler {
    batch_window: Duration,
}

impl DynamicBatchScheduler {
    pub fn new(batch_window: Duration) -> Self {
        Self { batch_window }
    }
}

#[async_trait]
impl Scheduler for DynamicBatchScheduler {
    type Request = DynamicBatchRequest;
    type Output = InferenceResult;

    async fn schedule(&self, requests: Vec<Self::Request>) -> Vec<Self::Output> {
        // All requests already collected; just run them
        let mut results = futures::future::join_all(
            requests.into_iter().map(|r| async move {
                let output = r.handle.await.unwrap_or_else(|_| "".to_string());
                InferenceResult { prompt_idx: r.idx, output }
            })
        ).await;
        results.sort_by_key(|r| r.prompt_idx);
        results
    }

    fn schedule_stream(&self, requests: Vec<Self::Request>) -> BoxStream<'static, Self::Output> {
        // For demo: yield outputs as they finish (could be grouped in batches for real batching)
        let stream = FuturesUnordered::from_iter(requests.into_iter().map(|r| async move {
            let output = r.handle.await.unwrap_or_else(|_| "".to_string());
            InferenceResult { prompt_idx: r.idx, output }
        }));
        Box::pin(stream)
    }
}

// =============================
// SchedulerSelector
// =============================

enum SchedulerSelector {
    Default(Arc<DefaultScheduler>),
    Priority(Arc<PriorityScheduler>),
    Deadline(Arc<DeadlineScheduler>),
    DynamicBatch(Arc<DynamicBatchScheduler>),
}

impl SchedulerSelector {
    fn default() -> Self {
        SchedulerSelector::Default(Arc::new(DefaultScheduler))
    }
}

// =============================
// LlmEngine Struct
// =============================

pub struct LlmEngine {
    queue: Arc<InferenceQueue>,
    config: Arc<VllmConfig>,
    state: Arc<Mutex<EngineState>>,
    scheduler: SchedulerSelector,
}

impl LlmEngine {
    pub async fn new(config: VllmConfig, scheduler_cfg: SchedulerConfig) -> Self {
        let workers = config.tensor_parallel_size.max(1);
        let model = Arc::new(Model::new(config.model.clone()).await);
        let queue = Arc::new(InferenceQueue::new(workers, model).await);

        let scheduler = match scheduler_cfg {
            SchedulerConfig::Default => SchedulerSelector::Default(Arc::new(DefaultScheduler)),
            SchedulerConfig::Priority => SchedulerSelector::Priority(Arc::new(PriorityScheduler)),
            SchedulerConfig::Deadline => SchedulerSelector::Deadline(Arc::new(DeadlineScheduler)),
            SchedulerConfig::DynamicBatch { batch_window } =>
                SchedulerSelector::DynamicBatch(Arc::new(DynamicBatchScheduler::new(batch_window))),
        };

        Self {
            queue,
            config: Arc::new(config),
            state: Arc::new(Mutex::new(EngineState::Running)),
            scheduler,
        }
    }

    /// Batch generate
    pub async fn generate<I, S2>(&self, prompts: I) -> Vec<InferenceResult>
    where
        I: IntoIterator<Item = S2>,
        S2: AsRef<str> + Send + 'static,
    {
        match &self.scheduler {
            SchedulerSelector::Default(sched) => {
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| (idx, tokio::spawn(self.queue.submit(p.as_ref().to_string()))))
                    .collect();
                sched.schedule(requests).await
            }
            SchedulerSelector::Priority(sched) => {
                // Example: prioritize even indices higher
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| PriorityRequest {
                        priority: if idx % 2 == 0 { 10 } else { 1 },
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule(requests).await
            }
            SchedulerSelector::Deadline(sched) => {
                let now = Instant::now();
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| DeadlineRequest {
                        deadline: now + Duration::from_millis((idx * 50) as u64),
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule(requests).await
            }
            SchedulerSelector::DynamicBatch(sched) => {
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| DynamicBatchRequest {
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule(requests).await
            }
        }
    }

    /// Streaming generation: yields each result as soon as it's ready.
    pub fn generate_stream<I, S2>(
        &self,
        prompts: I,
    ) -> BoxStream<'static, InferenceResult>
    where
        I: IntoIterator<Item = S2>,
        S2: AsRef<str> + Send + 'static,
    {
        match &self.scheduler {
            SchedulerSelector::Default(sched) => {
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| (idx, tokio::spawn(self.queue.submit(p.as_ref().to_string()))))
                    .collect();
                sched.schedule_stream(requests)
            }
            SchedulerSelector::Priority(sched) => {
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| PriorityRequest {
                        priority: if idx % 2 == 0 { 10 } else { 1 },
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule_stream(requests)
            }
            SchedulerSelector::Deadline(sched) => {
                let now = Instant::now();
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| DeadlineRequest {
                        deadline: now + Duration::from_millis((idx * 50) as u64),
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule_stream(requests)
            }
            SchedulerSelector::DynamicBatch(sched) => {
                let requests: Vec<_> = prompts
                    .into_iter()
                    .enumerate()
                    .map(|(idx, p)| DynamicBatchRequest {
                        idx,
                        handle: tokio::spawn(self.queue.submit(p.as_ref().to_string())),
                    })
                    .collect();
                sched.schedule_stream(requests)
            }
        }
    }

    pub fn config(&self) -> Arc<VllmConfig> {
        Arc::clone(&self.config)
    }

    pub async fn is_running(&self) -> bool {
        matches!(*self.state.lock().await, EngineState::Running)
    }

    pub async fn shutdown(&self) {
        let mut state = self.state.lock().await;
        *state = EngineState::Stopped;
    }
}

// =============================
// Example usage
// =============================

// #[tokio::main]
// async fn main() {
//     let config = VllmConfig::default();
//     let engine = LlmEngine::new(config, SchedulerConfig::Priority).await;
//     let prompts = vec!["Hello", "World"];
//     let results = engine.generate(prompts).await;
//     for r in results {
//         println!("Prompt {}: {}", r.prompt_idx, r.output);
//     }
//     // For streaming:
//     // let mut stream = engine.generate_stream(vec!["foo", "bar"]);
//     // while let Some(res) = stream.next().await {
//     //     println!("Streaming: {:?}", res);
//     // }
// }

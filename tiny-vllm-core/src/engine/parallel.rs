//! Parallel execution utilities.
//!
//! The initial version of this crate only provided stub functions so the rest
//! of the engine could compile.  This file now contains a very small thread
//! pool implementation used by the inference engine.  It is **not** a drop-in
//! replacement for the distributed setup of the original project, but it allows
//! running model computations on multiple threads within a single process.  The
//! process group APIs remain available and simply track the world size and rank
//! for now.

use std::sync::{Mutex, OnceLock};

use std::sync::atomic::{AtomicBool, AtomicUsize};

static WORLD_SIZE: OnceLock<AtomicUsize> = OnceLock::new();
static RANK: OnceLock<AtomicUsize> = OnceLock::new();
static INITIALIZED: OnceLock<AtomicBool> = OnceLock::new();

#[derive(Default)]
struct ParallelState {
    world_size: usize,
    rank: usize,
    initialized: bool,
}

static STATE: OnceLock<Mutex<ParallelState>> = OnceLock::new();

fn state() -> &'static Mutex<ParallelState> {
    STATE.get_or_init(|| Mutex::new(ParallelState::default()))
}

fn world_size() -> &'static AtomicUsize {
    WORLD_SIZE.get_or_init(|| AtomicUsize::new(1))
}

fn rank() -> &'static AtomicUsize {
    RANK.get_or_init(|| AtomicUsize::new(0))
}

fn initialized() -> &'static AtomicBool {
    INITIALIZED.get_or_init(|| AtomicBool::new(false))
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
    if s.initialized {
        s.world_size
    } else {
        1
    }
}

/// Return the rank of the current process within the process group.
pub fn get_rank() -> usize {
    let s = state().lock().unwrap();
    if s.initialized {
        s.rank
    } else {
        0
    }
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

// ----- Thread pool implementation -----

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

type Job = Box<dyn FnOnce() + Send + 'static>;

enum Message {
    Job(Job),
    Terminate,
}

/// Simple thread pool for executing jobs in parallel.
pub struct ThreadPool {
    sender: mpsc::Sender<Message>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl ThreadPool {
    /// Create a new thread pool with `size` worker threads.
    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let (tx, rx) = mpsc::channel::<Message>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(size);
        for _ in 0..size {
            let r = Arc::clone(&rx);
            workers.push(thread::spawn(move || loop {
                let msg = { r.lock().unwrap().recv().unwrap() };
                match msg {
                    Message::Job(job) => job(),
                    Message::Terminate => break,
                }
            }));
        }
        Self {
            sender: tx,
            workers,
        }
    }

    /// Execute a function on the thread pool.
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender.send(Message::Job(Box::new(f))).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        for _ in &self.workers {
            let _ = self.sender.send(Message::Terminate);
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

// ----- Task scheduler built on top of the thread pool -----

/// Handle returned when a task is scheduled. The result can be obtained using
/// [`TaskHandle::join`].
pub struct TaskHandle<R> {
    receiver: mpsc::Receiver<R>,
}

impl<R> TaskHandle<R> {
    /// Wait for the task to complete and return its result.
    pub fn join(self) -> R {
        self.receiver.recv().unwrap()
    }
}

/// Simple scheduler that dispatches work to a [`ThreadPool`].
pub struct TaskScheduler {
    pool: ThreadPool,
}

impl TaskScheduler {
    /// Create a scheduler backed by `num_threads` workers.
    pub fn new(num_threads: usize) -> Self {
        Self {
            pool: ThreadPool::new(num_threads.max(1)),
        }
    }

    /// Spawn a task returning a [`TaskHandle`].
    pub fn spawn<F, R>(&self, f: F) -> TaskHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        self.pool.execute(move || {
            let res = f();
            let _ = tx.send(res);
        });
        TaskHandle { receiver: rx }
    }

    /// Convenience helper to wait for multiple tasks.
    pub fn join_all<R>(handles: Vec<TaskHandle<R>>) -> Vec<R> {
        handles.into_iter().map(|h| h.join()).collect()
    }
}

/// Apply `func` to each item in `inputs` using up to `num_threads` threads.
pub fn parallel_map<I, O, F>(inputs: Vec<I>, func: F, num_threads: usize) -> Vec<O>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(I) -> O + Send + Sync + 'static,
{
    if num_threads <= 1 || inputs.len() <= 1 {
        return inputs.into_iter().map(func).collect();
    }

    let pool = ThreadPool::new(num_threads);
    let func = Arc::new(func);
    let len = inputs.len();
    let (tx, rx) = mpsc::channel();
    for (idx, item) in inputs.into_iter().enumerate() {
        let tx = tx.clone();
        let f = Arc::clone(&func);
        pool.execute(move || {
            let out = f(item);
            tx.send((idx, out)).unwrap();
        });
    }
    drop(tx);
    drop(pool); // wait for workers

    let mut results = Vec::with_capacity(len);
    for pair in rx.iter() {
        results.push(pair);
    }
    results.sort_by_key(|(idx, _)| *idx);
    results.into_iter().map(|(_, v)| v).collect()
}

// ----- Inference request queue -----

/// Simple representation of an inference request.
#[derive(Clone, Debug)]
pub struct InferenceRequest {
    id: u64,
    prompt: String,
    cancelled: Arc<AtomicBool>,
}

impl InferenceRequest {
    /// Create a new request with a unique identifier.
    pub fn new(prompt: String) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Self {
            id,
            prompt,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Unique identifier for the request.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Reference to the request prompt.
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Cancel the request.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether the request has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Handle returned to the caller for a queued request.
pub struct InferenceHandle {
    id: u64,
    receiver: mpsc::Receiver<String>,
    cancel_flag: Arc<AtomicBool>,
}

impl InferenceHandle {
    /// Cancel the underlying request.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// Wait for the request to finish and return the result.
    pub fn wait(self) -> Option<String> {
        self.receiver.recv().ok()
    }

    /// Identifier of the request.
    pub fn id(&self) -> u64 {
        self.id
    }
}

/// Thread-safe queue processing inference requests using a fixed set of workers.
use crate::model::Model;

#[derive(Debug)]
pub struct InferenceQueue {
    sender: mpsc::Sender<(InferenceRequest, mpsc::Sender<String>)>,
    workers: Vec<thread::JoinHandle<()>>,
    model: Arc<Model>,
}

impl InferenceQueue {
    /// Create a new queue with `num_workers` worker threads.
    pub fn new(num_workers: usize, model: Arc<Model>) -> Self {
        assert!(num_workers > 0);
        let (tx, rx) = mpsc::channel::<(InferenceRequest, mpsc::Sender<String>)>();
        let rx = Arc::new(Mutex::new(rx));
        let mut workers = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let r = Arc::clone(&rx);
            let model_clone = Arc::clone(&model);
            workers.push(thread::spawn(move || loop {
                let (req, result_tx) = match r.lock().unwrap().recv() {
                    Ok(v) => v,
                    Err(_) => break,
                };
                if req.is_cancelled() {
                    let _ = result_tx.send(String::new());
                    continue;
                }
                let out = model_clone.generate(req.prompt());
                let _ = result_tx.send(out);
            }));
        }
        Self {
            sender: tx,
            workers,
            model,
        }
    }

    /// Submit a prompt to the queue and obtain a handle to await the result.
    pub fn submit(&self, prompt: String) -> InferenceHandle {
        let req = InferenceRequest::new(prompt);
        let req_id = req.id();
        let cancel_flag = req.cancelled.clone();
        let (tx, rx) = mpsc::channel();
        self.sender.send((req, tx)).unwrap();
        InferenceHandle {
            id: req_id,
            receiver: rx,
            cancel_flag,
        }
    }
}

impl Drop for InferenceQueue {
    fn drop(&mut self) {
        // Close the channel so workers can exit.
        let (tx, _) = mpsc::channel();
        let old = std::mem::replace(&mut self.sender, tx);
        drop(old);
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_process_group_state() {
        destroy_process_group();
        init_process_group(4, 1);
        assert_eq!(get_world_size(), 4);
        assert_eq!(get_rank(), 1);
        destroy_process_group();
    }

    #[test]
    fn test_parallel_map() {
        let input = vec![1, 2, 3, 4];
        let result = parallel_map(input, |v| v * v, 2);
        assert_eq!(result, vec![1, 4, 9, 16]);
    }

    #[test]
    fn test_task_scheduler_basic() {
        let sched = TaskScheduler::new(2);
        let handles: Vec<_> = (0..4).map(|i| sched.spawn(move || i * 2)).collect();
        let results = TaskScheduler::join_all(handles);
        assert_eq!(results, vec![0, 2, 4, 6]);
    }

    #[test]
    fn test_task_scheduler_parallel() {
        use std::sync::{Arc, Barrier};
        let sched = TaskScheduler::new(4);
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let barrier_clone = Arc::clone(&barrier);
                sched.spawn(move || {
                    barrier_clone.wait(); // Ensure all threads start together
                    thread::sleep(Duration::from_millis(100));
                    1
                })
            })
            .collect();
        let results = TaskScheduler::join_all(handles);
        assert_eq!(results, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_inference_queue_basic() {
        let model = Arc::new(crate::model::Model::new("dummy".to_string()));
        let queue = InferenceQueue::new(2, Arc::clone(&model));
        let handles: Vec<_> = (0..5).map(|i| queue.submit(format!("req{}", i))).collect();
        let mut results: Vec<_> = handles.into_iter().map(|h| h.wait().unwrap()).collect();
        results.sort();
        let expected = vec![
            "dummy: 0.717565".to_string(),
            "dummy: 0.723292".to_string(),
            "dummy: 0.729033".to_string(),
            "dummy: 0.734785".to_string(),
            "dummy: 0.740554".to_string(),
        ];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_inference_queue_cancel() {
        let model = Arc::new(crate::model::Model::new("dummy".to_string()));
        let queue = InferenceQueue::new(1, Arc::clone(&model));
        let handle = queue.submit("slow".to_string());
        handle.cancel();
        let result = handle.wait().unwrap();
        assert!(result.is_empty());
    }
}

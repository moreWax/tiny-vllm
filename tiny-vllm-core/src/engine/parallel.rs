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

use std::sync::atomic::{AtomicUsize, AtomicBool};

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

// ----- Thread pool implementation -----

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
        Self { sender: tx, workers }
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
        Self { pool: ThreadPool::new(num_threads.max(1)) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use std::thread;

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
        let sched = TaskScheduler::new(4);
        let start = Instant::now();
        let handles: Vec<_> = (0..4)
            .map(|_| sched.spawn(|| {
                thread::sleep(Duration::from_millis(100));
                1
            }))
            .collect();
        let results = TaskScheduler::join_all(handles);
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(400));
        assert_eq!(results, vec![1, 1, 1, 1]);
    }
}


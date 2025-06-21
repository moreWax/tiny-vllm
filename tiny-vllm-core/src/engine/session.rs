//! Inference session management.
//!
//! This is a very small placeholder implementation that mirrors the
//! Python `engine.session` module. A session simply tracks a unique
//! identifier and the model it was created for. Future versions will
//! store cached KV states and scheduler metadata.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Debug)]
pub struct Session {
    id: u64,
    model: String,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

impl Session {
    /// Create a new session for the given model identifier.
    pub fn new(model: String) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        Self { id, model }
    }

    /// Unique identifier of this session.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Model identifier associated with the session.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Reset the session state. In this stub implementation this is a no-op.
    pub fn reset(&mut self) {
        // placeholder for cache clearing
    }
}

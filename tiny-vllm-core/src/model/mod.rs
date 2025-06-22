pub mod layers;

use serde::{Deserialize, Serialize};
use crate::config::VllmConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub config: VllmConfig,
}

impl Model {
    /// Create a new model representation from the given model identifier.
    pub fn new(model: String) -> Self {
        Self {
            config: VllmConfig {
                model,
                ..Default::default()
            },
        }
    }

    /// Return the underlying model identifier.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Generate a completion for the provided prompt.
    ///
    /// This is a very small stand-in for the real model forward pass. It
    /// simply echoes the prompt prefixed by the model identifier. The goal is
    /// to exercise the scheduling and session code while the heavy weight
    /// model integration is developed.
    pub fn generate(&self, prompt: &str) -> String {
        format!("{}: {}", self.model(), prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_basic() {
        let m = Model::new("test-model".to_string());
        assert_eq!(m.model(), "test-model");
        assert_eq!(m.config.model, "test-model".to_string());
    }
}

use serde::{Deserialize, Serialize};

use crate::config::VllmConfig;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Model {
    pub config: VllmConfig,
}

impl Model {
    /// Create a new model representation from the given model identifier.
    /// The identifier usually corresponds to a HuggingFace model name or path.
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

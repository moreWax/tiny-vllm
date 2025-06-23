pub mod layers;

use crate::config::VllmConfig;
use crate::model::layers::{LinearLayer, SiluAndMul};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Model {
    pub config: VllmConfig,
    fc1: LinearLayer,
    act: SiluAndMul,
    fc2: LinearLayer,
}

impl Model {
    /// Create a new model representation from the given model identifier.
    pub fn new(model: String) -> Self {
        Self {
            config: VllmConfig {
                model,
                ..Default::default()
            },
            fc1: LinearLayer::new(
                vec![
                    vec![0.03, 0.04],
                    vec![0.05, 0.06],
                    vec![0.07, 0.08],
                    vec![0.09, 0.10],
                ],
                Some(vec![0.0; 4]),
            ),
            act: SiluAndMul::new(),
            fc2: LinearLayer::new(vec![vec![0.5, -0.25]], Some(vec![0.1])),
        }
    }

    /// Return the underlying model identifier.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    /// Compute a toy forward pass over the given prompt and return a
    /// formatted string with the result.  The prompt is reduced to two
    /// features (length and average character code) which are fed through a
    /// small neural network built from the primitive layers.  The output is a
    /// single floating point value which is returned as a string prefixed by
    /// the model name.
    pub fn generate(&self, prompt: &str) -> String {
        let len = prompt.len() as f32;
        let avg = if len > 0.0 {
            prompt.bytes().map(|b| b as f32).sum::<f32>() / len
        } else {
            0.0
        };
        let mut x = vec![vec![len, avg]];
        x = self.fc1.forward(x);
        x = self.act.forward(x);
        x = self.fc2.forward(x);
        let val = x[0][0];
        format!("{}: {:.6}", self.model(), val)
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

    #[test]
    fn test_generate_output() {
        let m = Model::new("demo".to_string());
        let out = m.generate("abc");
        assert_eq!(out, "demo: 0.808687".to_string());
    }
}

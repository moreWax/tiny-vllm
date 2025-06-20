//! Neural network assembly and forward pass logic.
//!
//! This is a very small placeholder implementation used during the
//! early stages of the Rust port.  It mimics the Python
//! `tiny_vllm.model.network` module by providing a sequential network
//! structure that owns a list of layers.  Real layers and GPU execution
//! will be implemented in later epochs.

use std::sync::Arc;

/// Lightweight tensor type used for testing the network plumbing.  It
/// simply stores a flat vector of `f32` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor from raw data.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }
}

/// Trait implemented by all network layers.
pub trait Layer: Send + Sync {
    fn forward(&self, input: Tensor) -> Tensor;
}

/// Identity layer used as a default placeholder.  It simply returns the
/// input tensor unchanged.
pub struct IdentityLayer;

impl Layer for IdentityLayer {
    fn forward(&self, input: Tensor) -> Tensor {
        input
    }
}

/// Sequential network holding an ordered list of layers.  During the
/// forward pass each layer receives the output of the previous one.
pub struct Network {
    layers: Vec<Arc<dyn Layer>>,
}

impl Network {
    /// Create an empty network.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Append a layer to the network.
    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Arc::new(layer));
    }

    /// Execute the forward pass over all layers.
    pub fn forward(&self, mut input: Tensor) -> Tensor {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_network() {
        let mut net = Network::new();
        net.add_layer(IdentityLayer);
        net.add_layer(IdentityLayer);

        let input = Tensor::new(vec![1.0, 2.0, 3.0]);
        let output = net.forward(input.clone());
        assert_eq!(output, input);
    }
}

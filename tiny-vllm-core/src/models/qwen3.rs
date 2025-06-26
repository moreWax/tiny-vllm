//! Minimal Qwen3 model stubs.
//!
//! These structures only provide the functionality required for the
//! `ModelRunner` tests and are not meant to implement the real model.

use candle_core::{Tensor, Device, DType, Result as CandleResult};

#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub tensor_parallel_size: usize,
    pub head_dim: usize,
}

impl Qwen3Config {
    pub fn from_config<_C>(_config: &_C) -> Self {
        Self { num_hidden_layers: 2, num_key_value_heads: 2, tensor_parallel_size: 1, head_dim: 64 }
    }

    pub fn head_dim(&self) -> usize { self.head_dim }
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Self { num_hidden_layers: 2, num_key_value_heads: 2, tensor_parallel_size: 1, head_dim: 64 }
    }
}

#[derive(Debug)]
pub struct Qwen3Model {
    config: Qwen3Config,
    _device: Device,
    _dtype: DType,
}

impl Qwen3Model {
    pub fn new(config: Qwen3Config, _rank: usize, device: &Device, dtype: DType) -> CandleResult<Self> {
        Ok(Self { config, _device: device.clone(), _dtype: dtype })
    }

    pub fn forward(&self, input_ids: &Tensor, _position_ids: &Tensor) -> CandleResult<Tensor> {
        let bs = input_ids.dim(0)?;
        Tensor::zeros((bs, self.config.head_dim), DType::F32, input_ids.device())
    }

    pub fn config(&self) -> &Qwen3Config { &self.config }
}

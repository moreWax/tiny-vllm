//! Simplified token sampler.
//!
//! The real project implements a sophisticated sampling kernel.  For our
//! purposes a very small stub is sufficient.

use candle_core::{Device, Tensor, DType, Result as CandleResult};
use crate::sampling_params::SamplingParams;

#[derive(Debug)]
pub struct Sampler {
    _device: Device,
}

impl Sampler {
    pub fn new(device: &Device) -> Self {
        Self { _device: device.clone() }
    }

    /// Sample tokens from the provided logits.
    pub fn batch_sample(&self, logits: &Tensor, params: &[SamplingParams]) -> CandleResult<Tensor> {
        // This stub simply selects token zero for every batch element.
        let bs = params.len();
        Tensor::zeros((bs,), DType::I64, logits.device())
    }
}

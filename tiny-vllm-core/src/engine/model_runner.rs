//! Model runner for executing inference with multi-model and async weight loading support.
//! Follows tiny-vllm design guidelines and is structured for future ONNX extensibility.

use std::sync::Arc;
use tokio::sync::Mutex;

// Core tensor/types used by ModelRunner and models
use candle_core::{DType, Device, Result, Tensor, D};
use anyhow::Result as AnyResult;

// Model imports (use correct model name)
use crate::models::qwen3::{Qwen3ForCausalLM, Qwen3Config};
// TODO: When LLaMa and other models are implemented, add imports here.

// TODO: Uncomment and use these for full engine config/model type support:
// use crate::utils::config::{Config, EngineConfig, ModelType};
// use crate::utils::progress::{progress_worker, ProgressReporter};

use crate::config::Config;

use crate::engine::sequence::Sequence;
use crate::layers::sampler::Sampler;
use crate::utils::context::{Context, set_context};
// TODO: use crate::models::layers::VarBuilderX (to be implemented in models/layers/mod.rs)

#[derive(Debug)]
pub enum Model {
    Qwen3(Qwen3ForCausalLM),
    // LLaMa(LLaMaForCausalLM), // TODO: Add LLaMa support
    // Onnx(ONNXForCausalLM),   // TODO: For ONNX extensibility
}

#[derive(Debug)]
pub struct ModelRunner {
    model: Model,
    sampler: Sampler,
    kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    device: Device,
    dtype: DType,
    block_size: usize,
    num_blocks: usize,
}

impl ModelRunner {
    // For simplicity, using Config. If you want EngineConfig, adjust accordingly.
    pub async fn new(model_type: &str, config: &Config, dtype: DType, device: Device) -> Result<Self> {
        let model = match model_type {
            "qwen3" => {
                let model = Qwen3ForCausalLM::load(config, dtype, &device).await?;
                Model::Qwen3(model)
            }
            // "llama" => { ... }
            // "onnx" => { ... }
            _ => candle_core::bail!("Unsupported model type: {model_type}"),
        };

        let kv_cache = Self::init_kv_cache(config, dtype, &device)?;
        let sampler = Sampler::new(&device);

        Ok(Self {
            model,
            sampler,
            kv_cache: Arc::new(Mutex::new(kv_cache)),
            device,
            dtype,
            block_size: config.kvcache_block_size,
            num_blocks: config.num_kvcache_blocks.unwrap_or(1),
        })
    }

    pub async fn load_weights(&self, weights_path: &std::path::Path) -> Result<()> {
        match &self.model {
            Model::Qwen3(model) => model.load_weights(weights_path).await,
            // Model::LLaMa(model) => model.load_weights(weights_path).await,
            // Model::Onnx(model) => model.load_weights(weights_path).await,
        }
    }

    pub async fn run(&self, sequences: &[Sequence], is_prefill: bool) -> Result<Vec<u32>> {
        let (input_ids, positions, input_metadata) = if is_prefill {
            self.prepare_prefill(sequences)?
        } else {
            self.prepare_decode(sequences)?
        };

        let logits = match &self.model {
            Model::Qwen3(model) => {
                let kv_cache_guard = self.kv_cache.lock().await;
                model.forward(&input_ids, &positions, Some(&*kv_cache_guard), &input_metadata)?
            }
            // Model::LLaMa(model) => { ... }
            // Model::Onnx(model) => { ... }
        };

        let output_ids = self.sample(&logits)?;
        Ok(output_ids)
    }

    fn sample(&self, logits: &Tensor) -> Result<Vec<u32>> {
        logits.argmax(D::Minus1)?.to_vec1::<u32>()
    }

    fn init_kv_cache(config: &Config, dtype: DType, device: &Device) -> Result<Vec<(Tensor, Tensor)>> {
        let num_blocks = config.num_kvcache_block_size.unwrap_or(1);
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        let mut cache = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = Tensor::zeros((num_blocks, config.kvcache_block_size, num_heads, head_dim), dtype, device)?;
            let v = Tensor::zeros((num_blocks, config.kvcache_block_size, num_heads, head_dim), dtype, device)?;
            cache.push((k, v));
        }
        Ok(cache)
    }

    fn prepare_prefill(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor, crate::utils::context::InputMetadata)> {
        // TODO: Implement metadata logic based on runner.rs (slot mapping, cu_seqlens, etc.)
        unimplemented!("Prefill preparation is not yet implemented")
    }

    fn prepare_decode(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor, crate::utils::context::InputMetadata)> {
        // TODO: Implement decode logic based on runner.rs (block tables, slot mapping, etc.)
        unimplemented!("Decode preparation is not yet implemented")
    }
}

// TODO: Implement and export VarBuilderX in models/layers/mod.rs for weight loading and variable building.

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add tests for ModelRunner creation, weight loading, run, and error paths.
}

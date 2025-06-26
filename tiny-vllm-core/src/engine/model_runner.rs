//! Model runner for executing inference with multi-model and async weight loading support.
//! Follows tiny-vllm design guidelines and is structured for future ONNX extensibility.

use std::sync::Arc;
use tokio::sync::Mutex;

use candle_core::{DType, Device, Result, Tensor, D};
use anyhow::Result as AnyResult;

// Only use the real model struct; do not define or use any stub Qwen3Config.
use crate::models::qwen3::Qwen3ForCausalLM;

// TODO: Uncomment and use these for full engine config/model type support when ready.
use crate::utils::config::{Config, EngineConfig, ModelType};
use crate::utils::progress::{progress_worker, ProgressReporter};

use crate::engine::sequence::Sequence;
use crate::utils::context::{Context, set_context};

// TODO: use crate::models::layers::VarBuilderX (to be implemented in models/layers/mod.rs)
// TODO: use attention_rs::InputMetaData (to be implemented in layers/attention.rs)

#[derive(Debug)]
pub enum Model {
    Qwen3(Qwen3ForCausalLM),
    // LLaMa(LLaMaForCausalLM), // TODO: Add LLaMa support
    // Onnx(ONNXForCausalLM),   // TODO: For ONNX extensibility
}

#[derive(Debug)]
pub struct ModelRunner {
    model: Model,
    kv_cache: Arc<Mutex<Vec<(Tensor, Tensor)>>>,
    device: Device,
    config: EngineCong,
}

impl ModelRunner {
    pub async fn new(
        model_type: ModelType,
        vb: &VarBuilderX,
        econfig: &EngineConfig,
        config: &Config, 
        dtype: DType,
        is_rope_i: bool,
        device: Device,
    ) -> Result<Self> {
        // TODO:
        // implement ProgressReporter in utils/progress.rs
        let reporter = Arc::new(RwLock::new(ProgressReporter::new(0)));
        // TODO:
        // implement progress_worker in utils/progress.rs
        progress_worker(Some(1), config.num_hidden_layers, Arc::clone(&reporter));
        let model = match model_type {
            ModelType::Qwen3 => {
                let model = Model::Qwen3(Qwen3ForCausalLM::new(
                    vb,
                    config,
                    dtype,
                    is_rope_i,
                    &device),
                    Arc::clone(&reporter),
                )?),                         
            _ => { 
                candle_core::bail!("Unsupported model type: {model_type}"),
            }
        };

        let kv_cache = Self::init_kv_cache(config, dtype, &device)?;

        Ok(Self {
            model,
            kv_cache: Arc::new(Mutex::new(kv_cache)),
            device,
            config: econfig.clone(),
        })
    }

    // Stopped Auditing here

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
        let num_blocks = config.num_kvcache_blocks.unwrap_or(1);
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
        // TODO: Implement using InputMetaData from attention_rs when layers/attention.rs is ready.
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

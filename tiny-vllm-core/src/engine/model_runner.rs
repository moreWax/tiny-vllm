//! Model runner for executing inference with optimizations
//!
//! This module provides a lightweight `ModelRunner` capable of executing a
//! forward pass on the stub Qwen3 model.  The implementation mirrors the API of
//! the original project but keeps allocations to a minimum.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Tensor, Device, DType};
use anyhow::Result;

use crate::config::Config;
use crate::models::qwen3::{Qwen3Model, Qwen3Config};
use crate::engine::sequence::Sequence;
use crate::utils::context::{Context, set_context};
use crate::layers::sampler::Sampler;

/// Model runner for executing inference
#[derive(Debug)]
pub struct ModelRunner {
    model: Qwen3Model,
    sampler: Sampler,
    k_caches: Vec<Arc<Tensor>>,
    v_caches: Vec<Arc<Tensor>>,
    block_size: usize,
    num_blocks: usize,
    device: Device,
    dtype: DType,
    cuda_graphs_enabled: bool,
    cuda_graphs: HashMap<usize, CudaGraph>,
}

#[derive(Debug)]
struct CudaGraph {
    batch_size: usize,
    input_tensors: Vec<Tensor>,
    output_tensors: Vec<Tensor>,
    is_captured: bool,
}

impl ModelRunner {
    pub fn new(config: &Config) -> Result<Self> {
        let device = Self::get_device(&config.device)?;
        let dtype = Self::get_dtype(&config.dtype)?;
        let model_config = Qwen3Config::from_config(&Config::default());
        let model = Qwen3Model::new(model_config.clone(), 0, &device, dtype)?;
        let sampler = Sampler::new(&device);
        let (k, v) = Self::create_kv_cache(&model_config, config.num_kvcache_blocks.unwrap_or(1), config.kvcache_block_size, &device, dtype)?;
        Ok(Self { model, sampler, k_caches: k, v_caches: v, block_size: config.kvcache_block_size, num_blocks: config.num_kvcache_blocks.unwrap_or(1), device, dtype, cuda_graphs_enabled: !config.enforce_eager, cuda_graphs: HashMap::new() })
    }

    pub fn execute_model(&mut self, sequences: &[Sequence], is_prefill: bool) -> Result<Tensor> {
        let (input_ids, position_ids) = self.prepare_inputs(sequences, is_prefill)?;
        let ctx = self.create_context(sequences, is_prefill)?;
        set_context(ctx)?;
        let logits = self.model.forward(&input_ids, &position_ids)?;
        Ok(logits)
    }

    pub fn sample_tokens(&self, logits: &Tensor, sequences: &[Sequence]) -> Result<Vec<i64>> {
        let params: Vec<_> = sequences.iter().map(|s| s.sampling_params.clone()).collect();
        let t = self.sampler.batch_sample(logits, &params)?;
        Ok(t.to_vec1::<i64>()?)
    }

    fn prepare_inputs(&self, sequences: &[Sequence], is_prefill: bool) -> Result<(Tensor, Tensor)> {
        if is_prefill { self.prepare_prefill_inputs(sequences) } else { self.prepare_decode_inputs(sequences) }
    }

    fn prepare_prefill_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let mut all_input = Vec::new();
        let mut all_pos = Vec::new();
        for seq in sequences {
            let ids = seq.all_token_ids();
            all_input.extend_from_slice(ids);
            all_pos.extend(0..ids.len() as i64);
        }
        let len = all_input.len();
        let ids = Tensor::from_vec(all_input, (len,), &self.device)?;
        let pos = Tensor::from_vec(all_pos, (len,), &self.device)?;
        Ok((ids, pos))
    }

    fn prepare_decode_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let batch = sequences.len();
        let ids: Vec<i64> = sequences.iter().map(|s| s.last_token).collect();
        let pos: Vec<i64> = sequences.iter().map(|s| s.len() as i64 - 1).collect();
        let ids = Tensor::from_vec(ids, (batch,), &self.device)?;
        let pos = Tensor::from_vec(pos, (batch,), &self.device)?;
        Ok((ids, pos))
    }

    fn create_context(&self, sequences: &[Sequence], is_prefill: bool) -> Result<Context> {
        if is_prefill { self.create_prefill_context(sequences) } else { self.create_decode_context(sequences) }
    }

    fn create_prefill_context(&self, sequences: &[Sequence]) -> Result<Context> {
        let mut cu_q = vec![0i64];
        let mut cu_k = vec![0i64];
        let mut slot = Vec::new();
        let mut cur = 0;
        let mut max_q = 0;
        let mut max_k = 0;
        for seq in sequences {
            let l = seq.len();
            cur += l;
            cu_q.push(cur as i64);
            cu_k.push(cur as i64);
            max_q = max_q.max(l);
            max_k = max_k.max(l);
            slot.extend(0..l as i64);
        }
        let cu_q_t = Tensor::from_vec(cu_q, (sequences.len()+1,), &self.device)?;
        let cu_k_t = Tensor::from_vec(cu_k, (sequences.len()+1,), &self.device)?;
        let slot_t = Tensor::from_vec(slot, (cur,), &self.device)?;
        Ok(Context::prefill(cu_q_t, cu_k_t, max_q, max_k, slot_t, None))
    }

    fn create_decode_context(&self, sequences: &[Sequence]) -> Result<Context> {
        let bs = sequences.len();
        let slot: Vec<i64> = (0..bs as i64).collect();
        let ctx_len: Vec<i64> = sequences.iter().map(|s| s.len() as i64).collect();
        let max_blocks = sequences.iter().map(|s| s.num_blocks()).max().unwrap_or(1);
        let mut tables = Vec::new();
        for seq in sequences {
            let mut blocks: Vec<i64> = seq.block_table.iter().map(|&b| b as i64).collect();
            while blocks.len() < max_blocks { blocks.push(-1); }
            tables.extend(blocks);
        }
        let slot_t = Tensor::from_vec(slot, (bs,), &self.device)?;
        let ctx_t = Tensor::from_vec(ctx_len, (bs,), &self.device)?;
        let table_t = Tensor::from_vec(tables, (bs, max_blocks), &self.device)?;
        Ok(Context::decode(slot_t, ctx_t, table_t))
    }

    fn create_kv_cache(config: &Qwen3Config, num_blocks: usize, block_size: usize, device: &Device, dtype: DType) -> Result<(Vec<Arc<Tensor>>, Vec<Arc<Tensor>>)> {
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_key_value_heads / config.tensor_parallel_size;
        let head_dim = config.head_dim();
        let mut k_caches = Vec::new();
        let mut v_caches = Vec::new();
        for _ in 0..num_layers {
            let k = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype, device)?;
            let v = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype, device)?;
            k_caches.push(Arc::new(k));
            v_caches.push(Arc::new(v));
        }
        Ok((k_caches, v_caches))
    }

    fn get_device(s: &str) -> Result<Device> {
        match s {
            "cpu" => Ok(Device::Cpu),
            _ => Err(anyhow::anyhow!("unsupported device")),
        }
    }

    fn get_dtype(s: &str) -> Result<DType> {
        match s {
            "float32" | "f32" => Ok(DType::F32),
            "float16" | "f16" => Ok(DType::F16),
            "bfloat16" | "bf16" => Ok(DType::BF16),
            _ => Err(anyhow::anyhow!("unsupported dtype")),
        }
    }

    pub fn config(&self) -> &Qwen3Config { self.model.config() }
    pub fn device(&self) -> &Device { &self.device }
    pub fn dtype(&self) -> DType { self.dtype }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_params::SamplingParams;

    fn create_test_config() -> Config {
        Config::default().with_device("cpu").with_dtype("float32").with_max_num_seqs(4)
    }

    #[test]
    fn test_model_runner_creation() {
        let config = create_test_config();
        let runner = ModelRunner::new(&config).unwrap();
        assert!(matches!(*runner.device(), Device::Cpu));
        assert_eq!(runner.dtype(), DType::F32);
    }

    #[test]
    fn test_device_parsing() {
        assert!(matches!(ModelRunner::get_device("cpu").unwrap(), Device::Cpu));
        assert!(ModelRunner::get_device("invalid").is_err());
    }

    #[test]
    fn test_dtype_parsing() {
        assert_eq!(ModelRunner::get_dtype("float32").unwrap(), DType::F32);
        assert_eq!(ModelRunner::get_dtype("f32").unwrap(), DType::F32);
        assert!(ModelRunner::get_dtype("invalid").is_err());
    }

    #[test]
    fn test_prepare_decode_inputs() {
        let config = create_test_config();
        let runner = ModelRunner::new(&config).unwrap();
        let sequences = vec![
            Sequence::new(vec![1,2,3], SamplingParams::default()),
            Sequence::new(vec![4,5], SamplingParams::default()),
        ];
        let (ids, pos) = runner.prepare_decode_inputs(&sequences).unwrap();
        assert_eq!(ids.dims(), [2]);
        assert_eq!(pos.dims(), [2]);
        let id_vals: Vec<i64> = ids.to_vec1().unwrap();
        let pos_vals: Vec<i64> = pos.to_vec1().unwrap();
        assert_eq!(id_vals, vec![3,5]);
        assert_eq!(pos_vals, vec![2,1]);
    }

    #[test]
    fn test_kv_cache_creation() {
        let config = Qwen3Config::default();
        let device = Device::Cpu;
        let dtype = DType::F32;
        let (k,v) = ModelRunner::create_kv_cache(&config, 100, 16, &device, dtype).unwrap();
        assert_eq!(k.len(), config.num_hidden_layers);
        assert_eq!(v.len(), config.num_hidden_layers);
        let expected = [100,16, config.num_key_value_heads, config.head_dim()];
        assert_eq!(k[0].dims(), expected);
        assert_eq!(v[0].dims(), expected);
    }
}

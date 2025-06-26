//! Async, extensible ModelRunner with CUDA and MLX support
//!
//! This module provides a high-performance, async-ready model runner
//! for Qwen3 and future transformer models. It supports CPU, CUDA, and MLX (Apple Silicon)
//! via feature flags, and is designed for minimal allocations, zero-cost abstractions,
//! and easy ONNX extensibility in the future.

use std::collections::HashMap;
use std::sync::Arc;
use std::path::Path;

use candle_core::{Tensor, Device, DType};
use anyhow::{Result, bail};

use crate::config::Config;
use crate::models::qwen3::{Qwen3Model, Qwen3Config};
use crate::engine::sequence::Sequence;
use crate::utils::context::{Context, set_context};
use crate::layers::sampler::{Sampler, SamplingParams};

#[cfg(feature = "cuda")]
use candle_core::Device as CudaDevice;

#[cfg(feature = "mlx")]
use candle_core::Device as MlxDevice;

use tokio::sync::Mutex;

/// Enum for backends, for future extensibility (ONNX, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "mlx")]
    Mlx,
    #[allow(dead_code)]
    Onnx, // Not yet implemented
}

impl Backend {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => Ok(Self::Cuda),
            #[cfg(feature = "mlx")]
            "mlx" | "metal" | "apple" => Ok(Self::Mlx),
            "onnx" => Ok(Self::Onnx),
            _ => bail!("Unsupported backend: {s}"),
        }
    }
}

/// Main ModelRunner struct
pub struct ModelRunner {
    backend: Backend,
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
    /// Mutex for async weight loading, or other shared mutable state
    state_lock: Mutex<()>,
}

/// CUDA Graph representation (and future MLX/ONNX graph stubs)
#[derive(Debug)]
struct CudaGraph {
    batch_size: usize,
    input_tensors: Vec<Tensor>,
    output_tensors: Vec<Tensor>,
    is_captured: bool,
}

impl ModelRunner {
    pub async fn new(config: &Config) -> Result<Self> {
        let backend = Backend::from_str(&config.device)?;
        let device = get_device(&backend).await?;
        let dtype = get_dtype(&config.dtype)?;

        let model_config = Qwen3Config::from_config(config);
        let model = Qwen3Model::new(model_config.clone(), 0, &device, dtype)?;

        let sampler = Sampler::new(&device);

        let (k_caches, v_caches) = Self::create_kv_cache(
            &model_config,
            config.num_kvcache_blocks.unwrap_or(1000),
            config.kvcache_block_size,
            &device,
            dtype,
        )?;

        Ok(Self {
            backend,
            model,
            sampler,
            k_caches,
            v_caches,
            block_size: config.kvcache_block_size,
            num_blocks: config.num_kvcache_blocks.unwrap_or(1000),
            device,
            dtype,
            cuda_graphs_enabled: !config.enforce_eager,
            cuda_graphs: HashMap::new(),
            state_lock: Mutex::new(()),
        })
    }

    pub async fn load_weights(&self, weights_path: &Path) -> Result<()> {
        // Acquire lock for exclusive access during weight loading
        let _lock = self.state_lock.lock().await;
        tracing::info!("Loading model weights from: {:?}", weights_path);
        // TODO: Actually load safetensors files, create VarBuilder, call model.load_weights
        Ok(())
    }

    pub async fn execute_model(
        &mut self,
        sequences: &[Sequence],
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (input_ids, position_ids) = self.prepare_inputs(sequences, is_prefill)?;
        let context = self.create_context(sequences, is_prefill)?;
        set_context(context)?;
        let logits = match (self.cuda_graphs_enabled, self.backend, is_prefill) {
            (true, Backend::Cuda, false) => self.execute_with_cuda_graph(&input_ids, &position_ids).await?,
            _ => self.model.forward(&input_ids, &position_ids)?,
        };
        Ok(logits)
    }

    pub fn sample_tokens(&self, logits: &Tensor, sequences: &[Sequence]) -> Result<Vec<i64>> {
        let params: Vec<SamplingParams> = sequences.iter().map(|s| s.sampling_params.clone()).collect();
        let t = self.sampler.batch_sample(logits, &params)?;
        Ok(t.to_vec1::<i64>()?)
    }

    fn prepare_inputs(&self, sequences: &[Sequence], is_prefill: bool) -> Result<(Tensor, Tensor)> {
        match is_prefill {
            true => self.prepare_prefill_inputs(sequences),
            false => self.prepare_decode_inputs(sequences),
        }
    }

    fn prepare_prefill_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let (all_input, all_pos) = sequences.iter().flat_map(|seq| {
            let ids = seq.all_token_ids();
            let poss = (0..ids.len() as i64).collect::<Vec<_>>();
            ids.iter().copied().zip(poss).collect::<Vec<_>>()
        }).unzip::<_, _, Vec<_>, Vec<_>>();
        let len = all_input.len();
        let ids = Tensor::from_vec(all_input, (len,), &self.device)?;
        let pos = Tensor::from_vec(all_pos, (len,), &self.device)?;
        Ok((ids, pos))
    }

    fn prepare_decode_inputs(&self, sequences: &[Sequence]) -> Result<(Tensor, Tensor)> {
        let ids: Vec<i64> = sequences.iter().map(|s| s.last_token).collect();
        let pos: Vec<i64> = sequences.iter().map(|s| s.len() as i64 - 1).collect();
        let batch = sequences.len();
        let ids = Tensor::from_vec(ids, (batch,), &self.device)?;
        let pos = Tensor::from_vec(pos, (batch,), &self.device)?;
        Ok((ids, pos))
    }

    fn create_context(&self, sequences: &[Sequence], is_prefill: bool) -> Result<Context> {
        match is_prefill {
            true => self.create_prefill_context(sequences),
            false => self.create_decode_context(sequences),
        }
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

    fn create_kv_cache(
        config: &Qwen3Config,
        num_blocks: usize,
        block_size: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Vec<Arc<Tensor>>, Vec<Arc<Tensor>>)> {
        let num_layers = config.num_hidden_layers;
        let num_kv_heads = config.num_key_value_heads / config.tensor_parallel_size;
        let head_dim = config.head_dim();
        let mut k_caches = Vec::with_capacity(num_layers);
        let mut v_caches = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype, device)?;
            let v = Tensor::zeros((num_blocks, block_size, num_kv_heads, head_dim), dtype, device)?;
            k_caches.push(Arc::new(k));
            v_caches.push(Arc::new(v));
        }
        Ok((k_caches, v_caches))
    }

    async fn execute_with_cuda_graph(&mut self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Placeholder CUDA graph logic (future: MLX, ONNX graph support)
        let batch_size = input_ids.dims()[0];
        let entry = self.cuda_graphs.entry(batch_size).or_insert_with(|| CudaGraph {
            batch_size,
            input_tensors: vec![],
            output_tensors: vec![],
            is_captured: false,
        });
        match entry.is_captured {
            true => {
                // TODO: Actually execute CUDA graph
                self.model.forward(input_ids, position_ids)
            }
            false => self.model.forward(input_ids, position_ids),
        }
    }

    pub fn config(&self) -> &Qwen3Config { self.model.config() }
    pub fn device(&self) -> &Device { &self.device }
    pub fn dtype(&self) -> DType { self.dtype }
}

// Device selection with feature flags, async
async fn get_device(backend: &Backend) -> Result<Device> {
    match backend {
        Backend::Cpu => Ok(Device::Cpu),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Device::new_cuda(0).map_err(|e| anyhow::anyhow!("Failed to create CUDA device: {}", e)),
        #[cfg(feature = "mlx")]
        Backend::Mlx => Device::new_metal(0).map_err(|e| anyhow::anyhow!("Failed to create MLX device: {}", e)),
        #[allow(unreachable_patterns)]
        _ => bail!("Unsupported backend/device"),
    }
}

fn get_dtype(s: &str) -> Result<DType> {
    match s.to_lowercase().as_str() {
        "float32" | "f32" => Ok(DType::F32),
        "float16" | "f16" => Ok(DType::F16),
        "bfloat16" | "bf16" => Ok(DType::BF16),
        _ => bail!("Unsupported dtype: {}", s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling_params::SamplingParams;

    fn create_test_config(device: &str) -> Config {
        Config::default().with_device(device).with_dtype("float32").with_max_num_seqs(4)
    }

    #[tokio::test]
    async fn test_model_runner_creation_cpu() {
        let config = create_test_config("cpu");
        let runner = ModelRunner::new(&config).await.unwrap();
        assert!(matches!(*runner.device(), Device::Cpu));
        assert_eq!(runner.dtype(), DType::F32);
    }

    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn test_model_runner_creation_cuda() {
        let config = create_test_config("cuda");
        let runner = ModelRunner::new(&config).await.unwrap();
        assert!(matches!(*runner.device(), Device::Cuda(_)));
    }

    #[cfg(feature = "mlx")]
    #[tokio::test]
    async fn test_model_runner_creation_mlx() {
        let config = create_test_config("mlx");
        let runner = ModelRunner::new(&config).await.unwrap();
        // Add your MLX device assertions
    }

    #[tokio::test]
    async fn test_weight_loading() {
        let config = create_test_config("cpu");
        let runner = ModelRunner::new(&config).await.unwrap();
        runner.load_weights(Path::new("/does/not/exist")).await.unwrap();
    }

    #[test]
    fn test_prepare_decode_inputs() {
        let config = create_test_config("cpu");
        let rt = tokio::runtime::Runtime::new().unwrap();
        let runner = rt.block_on(ModelRunner::new(&config)).unwrap();
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

//! Qwen3 transformer model implementation for tiny-vllm
//! Clean, idiomatic, and extensible for real LLM inference.

use std::collections::HashMap;
use std::iter::zip;
use std::sync::{Arc, RwLock};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{var_builder::Shard, Module, RmsNorm};
use crate::models::layers::attention::Attention;
use crate::models::layers::linear::{linear_no_bias_x as linear_no_bias, LinearX as Linear};
use crate::models::layers::mask::get_attention_casual_mask;
use crate::models::layers::mlp::MLP;
use crate::models::layers::others::{embedding, rms_norm};
use crate::models::layers::rotary_emb::RotaryEmbedding;
use crate::models::layers::VarBuilderX;
use crate::utils::config::Config;
use crate::utils::progress::{ProgressLike, ProgressReporter};
use attention_rs::InputMetadata;

#[derive(Debug, Clone)]
pub struct Qwen3DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3DecoderLayer {
    pub fn new(
        vb: VarBuilderX,
        rotary_emb: Arc<RotaryEmbedding>,
        config: &Config,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();

        let self_attn = Attention::new(
            if is_qvar_builder { vb.clone() } else { vb.pp("self_attn").clone() },
            rotary_emb.clone(),
            config,
            dtype,
        )?;

        let mlp = MLP::new(
            if is_qvar_builder { vb.clone() } else { vb.pp("mlp").clone() },
            config,
            dtype,
        )?;

        let key_map = HashMap::from([
            ("input_layernorm", "attn_norm"),
            ("post_attention_layernorm", "ffn_norm"),
        ]);

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["input_layernorm"]).clone()
            } else {
                vb.pp("input_layernorm").clone()
            },
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["post_attention_layernorm"]).clone()
            } else {
                vb.pp("post_attention_layernorm").clone()
            },
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let attn_output = self.self_attn.forward(&xs, attention_mask, positions, cache, input_metadata)?;
        let xs = (&attn_output + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual + mlp_output
    }
}

#[derive(Debug)]
pub struct Qwen3ForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    config: Config,
    dtype: DType,
}

impl Qwen3ForCausalLM {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: &VarBuilderX,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        progress_reporter: Arc<RwLock<ProgressReporter>>,
    ) -> Result<Self> {
        let key_map = HashMap::from([
            ("model.embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("model.norm", "output_norm"),
            ("model.layers", "blk"),
        ]);
        let reporter = progress_reporter.clone();
        let is_qvar_builder = vb.is_qvar_builder();

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(key_map["model.embed_tokens"])
            } else {
                vb.pp("model.embed_tokens")
            },
        )?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, config, &vb.device(), is_rope_i)?);

        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let layer_vb = vb.pp(&format!(
                    "{}.{}",
                    if is_qvar_builder { key_map["model.layers"] } else { "model.layers" },
                    i
                ));
                let layer = Qwen3DecoderLayer::new(layer_vb, rotary_emb.clone(), config, dtype)?;
                reporter.write().unwrap().set_progress(i + 1);
                Ok(layer)
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(key_map["model.norm"])
            } else {
                vb.pp("model.norm")
            },
        )?;

        let lm_head = linear_no_bias(
            config.hidden_size,
            vocab_size,
            if config.tie_word_embeddings.unwrap_or(false) {
                if is_qvar_builder { vb.pp(key_map["model.embed_tokens"]) }
                else { vb.pp("model.embed_tokens") }
            } else {
                if is_qvar_builder { vb.pp(key_map["lm_head"]) }
                else { vb.pp("lm_head") }
            },
            Shard::default(),
            &None,
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            config: config.clone(),
            dtype,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dims1()?;
        let attention_mask = get_attention_casual_mask(
            &self.device,
            self.dtype,
            seq_len,
            positions,
            self.config.sliding_window,
        );
        let mut xs = self.embed_tokens.forward(input_ids)?;

        if let Some(kv_caches) = kv_caches {
            for ((k_cache, v_cache), layer) in zip(kv_caches.iter(), self.layers.iter()) {
                xs = layer.forward(&xs, attention_mask.as_ref(), positions, Some((k_cache, v_cache)), input_metadata)?;
            }
        } else {
            for layer in &self.layers {
                xs = layer.forward(&xs, attention_mask.as_ref(), positions, None, input_metadata)?;
            }
        }

        if let Some(cu_seqlens_q) = &input_metadata.cu_seqlens_q {
            let indices = &cu_seqlens_q.to_vec1::<u32>()?[1..];
            let indices: Vec<_> = indices.iter().map(|x| x - 1).collect();
            let length = indices.len();
            xs = xs.index_select(&Tensor::from_vec(indices, (length,), xs.device())?, 0)?;
        }
        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs)
    }
}

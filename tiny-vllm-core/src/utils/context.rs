//! Execution context utilities.
//!
//! The real project maintains an execution context used by the attention
//! kernels.  For the purpose of these exercises we only implement a very
//! lightweight placeholder so that the higher level components can compile.

use candle_core::Tensor;
use anyhow::Result;

#[derive(Debug, Clone, Default)]
pub struct Context {
    pub slot_mapping: Option<Tensor>,
    pub context_lens: Option<Tensor>,
    pub block_tables: Option<Tensor>,
    pub cu_seqlens_q: Option<Tensor>,
    pub cu_seqlens_k: Option<Tensor>,
}

impl Context {
    pub fn prefill(
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        slot_mapping: Tensor,
        block_tables: Option<Tensor>,
    ) -> Self {
        Self {
            cu_seqlens_q: Some(cu_seqlens_q),
            cu_seqlens_k: Some(cu_seqlens_k),
            slot_mapping: Some(slot_mapping),
            block_tables,
            context_lens: None,
        }
    }

    pub fn decode(
        slot_mapping: Tensor,
        context_lens: Tensor,
        block_tables: Tensor,
    ) -> Self {
        Self {
            slot_mapping: Some(slot_mapping),
            context_lens: Some(context_lens),
            block_tables: Some(block_tables),
            cu_seqlens_q: None,
            cu_seqlens_k: None,
        }
    }
}

/// Set the current execution context.
///
/// The implementation only stores the context in a thread local for now.
use std::cell::RefCell;
thread_local! {
    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
}

pub fn set_context(ctx: Context) -> Result<()> {
    CONTEXT.with(|c| *c.borrow_mut() = Some(ctx));
    Ok(())
}

#[allow(dead_code)]
pub fn get_context() -> Option<Context> {
    CONTEXT.with(|c| c.borrow().clone())
}

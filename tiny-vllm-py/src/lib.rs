use pyo3::prelude::*;
use tiny_vllm_core::helpers;
use tiny_vllm_core::{config, cuda_utils};
use tiny_vllm_core::engine::parallel;


fn to_py_err(err: anyhow::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

#[pyfunction]
fn get_device() -> PyResult<String> {
    cuda_utils::get_device().map_err(to_py_err)
}

#[pyfunction]
fn get_gpu_memory() -> PyResult<u64> {
    cuda_utils::get_gpu_memory().map_err(to_py_err)
}

#[pyfunction]
fn get_gpu_memory_utilization() -> PyResult<f32> {
    cuda_utils::get_gpu_memory_utilization().map_err(to_py_err)
}

// ----- Parallel helpers -----
#[pyfunction]
fn init_process_group(world_size: usize, rank: usize) {
    parallel::init_process_group(world_size, rank);
}

#[pyfunction]
fn destroy_process_group() {
    parallel::destroy_process_group();
}

#[pyfunction]
fn get_rank() -> usize {
    parallel::get_rank()
}

#[pyfunction]
fn get_world_size() -> usize {
    parallel::get_world_size()
}

#[pyfunction]
fn barrier() {
    parallel::barrier();
}

// ----- Default settings helpers -----
#[pyfunction]
fn default_max_num_batched_tokens() -> usize {
    config::settings::MAX_NUM_BATCHED_TOKENS
}

#[pyfunction]
fn default_max_num_seqs() -> usize {
    config::settings::MAX_NUM_SEQS
}

#[pyfunction]
fn default_max_model_len() -> usize {
    config::settings::MAX_MODEL_LEN
}

#[pyfunction]
fn default_gpu_memory_utilization() -> f32 {
    config::settings::GPU_MEMORY_UTILIZATION
}

#[pyfunction]
fn default_tensor_parallel_size() -> usize {
    config::settings::TENSOR_PARALLEL_SIZE
}

#[pyfunction]
fn default_enforce_eager() -> bool {
    config::settings::ENFORCE_EAGER
}

#[pyfunction]
fn default_kvcache_block_size() -> usize {
    config::settings::KVCACHE_BLOCK_SIZE
}

#[pyfunction]
fn default_num_kvcache_blocks() -> isize {
    config::settings::NUM_KVCACHE_BLOCKS
}

#[pyfunction]
fn default_eos() -> i64 {
    config::settings::EOS
}

#[pymodule]
fn tiny_vllm_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_device, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory, m)?)?;
    m.add_function(wrap_pyfunction!(get_gpu_memory_utilization, m)?)?;
    m.add_function(wrap_pyfunction!(clamp, m)?)?;
    m.add_function(wrap_pyfunction!(flatten, m)?)?;
    m.add_function(wrap_pyfunction!(chunked, m)?)?;
    // parallel helpers
    m.add_function(wrap_pyfunction!(init_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(destroy_process_group, m)?)?;
    m.add_function(wrap_pyfunction!(get_rank, m)?)?;
    m.add_function(wrap_pyfunction!(get_world_size, m)?)?;
    m.add_function(wrap_pyfunction!(barrier, m)?)?;

    // default setting helpers
    m.add_function(wrap_pyfunction!(default_max_num_batched_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(default_max_num_seqs, m)?)?;
    m.add_function(wrap_pyfunction!(default_max_model_len, m)?)?;
    m.add_function(wrap_pyfunction!(default_gpu_memory_utilization, m)?)?;
    m.add_function(wrap_pyfunction!(default_tensor_parallel_size, m)?)?;
    m.add_function(wrap_pyfunction!(default_enforce_eager, m)?)?;
    m.add_function(wrap_pyfunction!(default_kvcache_block_size, m)?)?;
    m.add_function(wrap_pyfunction!(default_num_kvcache_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(default_eos, m)?)?;
  
    Ok(())
}

#[pyfunction]
fn clamp(value: i64, min_value: i64, max_value: i64) -> PyResult<i64> {
    Ok(helpers::clamp(value, min_value, max_value))
}

#[pyfunction]
fn flatten(list_of_lists: Vec<Vec<i64>>) -> PyResult<Vec<i64>> {
    Ok(helpers::flatten(list_of_lists))
}

#[pyfunction]
fn chunked(lst: Vec<i64>, size: usize) -> PyResult<Vec<Vec<i64>>> {
    Ok(helpers::chunked(lst, size))
}


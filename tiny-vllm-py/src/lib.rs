use pyo3::prelude::*;
use pyo3::Bound;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use tiny_vllm_core::helpers;
use tiny_vllm_core::{config, cuda_utils};
use tiny_vllm_core::layers::{activation::SiluAndMul as SiluCore, linear::Linear as LinearCore};


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
    m.add_class::<LinearLayer>()?;
    m.add_class::<SiluAndMul>()?;

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


#[pyclass]
struct LinearLayer {
    inner: LinearCore,
}

#[pymethods]
impl LinearLayer {
    #[new]
    fn new(weight: PyReadonlyArray2<f32>, bias: Option<PyReadonlyArray1<f32>>) -> Self {
        let weight = weight.as_array().to_owned();
        let bias = bias.map(|b| b.as_array().to_owned());
        Self { inner: LinearCore::new(weight, bias) }
    }

    fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> Bound<'py, PyArray2<f32>> {
        let x = x.as_array().to_owned();
        let y = self.inner.forward(&x);
        y.into_pyarray_bound(py)
    }
}

#[pyclass]
struct SiluAndMul {
    inner: SiluCore,
}

#[pymethods]
impl SiluAndMul {
    #[new]
    fn new() -> Self {
        Self { inner: SiluCore::default() }
    }

    fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> Bound<'py, PyArray2<f32>> {
        let x = x.as_array().to_owned();
        let y = self.inner.forward(&x);
        y.into_pyarray_bound(py)
    }
}

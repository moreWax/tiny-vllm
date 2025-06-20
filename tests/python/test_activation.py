import numpy as np
from tiny_vllm.model.layers import SiluAndMul


def silu_and_mul_np(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    a, b = x[:, :half], x[:, half:]
    return (a / (1.0 + np.exp(-a))) * b


def test_silu_and_mul_forward():
    layer_rs = SiluAndMul()
    x = np.random.randn(2, 8).astype(np.float32)
    out_np = silu_and_mul_np(x)
    out_rs = layer_rs.forward(x)
    assert np.allclose(out_rs, out_np, atol=1e-5)

import numpy as np
from tiny_vllm.model.layers import LinearLayer


def linear_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def test_linear_forward():
    weight = np.random.randn(4, 3).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    layer = LinearLayer(weight, bias)
    x = np.random.randn(2, 3).astype(np.float32)
    y_rs = layer.forward(x)
    y_np = linear_np(x, weight, bias)
    assert np.allclose(y_rs, y_np, atol=1e-5)

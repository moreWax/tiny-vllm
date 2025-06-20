import math

from tiny_vllm.model import layers


def almost_equal(a, b, tol=1e-5):
    return abs(a - b) < tol


def test_linear_layer():
    layer = layers.LinearLayer([[1.0, 2.0], [3.0, 4.0]], bias=[1.0, -1.0])
    out = layer.forward([[1.0, 0.5]])
    expected = [[1.0*1.0 + 0.5*2.0 + 1.0,
                 1.0*3.0 + 0.5*4.0 - 1.0]]
    for o, e in zip(out[0], expected[0]):
        assert almost_equal(o, e)


def test_silu_and_mul():
    layer = layers.SiluAndMul()
    x = [[1.0, 2.0, 0.5, 0.5]]
    out = layer.forward(x)
    def silu(v):
        return v / (1.0 + math.exp(-v))
    expected = [[silu(1.0)*0.5, silu(2.0)*0.5]]
    for o, e in zip(out[0], expected[0]):
        assert almost_equal(o, e)


def test_rmsnorm():
    layer = layers.RMSNorm(2, 1e-6)
    x = [[3.0, 4.0]]
    out = layer.forward(x)
    var = (3.0**2 + 4.0**2) / 2
    inv_rms = 1.0 / math.sqrt(var + 1e-6)
    expected = [[3.0 * inv_rms, 4.0 * inv_rms]]
    for o, e in zip(out[0], expected[0]):
        assert almost_equal(o, e)

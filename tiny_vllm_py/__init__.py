import numpy as np

# ----- Device helpers -----

def get_device() -> str:
    return "cpu"

def get_gpu_memory() -> int:
    return 0

def get_gpu_memory_utilization() -> float:
    return 0.0

# ----- Default constants -----

def default_max_num_batched_tokens() -> int:
    return 32768

def default_max_num_seqs() -> int:
    return 512

def default_max_model_len() -> int:
    return 4096

def default_gpu_memory_utilization() -> float:
    return 0.9

def default_tensor_parallel_size() -> int:
    return 1

def default_enforce_eager() -> bool:
    return False

def default_kvcache_block_size() -> int:
    return 256

def default_num_kvcache_blocks() -> int:
    return -1

def default_eos() -> int:
    return -1

# ----- Helper functions -----

def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def flatten(data):
    return [item for sublist in data for item in sublist]


def chunked(data, size):
    if size <= 0:
        return []
    return [data[i:i + size] for i in range(0, len(data), size)]

# ----- Layers -----
class LinearLayer:
    def __init__(self, weight, bias=None):
        self.weight = np.array(weight, dtype=np.float32)
        if bias is None:
            self.bias = np.zeros(self.weight.shape[0], dtype=np.float32)
        else:
            self.bias = np.array(bias, dtype=np.float32)

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        return x @ self.weight.T + self.bias


class SiluAndMul:
    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        half = x.shape[-1] // 2
        a = x[:, :half]
        b = x[:, half:]
        return (a / (1.0 + np.exp(-a))) * b


class RMSNorm:
    def __init__(self, dim, epsilon=1e-6):
        self.dim = dim
        self.epsilon = epsilon

    def forward(self, x):
        x = np.array(x, dtype=np.float32)
        variance = np.mean(x * x, axis=-1, keepdims=True)
        inv_rms = 1.0 / np.sqrt(variance + self.epsilon)
        return x * inv_rms

# ----- Model and Engine -----
class Model:
    def __init__(self, model: str):
        self.model = model

        # Small neural network matching the Rust implementation
        self.fc1 = LinearLayer(
            [
                [0.03, 0.04],
                [0.05, 0.06],
                [0.07, 0.08],
                [0.09, 0.10],
            ],
            bias=[0.0, 0.0, 0.0, 0.0],
        )
        self.act = SiluAndMul()
        self.fc2 = LinearLayer([[0.5, -0.25]], bias=[0.1])

    def generate(self, prompt: str) -> str:
        length = float(len(prompt))
        avg = sum(ord(c) for c in prompt) / length if length > 0 else 0.0
        x = [[length, avg]]
        x = self.fc1.forward(x)
        x = self.act.forward(x)
        x = self.fc2.forward(x)
        val = float(x[0][0])
        return f"{self.model}: {val:.6f}"


class Engine:
    def __init__(self, num_threads: int = 1):
        self.num_threads = num_threads


class Session:
    pass

__all__ = [
    "get_device",
    "get_gpu_memory",
    "get_gpu_memory_utilization",
    "default_max_num_batched_tokens",
    "default_max_num_seqs",
    "default_max_model_len",
    "default_gpu_memory_utilization",
    "default_tensor_parallel_size",
    "default_enforce_eager",
    "default_kvcache_block_size",
    "default_num_kvcache_blocks",
    "default_eos",
    "clamp",
    "flatten",
    "chunked",
    "LinearLayer",
    "SiluAndMul",
    "RMSNorm",
    "Model",
    "Engine",
    "Session",
]

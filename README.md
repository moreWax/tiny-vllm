# Tiny-vLLM

Tiny-vLLM is a ground-up rewrite (port) of the vLLM inference engine from Python to Rust. The repository currently contains a lightweight Python implementation (`nanovllm`) used as a reference. As development progresses, the Rust crate will mirror the original architecture while exposing a similar API via PyO3.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~1,200 lines of Python code and a growing Rust port
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Quick Start (Rust)

Ensure a recent Rust toolchain is installed. Build the library and run tests with:

```bash
cargo build --release
cargo test
```

The Python example (`example.py`) still works with the reference implementation. As Rust code lands, bindings will be exposed so usage remains largely the same:

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) explains the module layout and Python ↔ Rust mapping.
- [ROADMAP.md](ROADMAP.md) outlines the planned cycles and milestones.
- [CONTRIBUTING.md](CONTRIBUTING.md) describes the workflow for porting epochs.
- [docs/porting_plan.md](docs/porting_plan.md) tracks the detailed file-by-file plan.


## Benchmark

See `bench.py` for the original benchmark setup. Preliminary results on a RTX 4070 running the Python engine show:

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens


| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.95    | 1353.86               |
| Nano-vLLM      | 133,966     | 101.90   | 1314.65               |

We will add Rust benchmark numbers in [BENCHMARKS.md](BENCHMARKS.md) as the port progresses.
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |
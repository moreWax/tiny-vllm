# Python to Rust Porting Tasks

This document breaks down the porting process into epochs. Each epoch corresponds to a single Python file that will be translated into Rust. The order roughly follows the cycle-based plan (configuration & utilities, layers, models, engine, API).

## Cycle 1: Configuration and Utilities
1. `nanovllm/config.py` – implement Config struct and validation logic in Rust.
2. `nanovllm/utils/context.py` – implement context management utilities.
3. `nanovllm/utils/loader.py` – implement model weight loading helpers.
4. `nanovllm/utils/memory.py` – implement GPU memory query helper.

## Cycle 2: Layers
5. `nanovllm/layers/activation.py` – port SiluAndMul activation layer.
6. `nanovllm/layers/attention.py` – port Attention module and kernel helpers.
7. `nanovllm/layers/embed_head.py` – port embedding and LM head layers.
8. `nanovllm/layers/layernorm.py` – port RMSNorm implementation.
9. `nanovllm/layers/linear.py` – port linear layers (replicated/parallel).
10. `nanovllm/layers/rotary_embedding.py` – port rotary embedding utilities.
11. `nanovllm/layers/sampler.py` – port sampling layer.

## Cycle 3: Model Definition
12. `nanovllm/models/qwen3.py` – port Qwen3 model and related classes.

## Cycle 4: Engine Core
13. `nanovllm/engine/sequence.py` – port Sequence data structure.
14. `nanovllm/engine/block_manager.py` – port block management logic.
15. `nanovllm/engine/scheduler.py` – port Scheduler.
16. `nanovllm/engine/model_runner.py` – port ModelRunner including CUDA-graph logic (adapted for CPU).
17. `nanovllm/engine/llm_engine.py` – port LLMEngine orchestration.

## Cycle 5: Public API
18. `nanovllm/sampling_params.py` – port SamplingParams dataclass.
19. `nanovllm/llm.py` – port top-level LLM class.
20. `nanovllm/__init__.py` – re-export Rust bindings to maintain API.

## Cycle 6: Examples and Benchmarks (optional)
21. `example.py` – reimplement example using Rust library.
22. `bench.py` – port benchmark script if desired.

Each task should follow the workflow described in the main porting plan: create stubs, implement functionality, test against the Python version via PyO3 bindings, and commit once the epoch passes all checks.

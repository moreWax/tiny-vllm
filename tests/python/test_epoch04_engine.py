import tiny_vllm.engine as engine

def test_engine_init():
    e = engine.Engine(num_threads=3)
    assert e.num_threads == 3

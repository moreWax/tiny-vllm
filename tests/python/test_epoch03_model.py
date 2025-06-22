import tiny_vllm.model as model


def test_model_instantiation():
    m = model.Model("demo-model")
    assert m.model == "demo-model"


def test_model_generate():
    m = model.Model("demo")
    out = m.generate("abc")
    assert out == "demo: 0.808687"


import tiny_vllm.model as model


def test_model_instantiation():
    m = model.Model("demo-model")
    assert m.model == "demo-model"

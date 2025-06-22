#!/usr/bin/env python3
"""Test script to verify all components are working correctly."""

def test_rust_module():
    """Test that the Rust module can be imported and basic functions work."""
    import tiny_vllm_py

    # Basic stub functionality should return default values
    device = tiny_vllm_py.get_device()
    memory = tiny_vllm_py.get_gpu_memory()
    utilization = tiny_vllm_py.get_gpu_memory_utilization()

    assert device == "cpu"
    assert memory == 0
    assert utilization == 0.0

def test_python_module():
    """Test that the Python modules can be imported."""
    import importlib.util
    import pathlib

    path = pathlib.Path('nanovllm/sampling_params.py')
    spec = importlib.util.spec_from_file_location('nano_params', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    params = module.SamplingParams(temperature=0.8)
    assert params.temperature == 0.8

def test_transformers():
    """Test that transformers can be imported."""
    from transformers import AutoTokenizer

    assert AutoTokenizer is not None

if __name__ == "__main__":
    print("=== Testing Tiny-vLLM Components ===\n")
    
    rust_ok = test_rust_module()
    python_ok = test_python_module()
    transformers_ok = test_transformers()
    
    print(f"\n=== Results ===")
    print(f"Rust module: {'✓ PASS' if rust_ok else '✗ FAIL'}")
    print(f"Python module: {'✓ PASS' if python_ok else '✗ FAIL'}")
    print(f"Transformers: {'✓ PASS' if transformers_ok else '✗ FAIL'}")
    
    if rust_ok and python_ok and transformers_ok:
        print("\n🎉 All tests passed!")
        print("You can now run:")
        print("  python demo.py")
        print("  python example.py")
        exit(0)
    else:
        print("\n⚠ Some tests failed - check the fixes above")
        exit(1)

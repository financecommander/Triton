# Integration Tests

This directory contains comprehensive integration tests for the Triton DSL compiler and ternary neural network framework.

## Overview

The integration test suite provides end-to-end testing of the complete pipeline from DSL to executable PyTorch models, covering compilation, quantization, training, export, performance, and error handling.

## Test Structure

### Test Files

1. **test_codegen.py** (8 tests)
   - Tests PyTorch code generation from AST
   - Validates code syntax and executability
   - Tests parameter extraction

2. **test_resnet18_compilation.py** (26 tests)
   - ResNet18 model instantiation and architecture
   - Forward and backward passes
   - Parameter counting and quantization
   - Performance benchmarking
   - GPU support

3. **test_mobilenet_compilation.py** (23 tests)
   - MobileNetV2 compilation and execution
   - Width multiplier variations
   - Efficiency comparisons with ResNet18
   - Model persistence

4. **test_custom_model_compilation.py** (28 tests)
   - Custom layer compilation from DSL
   - Parameter shapes and ternary packing
   - Sequential model composition
   - Code generation consistency

5. **test_ternary_quantization_e2e.py** (33 tests)
   - Single tensor quantization
   - Threshold calibration
   - Pack/unpack operations
   - Model-level quantization
   - Sparsity control

6. **test_quantization_pipeline.py** (27 tests)
   - FP32 → FP16 conversion
   - FP16 → INT8 dynamic quantization
   - FP16 → Ternary quantization
   - Mixed precision models
   - Calibration accuracy
   - Memory comparisons

7. **test_training_integration.py** (19 tests)
   - Training setup and loops
   - Gradient flow validation
   - Learning rate scheduling
   - Mixed precision training
   - Gradient clipping and regularization
   - Checkpoint management

8. **test_export_pipeline.py** (24 tests)
   - PyTorch model save/load
   - ONNX export and verification
   - TorchScript (trace and script)
   - HuggingFace Hub upload (mocked)
   - GitHub Releases (mocked)
   - Multi-format export

9. **test_performance_benchmarks.py** (27 tests)
   - Code generation speed
   - Compilation speed scaling
   - Inference speed benchmarks
   - Memory usage analysis
   - Batch size scaling
   - GPU vs CPU comparisons

10. **test_error_handling.py** (27 tests)
    - Invalid layer names and shapes
    - Mismatched inputs
    - NaN and Inf handling
    - Gradient explosion/vanishing
    - Device mismatches
    - Error message clarity

### Support Files

- **conftest.py** - Pytest fixtures for models, inputs, devices, configurations
- **test_utils.py** - Helper functions for benchmarking, validation, comparison
- **test_adapters.py** - API compatibility wrappers for testing

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/
```

### Run Specific Test File
```bash
pytest tests/integration/test_resnet18_compilation.py -v
```

### Run Specific Test Class
```bash
pytest tests/integration/test_custom_model_compilation.py::TestCustomModelCompilation -v
```

### Run Specific Test
```bash
pytest tests/integration/test_codegen.py::TestPyTorchCodeGenerator::test_codegen_initialization -v
```

### Run with Coverage
```bash
pytest tests/integration/ --cov=backend --cov=compiler --cov=models --cov-report=html
```

### Run GPU Tests Only
```bash
pytest tests/integration/ -v -m "not skipif"  # Runs tests that aren't skipped due to CUDA
```

### Run Fast Tests (Skip Slow Benchmarks)
```bash
pytest tests/integration/ -v -k "not benchmark"
```

## Test Categories

### By Functionality
- **Compilation**: test_codegen, test_resnet18_compilation, test_mobilenet_compilation, test_custom_model_compilation
- **Quantization**: test_ternary_quantization_e2e, test_quantization_pipeline
- **Training**: test_training_integration
- **Export**: test_export_pipeline
- **Performance**: test_performance_benchmarks
- **Reliability**: test_error_handling

### By Speed
- **Fast** (< 1s): Most unit-style tests
- **Medium** (1-10s): Model instantiation and forward passes
- **Slow** (> 10s): Training loops, benchmarking, large model tests

## Key Features

### Fixtures
- **device**: Auto-detects CUDA/CPU
- **temp_dir**: Temporary directory for test files
- **compiled_simple_model**: Pre-compiled simple ternary layer
- **reference_pytorch_model**: Standard PyTorch model for comparison
- **mock_dataloader**: Mock dataset for training tests
- **benchmark_config**: Configuration for performance tests

### Utilities
- `measure_inference_time()`: Benchmark inference speed
- `measure_memory_usage()`: Measure model memory footprint
- `count_parameters()`: Count model parameters
- `validate_output_shape()`: Verify tensor shapes
- `compare_model_outputs()`: Compare two model outputs
- `validate_gradients()`: Check gradient computation
- `validate_ternary_weights()`: Verify ternary quantization

### Parameterized Tests
Many tests use `@pytest.mark.parametrize` to run multiple scenarios:
- Different batch sizes
- Various model architectures
- Multiple quantization thresholds
- Different input dimensions

## Dependencies

### Required
- pytest >= 7.4.0
- pytest-cov >= 4.1.0
- torch >= 2.1.0
- numpy >= 1.24.0
- psutil (for memory testing)

### Optional (for full functionality)
- CUDA-enabled GPU (for GPU tests)
- onnx >= 1.17.0 (for ONNX export tests)
- onnxruntime >= 1.15.0 (for ONNX verification)
- huggingface-hub (for HF Hub tests)
- PyGithub (for GitHub release tests)

## Test Design Principles

1. **Isolation**: Each test is independent and can run alone
2. **Repeatability**: Tests use fixed random seeds where applicable
3. **Coverage**: Tests cover happy paths, edge cases, and error conditions
4. **Documentation**: Each test has a clear docstring
5. **Performance**: Fast tests run first; slow tests are clearly marked
6. **Portability**: GPU tests skip gracefully on CPU-only systems
7. **Mocking**: External services (HF, GitHub) are mocked

## Adding New Tests

### Template for New Test File
```python
"""
Integration tests for [feature name].
Tests [description of what is tested].
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.integration.test_utils import (
    measure_inference_time,
    validate_output_shape,
)


class Test[FeatureName]:
    """Test [feature name] functionality."""
    
    def test_basic_functionality(self):
        """Test basic [feature] functionality."""
        # Arrange
        # Act
        # Assert
        pass
```

### Best Practices
1. Group related tests in classes
2. Use descriptive test names (test_what_when_expected)
3. Use fixtures for common setup
4. Add docstrings to all tests
5. Use assertions with clear messages
6. Clean up resources (use fixtures with cleanup)
7. Skip tests that require unavailable resources

## Continuous Integration

These tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly builds

### CI Configuration
- Tests run on Ubuntu, Windows, macOS
- Multiple Python versions (3.9, 3.10, 3.11, 3.12)
- CPU and GPU environments
- Coverage reporting to Codecov

## Coverage Goals

- **Target**: 90%+ coverage
- **Current**: [To be measured]
- **Focus Areas**: 
  - Code generation: 95%+
  - Quantization: 90%+
  - Export: 85%+
  - Error handling: 80%+

## Troubleshooting

### Tests Fail to Import
```bash
# Ensure dependencies are installed
pip install -r requirements-dev.txt
```

### CUDA Tests Fail
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Slow Test Performance
```bash
# Run with fewer warmup iterations
pytest tests/integration/ --benchmark-warmup=1
```

### Memory Issues
```bash
# Run tests with smaller batch sizes
pytest tests/integration/ -k "not large_batch"
```

## Contributing

When adding new features:
1. Write integration tests alongside implementation
2. Ensure tests pass locally
3. Update this README with new test information
4. Add fixtures to conftest.py if reusable
5. Add utilities to test_utils.py if generally useful

## License

MIT License - See LICENSE file in repository root.

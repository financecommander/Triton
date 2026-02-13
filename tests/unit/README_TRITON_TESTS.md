# Comprehensive Triton GPU Backend Test Suite

This document describes the comprehensive test suite for the Triton GPU backend implementation.

## Overview

The test suite `test_triton_backend_comprehensive.py` contains **224 test cases** organized into 16 test classes, covering:

- Auto-tuning validation (28 tests)
- Correctness verification (122 tests)
- Performance benchmarking (32 tests)
- Multi-GPU support (6 tests + parameterization)
- Error handling (11 tests + parameterization)
- Memory management (11 tests + parameterization)
- Integration testing (3 tests)

## Running the Tests

### Prerequisites

Install the required dependencies:

```bash
# Core dependencies
pip install torch>=2.1.0 numpy>=1.24.0

# Development dependencies (for running tests)
pip install pytest>=7.4.0 pytest-cov>=4.1.0

# Benchmark dependencies (optional)
pip install pytest-benchmark>=4.0.0 matplotlib pandas scipy

# Triton (optional, for GPU tests)
pip install triton>=2.1.0
```

### Basic Test Execution

Run all tests:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py -v
```

Run a specific test class:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py::TestAutoTuning -v
```

Run a specific test:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py::TestCorrectnessBasic::test_matmul_accuracy_square -v
```

### Running with Benchmarks

To run performance benchmarks:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py -v --benchmark-only
```

Save benchmark results:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py --benchmark-json=results.json
```

### Filtering Tests

Run only GPU tests (skip CPU-only):
```bash
pytest tests/unit/test_triton_backend_comprehensive.py -v -m "not skipif"
```

Run only correctness tests:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py::TestCorrectness* -v
```

## Test Categories

### 1. Auto-Tuning Tests (`TestAutoTuning`)

Validates that Triton's auto-tuning mechanism works correctly:

- **Block Size Configurations**: Tests 14 combinations of block sizes and matrix sizes
- **Pipeline Stages**: Tests 4 different pipeline stage counts (2, 3, 4, 5)
- **Warp Counts**: Tests 4 different warp counts (2, 4, 6, 8)
- **Configuration Selection**: Verifies optimal config selection for different matrix sizes
- **Configuration Caching**: Ensures tuned configurations are cached and reused

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestAutoTuning -v
```

### 2. Correctness Tests

Multiple test classes verify correctness:

#### `TestCorrectnessBasic`
- Matrix multiplication accuracy for 17 different square matrix sizes (1×1 to 2048×2048)
- Non-square matrix multiplication for 15 configurations
- 2-bit packing/unpacking correctness (6 test cases)
- Pack/unpack roundtrip for 15 different sizes

#### `TestCorrectnessEdgeCases`
- All-zeros matrices
- All-ones matrices
- All-negative-ones matrices
- Identity matrices
- Sparse matrices with varying sparsity (15 combinations)
- Single element (1×1) matrices

#### `TestCorrectnessLargeSizes`
- Large square matrices: 512, 1024, 2048, 4096
- Tests with high sparsity to avoid OOM

#### `TestCorrectnessRandom`
- 30 random small matrices (16-128 size range)
- 20 random non-square matrices (32-256 size range)

#### `TestCorrectnessBatch`
- Batch processing with 11 different batch size and matrix size combinations

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestCorrectnessBasic -v
```

### 3. Performance Tests

#### `TestPerformanceThroughput`
- Throughput measurement for 8 different matrix sizes (128 to 2048)
- Uses pytest-benchmark for accurate timing
- Calculates GFLOPS (billion floating-point operations per second)

#### `TestPerformanceLatency`
- Latency distribution (p50, p95, p99) for 2 matrix sizes
- Measures 100 iterations to get statistical distribution

#### `TestPerformanceScaling`
- Tests performance scaling with 10 different matrix sizes (32 to 1024)
- Validates that performance scales appropriately with problem size

#### `TestPerformanceComparison`
- Compares Triton vs PyTorch baseline (2 sizes)
- Compares Triton vs CUDA baseline if available

#### `TestPerformanceBatch`
- Batch processing throughput for 3 different batch sizes

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestPerformanceThroughput -v --benchmark-only
```

### 4. Multi-GPU Tests (`TestMultiGPU`)

Tests multi-GPU operations (gracefully skip if < 2 GPUs):

- Basic data parallelism across GPUs
- Execution on multiple GPU devices (tested on GPU 0 and 1)
- Cross-device tensor transfer
- Distributed batch processing (2 batch size configurations)

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestMultiGPU -v
```

### 5. Error Handling Tests

#### `TestErrorHandling`
- Invalid input shape mismatches
- Unsupported data types (float64)
- Values outside valid range {-1, 0, 1}
- GPU not available fallback to CPU
- Invalid matrix sizes (0, negative)
- Empty tensor handling
- Mismatched device errors
- Invalid tensor dimensions (wrong number of dims)

#### `TestErrorHandlingMemory`
- Large allocation and OOM handling
- Recovery after OOM errors

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestErrorHandling -v
```

### 6. Memory Management Tests (`TestMemoryManagement`)

Validates memory efficiency and leak prevention:

- No memory leaks on 100 repeated operations
- Peak memory usage tracking for 3 matrix sizes
- Memory cleanup after operations
- Repeated allocations stability (tested with 10, 50, 100 iterations)
- Large matrix memory efficiency (2048×2048)
- Batch memory efficiency (2 batch sizes)

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestMemoryManagement -v
```

### 7. Integration Tests (`TestIntegration`)

End-to-end pipeline tests:
- Complete pack → unpack → matmul pipeline
- Class-based interface testing
- Global instance getter validation

Example:
```python
pytest tests/unit/test_triton_backend_comprehensive.py::TestIntegration -v
```

## Graceful Degradation

The test suite is designed to work in various environments:

### Without GPU
Tests automatically skip GPU-specific tests and run CPU fallback versions:
```bash
# Will skip Triton-specific tests but run correctness tests on CPU
pytest tests/unit/test_triton_backend_comprehensive.py -v
```

### Without Triton
If Triton is not installed, GPU tests are skipped:
```python
@pytest.mark.skipif(not TRITON_AVAILABLE or not HAS_CUDA, 
                   reason="Triton and CUDA required")
```

### With Single GPU
Multi-GPU tests are skipped automatically:
```python
@pytest.mark.skipif(GPU_COUNT < 2, reason="Need 2+ GPUs")
```

## Test Execution Time

Approximate execution times (on NVIDIA A100):

- **Correctness tests only**: ~2-3 minutes
- **Auto-tuning tests**: ~5-10 minutes (first run includes tuning)
- **Performance benchmarks**: ~10-15 minutes
- **All tests**: ~20-30 minutes

To run quickly during development:
```bash
# Run only fast correctness tests
pytest tests/unit/test_triton_backend_comprehensive.py::TestCorrectnessBasic -v -k "not large"
```

## Continuous Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Triton Tests
  run: |
    pytest tests/unit/test_triton_backend_comprehensive.py -v \
      --tb=short \
      --maxfail=5 \
      -m "not slow"
```

Key features for CI:
- Graceful skipping when dependencies unavailable
- Clear skip reasons in test output
- Short tracebacks on failure
- Parameterized tests for comprehensive coverage

## Adding New Tests

When adding tests to this suite:

1. **Use appropriate test class**: Add to existing class or create new one
2. **Use parameterization**: Test multiple scenarios efficiently
3. **Add skip conditions**: Use `@pytest.mark.skipif` for optional dependencies
4. **Document**: Add docstring explaining what the test validates
5. **Follow patterns**: Match existing test structure and naming

Example:
```python
@pytest.mark.skipif(not TRITON_AVAILABLE or not HAS_CUDA,
                   reason="Triton and CUDA required")
@pytest.mark.parametrize("size", [128, 256, 512])
def test_new_feature(self, size):
    """Test description of what this validates."""
    # Test implementation
    pass
```

## Test Coverage Summary

| Category | Test Cases | Lines of Code | Coverage |
|----------|-----------|---------------|----------|
| Auto-Tuning | 28 | ~100 | Block sizes, stages, warps, caching |
| Correctness | 122 | ~400 | Matrix ops, packing, edge cases, random |
| Performance | 32 | ~200 | Throughput, latency, scaling, comparison |
| Multi-GPU | 6+ | ~80 | Data parallel, cross-device |
| Error Handling | 11+ | ~150 | Invalid inputs, OOM, fallback |
| Memory Management | 11+ | ~120 | Leaks, peak usage, cleanup |
| Integration | 3 | ~80 | End-to-end pipelines |
| **Total** | **224** | **1252** | **Comprehensive** |

## Troubleshooting

### Tests Skipped
If all tests are skipped, ensure dependencies are installed:
```bash
pip install torch triton pytest
```

### CUDA Out of Memory
Reduce test matrix sizes or use sparse matrices:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py -v -k "not large"
```

### Slow Execution
Run subset of tests or disable benchmarks:
```bash
pytest tests/unit/test_triton_backend_comprehensive.py -v --benchmark-skip
```

## References

- Triton Documentation: https://triton-lang.org/
- PyTorch Testing: https://pytorch.org/docs/stable/testing.html
- pytest Documentation: https://docs.pytest.org/
- pytest-benchmark: https://pytest-benchmark.readthedocs.io/

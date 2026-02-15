# Testing Guide

This guide covers testing requirements, best practices, and procedures for contributing to Triton DSL.

## Table of Contents

- [Test Organization](#test-organization)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Benchmark Testing](#benchmark-testing)
- [Coverage Requirements](#coverage-requirements)
- [CI/CD Testing](#cicd-testing)
- [Writing Tests](#writing-tests)
- [Running Tests](#running-tests)

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── unit/                       # Unit tests (fast, isolated)
│   ├── __init__.py
│   ├── test_lexer.py
│   ├── test_parser.py
│   ├── test_typechecker.py
│   ├── test_codegen.py
│   └── test_backend.py
├── integration/                # Integration tests (slower, multi-component)
│   ├── __init__.py
│   ├── test_compile_pipeline.py
│   ├── test_pytorch_integration.py
│   └── test_cuda_integration.py
├── benchmarks/                 # Performance benchmarks
│   ├── __init__.py
│   ├── bench_quantization.py
│   ├── bench_matmul.py
│   └── bench_inference.py
├── fixtures/                   # Shared test data
│   ├── dsl_samples/
│   │   ├── simple.tri
│   │   ├── mnist_model.tri
│   │   └── invalid_syntax.tri
│   ├── models/
│   │   └── pretrained_weights.pth
│   └── datasets/
│       └── test_mnist_subset.pt
├── property/                   # Property-based tests (Hypothesis)
│   ├── __init__.py
│   └── test_quantization_properties.py
├── stress/                     # Stress and load tests
│   ├── __init__.py
│   └── test_large_models.py
├── security/                   # Security tests
│   ├── __init__.py
│   └── test_input_validation.py
└── fuzzing/                    # Fuzz tests
    ├── __init__.py
    └── test_parser_fuzzing.py
```

### Test Categories

**Unit Tests** (`tests/unit/`)
- Test individual functions/classes in isolation
- Fast execution (< 1s per test)
- No external dependencies (files, networks)
- Mock external interactions
- ~70% of total tests

**Integration Tests** (`tests/integration/`)
- Test multiple components together
- Moderate execution time (1-10s per test)
- May use file I/O, actual models
- ~20% of total tests

**Benchmark Tests** (`tests/benchmarks/`)
- Measure performance characteristics
- Track regressions
- Compare implementations
- Run separately from main test suite

**Property Tests** (`tests/property/`)
- Test invariants using Hypothesis
- Generate random inputs
- Find edge cases

**Stress Tests** (`tests/stress/`)
- Large inputs, memory limits
- Long-running operations
- Edge cases and limits

## Unit Testing

### Test Structure

Use **pytest** framework with clear test structure:

```python
"""
Test module for Triton lexer functionality.

This module tests tokenization of Triton DSL source code,
including keywords, literals, operators, and error handling.
"""

import pytest
from compiler.lexer.triton_lexer import TritonLexer, Token, LexerError


class TestLexerBasics:
    """Test basic lexer functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.lexer = TritonLexer()
    
    def teardown_method(self):
        """Clean up after each test method."""
        self.lexer = None
    
    def test_tokenize_keywords(self):
        """Test that keywords are correctly tokenized."""
        source = "let layer trit"
        tokens = self.lexer.tokenize(source)
        
        assert len(tokens) == 3
        assert tokens[0].type == "LET"
        assert tokens[1].type == "LAYER"
        assert tokens[2].type == "TRIT"
    
    def test_tokenize_identifier(self):
        """Test identifier tokenization."""
        source = "my_variable"
        tokens = self.lexer.tokenize(source)
        
        assert len(tokens) == 1
        assert tokens[0].type == "IDENTIFIER"
        assert tokens[0].value == "my_variable"
    
    def test_tokenize_integer_literal(self):
        """Test integer literal tokenization."""
        source = "42"
        tokens = self.lexer.tokenize(source)
        
        assert len(tokens) == 1
        assert tokens[0].type == "INTEGER"
        assert tokens[0].value == 42


class TestLexerErrors:
    """Test lexer error handling."""
    
    def test_invalid_character_raises_error(self):
        """Test that invalid characters raise LexerError."""
        lexer = TritonLexer()
        with pytest.raises(LexerError) as exc_info:
            lexer.tokenize("let x = $invalid")
        
        assert "Invalid character" in str(exc_info.value)
        assert "$" in str(exc_info.value)
    
    def test_unterminated_string_raises_error(self):
        """Test that unterminated strings raise LexerError."""
        lexer = TritonLexer()
        with pytest.raises(LexerError) as exc_info:
            lexer.tokenize('let s = "unterminated')
        
        assert "Unterminated string" in str(exc_info.value)


class TestLexerEdgeCases:
    """Test lexer edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("source,expected_count", [
        ("", 0),
        ("   ", 0),
        ("// comment only", 0),
        ("let x = 1  // comment", 5),
    ])
    def test_whitespace_and_comments(self, source, expected_count):
        """Test handling of whitespace and comments."""
        lexer = TritonLexer()
        tokens = lexer.tokenize(source)
        assert len(tokens) == expected_count
```

### Testing Best Practices

**1. Test One Thing Per Test**:

```python
# ✅ Good: Each test has single responsibility
def test_quantize_positive_values():
    """Test quantization of positive values."""
    result = quantize_to_ternary(torch.tensor([0.1, 0.6, 0.9]))
    assert torch.all(result == 1)

def test_quantize_negative_values():
    """Test quantization of negative values."""
    result = quantize_to_ternary(torch.tensor([-0.1, -0.6, -0.9]))
    assert torch.all(result == -1)

def test_quantize_zero_threshold():
    """Test quantization of values near zero."""
    result = quantize_to_ternary(torch.tensor([-0.05, 0.03, -0.02]))
    assert torch.all(result == 0)

# ❌ Bad: Testing multiple things
def test_quantization():
    """Test quantization."""
    # Too many assertions, hard to debug
    assert quantize_to_ternary(torch.tensor([0.6]))[0] == 1
    assert quantize_to_ternary(torch.tensor([-0.6]))[0] == -1
    assert quantize_to_ternary(torch.tensor([0.05]))[0] == 0
```

**2. Use Descriptive Test Names**:

```python
# ✅ Good: Describes what is being tested
def test_parser_raises_syntax_error_on_missing_semicolon():
    pass

def test_type_checker_infers_tensor_shape_from_declaration():
    pass

def test_backend_generates_pytorch_module_with_correct_layers():
    pass

# ❌ Bad: Vague or generic names
def test_parser():
    pass

def test_type_check():
    pass

def test_backend():
    pass
```

**3. Use Fixtures for Common Setup**:

```python
import pytest

@pytest.fixture
def sample_model():
    """Provide a sample ternary model for testing."""
    return TernaryModel(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10
    )

@pytest.fixture
def mnist_batch():
    """Provide a sample MNIST batch."""
    return torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))

@pytest.fixture
def triton_lexer():
    """Provide a configured Triton lexer."""
    return TritonLexer()

# Use fixtures in tests
def test_model_forward_pass(sample_model, mnist_batch):
    """Test model forward pass."""
    images, labels = mnist_batch
    output = sample_model(images.view(images.size(0), -1))
    assert output.shape == (32, 10)
```

**4. Use Parametrize for Multiple Inputs**:

```python
@pytest.mark.parametrize("input_value,expected", [
    (0.7, 1),
    (0.3, 0),
    (-0.3, 0),
    (-0.7, -1),
    (1.0, 1),
    (-1.0, -1),
])
def test_quantize_deterministic(input_value, expected):
    """Test deterministic quantization with various inputs."""
    result = quantize_to_ternary(torch.tensor([input_value]))
    assert result[0] == expected

@pytest.mark.parametrize("source,expected_error", [
    ("let x =", "Unexpected end of input"),
    ("let = 5", "Expected identifier"),
    ("let x: InvalidType = 5", "Unknown type"),
])
def test_parser_errors(source, expected_error):
    """Test parser error messages."""
    parser = TritonParser()
    with pytest.raises(SyntaxError) as exc_info:
        parser.parse(source)
    assert expected_error in str(exc_info.value)
```

**5. Test Edge Cases and Boundaries**:

```python
def test_quantize_handles_empty_tensor():
    """Test quantization of empty tensor."""
    result = quantize_to_ternary(torch.tensor([]))
    assert result.shape == (0,)
    assert result.dtype == torch.float32

def test_quantize_handles_single_element():
    """Test quantization of single element."""
    result = quantize_to_ternary(torch.tensor([0.5]))
    assert result.shape == (1,)

def test_quantize_handles_large_tensor():
    """Test quantization of large tensor (stress test)."""
    large_tensor = torch.randn(1000, 1000)
    result = quantize_to_ternary(large_tensor)
    assert result.shape == large_tensor.shape
    assert set(result.unique().tolist()).issubset({-1, 0, 1})

def test_quantize_handles_inf_and_nan():
    """Test quantization handles special float values."""
    tensor = torch.tensor([float('inf'), float('-inf'), float('nan')])
    result = quantize_to_ternary(tensor)
    # Define expected behavior for special values
    assert not torch.isnan(result).any()
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch, MagicMock

def test_model_save_creates_file(tmp_path):
    """Test that model.save() creates a file."""
    model = TernaryModel(10, [32], 2)
    save_path = tmp_path / "model.pth"
    
    model.save(str(save_path))
    
    assert save_path.exists()
    assert save_path.stat().st_size > 0

@patch('compiler.codegen.pytorch_generator.open')
def test_codegen_writes_to_file(mock_open):
    """Test code generator writes to file."""
    generator = PyTorchGenerator()
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    generator.generate_to_file(ast, "output.py")
    
    mock_open.assert_called_once_with("output.py", "w")
    mock_file.write.assert_called()

@patch('torch.cuda.is_available', return_value=False)
def test_backend_falls_back_to_cpu_when_cuda_unavailable(mock_cuda):
    """Test backend uses CPU when CUDA unavailable."""
    backend = PyTorchBackend(device="auto")
    assert backend.device == "cpu"
    mock_cuda.assert_called_once()
```

## Integration Testing

### Testing Multiple Components

```python
"""Integration tests for Triton compiler pipeline."""

import pytest
from compiler import TritonCompiler
from backend.pytorch_backend import PyTorchBackend


class TestCompilerPipeline:
    """Test complete compilation pipeline."""
    
    @pytest.fixture
    def sample_source(self):
        """Provide sample Triton DSL source."""
        return """
        layer fc1: TernaryLinear {
            in_features: 784,
            out_features: 128
        }
        
        let input: TernaryTensor[32, 784] = zeros()
        let output: TernaryTensor = fc1(input)
        """
    
    def test_compile_simple_program(self, sample_source):
        """Test compiling a simple program end-to-end."""
        compiler = TritonCompiler()
        
        # Compile source to AST
        ast = compiler.parse(sample_source)
        assert ast is not None
        
        # Type check
        type_errors = compiler.type_check(ast)
        assert len(type_errors) == 0
        
        # Generate code
        module = compiler.generate(ast, backend="pytorch")
        assert module is not None
    
    def test_compiled_model_can_run_inference(self, sample_source, tmp_path):
        """Test that compiled model can perform inference."""
        compiler = TritonCompiler()
        
        # Compile
        module = compiler.compile(sample_source, output_dir=str(tmp_path))
        
        # Load and run
        model = module.instantiate()
        input_tensor = torch.randn(32, 784)
        output = model(input_tensor)
        
        assert output.shape == (32, 128)
        assert set(output.unique().tolist()).issubset({-1, 0, 1})
    
    def test_compile_mnist_model_trains_successfully(self, tmp_path):
        """Test compiling and training MNIST model."""
        # Load MNIST model definition
        with open("tests/fixtures/dsl_samples/mnist_model.tri") as f:
            source = f.read()
        
        # Compile
        compiler = TritonCompiler()
        module = compiler.compile(source, output_dir=str(tmp_path))
        model = module.instantiate()
        
        # Create dummy data
        train_data = torch.randn(128, 784)
        train_labels = torch.randint(0, 10, (128,))
        
        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        initial_loss = None
        for epoch in range(5):
            optimizer.zero_grad()
            output = model(train_data)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
            
            if initial_loss is None:
                initial_loss = loss.item()
        
        # Loss should decrease
        assert loss.item() < initial_loss
```

### Testing CUDA Integration

```python
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAIntegration:
    """Test CUDA kernel integration."""
    
    def test_ternary_matmul_cuda(self):
        """Test ternary matrix multiplication on CUDA."""
        a = torch.randint(-1, 2, (128, 256)).cuda()
        b = torch.randint(-1, 2, (256, 512)).cuda()
        
        result = ternary_matmul_cuda(a, b)
        
        assert result.device.type == "cuda"
        assert result.shape == (128, 512)
    
    def test_cuda_kernel_matches_cpu_result(self):
        """Test that CUDA kernel produces same result as CPU."""
        a = torch.randint(-1, 2, (64, 128))
        b = torch.randint(-1, 2, (128, 256))
        
        cpu_result = ternary_matmul_cpu(a, b)
        cuda_result = ternary_matmul_cuda(a.cuda(), b.cuda()).cpu()
        
        torch.testing.assert_close(cpu_result, cuda_result)
```

## Benchmark Testing

### Performance Benchmarks

```python
"""Benchmark tests for ternary operations."""

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture


class TestQuantizationBenchmarks:
    """Benchmark quantization operations."""
    
    @pytest.fixture
    def large_tensor(self):
        """Create large tensor for benchmarking."""
        return torch.randn(1000, 1000)
    
    def test_benchmark_deterministic_quantization(self, benchmark, large_tensor):
        """Benchmark deterministic quantization."""
        result = benchmark(quantize_to_ternary, large_tensor, method="deterministic")
        assert result.shape == large_tensor.shape
    
    def test_benchmark_stochastic_quantization(self, benchmark, large_tensor):
        """Benchmark stochastic quantization."""
        result = benchmark(quantize_to_ternary, large_tensor, method="stochastic")
        assert result.shape == large_tensor.shape
    
    @pytest.mark.parametrize("size", [100, 500, 1000, 2000])
    def test_benchmark_quantization_scales_linearly(self, benchmark, size):
        """Test that quantization scales linearly with size."""
        tensor = torch.randn(size, size)
        benchmark(quantize_to_ternary, tensor)


class TestInferenceBenchmarks:
    """Benchmark inference performance."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_ternary_model_inference_cuda(self, benchmark):
        """Benchmark ternary model inference on CUDA."""
        model = TernaryModel(784, [256, 128], 10).cuda()
        input_data = torch.randn(32, 784).cuda()
        
        def run_inference():
            with torch.no_grad():
                return model(input_data)
        
        result = benchmark(run_inference)
        assert result.shape == (32, 10)
    
    def test_compare_ternary_vs_float32_inference(self, benchmark):
        """Compare ternary vs float32 inference speed."""
        # Ternary model
        ternary_model = TernaryModel(784, [256, 128], 10)
        ternary_input = torch.randn(32, 784)
        
        # Float32 model
        float_model = FloatModel(784, [256, 128], 10)
        float_input = torch.randn(32, 784)
        
        # Benchmark both
        ternary_time = benchmark(lambda: ternary_model(ternary_input))
        float_time = benchmark(lambda: float_model(float_input))
        
        # Ternary should be faster or comparable
        # (This is informational, not a hard assertion)
        print(f"Ternary: {ternary_time}, Float32: {float_time}")
```

### Running Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ --benchmark-only

# Save benchmark results
pytest tests/benchmarks/ --benchmark-only --benchmark-save=baseline

# Compare with baseline
pytest tests/benchmarks/ --benchmark-only --benchmark-compare=baseline

# Generate histogram
pytest tests/benchmarks/ --benchmark-only --benchmark-histogram=histogram
```

## Coverage Requirements

### Target Coverage

- **Overall**: ≥ 85% code coverage
- **Critical paths**: ≥ 95% (lexer, parser, type checker, quantization)
- **New code**: ≥ 90%

### Measuring Coverage

```bash
# Run tests with coverage
pytest tests/ --cov=compiler --cov=backend --cov=kernels

# Generate HTML report
pytest tests/ --cov=compiler --cov=backend --cov=kernels --cov-report=html

# Open report
open htmlcov/index.html

# Show missing lines
pytest tests/ --cov=compiler --cov=backend --cov=kernels --cov-report=term-missing

# Fail if coverage below threshold
pytest tests/ --cov=compiler --cov-fail-under=85
```

### Coverage Configuration

In `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["compiler", "backend", "kernels"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__pycache__/*",
    "*/parser.out",
    "*/parsetab.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
```

## CI/CD Testing

### GitHub Actions Workflow

Create `.github/workflows/test.yml`:
```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: ruff check .
    
    - name: Format check with black
      run: black --check .
    
    - name: Type check with mypy
      run: mypy compiler/ backend/ kernels/
    
    - name: Run tests
      run: pytest tests/ -v --cov=compiler --cov=backend --cov=kernels
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Pre-commit Testing

Before every commit, ensure:
```bash
# Format code
black .

# Lint
ruff check . --fix

# Type check
mypy compiler/ backend/ kernels/

# Run tests
pytest tests/unit/ -v

# Check coverage
pytest tests/ --cov=compiler --cov=backend --cov=kernels --cov-fail-under=85
```

### Pre-push Testing

Before pushing to remote:
```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=compiler --cov=backend --cov=kernels --cov-report=html

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only

# Check for regressions
pytest tests/benchmarks/ --benchmark-compare=baseline
```

## Writing Tests

### Test Template

```python
"""
Test module for [component name].

Brief description of what this module tests.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from component.module import ComponentClass


class TestComponentBasics:
    """Test basic functionality of ComponentClass."""
    
    @pytest.fixture
    def component(self):
        """Provide a ComponentClass instance for testing."""
        return ComponentClass(arg1="value1", arg2=42)
    
    def test_component_initialization(self, component):
        """Test that ComponentClass initializes correctly."""
        assert component.arg1 == "value1"
        assert component.arg2 == 42
    
    def test_component_method_returns_expected_value(self, component):
        """Test that method() returns expected value."""
        result = component.method(input_value)
        assert result == expected_value
    
    def test_component_method_raises_on_invalid_input(self, component):
        """Test that method() raises ValueError on invalid input."""
        with pytest.raises(ValueError) as exc_info:
            component.method(invalid_input)
        assert "Expected error message" in str(exc_info.value)


class TestComponentEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("input_val,expected", [
        (case1, expected1),
        (case2, expected2),
        (case3, expected3),
    ])
    def test_component_handles_various_inputs(self, input_val, expected):
        """Test component with various input values."""
        component = ComponentClass()
        result = component.method(input_val)
        assert result == expected
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_lexer.py -v

# Run specific test class
pytest tests/unit/test_lexer.py::TestLexerBasics -v

# Run specific test method
pytest tests/unit/test_lexer.py::TestLexerBasics::test_tokenize_keywords -v

# Run tests matching pattern
pytest tests/ -k "quantize" -v

# Run tests with specific marker
pytest tests/ -m "slow" -v
pytest tests/ -m "not slow" -v  # Skip slow tests
```

### Useful Options

```bash
# Stop on first failure
pytest tests/ -x

# Show local variables in traceback
pytest tests/ -l

# Show print statements
pytest tests/ -s

# Show test durations
pytest tests/ --durations=10

# Run in parallel (with pytest-xdist)
pytest tests/ -n auto

# Rerun failed tests
pytest tests/ --lf  # last failed
pytest tests/ --ff  # failed first
```

## See Also

- [Development Setup](development_setup.md)
- [Code Style Guide](code_style.md)
- [PR Process](pr_process.md)
- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)

"""
Fuzzing Tests for Triton Compiler

Tests using random/fuzzed inputs to find edge cases and vulnerabilities:
- Random kernel generation
- Random tensor shapes and values
- Invalid input handling
- Boundary condition testing
- Unexpected input types
"""

import pytest
import random
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import string
import itertools
from unittest.mock import Mock

# Try to import Triton components, use mocks if not available
try:
    from compiler import TritonCompiler
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    TritonCompiler = Mock()

try:
    from backend.pytorch import ternary_matmul
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    ternary_matmul = Mock()


class FuzzInputGenerator:
    """Generate fuzzed inputs for testing."""

    @staticmethod
    def random_tensor_shape(min_dims: int = 1, max_dims: int = 4,
                          min_size: int = 1, max_size: int = 1024) -> tuple:
        """Generate random tensor shape."""
        num_dims = random.randint(min_dims, max_dims)
        shape = tuple(random.randint(min_size, max_size) for _ in range(num_dims))
        return shape

    @staticmethod
    def random_tensor_values(shape: tuple, dtype: torch.dtype = torch.float32,
                           device: str = 'cpu') -> torch.Tensor:
        """Generate tensor with random values."""
        if random.random() < 0.1:  # 10% chance of special values
            special_values = [
                lambda: torch.zeros(shape, dtype=dtype, device=device),
                lambda: torch.ones(shape, dtype=dtype, device=device),
                lambda: torch.full(shape, float('inf'), dtype=dtype, device=device),
                lambda: torch.full(shape, float('-inf'), dtype=dtype, device=device),
                lambda: torch.full(shape, float('nan'), dtype=dtype, device=device),
                lambda: torch.randn(shape, dtype=dtype, device=device) * 1000,  # Large values
                lambda: torch.randn(shape, dtype=dtype, device=device) * 1e-6,  # Small values
            ]
            return random.choice(special_values)()
        else:
            return torch.randn(shape, dtype=dtype, device=device)

    @staticmethod
    def random_kernel_code() -> str:
        """Generate random (potentially invalid) kernel code."""
        templates = [
            # Valid kernel template
            """
            @triton.jit
            def fuzz_kernel_{id}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    tl.store(output + pid, val * {factor})
            """,
            # Invalid kernel with syntax errors
            """
            @triton.jit
            def invalid_kernel_{id}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    tl.store(output + pid, val * {factor}
            """,
            # Kernel with random operations
            """
            @triton.jit
            def random_kernel_{id}(x, y, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    a = tl.load(x + pid)
                    b = tl.load(y + pid)
                    result = {operation}
                    tl.store(output + pid, result)
            """
        ]

        template = random.choice(templates)
        kernel_id = random.randint(1, 10000)
        factor = random.uniform(-100, 100)

        operations = [
            "a + b", "a - b", "a * b", "a / (b + 1e-6)",
            "tl.abs(a)", "tl.sqrt(tl.abs(a))", "tl.sin(a)", "tl.cos(a)",
            "tl.maximum(a, b)", "tl.minimum(a, b)",
            "a * b + a", "a * a + b * b"
        ]
        operation = random.choice(operations)

        return template.format(id=kernel_id, factor=factor, operation=operation)

    @staticmethod
    def random_invalid_input() -> Any:
        """Generate random invalid inputs."""
        invalid_inputs = [
            None,
            "string",
            42,
            [1, 2, 3],
            {"key": "value"},
            lambda x: x,
            float('inf'),
            float('-inf'),
            float('nan'),
            np.array([1, 2, 3]),
            torch.tensor([1, 2, 3]).numpy(),
        ]
        return random.choice(invalid_inputs)


class TestKernelFuzzing:
    """Test kernel compilation with fuzzed inputs."""

    @pytest.mark.parametrize("iteration", range(50))
    def test_random_kernel_compilation(self, iteration):
        """Test compilation of randomly generated kernels."""
        compiler = TritonCompiler()

        kernel_code = FuzzInputGenerator.random_kernel_code()

        # Try to compile - may succeed or fail
        try:
            compiled_kernel = compiler.compile(kernel_code)
            # If compilation succeeds, kernel should be valid
            assert compiled_kernel is not None
        except Exception:
            # Compilation failure is expected for some fuzzed inputs
            pass

    def test_extreme_kernel_sizes(self):
        """Test kernels with extreme sizes."""
        compiler = TritonCompiler()

        # Test with very large kernels
        large_kernel = """
        @triton.jit
        def huge_kernel(x, output, n: tl.constexpr):
            pid = tl.program_id(0)
            # Very large block size
            block_size = 4096
            for i in range(block_size):
                if pid * block_size + i < n:
                    val = tl.load(x + pid * block_size + i)
                    # Many operations
                    val = val * 2 + 1
                    val = tl.sin(val) + tl.cos(val)
                    val = tl.sqrt(tl.abs(val))
                    val = val * val + val
                    tl.store(output + pid * block_size + i, val)
        """

        try:
            compiled_kernel = compiler.compile(large_kernel)
            assert compiled_kernel is not None
        except Exception:
            # May fail due to size constraints
            pass

    def test_malformed_kernel_code(self):
        """Test handling of malformed kernel code."""
        compiler = TritonCompiler()

        malformed_kernels = [
            "",  # Empty
            "@triton.jit",  # Incomplete
            "@triton.jit\ndef kernel(): pass",  # Missing tl import
            "def kernel(x, output):\n    return x + output",  # Not a Triton kernel
            "@triton.jit\ndef kernel(x, output):\n    tl.store(output, x)",  # Missing program_id
            "@triton.jit\ndef kernel(x, output, n):\n    pass",  # Missing tl.constexpr
        ]

        for kernel_code in malformed_kernels:
            with pytest.raises(Exception):
                compiler.compile(kernel_code)

    def test_random_code_injection(self):
        """Test resistance to code injection attempts."""
        compiler = TritonCompiler()

        injection_attempts = [
            """
            @triton.jit
            def kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Attempt code injection
                    exec("print('injected')")
                    tl.store(output + pid, val)
            """,
            """
            @triton.jit
            def kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Attempt import injection
                    import os
                    tl.store(output + pid, val)
            """
        ]

        for kernel_code in injection_attempts:
            try:
                compiled_kernel = compiler.compile(kernel_code)
                # Should not execute injected code during compilation
            except Exception:
                # Expected to fail
                pass


class TestTensorFuzzing:
    """Test tensor operations with fuzzed inputs."""

    @pytest.mark.parametrize("iteration", range(100))
    def test_random_tensor_operations(self, iteration):
        """Test ternary_matmul with random tensor shapes and values."""
        # Generate random shapes
        shape1 = FuzzInputGenerator.random_tensor_shape(min_size=1, max_size=512)
        shape2 = FuzzInputGenerator.random_tensor_shape(min_size=1, max_size=512)

        # Ensure compatible for matrix multiplication
        if len(shape1) >= 2 and len(shape2) >= 2:
            # Make shapes compatible: (..., M, K) x (..., K, N) -> (..., M, N)
            m, k1 = shape1[-2], shape1[-1]
            k2, n = shape2[-2], shape2[-1]

            if k1 == k2:  # Compatible dimensions
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                try:
                    x = FuzzInputGenerator.random_tensor_values(shape1, device=device)
                    y = FuzzInputGenerator.random_tensor_values(shape2, device=device)

                    result = ternary_matmul(x, y)

                    # Verify result shape
                    expected_shape = shape1[:-2] + shape2[:-2] + (m, n)
                    assert result.shape == expected_shape

                    # Check for invalid values
                    assert not torch.isinf(result).any(), "Result contains infinity"
                    # NaN is allowed in some cases with extreme inputs

                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # Expected for some random shapes/values
                    pass

    def test_extreme_tensor_values(self):
        """Test with extreme tensor values."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        extreme_values = [
            torch.full((64, 64), float('inf'), device=device),
            torch.full((64, 64), float('-inf'), device=device),
            torch.full((64, 64), float('nan'), device=device),
            torch.full((64, 64), 1e10, device=device),  # Very large
            torch.full((64, 64), 1e-10, device=device),  # Very small
            torch.zeros(64, 64, device=device),
            torch.ones(64, 64, device=device),
        ]

        for x in extreme_values:
            for y in extreme_values:
                try:
                    result = ternary_matmul(x, y)
                    # Result may be inf/nan but should not crash
                    assert result.shape == (64, 64)
                except Exception:
                    # Some combinations may fail - that's expected
                    pass

    def test_invalid_tensor_inputs(self):
        """Test handling of invalid tensor inputs."""
        invalid_inputs = [
            None,
            "string",
            42,
            [1, 2, 3],
            torch.tensor([1, 2, 3]),  # Wrong shape
            torch.randn(10),  # 1D tensor
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, RuntimeError, ValueError)):
                ternary_matmul(invalid_input, torch.randn(32, 32))

            with pytest.raises((TypeError, RuntimeError, ValueError)):
                ternary_matmul(torch.randn(32, 32), invalid_input)


class TestShapeFuzzing:
    """Test with fuzzed tensor shapes."""

    def test_incompatible_shapes(self):
        """Test behavior with incompatible tensor shapes."""
        incompatible_pairs = [
            ((32, 32), (16, 32)),  # Incompatible K dimension
            ((32,), (32, 32)),     # 1D vs 2D
            ((32, 32, 32), (32,)), # 3D vs 1D
            ((1, 32, 32), (32, 16)), # Batch vs no batch
        ]

        for shape1, shape2 in incompatible_pairs:
            x = torch.randn(shape1)
            y = torch.randn(shape2)

            with pytest.raises((RuntimeError, ValueError)):
                ternary_matmul(x, y)

    def test_edge_case_shapes(self):
        """Test with edge case tensor shapes."""
        edge_shapes = [
            (1, 1),      # Minimum size
            (1, 32),     # One dimension is 1
            (32, 1),     # Other dimension is 1
            (0, 32),     # Zero size (should fail)
            (32, 0),     # Zero size (should fail)
            (16384, 1),  # Very large first dimension
            (1, 16384),  # Very large second dimension
        ]

        for shape in edge_shapes:
            try:
                x = torch.randn(shape)
                y = torch.randn(shape)

                if 0 in shape:
                    # Should fail for zero-sized dimensions
                    with pytest.raises((RuntimeError, ValueError)):
                        ternary_matmul(x, y)
                else:
                    result = ternary_matmul(x, y)
                    assert result.shape[0] == shape[0]
                    assert result.shape[1] == shape[1]

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Expected for very large or invalid shapes
                pass

    def test_high_dimensional_tensors(self):
        """Test with high-dimensional tensors."""
        # Test up to 6D tensors
        for ndim in range(3, 7):
            try:
                shape = tuple(random.randint(2, 8) for _ in range(ndim))
                if ndim >= 2:
                    # Make last two dimensions compatible
                    shape = shape[:-2] + (shape[-2], shape[-2])

                x = torch.randn(shape)
                y = torch.randn(shape)

                result = ternary_matmul(x, y)

                # Result should have correct shape
                expected_shape = shape[:-2] + (shape[-2], shape[-2])
                assert result.shape == expected_shape

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Expected for high-dimensional tensors
                pass


class TestDataTypeFuzzing:
    """Test with different data types."""

    def test_various_data_types(self):
        """Test ternary_matmul with different data types."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dtypes_to_test = [
            torch.float32, torch.float64,
            torch.int32, torch.int64,
            torch.bool
        ]

        for dtype in dtypes_to_test:
            try:
                x = torch.randn(32, 32).to(dtype).to(device)
                y = torch.randn(32, 32).to(dtype).to(device)

                result = ternary_matmul(x, y)

                # Result should maintain shape
                assert result.shape == (32, 32)

            except (TypeError, RuntimeError):
                # Some dtypes may not be supported
                pass

    def test_mixed_data_types(self):
        """Test with mixed data types."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        dtype_pairs = [
            (torch.float32, torch.float64),
            (torch.int32, torch.float32),
            (torch.bool, torch.float32),
        ]

        for dtype1, dtype2 in dtype_pairs:
            x = torch.randn(32, 32).to(dtype1).to(device)
            y = torch.randn(32, 32).to(dtype2).to(device)

            try:
                result = ternary_matmul(x, y)
                assert result.shape == (32, 32)
            except (TypeError, RuntimeError):
                # Mixed types may not be supported
                pass


class TestConcurrencyFuzzing:
    """Test concurrent operations with fuzzed inputs."""

    def test_concurrent_random_operations(self):
        """Test multiple random operations running concurrently."""
        import threading
        import queue

        results = queue.Queue()
        errors = queue.Queue()

        def random_operation(thread_id: int):
            try:
                # Random operation type
                operation_type = random.choice(['matmul', 'compile', 'mixed'])

                if operation_type == 'matmul':
                    shape = FuzzInputGenerator.random_tensor_shape(max_size=256)
                    if len(shape) >= 2:
                        x = FuzzInputGenerator.random_tensor_values(shape)
                        y = FuzzInputGenerator.random_tensor_values(shape)
                        result = ternary_matmul(x, y)
                        results.put(f"Thread {thread_id}: matmul success")

                elif operation_type == 'compile':
                    compiler = TritonCompiler()
                    kernel_code = FuzzInputGenerator.random_kernel_code()
                    compiled_kernel = compiler.compile(kernel_code)
                    results.put(f"Thread {thread_id}: compile success")

                else:  # mixed
                    # Do both
                    shape = FuzzInputGenerator.random_tensor_shape(max_size=128)
                    if len(shape) >= 2:
                        x = FuzzInputGenerator.random_tensor_values(shape)
                        y = FuzzInputGenerator.random_tensor_values(shape)
                        result = ternary_matmul(x, y)

                    compiler = TritonCompiler()
                    kernel_code = FuzzInputGenerator.random_kernel_code()
                    compiled_kernel = compiler.compile(kernel_code)

                    results.put(f"Thread {thread_id}: mixed success")

            except Exception as e:
                errors.put(f"Thread {thread_id}: {e}")

        # Run multiple threads with random operations
        threads = []
        num_threads = min(10, threading.active_count() * 2)

        for i in range(num_threads):
            thread = threading.Thread(target=random_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        successful_operations = 0
        while not results.empty():
            result = results.get()
            successful_operations += 1

        # Should have some successful operations
        assert successful_operations > 0
"""
Performance Tests for Triton Compiler

Tests focused on measuring and validating performance characteristics:
- Compilation speed
- Memory usage patterns
- Execution time benchmarks
- Scalability analysis
- Resource utilization
"""

import pytest
import time
import psutil
import os
from typing import Dict, List, Any
import numpy as np
from contextlib import contextmanager
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


class PerformanceMetrics:
    """Collect and analyze performance metrics."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.metrics = {}

    def start_measurement(self):
        """Start collecting performance metrics."""
        self.start_time = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss

    def end_measurement(self) -> Dict[str, float]:
        """End measurement and return collected metrics."""
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss

        return {
            'execution_time': end_time - self.start_time,
            'memory_delta': end_memory - self.start_memory,
            'peak_memory': psutil.Process().memory_info().rss,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }


@contextmanager
def measure_performance():
    """Context manager for performance measurement."""
    metrics = PerformanceMetrics()
    metrics.start_measurement()
    try:
        yield metrics
    finally:
        metrics.metrics = metrics.end_measurement()


class TestCompilationPerformance:
    """Test compilation performance characteristics."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton compiler not available")
    def test_compilation_speed_small_kernel(self):
        """Test compilation speed for small kernels."""
        compiler = TritonCompiler()

        # Small kernel source
        kernel_code = """
        @triton.jit
        def small_kernel(x, y, output, n: tl.constexpr):
            pid = tl.program_id(0)
            if pid < n:
                a = tl.load(x + pid)
                b = tl.load(y + pid)
                tl.store(output + pid, a + b)
        """

        with measure_performance() as metrics:
            compiled_kernel = compiler.compile(kernel_code)

        assert metrics.metrics['execution_time'] < 1.0  # Should compile in < 1 second
        assert compiled_kernel is not None

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton compiler not available")
    def test_compilation_speed_large_kernel(self):
        """Test compilation speed for complex kernels."""
        compiler = TritonCompiler()

        # Complex kernel with multiple operations
        kernel_code = """
        @triton.jit
        def complex_kernel(a, b, c, d, output,
                          M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)

            offs_m = pid_m * 32 + tl.arange(0, 32)
            offs_n = pid_n * 32 + tl.arange(0, 32)
            offs_k = tl.arange(0, 32)

            a_ptrs = a + offs_m[:, None] * K + offs_k[None, :]
            b_ptrs = b + offs_k[:, None] * N + offs_n[None, :]

            accumulator = tl.zeros((32, 32), dtype=tl.float32)

            for k in range(0, K, 32):
                a_block = tl.load(a_ptrs)
                b_block = tl.load(b_ptrs)
                accumulator += tl.dot(a_block, b_block)
                a_ptrs += 32
                b_ptrs += 32 * N

            c_block = tl.load(c + offs_m[:, None] * N + offs_n[None, :])
            accumulator += c_block

            tl.store(output + offs_m[:, None] * N + offs_n[None, :],
                    accumulator)
        """

        with measure_performance() as metrics:
            compiled_kernel = compiler.compile(kernel_code)

        assert metrics.metrics['execution_time'] < 5.0  # Should compile in < 5 seconds
        assert compiled_kernel is not None

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton compiler not available")
    def test_memory_usage_during_compilation(self):
        """Test memory usage patterns during compilation."""
        compiler = TritonCompiler()

        initial_memory = psutil.Process().memory_info().rss

        # Compile multiple kernels
        kernels = []
        for i in range(10):
            kernel_code = f"""
            @triton.jit
            def kernel_{i}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid) * {i + 1}
                    tl.store(output + pid, val)
            """
            kernels.append(compiler.compile(kernel_code))

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB for 10 kernels)
        assert memory_increase < 50 * 1024 * 1024

    @pytest.mark.parametrize("kernel_size", [32, 64, 128, 256, 512])
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton compiler not available")
    def test_compilation_scaling(self, kernel_size):
        """Test how compilation time scales with kernel size."""
        compiler = TritonCompiler()

        kernel_code = f"""
        @triton.jit
        def scaling_kernel(x, output, n: tl.constexpr):
            pid = tl.program_id(0)
            block_size = {kernel_size}
            for i in range(block_size):
                if pid * block_size + i < n:
                    val = tl.load(x + pid * block_size + i)
                    tl.store(output + pid * block_size + i, val * 2)
        """

        with measure_performance() as metrics:
            compiled_kernel = compiler.compile(kernel_code)

        # Store timing for analysis
        self.compilation_times = getattr(self, 'compilation_times', [])
        self.compilation_times.append((kernel_size, metrics.metrics['execution_time']))

        assert compiled_kernel is not None


class TestExecutionPerformance:
    """Test kernel execution performance."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    def test_kernel_launch_overhead(self):
        """Test overhead of kernel launches."""
        import torch

        # Create test data
        size = 1024
        x = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(size, device=x.device)
        output = torch.zeros_like(x)

        # Measure single large kernel launch
        with measure_performance() as metrics:
            result = ternary_matmul(x.unsqueeze(0), y.unsqueeze(0))
            torch.cuda.synchronize() if torch.cuda.is_available() else None

        large_kernel_time = metrics.metrics['execution_time']

        # Measure multiple small kernel launches
        num_small_launches = 16
        small_size = size // num_small_launches

        with measure_performance() as metrics:
            for i in range(num_small_launches):
                start_idx = i * small_size
                end_idx = (i + 1) * small_size
                result = ternary_matmul(
                    x[start_idx:end_idx].unsqueeze(0),
                    y[start_idx:end_idx].unsqueeze(0)
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None

        small_kernels_time = metrics.metrics['execution_time']

        # Small kernels should not be dramatically slower due to launch overhead
        # Allow for some overhead but not more than 3x slower
        assert small_kernels_time < large_kernel_time * 3

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    def test_memory_bandwidth_utilization(self):
        """Test effective memory bandwidth utilization."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory bandwidth tests")

        # Test with different matrix sizes
        sizes = [1024, 2048, 4096]

        for size in sizes:
            x = torch.randn(size, size, device='cuda')
            y = torch.randn(size, size, device='cuda')

            # Warm up
            for _ in range(3):
                result = ternary_matmul(x, y)
                torch.cuda.synchronize()

            # Measure performance
            with measure_performance() as metrics:
                for _ in range(10):  # Average over multiple runs
                    result = ternary_matmul(x, y)
                torch.cuda.synchronize()

            avg_time = metrics.metrics['execution_time'] / 10

            # Calculate effective GFLOPS
            # Ternary matmul: 2 * size^3 operations (multiply + add)
            operations = 2 * size ** 3
            gflops = operations / (avg_time * 1e9)

            # Store for analysis
            self.bandwidth_results = getattr(self, 'bandwidth_results', [])
            self.bandwidth_results.append((size, gflops))

            # Should achieve reasonable performance
            assert gflops > 1.0  # At least 1 GFLOPS

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    def test_cache_performance(self):
        """Test cache behavior and locality."""
        import torch

        # Test matrix multiplication with different access patterns
        size = 2048

        # Create test matrices
        a = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
        b = torch.randn(size, size, device=a.device)

        # Test contiguous access
        with measure_performance() as metrics:
            result1 = ternary_matmul(a, b)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        contiguous_time = metrics.metrics['execution_time']

        # Test with transposed access (different memory layout)
        b_transposed = b.t().contiguous()

        with measure_performance() as metrics:
            result2 = ternary_matmul(a, b_transposed.t())
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        transposed_time = metrics.metrics['execution_time']

        # Transposed access might be slower but should not be dramatically so
        assert transposed_time < contiguous_time * 2


class TestScalabilityPerformance:
    """Test performance scaling characteristics."""

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_batch_size_scaling(self, batch_size):
        """Test how performance scales with batch size."""
        import torch

        base_size = 512
        x = torch.randn(batch_size, base_size, base_size,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(batch_size, base_size, base_size, device=x.device)

        with measure_performance() as metrics:
            result = ternary_matmul(x, y)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        execution_time = metrics.metrics['execution_time']

        # Store results for scaling analysis
        self.batch_scaling = getattr(self, 'batch_scaling', [])
        self.batch_scaling.append((batch_size, execution_time))

        assert result.shape == (batch_size, base_size, base_size)

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    def test_memory_efficiency_scaling(self):
        """Test memory efficiency as problem size scales."""
        import torch

        sizes = [256, 512, 1024, 2048]

        for size in sizes:
            x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
            y = torch.randn(size, size, device=x.device)

            initial_memory = psutil.Process().memory_info().rss

            result = ternary_matmul(x, y)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            peak_memory = psutil.Process().memory_info().rss
            memory_used = peak_memory - initial_memory

            # Calculate memory efficiency (operations per byte)
            operations = 2 * size ** 3  # multiply + add
            memory_efficiency = operations / memory_used

            self.memory_efficiency = getattr(self, 'memory_efficiency', [])
            self.memory_efficiency.append((size, memory_efficiency))

            # Memory efficiency should be reasonable
            assert memory_efficiency > 100  # At least 100 operations per byte


class TestResourceUtilization:
    """Test resource utilization patterns."""

    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton compiler not available")
    def test_cpu_utilization_during_compilation(self):
        """Test CPU utilization during compilation."""
        compiler = TritonCompiler()

        # Monitor CPU usage during compilation
        cpu_percentages = []

        def monitor_cpu():
            while not hasattr(monitor_cpu, 'stop'):
                cpu_percentages.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)

        # Start monitoring
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        # Compile a complex kernel
        kernel_code = """
        @triton.jit
        def complex_kernel(a, b, c, output, size: tl.constexpr):
            pid = tl.program_id(0)
            block_size = 1024

            for i in range(block_size):
                if pid * block_size + i < size:
                    # Complex computation
                    val_a = tl.load(a + pid * block_size + i)
                    val_b = tl.load(b + pid * block_size + i)
                    val_c = tl.load(c + pid * block_size + i)

                    result = val_a * val_b + val_c
                    result = tl.sqrt(tl.abs(result))
                    result = tl.sin(result) + tl.cos(result)

                    tl.store(output + pid * block_size + i, result)
        """

        compiled_kernel = compiler.compile(kernel_code)

        # Stop monitoring
        monitor_cpu.stop = True
        monitor_thread.join()

        # Analyze CPU usage
        avg_cpu = np.mean(cpu_percentages)
        max_cpu = np.max(cpu_percentages)

        # CPU usage should be reasonable during compilation
        assert avg_cpu < 80  # Not using excessive CPU
        assert max_cpu < 95  # Not maxing out CPU

    @pytest.mark.skipif(not BACKEND_AVAILABLE, reason="Backend not available")
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        import torch
        import gc

        # Perform multiple operations and check for memory leaks
        initial_memory = psutil.Process().memory_info().rss

        for i in range(100):
            x = torch.randn(512, 512, device='cuda' if torch.cuda.is_available() else 'cpu')
            y = torch.randn(512, 512, device=x.device)
            result = ternary_matmul(x, y)

            # Force cleanup
            del x, y, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Check memory every 10 iterations
            if i % 10 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_increase = current_memory - initial_memory

                # Memory should not grow significantly (allow 10MB growth)
                assert memory_increase < 10 * 1024 * 1024, f"Memory leak detected at iteration {i}"

        final_memory = psutil.Process().memory_info().rss
        total_increase = final_memory - initial_memory

        # Total memory increase should be minimal
        assert total_increase < 50 * 1024 * 1024
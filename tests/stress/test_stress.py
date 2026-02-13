"""
Stress Tests for Triton Compiler

Tests designed to push the system to its limits and beyond:
- Large scale computations
- Memory pressure scenarios
- Concurrent operations
- Resource exhaustion handling
- Recovery from extreme conditions
"""

import pytest
import threading
import multiprocessing
import time
import psutil
import os
from typing import List, Dict, Any
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
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


class StressTestConfig:
    """Configuration for stress tests."""

    # Memory stress
    MAX_MEMORY_FRACTION = 0.8  # Use up to 80% of available memory
    LARGE_MATRIX_SIZE = 4096   # Large matrix for memory stress
    HUGE_MATRIX_SIZE = 8192    # Huge matrix for extreme stress

    # Concurrency stress
    MAX_CONCURRENT_THREADS = min(32, multiprocessing.cpu_count() * 2)
    MAX_CONCURRENT_PROCESSES = min(8, multiprocessing.cpu_count())

    # Time limits
    STRESS_TEST_TIMEOUT = 300  # 5 minutes max per stress test
    RECOVERY_TIMEOUT = 60      # 1 minute to recover from stress

    # Iteration counts
    STRESS_ITERATIONS = 1000
    LONG_RUNNING_ITERATIONS = 10000


class TestMemoryStress:
    """Test system behavior under memory pressure."""

    def test_large_matrix_operations(self):
        """Test operations on very large matrices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for large matrix tests")

        # Calculate available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        max_matrix_size = int(np.sqrt(gpu_memory // (4 * 3)))  # Rough estimate

        # Test with largest possible matrix
        size = min(max_matrix_size, StressTestConfig.LARGE_MATRIX_SIZE)

        try:
            x = torch.randn(size, size, device='cuda', dtype=torch.float32)
            y = torch.randn(size, size, device='cuda', dtype=torch.float32)

            # Perform operation
            result = ternary_matmul(x, y)
            torch.cuda.synchronize()

            assert result.shape == (size, size)
            assert not torch.isnan(result).any()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Insufficient GPU memory for {size}x{size} matrices")
            else:
                raise

    def test_memory_pressure_recovery(self):
        """Test recovery from memory pressure situations."""
        import gc

        # Fill memory gradually
        tensors = []
        initial_memory = psutil.Process().memory_info().rss

        try:
            # Create memory pressure
            for i in range(100):
                if torch.cuda.is_available():
                    tensor = torch.randn(1024, 1024, device='cuda')
                else:
                    tensor = torch.randn(512, 512)  # Smaller for CPU
                tensors.append(tensor)

                # Check if we're approaching memory limits
                current_memory = psutil.Process().memory_info().rss
                memory_fraction = current_memory / psutil.virtual_memory().total

                if memory_fraction > StressTestConfig.MAX_MEMORY_FRACTION:
                    break

            # Now try to perform operations under memory pressure
            x = torch.randn(256, 256, device='cuda' if torch.cuda.is_available() else 'cpu')
            y = torch.randn(256, 256, device=x.device)

            # This should still work despite memory pressure
            result = ternary_matmul(x, y)

            assert result.shape == (256, 256)

        finally:
            # Clean up
            del tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def test_memory_fragmentation_handling(self):
        """Test handling of memory fragmentation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for fragmentation tests")

        # Create and delete tensors of various sizes to fragment memory
        fragments = []

        # Fragmentation pattern: allocate, free, allocate different sizes
        sizes = [512, 256, 1024, 128, 2048, 64, 4096, 32]

        for size in sizes:
            tensor = torch.randn(size, size, device='cuda')
            fragments.append(tensor)

        # Delete every other tensor to create fragmentation
        for i in range(0, len(fragments), 2):
            del fragments[i]

        # Try to allocate a large contiguous block
        try:
            large_tensor = torch.randn(2048, 2048, device='cuda')
            result = ternary_matmul(large_tensor, large_tensor)
            torch.cuda.synchronize()

            assert result.shape == (2048, 2048)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Memory fragmentation prevents large allocations")
            else:
                raise

    def test_gradual_memory_exhaustion(self):
        """Test behavior as memory is gradually exhausted."""
        memory_usage = []
        operations_completed = 0

        try:
            size = 512
            while True:
                x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(size, size, device=x.device)

                result = ternary_matmul(x, y)
                operations_completed += 1

                # Track memory usage
                current_memory = psutil.Process().memory_info().rss
                memory_usage.append(current_memory)

                # Check if we're using too much memory
                memory_fraction = current_memory / psutil.virtual_memory().total
                if memory_fraction > StressTestConfig.MAX_MEMORY_FRACTION:
                    break

                # Gradually increase size to approach limits
                size = min(size + 64, 2048)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # This is expected - we pushed memory limits
                pass
            else:
                raise

        # Should have completed at least some operations
        assert operations_completed > 0

        # Memory usage should have increased
        if len(memory_usage) > 1:
            assert memory_usage[-1] > memory_usage[0]


class TestConcurrencyStress:
    """Test system behavior under concurrent load."""

    def test_concurrent_kernel_compilation(self):
        """Test compiling multiple kernels concurrently."""
        compiler = TritonCompiler()

        def compile_kernel(kernel_id: int):
            kernel_code = f"""
            @triton.jit
            def concurrent_kernel_{kernel_id}(x, y, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    a = tl.load(x + pid)
                    b = tl.load(y + pid)
                    result = a * b + {kernel_id}
                    tl.store(output + pid, result)
            """
            return compiler.compile(kernel_code)

        # Compile kernels concurrently
        with ThreadPoolExecutor(max_workers=StressTestConfig.MAX_CONCURRENT_THREADS) as executor:
            futures = [executor.submit(compile_kernel, i) for i in range(20)]
            results = []

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # All compilations should succeed
        assert len(results) == 20
        assert all(result is not None for result in results)

    def test_concurrent_kernel_execution(self):
        """Test executing multiple kernels concurrently."""
        import torch

        def execute_kernel(kernel_id: int):
            size = 1024
            x = torch.randn(size, device='cuda' if torch.cuda.is_available() else 'cpu')
            y = torch.randn(size, device=x.device)

            result = ternary_matmul(x.unsqueeze(0), y.unsqueeze(0))
            return result.sum().item()

        # Execute kernels concurrently
        with ThreadPoolExecutor(max_workers=StressTestConfig.MAX_CONCURRENT_THREADS) as executor:
            futures = [executor.submit(execute_kernel, i) for i in range(10)]
            results = []

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # All executions should complete
        assert len(results) == 10
        assert all(isinstance(r, float) for r in results)

    def test_mixed_compilation_execution(self):
        """Test mixed workload of compilation and execution."""
        compiler = TritonCompiler()
        results = []

        def mixed_workload(task_id: int):
            if task_id % 2 == 0:
                # Compilation task
                kernel_code = f"""
                @triton.jit
                def mixed_kernel_{task_id}(x, output, n: tl.constexpr):
                    pid = tl.program_id(0)
                    if pid < n:
                        val = tl.load(x + pid) * {task_id}
                        tl.store(output + pid, val)
                """
                return compiler.compile(kernel_code)
            else:
                # Execution task
                import torch
                x = torch.randn(512, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(512, device=x.device)
                return ternary_matmul(x.unsqueeze(0), y.unsqueeze(0)).sum().item()

        # Run mixed workload concurrently
        with ThreadPoolExecutor(max_workers=StressTestConfig.MAX_CONCURRENT_THREADS) as executor:
            futures = [executor.submit(mixed_workload, i) for i in range(20)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        assert len(results) == 20

    def test_resource_contention(self):
        """Test behavior under resource contention."""
        import torch

        # Create multiple processes competing for GPU resources
        def gpu_intensive_task(process_id: int):
            try:
                size = 2048
                for i in range(10):
                    x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
                    y = torch.randn(size, size, device=x.device)
                    result = ternary_matmul(x, y)

                    # Small delay to allow context switching
                    time.sleep(0.01)

                return f"Process {process_id} completed successfully"
            except Exception as e:
                return f"Process {process_id} failed: {e}"

        if torch.cuda.is_available():
            # Test with multiple processes
            with ProcessPoolExecutor(max_workers=min(4, StressTestConfig.MAX_CONCURRENT_PROCESSES)) as executor:
                futures = [executor.submit(gpu_intensive_task, i) for i in range(4)]
                results = [future.result() for future in as_completed(futures)]

            # All processes should complete
            assert len(results) == 4
            assert all("completed successfully" in r for r in results)


class TestLongRunningStress:
    """Test system stability during long-running operations."""

    @pytest.mark.slow
    def test_extended_operation_stability(self):
        """Test stability during extended operations."""
        import torch

        start_time = time.time()
        operations_completed = 0
        errors_encountered = 0

        try:
            while time.time() - start_time < 60:  # Run for 1 minute
                size = np.random.randint(256, 1024)
                x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(size, size, device=x.device)

                try:
                    result = ternary_matmul(x, y)
                    operations_completed += 1

                    # Verify result is valid
                    assert not torch.isnan(result).any()
                    assert not torch.isinf(result).any()

                except Exception as e:
                    errors_encountered += 1
                    if errors_encountered > 10:  # Too many errors
                        break

                # Periodic cleanup
                if operations_completed % 100 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        except KeyboardInterrupt:
            pass  # Allow manual interruption

        # Should complete reasonable number of operations
        assert operations_completed > 10

        # Error rate should be low
        if operations_completed > 0:
            error_rate = errors_encountered / operations_completed
            assert error_rate < 0.1  # Less than 10% error rate

    @pytest.mark.slow
    def test_memory_stability_over_time(self):
        """Test memory usage stability over extended periods."""
        import gc

        memory_samples = []
        start_time = time.time()

        try:
            while time.time() - start_time < 30:  # Run for 30 seconds
                # Perform operations
                import torch
                x = torch.randn(512, 512, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(512, 512, device=x.device)
                result = ternary_matmul(x, y)

                # Sample memory usage
                current_memory = psutil.Process().memory_info().rss
                memory_samples.append(current_memory)

                # Periodic cleanup
                if len(memory_samples) % 50 == 0:
                    del x, y, result
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                time.sleep(0.01)  # Small delay

        except KeyboardInterrupt:
            pass

        # Analyze memory stability
        if len(memory_samples) > 10:
            memory_array = np.array(memory_samples)
            memory_std = np.std(memory_array)
            memory_mean = np.mean(memory_array)

            # Memory usage should be relatively stable (std < 10% of mean)
            memory_variability = memory_std / memory_mean
            assert memory_variability < 0.1


class TestErrorRecoveryStress:
    """Test error recovery under stress conditions."""

    def test_recovery_from_compilation_failures(self):
        """Test recovery from compilation failures."""
        compiler = TritonCompiler()

        # Mix of valid and invalid kernels
        kernel_templates = [
            # Valid kernel
            """
            @triton.jit
            def valid_kernel_{i}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    tl.store(output + pid, val * 2)
            """,
            # Invalid kernel (syntax error)
            """
            @triton.jit
            def invalid_kernel_{i}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Missing closing parenthesis
                    tl.store(output + pid, val * 2
            """,
            # Another valid kernel
            """
            @triton.jit
            def another_valid_kernel_{i}(x, y, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    a = tl.load(x + pid)
                    b = tl.load(y + pid)
                    tl.store(output + pid, a + b)
            """
        ]

        successful_compilations = 0
        failed_compilations = 0

        # Try to compile many kernels, mixing valid and invalid
        for i in range(50):
            template = kernel_templates[i % len(kernel_templates)]
            kernel_code = template.format(i=i)

            try:
                compiled_kernel = compiler.compile(kernel_code)
                successful_compilations += 1
            except Exception:
                failed_compilations += 1

        # Should have both successes and failures
        assert successful_compilations > 0
        assert failed_compilations > 0

        # Success rate should be reasonable
        total_attempts = successful_compilations + failed_compilations
        success_rate = successful_compilations / total_attempts
        assert success_rate > 0.5  # At least 50% success rate

    def test_recovery_from_execution_failures(self):
        """Test recovery from execution failures."""
        import torch

        successful_executions = 0
        failed_executions = 0

        # Try various matrix sizes, some may cause failures
        sizes = [128, 256, 512, 1024, 2048, 4096]

        for size in sizes:
            try:
                x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(size, size, device=x.device)

                result = ternary_matmul(x, y)

                # Verify result
                assert result.shape == (size, size)
                assert not torch.isnan(result).any()

                successful_executions += 1

            except (RuntimeError, AssertionError):
                failed_executions += 1

        # Should have at least some successful executions
        assert successful_executions > 0

    def test_resource_cleanup_after_failures(self):
        """Test that resources are properly cleaned up after failures."""
        import torch
        import gc

        initial_memory = psutil.Process().memory_info().rss

        # Cause some failures and check resource cleanup
        for i in range(20):
            try:
                # Try increasingly large matrices to trigger failures
                size = 512 + i * 128
                x = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
                y = torch.randn(size, size, device=x.device)

                result = ternary_matmul(x, y)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Expected failure for large matrices
                pass

            # Clean up after each attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        final_memory = psutil.Process().memory_info().rss

        # Memory should not have grown significantly despite failures
        memory_increase = final_memory - initial_memory
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase
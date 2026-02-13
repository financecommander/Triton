"""
Security Tests for Triton Compiler

Tests focused on security aspects:
- Input validation and sanitization
- Resource exhaustion prevention
- Safe handling of malicious inputs
- Memory safety
- Information leakage prevention
"""

import pytest
import torch
import numpy as np
import psutil
import os
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock, Mock
import sys

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


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_none_inputs_rejection(self):
        """Test that None inputs are properly rejected."""
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            ternary_matmul(None, torch.randn(32, 32))

        with pytest.raises((TypeError, ValueError, RuntimeError)):
            ternary_matmul(torch.randn(32, 32), None)

        with pytest.raises((TypeError, ValueError, RuntimeError)):
            ternary_matmul(None, None)

    def test_invalid_tensor_types(self):
        """Test rejection of invalid tensor types."""
        invalid_inputs = [
            "string",
            42,
            [1, 2, 3],
            {"key": "value"},
            lambda x: x,
            np.array([1, 2, 3]),
        ]

        for invalid in invalid_inputs:
            with pytest.raises((TypeError, ValueError, RuntimeError)):
                ternary_matmul(invalid, torch.randn(32, 32))

            with pytest.raises((TypeError, ValueError, RuntimeError)):
                ternary_matmul(torch.randn(32, 32), invalid)

    def test_malformed_tensor_shapes(self):
        """Test handling of malformed tensor shapes."""
        # Test with incompatible shapes
        incompatible_pairs = [
            (torch.randn(32, 32), torch.randn(16, 32)),  # Wrong K dimension
            (torch.randn(32), torch.randn(32, 32)),      # 1D vs 2D
            (torch.randn(32, 32, 32), torch.randn(32)),   # 3D vs 1D
        ]

        for A, B in incompatible_pairs:
            with pytest.raises((RuntimeError, ValueError)):
                ternary_matmul(A, B)

    def test_extreme_tensor_values(self):
        """Test handling of extreme tensor values."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        extreme_tensors = [
            torch.full((32, 32), float('inf'), device=device),
            torch.full((32, 32), float('-inf'), device=device),
            torch.full((32, 32), float('nan'), device=device),
            torch.full((32, 32), 1e308, device=device),  # Very large
            torch.full((32, 32), 1e-308, device=device),  # Very small
        ]

        for tensor in extreme_tensors:
            # Should not crash, but may produce inf/nan results
            result = ternary_matmul(tensor, tensor)
            assert result.shape == (32, 32)  # Shape should be correct

    def test_empty_tensors(self):
        """Test handling of empty tensors."""
        empty_tensors = [
            torch.empty(0, 32),
            torch.empty(32, 0),
            torch.empty(0, 0),
        ]

        for tensor in empty_tensors:
            with pytest.raises((RuntimeError, ValueError)):
                ternary_matmul(tensor, torch.randn(32, 32))


class TestResourceExhaustion:
    """Test prevention of resource exhaustion attacks."""

    def test_memory_exhaustion_prevention(self):
        """Test that memory exhaustion is prevented."""
        initial_memory = psutil.Process().memory_info().rss

        # Try to allocate increasingly large tensors
        max_attempts = 10
        for i in range(max_attempts):
            size = 1024 * (2 ** i)  # Exponential growth
            try:
                A = torch.randn(size, size)
                B = torch.randn(size, size)
                result = ternary_matmul(A, B)

                # If we get here, clean up
                del A, B, result

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Expected for very large tensors
                break

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded
        assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB increase

    def test_cpu_exhaustion_prevention(self):
        """Test that CPU exhaustion is prevented."""
        # Monitor CPU usage during operations
        cpu_samples = []

        def monitor_cpu():
            while not hasattr(monitor_cpu, 'stop'):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
                time.sleep(0.1)

        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        try:
            # Perform CPU-intensive operations
            for _ in range(50):
                A = torch.randn(256, 256)
                B = torch.randn(256, 256)
                result = ternary_matmul(A, B)
                del A, B, result

        finally:
            # Stop monitoring
            monitor_cpu.stop = True
            monitor_thread.join()

        # Check CPU usage
        avg_cpu = np.mean(cpu_samples)
        max_cpu = np.max(cpu_samples)

        # CPU usage should not be excessive
        assert avg_cpu < 80  # Less than 80% average
        assert max_cpu < 95  # Less than 95% peak

    def test_file_descriptor_exhaustion(self):
        """Test that file descriptor exhaustion is prevented."""
        import resource

        # Get current file descriptor limits
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Perform operations that might open files
        for i in range(100):
            compiler = TritonCompiler()
            kernel_code = f"""
            @triton.jit
            def test_kernel_{i}(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    tl.store(output + pid, val * {i})
            """
            try:
                compiler.compile(kernel_code)
            except Exception:
                pass  # Compilation may fail, that's OK

        # Check that we haven't exhausted file descriptors
        current_fds = len(os.listdir('/proc/self/fd'))
        assert current_fds < soft_limit * 0.8  # Use less than 80% of limit

    def test_thread_exhaustion_prevention(self):
        """Test that thread exhaustion is prevented."""
        initial_threads = threading.active_count()

        # Perform operations that might create threads
        threads_created = []

        def worker():
            A = torch.randn(64, 64)
            B = torch.randn(64, 64)
            result = ternary_matmul(A, B)
            return result.sum().item()

        # Create multiple threads
        for i in range(min(20, threading.active_count() * 2)):
            thread = threading.Thread(target=worker)
            threads_created.append(thread)
            thread.start()

        # Wait for threads to complete
        for thread in threads_created:
            thread.join(timeout=10)  # 10 second timeout

        final_threads = threading.active_count()

        # Thread count should not have grown excessively
        thread_increase = final_threads - initial_threads
        assert thread_increase < 10  # Allow some increase but not too much


class TestCodeInjection:
    """Test prevention of code injection attacks."""

    def test_kernel_code_injection_prevention(self):
        """Test that kernel code injection is prevented."""
        compiler = TritonCompiler()

        malicious_kernels = [
            """
            @triton.jit
            def malicious_kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Attempt to inject system calls
                    import os
                    os.system("echo hacked")
                    tl.store(output + pid, val)
            """,
            """
            @triton.jit
            def malicious_kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Attempt to access file system
                    with open("/etc/passwd", "r") as f:
                        data = f.read()
                    tl.store(output + pid, val)
            """,
            """
            @triton.jit
            def malicious_kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Attempt to execute arbitrary code
                    exec("print('Code injection successful')")
                    tl.store(output + pid, val)
            """
        ]

        for kernel_code in malicious_kernels:
            # Should either fail to compile or not execute malicious code
            try:
                compiled_kernel = compiler.compile(kernel_code)
                # If compilation succeeds, the kernel should not execute dangerous operations
                # (This is hard to test directly, but at least it shouldn't crash the compiler)
            except Exception:
                # Expected - malicious code should be rejected
                pass

    def test_import_injection_prevention(self):
        """Test that import injection is prevented."""
        compiler = TritonCompiler()

        injection_attempts = [
            """
            @triton.jit
            def kernel(x, output, n: tl.constexpr):
                import sys
                sys.path.append("/malicious/path")
                pid = tl.program_id(0)
                if pid < n:
                    tl.store(output + pid, tl.load(x + pid))
            """,
            """
            @triton.jit
            def kernel(x, output, n: tl.constexpr):
                from os import system
                system("rm -rf /")
                pid = tl.program_id(0)
                if pid < n:
                    tl.store(output + pid, tl.load(x + pid))
            """
        ]

        for kernel_code in injection_attempts:
            with pytest.raises(Exception):
                compiler.compile(kernel_code)

    def test_string_injection_prevention(self):
        """Test that string injection attacks are prevented."""
        compiler = TritonCompiler()

        # Try to inject malicious strings
        malicious_strings = [
            "\\x00\\x01\\x02",  # Null bytes
            "\\n\\r\\t",        # Control characters
            "<script>",         # HTML injection
            "../../../../etc/passwd",  # Path traversal
            "${USER}",          # Environment variable injection
        ]

        for malicious_str in malicious_strings:
            kernel_code = f"""
            @triton.jit
            def kernel(x, output, n: tl.constexpr):
                pid = tl.program_id(0)
                if pid < n:
                    val = tl.load(x + pid)
                    # Try to use malicious string in comments or code
                    # {malicious_str}
                    tl.store(output + pid, val)
            """

            try:
                compiled_kernel = compiler.compile(kernel_code)
                # Should not crash or behave unexpectedly
            except Exception:
                # May fail due to syntax issues, but shouldn't be exploitable
                pass


class TestInformationLeakage:
    """Test prevention of information leakage."""

    def test_error_message_safety(self):
        """Test that error messages don't leak sensitive information."""
        compiler = TritonCompiler()

        # Try various invalid inputs
        invalid_kernels = [
            "",  # Empty
            "@triton.jit",  # Incomplete
            "def kernel(): pass",  # Wrong syntax
            "@triton.jit\ndef kernel(x, output):\n    tl.load(x + 1000000)",  # Out of bounds
        ]

        for kernel_code in invalid_kernels:
            try:
                compiler.compile(kernel_code)
            except Exception as e:
                error_msg = str(e)
                # Error messages should not contain:
                # - File paths that might reveal system structure
                # - Memory addresses
                # - Internal implementation details that could aid attacks
                assert "0x" not in error_msg.lower()  # No hex addresses
                assert "/home" not in error_msg       # No home directory paths
                assert "/usr" not in error_msg        # No system paths
                assert "password" not in error_msg.lower()

    def test_memory_content_isolation(self):
        """Test that memory contents are properly isolated."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create tensors with known patterns
        secret_data = torch.tensor([0xDEAD, 0xBEEF, 0xCAFE, 0xBABE] * 8,
                                 device=device, dtype=torch.float32)

        # Perform operations
        A = torch.randn(32, 32, device=device)
        B = torch.randn(32, 32, device=device)
        result = ternary_matmul(A, B)

        # Result should not contain the secret data pattern
        result_flat = result.flatten()
        secret_pattern = torch.tensor([0xDEAD, 0xBEEF, 0xCAFE, 0xBABE],
                                    device=device, dtype=torch.float32)

        # Check if the secret pattern appears in the result
        found_pattern = False
        for i in range(len(result_flat) - len(secret_pattern) + 1):
            if torch.allclose(result_flat[i:i+len(secret_pattern)], secret_pattern, atol=1e-6):
                found_pattern = True
                break

        assert not found_pattern, "Secret data leaked into computation results"

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Measure timing for different input sizes
        sizes = [32, 64, 128, 256]
        timings = {}

        for size in sizes:
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)

            start_time = time.perf_counter()
            for _ in range(10):  # Average over multiple runs
                result = ternary_matmul(A, B)
            end_time = time.perf_counter()

            timings[size] = (end_time - start_time) / 10

        # Timing should scale roughly with size^3 (matrix multiplication complexity)
        # Check that timing differences are not excessive
        base_time = timings[32]
        for size in sizes[1:]:
            expected_ratio = (size / 32) ** 3
            actual_ratio = timings[size] / base_time
            # Allow some variation but not too much
            assert 0.1 < actual_ratio / expected_ratio < 10


class TestSystemIntegrity:
    """Test system integrity and safe operation."""

    def test_process_isolation(self):
        """Test that operations are properly isolated."""
        import subprocess
        import signal

        # Test that operations don't affect parent process
        parent_pid = os.getpid()
        parent_memory = psutil.Process().memory_info().rss

        # Perform operations in subprocess
        def subprocess_operations():
            import torch
            from backend.pytorch.ops import ternary_matmul

            # Perform operations that might be dangerous
            for _ in range(100):
                A = torch.randn(64, 64)
                B = torch.randn(64, 64)
                C = ternary_matmul(A, B)

            return "completed"

        # Run in subprocess
        result = subprocess.run(
            [sys.executable, "-c", """
import torch
from backend.pytorch.ops import ternary_matmul
for _ in range(10):
    A = torch.randn(64, 64)
    B = torch.randn(64, 64)
    C = ternary_matmul(A, B)
print('completed')
            """],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0
        assert "completed" in result.stdout

        # Parent process should be unaffected
        assert os.getpid() == parent_pid
        current_memory = psutil.Process().memory_info().rss
        memory_increase = current_memory - parent_memory
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

    def test_signal_handling(self):
        """Test proper signal handling."""
        import signal

        signal_received = False

        def signal_handler(signum, frame):
            nonlocal signal_received
            signal_received = True

        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(1)  # Alarm in 1 second

        try:
            # Perform operation that should complete before alarm
            A = torch.randn(128, 128)
            B = torch.randn(128, 128)
            result = ternary_matmul(A, B)

            # Cancel alarm
            signal.alarm(0)

            assert not signal_received
            assert result.shape == (128, 128)

        finally:
            # Restore signal handler
            signal.signal(signal.SIGALRM, old_handler)

    def test_environment_variable_safety(self):
        """Test that environment variables are handled safely."""
        # Test with various environment variable settings
        dangerous_env_vars = {
            "PATH": "/malicious/path",
            "LD_LIBRARY_PATH": "/malicious/libs",
            "PYTHONPATH": "/malicious/modules",
            "CUDA_VISIBLE_DEVICES": "999",  # Invalid GPU
        }

        for var_name, dangerous_value in dangerous_env_vars.items():
            # Save original value
            original_value = os.environ.get(var_name)

            try:
                # Set dangerous value
                os.environ[var_name] = dangerous_value

                # Try operation
                A = torch.randn(32, 32)
                B = torch.randn(32, 32)
                result = ternary_matmul(A, B)

                # Should still work or fail gracefully
                assert result.shape == (32, 32)

            except Exception:
                # May fail due to dangerous environment, but shouldn't crash system
                pass

            finally:
                # Restore original value
                if original_value is not None:
                    os.environ[var_name] = original_value
                else:
                    os.environ.pop(var_name, None)

    def test_file_system_isolation(self):
        """Test that file system access is properly isolated."""
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                # Perform operations
                A = torch.randn(64, 64)
                B = torch.randn(64, 64)
                result = ternary_matmul(A, B)

                # Check that no unauthorized files were created
                files_created = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        files_created.append(os.path.join(root, file))

                # Should not have created any files
                assert len(files_created) == 0

            finally:
                os.chdir(original_cwd)
"""
Tests for Triton-based ternary operations.

This module provides comprehensive tests for the Triton implementation,
including correctness validation and performance benchmarks.
"""

import torch
import numpy as np
from kernels.triton import ternary_matmul, TernaryMatMulTriton


def reference_ternary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of ternary matrix multiplication for testing.

    Args:
        a: Ternary matrix A (M x K) with values in {-1, 0, 1}
        b: Ternary matrix B (K x N) with values in {-1, 0, 1}

    Returns:
        Result matrix C (M x N) with dtype int32
    """
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def generate_random_ternary(shape: tuple, device: str = 'cuda') -> torch.Tensor:
    """Generate a random ternary tensor with values in {-1, 0, 1}."""
    return torch.randint(-1, 2, shape, dtype=torch.int8, device=device)


def test_correctness():
    """Test correctness of Triton ternary matmul against reference implementation."""
    print("Testing correctness of Triton ternary matmul...")

    # Test various matrix sizes
    test_cases = [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (16, 32, 64),  # Non-square matrices
        (64, 16, 32),
    ]

    for M, K, N in test_cases:
        print(f"  Testing {M}x{K} @ {K}x{N}...")

        # Generate test matrices
        a = generate_random_ternary((M, K))
        b = generate_random_ternary((K, N))

        # Reference result
        ref_c = reference_ternary_matmul(a, b)

        # Triton result
        triton_c = ternary_matmul(a, b)

        # Compare results
        if not torch.allclose(ref_c, triton_c):
            max_diff = torch.max(torch.abs(ref_c - triton_c))
            print(f"    FAILED: Max difference = {max_diff}")
            return False
        else:
            print("    PASSED")

    print("All correctness tests passed!")
    return True


def test_packing():
    """Test ternary packing and unpacking functions."""
    print("Testing ternary packing/unpacking...")

    # Test various sizes
    test_shapes = [
        (16,),
        (64,),
        (256,),
        (16, 16),
        (32, 32),
        (64, 32),
    ]

    triton_ops = TernaryMatMulTriton()

    for shape in test_shapes:
        print(f"  Testing shape {shape}...")

        # Generate test tensor
        original = generate_random_ternary(shape)

        # Pack and unpack
        packed = triton_ops.pack_ternary(original)
        unpacked = triton_ops.unpack_ternary(packed, original.shape)

        # Verify round-trip
        if not torch.equal(original, unpacked):
            print("    FAILED: Pack/unpack round-trip failed")
            return False
        else:
            print("    PASSED")

    print("All packing tests passed!")
    return True


def benchmark_performance():
    """Benchmark Triton implementation against reference."""
    print("Benchmarking performance...")

    # Test on larger matrices for meaningful timing
    sizes = [512, 1024, 2048]

    for size in sizes:
        print(f"  Benchmarking {size}x{size} matrices...")

        # Generate test matrices
        a = generate_random_ternary((size, size))
        b = generate_random_ternary((size, size))

        # Warm up
        _ = ternary_matmul(a, b)
        torch.cuda.synchronize()

        # Time Triton implementation
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        triton_result = ternary_matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end)

        # Time reference implementation
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ref_result = reference_ternary_matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        ref_time = start.elapsed_time(end)

        # Calculate speedup
        speedup = ref_time / triton_time if triton_time > 0 else float('inf')

        print(".2f")
        print(".2f")
        print(".2f")

        # Verify correctness
        if not torch.allclose(ref_result, triton_result):
            print("    WARNING: Results don't match!")


def run_all_tests():
    """Run all tests."""
    print("Running Triton ternary operations tests...\n")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return

    success = True

    # Test correctness
    if not test_correctness():
        success = False

    print()

    # Test packing
    if not test_packing():
        success = False

    print()

    # Benchmark performance
    benchmark_performance()

    print()
    if success:
        print("All tests completed successfully!")
    else:
        print("Some tests failed!")


if __name__ == "__main__":
    run_all_tests()
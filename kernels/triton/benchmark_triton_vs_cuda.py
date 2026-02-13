"""
Performance benchmark for Triton vs CUDA ternary matrix multiplication.

This script compares the performance of the new Triton implementation
against the existing CUDA implementation and measures the speedup achieved.

Usage:
    python benchmark_triton_vs_cuda.py

Requirements:
    - CUDA-capable GPU (A100/H100 recommended)
    - PyTorch with CUDA support
    - Existing CUDA implementation in kernels/cuda/
"""

import torch
import time
import numpy as np
from typing import List, Tuple, Dict
from kernels.triton import ternary_matmul, TernaryMatMulTriton


def load_cuda_implementation():
    """Load the existing CUDA implementation for comparison."""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cuda'))

        # Import the CUDA implementation
        from ternary_ops import ternary_matmul as cuda_ternary_matmul
        return cuda_ternary_matmul
    except ImportError as e:
        print(f"Warning: Could not load CUDA implementation: {e}")
        return None


def generate_test_matrices(sizes: List[Tuple[int, int, int]], device: str = 'cuda') -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate test matrices for benchmarking."""
    matrices = []
    for M, K, N in sizes:
        # Generate ternary matrices with some structure for realistic benchmarking
        a = torch.randint(-1, 2, (M, K), dtype=torch.int8, device=device)
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device=device)
        matrices.append((a, b))
    return matrices


def benchmark_implementation(func, matrices: List[Tuple[torch.Tensor, torch.Tensor]],
                           name: str, warmup_runs: int = 3, benchmark_runs: int = 10) -> Dict:
    """Benchmark a matrix multiplication implementation."""
    print(f"Benchmarking {name}...")

    results = {
        'name': name,
        'sizes': [],
        'times': [],
        'gflops': [],
        'throughput': []
    }

    for i, (a, b) in enumerate(matrices):
        M, K = a.shape
        K2, N = b.shape

        # Warmup runs
        for _ in range(warmup_runs):
            _ = func(a, b)
        torch.cuda.synchronize()

        # Benchmark runs
        times = []
        for _ in range(benchmark_runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            c = func(a, b)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)  # Convert to seconds

        avg_time = np.mean(times)
        min_time = np.min(times)

        # Calculate performance metrics
        # For ternary matmul: each output element requires K operations
        # Each operation is a multiply-accumulate with ternary values
        operations = 2 * M * N * K  # 2 because we count both multiply and add
        gflops = operations / (min_time * 1e9)
        throughput = M * N / min_time / 1e6  # M*N elements per second (millions)

        results['sizes'].append((M, K, N))
        results['times'].append(min_time)
        results['gflops'].append(gflops)
        results['throughput'].append(throughput)

        print(".3f"
    return results


def compare_implementations(cuda_results: Dict, triton_results: Dict) -> Dict:
    """Compare CUDA and Triton implementations."""
    comparison = {
        'sizes': cuda_results['sizes'],
        'cuda_times': cuda_results['times'],
        'triton_times': triton_results['times'],
        'speedups': [],
        'cuda_gflops': cuda_results['gflops'],
        'triton_gflops': triton_results['gflops'],
        'gflops_improvements': []
    }

    for cuda_time, triton_time, cuda_gflops, triton_gflops in zip(
        cuda_results['times'], triton_results['times'],
        cuda_results['gflops'], triton_results['gflops']
    ):
        speedup = cuda_time / triton_time if triton_time > 0 else float('inf')
        gflops_improvement = (triton_gflops - cuda_gflops) / cuda_gflops * 100 if cuda_gflops > 0 else 0

        comparison['speedups'].append(speedup)
        comparison['gflops_improvements'].append(gflops_improvement)

    return comparison


def print_comparison_table(comparison: Dict):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: Triton vs CUDA")
    print("="*80)
    print("<12")
    print("-" * 80)

    for i, (size, cuda_time, triton_time, speedup, cuda_gflops, triton_gflops, gflops_imp) in enumerate(zip(
        comparison['sizes'], comparison['cuda_times'], comparison['triton_times'],
        comparison['speedups'], comparison['cuda_gflops'], comparison['triton_gflops'],
        comparison['gflops_improvements']
    )):
        print("<12")

    print("-" * 80)

    # Summary statistics
    avg_speedup = np.mean(comparison['speedups'])
    avg_gflops_imp = np.mean(comparison['gflops_improvements'])

    print(".2f")
    print(".1f")

    if avg_speedup >= 1.2:
        print("🎉 SUCCESS: Achieved 20%+ performance improvement target!")
    else:
        print("⚠️  WARNING: Did not meet 20% performance improvement target")


def run_benchmarks():
    """Run comprehensive benchmarks comparing Triton vs CUDA."""
    print("Triton vs CUDA Ternary Matrix Multiplication Benchmark")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires a CUDA GPU.")
        return

    device = torch.device('cuda')
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # Load CUDA implementation
    cuda_func = load_cuda_implementation()
    if cuda_func is None:
        print("❌ Could not load CUDA implementation for comparison.")
        return

    # Test matrix sizes (suitable for A100/H100 benchmarking)
    test_sizes = [
        (1024, 1024, 1024),    # Square matrices
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 2048, 1024),    # Rectangular matrices
        (2048, 1024, 2048),
        (1024, 4096, 1024),
    ]

    print(f"Generating test matrices for {len(test_sizes)} size configurations...")
    matrices = generate_test_matrices(test_sizes, device=str(device))

    # Benchmark CUDA implementation
    cuda_results = benchmark_implementation(cuda_func, matrices, "CUDA")

    # Benchmark Triton implementation
    triton_func = lambda a, b: ternary_matmul(a, b)
    triton_results = benchmark_implementation(triton_func, matrices, "Triton")

    # Compare results
    comparison = compare_implementations(cuda_results, triton_results)
    print_comparison_table(comparison)

    # Verify correctness on small matrices
    print("\n" + "-"*60)
    print("CORRECTNESS VERIFICATION")
    print("-"*60)

    small_a = torch.randint(-1, 2, (64, 64), dtype=torch.int8, device=device)
    small_b = torch.randint(-1, 2, (64, 64), dtype=torch.int8, device=device)

    cuda_result = cuda_func(small_a, small_b)
    triton_result = ternary_matmul(small_a, small_b)

    if torch.allclose(cuda_result, triton_result):
        print("✓ Correctness verified: Triton and CUDA produce identical results")
    else:
        max_diff = torch.max(torch.abs(cuda_result - triton_result))
        print(f"❌ Correctness check failed: Max difference = {max_diff}")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    run_benchmarks()
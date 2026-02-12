"""
Benchmark script for ternary matrix multiplication

This script compares the performance of the optimized CUDA ternary matmul
kernel against standard PyTorch operations.
"""

import time
import torch
import numpy as np
from typing import Tuple, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from kernels.cuda.ternary_ops import ternary_matmul, get_ternary_matmul
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import CUDA operations: {e}")
    CUDA_AVAILABLE = False


def generate_ternary_matrix(M: int, N: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate a random ternary matrix with values in {-1, 0, 1}.
    
    Args:
        M: Number of rows
        N: Number of columns
        device: Device to create tensor on
    
    Returns:
        Ternary matrix
    """
    # Generate random integers in {-1, 0, 1}
    matrix = torch.randint(-1, 2, (M, N), dtype=torch.int8, device=device)
    return matrix


def naive_ternary_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Naive implementation of ternary matrix multiplication using PyTorch.
    
    Args:
        A: Matrix A (M x K)
        B: Matrix B (K x N)
    
    Returns:
        Result matrix C (M x N)
    """
    return torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.int16)


def benchmark_operation(
    func,
    A: torch.Tensor,
    B: torch.Tensor,
    warmup_iters: int = 5,
    benchmark_iters: int = 20
) -> Tuple[float, torch.Tensor]:
    """
    Benchmark an operation.
    
    Args:
        func: Function to benchmark
        A: Matrix A
        B: Matrix B
        warmup_iters: Number of warmup iterations
        benchmark_iters: Number of benchmark iterations
    
    Returns:
        Tuple of (average time in ms, result)
    """
    # Warmup
    for _ in range(warmup_iters):
        result = func(A, B)
    
    if A.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_iters):
        start = time.perf_counter()
        result = func(A, B)
        if A.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time, result


def verify_correctness(
    A: torch.Tensor,
    B: torch.Tensor,
    result_cuda: torch.Tensor,
    result_naive: torch.Tensor,
    tolerance: float = 1e-3
) -> bool:
    """
    Verify that CUDA result matches naive implementation.
    
    Args:
        A: Matrix A
        B: Matrix B
        result_cuda: Result from CUDA kernel
        result_naive: Result from naive implementation
        tolerance: Relative tolerance for comparison
    
    Returns:
        True if results match
    """
    diff = torch.abs(result_cuda.float() - result_naive.float())
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    
    # Check if all values are close
    matches = torch.allclose(
        result_cuda.float(),
        result_naive.float(),
        rtol=tolerance,
        atol=1.0
    )
    
    return matches


def run_benchmark_suite():
    """Run a comprehensive benchmark suite."""
    print("=" * 80)
    print("Ternary Matrix Multiplication Benchmark")
    print("=" * 80)
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Running CPU-only benchmarks.")
        device = 'cpu'
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    
    if not CUDA_AVAILABLE and device == 'cuda':
        print("Warning: CUDA operations not available. Skipping CUDA benchmarks.")
        device = 'cpu'
    
    print()
    
    # Test sizes: (M, K, N)
    test_sizes = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]
    
    print(f"Device: {device}")
    print()
    
    for M, K, N in test_sizes:
        print(f"Matrix dimensions: ({M} x {K}) @ ({K} x {N})")
        print("-" * 80)
        
        # Generate test matrices
        A = generate_ternary_matrix(M, K, device=device)
        B = generate_ternary_matrix(K, N, device=device)
        
        # Benchmark naive implementation
        avg_time_naive, std_time_naive, result_naive = benchmark_operation(
            naive_ternary_matmul, A, B
        )
        print(f"Naive PyTorch: {avg_time_naive:.3f} ± {std_time_naive:.3f} ms")
        
        # Benchmark CUDA kernel if available
        if CUDA_AVAILABLE and device == 'cuda':
            try:
                avg_time_cuda, std_time_cuda, result_cuda = benchmark_operation(
                    ternary_matmul, A, B
                )
                print(f"CUDA Kernel:   {avg_time_cuda:.3f} ± {std_time_cuda:.3f} ms")
                
                # Calculate speedup
                speedup = avg_time_naive / avg_time_cuda
                print(f"Speedup:       {speedup:.2f}x")
                print()
                
                # Verify correctness
                print("Correctness check:")
                is_correct = verify_correctness(A, B, result_cuda, result_naive)
                print(f"  Results match: {is_correct}")
                
            except Exception as e:
                print(f"Error running CUDA kernel: {e}")
        
        print()
    
    # Memory efficiency test
    print("=" * 80)
    print("Memory Efficiency Analysis")
    print("=" * 80)
    print()
    
    M, K, N = 1024, 1024, 1024
    A = generate_ternary_matrix(M, K, device=device)
    B = generate_ternary_matrix(K, N, device=device)
    
    # Calculate theoretical memory savings
    unpacked_size = M * K + K * N  # int8 for each element
    packed_size = (M * K + 3) // 4 + (K * N + 3) // 4  # 4 trits per byte
    compression_ratio = unpacked_size / packed_size
    
    print(f"Matrix A size: {M} x {K}")
    print(f"Matrix B size: {K} x {N}")
    print(f"Unpacked storage: {unpacked_size / 1024 / 1024:.2f} MB")
    print(f"Packed storage: {packed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print()
    
    # Test packing/unpacking if CUDA available
    if CUDA_AVAILABLE and device == 'cuda':
        try:
            matmul_op = get_ternary_matmul()
            
            # Test packing
            start = time.perf_counter()
            A_packed = matmul_op.pack_ternary(A)
            pack_time = (time.perf_counter() - start) * 1000
            
            print(f"Packing time: {pack_time:.3f} ms")
            print(f"Packed size: {A_packed.numel()} bytes ({A_packed.numel() / 1024 / 1024:.2f} MB)")
            
            # Test unpacking
            start = time.perf_counter()
            A_unpacked = matmul_op.unpack_ternary(A_packed, M * K)
            unpack_time = (time.perf_counter() - start) * 1000
            
            print(f"Unpacking time: {unpack_time:.3f} ms")
            
            # Verify pack/unpack correctness
            A_flat = A.flatten()
            matches = torch.all(A_flat == A_unpacked[:A_flat.numel()])
            print(f"Pack/unpack correct: {matches.item()}")
            
        except Exception as e:
            print(f"Error testing packing: {e}")
    
    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_suite()

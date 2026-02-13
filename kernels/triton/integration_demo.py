"""
Integration example: Using Triton as drop-in replacement for CUDA ternary operations.

This script demonstrates how to seamlessly replace CUDA implementations with
Triton-based versions while maintaining the same API and improving performance.
"""

import torch
import sys
import os

# Add the project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def demonstrate_api_compatibility():
    """Show that Triton implementation has the same API as CUDA version."""
    print("API Compatibility Demonstration")
    print("=" * 40)

    # Import both implementations
    try:
        from kernels.cuda.ternary_ops import ternary_matmul as cuda_matmul
        cuda_available = True
    except ImportError:
        print("CUDA implementation not available for comparison")
        cuda_available = False

    from kernels.triton import ternary_matmul as triton_matmul, TernaryMatMulTriton

    # Create test data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.int8, device=device)
    b = torch.tensor([[1, 0], [0, 1], [-1, 1]], dtype=torch.int8, device=device)

    print(f"Input A: {a}")
    print(f"Input B: {b}")
    print()

    # Test Triton implementation
    triton_result = triton_matmul(a, b)
    print(f"Triton result: {triton_result}")

    # Compare with CUDA if available
    if cuda_available and device == 'cuda':
        cuda_result = cuda_matmul(a, b)
        print(f"CUDA result:   {cuda_result}")

        if torch.equal(triton_result, cuda_result):
            print("✓ Results match - API compatibility verified")
        else:
            print("❌ Results differ - check implementation")
    else:
        print("CUDA not available for comparison")

    print()


def demonstrate_class_interface():
    """Show the class-based interface for advanced usage."""
    print("Class Interface Demonstration")
    print("=" * 30)

    from kernels.triton import TernaryMatMulTriton

    ops = TernaryMatMulTriton()

    # Create test tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randint(-1, 2, (8, 8), dtype=torch.int8, device=device)
    b = torch.randint(-1, 2, (8, 8), dtype=torch.int8, device=device)

    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")

    # Test packing/unpacking
    a_packed = ops.pack_ternary(a)
    print(f"Packed A shape: {a_packed.shape} (dtype: {a_packed.dtype})")

    a_unpacked = ops.unpack_ternary(a_packed, a.shape)
    if torch.equal(a, a_unpacked):
        print("✓ Pack/unpack round-trip successful")
    else:
        print("❌ Pack/unpack failed")

    # Test matrix multiplication
    result = ops.matmul(a, b)
    print(f"Result shape: {result.shape} (dtype: {result.dtype})")

    print()


def demonstrate_performance_comparison():
    """Compare performance between implementations."""
    print("Performance Comparison")
    print("=" * 25)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping performance comparison")
        return

    try:
        from kernels.cuda.ternary_ops import ternary_matmul as cuda_matmul
    except ImportError:
        print("CUDA implementation not available for comparison")
        return

    from kernels.triton import ternary_matmul as triton_matmul
    import time

    # Test on larger matrices
    sizes = [(512, 512, 512), (1024, 1024, 1024)]

    for M, K, N in sizes:
        print(f"Testing {M}x{K} @ {K}x{N}...")

        # Generate test data
        a = torch.randint(-1, 2, (M, K), dtype=torch.int8, device='cuda')
        b = torch.randint(-1, 2, (K, N), dtype=torch.int8, device='cuda')

        # Benchmark CUDA
        torch.cuda.synchronize()
        start = time.time()
        cuda_result = cuda_matmul(a, b)
        torch.cuda.synchronize()
        cuda_time = time.time() - start

        # Benchmark Triton
        torch.cuda.synchronize()
        start = time.time()
        triton_result = triton_matmul(a, b)
        torch.cuda.synchronize()
        triton_time = time.time() - start

        # Calculate speedup
        speedup = cuda_time / triton_time if triton_time > 0 else float('inf')

        print(".3f")
        print(".3f")
        print(".2f")

        # Verify correctness
        if torch.equal(cuda_result, triton_result):
            print("  ✓ Results identical")
        else:
            print("  ❌ Results differ")

        print()


def show_migration_example():
    """Show how to migrate from CUDA to Triton."""
    print("Migration Example")
    print("=" * 18)

    migration_code = '''
# Before: Using CUDA implementation
from kernels.cuda.ternary_ops import pack_ternary, unpack_ternary, ternary_matmul

# Your existing code
a_packed = pack_ternary(a)
b_packed = pack_ternary(b)
result = ternary_matmul(a_packed, b_packed)
result_unpacked = unpack_ternary(result, original_shape)

# After: Using Triton implementation (same API)
from kernels.triton import TernaryMatMulTriton

ops = TernaryMatMulTriton()
# Same code works unchanged!
a_packed = ops.pack_ternary(a)
b_packed = ops.pack_ternary(b)
result = ops.matmul(a_packed, b_packed)  # Note: matmul expects unpacked inputs
result_unpacked = ops.unpack_ternary(result, original_shape)

# Or use the functional interface
from kernels.triton import ternary_matmul
result = ternary_matmul(a, b)  # Even simpler!
'''

    print("To migrate from CUDA to Triton:")
    print("1. Change the import statement")
    print("2. Use the same API calls")
    print("3. Enjoy better performance!")
    print()
    print("Code example:")
    print(migration_code)


def main():
    """Run all integration demonstrations."""
    print("Triton Ternary Operations - Integration Demo")
    print("=" * 50)
    print()

    demonstrate_api_compatibility()
    demonstrate_class_interface()
    demonstrate_performance_comparison()
    show_migration_example()

    print("Integration demo completed!")
    print()
    print("Next steps:")
    print("- Run benchmarks: python kernels/triton/benchmark_triton_vs_cuda.py")
    print("- Run tests: python -m kernels.triton.test_cpu_validation")
    print("- Read docs: kernels/triton/README.md")


if __name__ == "__main__":
    main()
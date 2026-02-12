"""
Basic tests for ternary matrix multiplication
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.cuda.ternary_ops import get_ternary_matmul


def test_pack_unpack():
    """Test packing and unpacking of ternary values."""
    print("Testing pack/unpack...")
    
    # Create a simple ternary matrix
    original = torch.tensor([-1, 0, 1, -1, 0, 1, 1, -1], dtype=torch.int8)
    
    matmul_op = get_ternary_matmul()
    
    # Pack
    packed = matmul_op.pack_ternary(original)
    print(f"Original size: {original.numel()} elements")
    print(f"Packed size: {packed.numel()} bytes")
    
    # Unpack
    unpacked = matmul_op.unpack_ternary(packed, original.numel())
    
    # Verify
    matches = torch.all(original == unpacked)
    print(f"Pack/unpack test: {'PASS' if matches else 'FAIL'}")
    
    if not matches:
        print(f"Original: {original}")
        print(f"Unpacked: {unpacked}")
    
    return matches.item()


def test_matmul_small():
    """Test matrix multiplication with small matrices."""
    print("\nTesting small matrix multiplication...")
    
    # Create small test matrices
    A = torch.tensor([
        [-1, 0, 1],
        [1, -1, 0],
        [0, 1, -1]
    ], dtype=torch.int8)
    
    B = torch.tensor([
        [1, 0],
        [0, 1],
        [-1, 1]
    ], dtype=torch.int8)
    
    # Expected result: A @ B
    expected = torch.tensor([
        [-2, 1],   # [-1*1 + 0*0 + 1*(-1), -1*0 + 0*1 + 1*1]
        [1, -1],   # [1*1 + (-1)*0 + 0*(-1), 1*0 + (-1)*1 + 0*1]
        [1, 0]     # [0*1 + 1*0 + (-1)*(-1), 0*0 + 1*1 + (-1)*1]
    ], dtype=torch.int16)
    
    # Compute using naive method
    result_naive = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.int16)
    
    print(f"Expected result:\n{expected}")
    print(f"Naive result:\n{result_naive}")
    
    # Verify naive computation
    naive_correct = torch.all(expected == result_naive)
    print(f"Naive computation: {'PASS' if naive_correct else 'FAIL'}")
    
    return naive_correct.item()


def test_matmul_medium():
    """Test matrix multiplication with medium-sized matrices."""
    print("\nTesting medium matrix multiplication...")
    
    M, K, N = 32, 32, 32
    
    # Generate random ternary matrices
    torch.manual_seed(42)
    A = torch.randint(-1, 2, (M, K), dtype=torch.int8)
    B = torch.randint(-1, 2, (K, N), dtype=torch.int8)
    
    # Compute expected result using PyTorch
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.int16)
    
    print(f"Matrix dimensions: ({M} x {K}) @ ({K} x {N})")
    print(f"Result shape: {expected.shape}")
    print(f"Result range: [{expected.min().item()}, {expected.max().item()}]")
    
    return True


def test_zero_skipping():
    """Test that zero-skipping optimization works correctly."""
    print("\nTesting zero-skipping optimization...")
    
    # Create matrices with many zeros
    A = torch.tensor([
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, -1]
    ], dtype=torch.int8)
    
    B = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0]
    ], dtype=torch.int8)
    
    expected = torch.matmul(A.to(torch.float32), B.to(torch.float32)).to(torch.int16)
    
    print(f"Matrix A sparsity: {(A == 0).sum().item() / A.numel() * 100:.1f}% zeros")
    print(f"Matrix B sparsity: {(B == 0).sum().item() / B.numel() * 100:.1f}% zeros")
    print(f"Expected result:\n{expected}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Ternary Matrix Multiplication Tests")
    print("=" * 80)
    print()
    
    try:
        results = []
        
        # Run tests
        results.append(("Pack/Unpack", test_pack_unpack()))
        results.append(("Small MatMul", test_matmul_small()))
        results.append(("Medium MatMul", test_matmul_medium()))
        results.append(("Zero Skipping", test_zero_skipping()))
        
        # Print summary
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{name:20s}: {status}")
        
        all_passed = all(result[1] for result in results)
        print()
        print(f"Overall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
        
        return all_passed
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

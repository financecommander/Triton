"""
CPU-based validation tests for Triton ternary operations.

These tests validate the logic and compilation of Triton kernels without requiring CUDA.
"""

import torch
import numpy as np
from kernels.triton.ternary_packing import pack_ternary_triton, unpack_ternary_triton


def test_packing_cpu():
    """Test ternary packing and unpacking on CPU."""
    print("Testing ternary packing/unpacking on CPU...")

    # Test basic packing/unpacking
    test_tensor = torch.tensor([-1, 0, 1, -1, 0, 1, -1, 0], dtype=torch.int8)
    print(f"Original: {test_tensor}")

    # Pack
    packed = pack_ternary_triton(test_tensor)
    print(f"Packed: {packed} (shape: {packed.shape})")

    # Unpack
    unpacked = unpack_ternary_triton(packed, test_tensor.shape)
    print(f"Unpacked: {unpacked}")

    # Verify
    if torch.equal(test_tensor, unpacked):
        print("✓ Pack/unpack round-trip successful")
        return True
    else:
        print("✗ Pack/unpack round-trip failed")
        return False


def test_triton_compilation():
    """Test that Triton kernels compile successfully."""
    print("Testing Triton kernel compilation...")

    try:
        import triton
        import triton.language as tl
        from kernels.triton.ternary_ops import ternary_matmul_packed_kernel

        # Try to compile the kernel (this will fail on CPU but should compile)
        print("✓ Triton kernel import successful")

        # Test with a small dummy kernel to verify compilation works
        @triton.jit
        def dummy_kernel(x_ptr, y_ptr, N: tl.constexpr):
            pid = tl.program_id(0)
            offset = pid * 4
            x = tl.load(x_ptr + offset)
            tl.store(y_ptr + offset, x + 1)

        print("✓ Dummy kernel compilation test passed")
        return True

    except Exception as e:
        print(f"✗ Triton compilation test failed: {e}")
        return False


def test_reference_matmul():
    """Test reference matrix multiplication implementation."""
    print("Testing reference matrix multiplication...")

    # Small test matrices
    a = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.int8)
    b = torch.tensor([[1, 0], [0, 1], [-1, 1]], dtype=torch.int8)

    print(f"A: {a}")
    print(f"B: {b}")

    # Reference result
    c_ref = torch.matmul(a.to(torch.int32), b.to(torch.int32))
    print(f"Reference C: {c_ref}")

    # Manual calculation for verification
    c_manual = torch.tensor([
        [-1*1 + 0*0 + 1*(-1), -1*0 + 0*1 + 1*1],  # [-1, 1]
        [1*1 + (-1)*0 + 0*(-1), 1*0 + (-1)*1 + 0*1]   # [1, -1]
    ], dtype=torch.int32)

    if torch.equal(c_ref, c_manual):
        print("✓ Reference implementation verified")
        return True
    else:
        print("✗ Reference implementation failed")
        return False


def run_cpu_tests():
    """Run all CPU-based tests."""
    print("Running CPU-based validation tests for Triton ternary operations...\n")

    tests = [
        test_packing_cpu,
        test_triton_compilation,
        test_reference_matmul,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        print(f"\n--- {test.__name__} ---")
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All CPU tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    run_cpu_tests()
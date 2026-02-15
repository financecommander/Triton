"""
Tests for GPU Optimization Module.

Tests cover:
- Vectorized packing/unpacking
- Memory layout optimization
- GPU kernel dispatch (CPU fallback)
- Batch processing optimization
- GPUOptimizer class interface
"""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.triton_gpu.gpu_optimizer import (
    GPUOptimizer,
    gpu_ternary_matmul,
    batch_ternary_matmul,
    pack_ternary_vectorized,
    unpack_ternary_vectorized,
    optimize_memory_layout,
    ensure_contiguous_layout,
)

HAS_CUDA = torch.cuda.is_available()


# ============================================================================
# Vectorized Packing Tests
# ============================================================================


class TestVectorizedPacking:
    """Test vectorized pack/unpack operations."""

    def test_pack_basic(self):
        """Test basic packing of 8 ternary values."""
        tensor = torch.tensor([-1, 0, 1, -1, 0, 1, 1, 0], dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        assert packed.dtype == torch.uint8
        assert packed.numel() == 2

    def test_unpack_basic(self):
        """Test basic unpacking recovers original values."""
        tensor = torch.tensor([-1, 0, 1, -1, 0, 1, 1, 0], dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        unpacked = unpack_ternary_vectorized(packed, 8)
        assert torch.equal(tensor, unpacked)

    def test_roundtrip_non_multiple_of_4(self):
        """Test packing/unpacking when length is not a multiple of 4."""
        for length in [1, 2, 3, 5, 7, 13]:
            tensor = torch.randint(-1, 2, (length,), dtype=torch.int8)
            packed = pack_ternary_vectorized(tensor)
            unpacked = unpack_ternary_vectorized(packed, length)
            assert torch.equal(tensor, unpacked), f"Failed for length {length}"

    def test_all_minus_ones(self):
        """Test packing of all -1 values."""
        tensor = torch.full((8,), -1, dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        unpacked = unpack_ternary_vectorized(packed, 8)
        assert torch.equal(tensor, unpacked)

    def test_all_zeros(self):
        """Test packing of all 0 values."""
        tensor = torch.zeros(8, dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        unpacked = unpack_ternary_vectorized(packed, 8)
        assert torch.equal(tensor, unpacked)

    def test_all_ones(self):
        """Test packing of all 1 values."""
        tensor = torch.ones(8, dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        unpacked = unpack_ternary_vectorized(packed, 8)
        assert torch.equal(tensor, unpacked)

    def test_large_tensor(self):
        """Test packing/unpacking of a large tensor."""
        tensor = torch.randint(-1, 2, (10000,), dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        unpacked = unpack_ternary_vectorized(packed, 10000)
        assert torch.equal(tensor, unpacked)

    def test_compression_ratio(self):
        """Test that packing achieves 4x compression."""
        tensor = torch.randint(-1, 2, (1024,), dtype=torch.int8)
        packed = pack_ternary_vectorized(tensor)
        assert packed.numel() == 256  # 1024 / 4


# ============================================================================
# Memory Layout Optimization Tests
# ============================================================================


class TestMemoryLayout:
    """Test memory layout optimization."""

    def test_ensure_contiguous(self):
        """Test that non-contiguous tensors are made contiguous."""
        tensor = torch.randn(4, 4).t()  # Transpose makes it non-contiguous
        assert not tensor.is_contiguous()
        result = ensure_contiguous_layout(tensor)
        assert result.is_contiguous()

    def test_already_contiguous(self):
        """Test that contiguous tensors are returned as-is."""
        tensor = torch.randn(4, 4)
        assert tensor.is_contiguous()
        result = ensure_contiguous_layout(tensor)
        assert result is tensor  # Same object

    def test_optimize_float32_to_ternary(self):
        """Test that float32 weights are ternarized with correct layout."""
        weight = torch.tensor([[0.5, -0.3, 0.01, 0.8]], dtype=torch.float32)
        optimized = optimize_memory_layout(weight)
        assert optimized.dtype == torch.int8
        assert optimized.is_contiguous()
        # 0.5 -> 1, -0.3 -> -1, 0.01 -> 0 (below threshold), 0.8 -> 1
        expected = torch.tensor([[1, -1, 0, 1]], dtype=torch.int8)
        assert torch.equal(optimized, expected)

    def test_optimize_int8_passthrough(self):
        """Test that int8 tensors are clamped and returned as int8."""
        weight = torch.tensor([[-2, 0, 1, 3]], dtype=torch.int8)
        optimized = optimize_memory_layout(weight)
        assert optimized.dtype == torch.int8
        expected = torch.tensor([[-1, 0, 1, 1]], dtype=torch.int8)
        assert torch.equal(optimized, expected)


# ============================================================================
# GPU Matmul Tests (CPU fallback path)
# ============================================================================


class TestGPUTernaryMatmul:
    """Test GPU ternary matrix multiplication."""

    def test_basic_matmul(self):
        """Test basic ternary matrix multiplication."""
        a = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.int8)
        b = torch.tensor([[1, 0], [0, 1], [-1, 1]], dtype=torch.int8)
        result = gpu_ternary_matmul(a, b)
        expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        assert torch.equal(result, expected)

    def test_identity_matmul(self):
        """Test matmul with identity-like ternary matrix."""
        eye = torch.eye(4, dtype=torch.int8)
        a = torch.randint(-1, 2, (4, 4), dtype=torch.int8)
        result = gpu_ternary_matmul(a, eye)
        expected = torch.matmul(a.to(torch.int32), eye.to(torch.int32))
        assert torch.equal(result, expected)

    def test_zero_matrix(self):
        """Test matmul with zero matrix."""
        a = torch.zeros(4, 4, dtype=torch.int8)
        b = torch.randint(-1, 2, (4, 4), dtype=torch.int8)
        result = gpu_ternary_matmul(a, b)
        assert torch.all(result == 0)

    @pytest.mark.parametrize("m,k,n", [(4, 8, 4), (16, 32, 16), (64, 64, 64)])
    def test_various_sizes(self, m, k, n):
        """Test matmul with various matrix sizes."""
        a = torch.randint(-1, 2, (m, k), dtype=torch.int8)
        b = torch.randint(-1, 2, (k, n), dtype=torch.int8)
        result = gpu_ternary_matmul(a, b)
        expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        assert torch.equal(result, expected)

    def test_non_contiguous_input(self):
        """Test matmul with non-contiguous inputs."""
        a_full = torch.randint(-1, 2, (8, 4), dtype=torch.int8)
        a = a_full.t()  # 4x8, non-contiguous
        b = torch.randint(-1, 2, (8, 4), dtype=torch.int8)
        # Should handle non-contiguous gracefully
        result = gpu_ternary_matmul(a, b)
        assert result.shape == (4, 4)


# ============================================================================
# Batch Processing Tests
# ============================================================================


class TestBatchMatmul:
    """Test batch processing optimization."""

    def test_uniform_batch(self):
        """Test batch matmul with same-shaped inputs."""
        a_batch = [torch.randint(-1, 2, (4, 4), dtype=torch.int8) for _ in range(3)]
        b_batch = [torch.randint(-1, 2, (4, 4), dtype=torch.int8) for _ in range(3)]
        results = batch_ternary_matmul(a_batch, b_batch)

        assert len(results) == 3
        for a, b, r in zip(a_batch, b_batch, results):
            expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
            assert torch.equal(r, expected)

    def test_mixed_shapes(self):
        """Test batch matmul with different shapes falls back correctly."""
        a_batch = [
            torch.randint(-1, 2, (4, 4), dtype=torch.int8),
            torch.randint(-1, 2, (8, 8), dtype=torch.int8),
        ]
        b_batch = [
            torch.randint(-1, 2, (4, 4), dtype=torch.int8),
            torch.randint(-1, 2, (8, 8), dtype=torch.int8),
        ]
        results = batch_ternary_matmul(a_batch, b_batch)

        assert len(results) == 2
        assert results[0].shape == (4, 4)
        assert results[1].shape == (8, 8)

    def test_empty_batch(self):
        """Test batch matmul with empty input."""
        results = batch_ternary_matmul([], [])
        assert results == []

    def test_single_item_batch(self):
        """Test batch matmul with single item."""
        a = torch.randint(-1, 2, (4, 4), dtype=torch.int8)
        b = torch.randint(-1, 2, (4, 4), dtype=torch.int8)
        results = batch_ternary_matmul([a], [b])
        expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        assert len(results) == 1
        assert torch.equal(results[0], expected)

    def test_batch_size_mismatch(self):
        """Test that mismatched batch sizes raise ValueError."""
        a_batch = [torch.randint(-1, 2, (4, 4), dtype=torch.int8)]
        b_batch = [
            torch.randint(-1, 2, (4, 4), dtype=torch.int8),
            torch.randint(-1, 2, (4, 4), dtype=torch.int8),
        ]
        with pytest.raises(ValueError, match="Batch size mismatch"):
            batch_ternary_matmul(a_batch, b_batch)

    def test_large_uniform_batch(self):
        """Test batch processing with larger batch."""
        batch_size = 16
        a_batch = [torch.randint(-1, 2, (8, 8), dtype=torch.int8) for _ in range(batch_size)]
        b_batch = [torch.randint(-1, 2, (8, 8), dtype=torch.int8) for _ in range(batch_size)]
        results = batch_ternary_matmul(a_batch, b_batch)
        assert len(results) == batch_size

        for a, b, r in zip(a_batch, b_batch, results):
            expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
            assert torch.equal(r, expected)


# ============================================================================
# GPUOptimizer Class Tests
# ============================================================================


class TestGPUOptimizer:
    """Test GPUOptimizer class."""

    def test_init_default(self):
        """Test default initialization."""
        opt = GPUOptimizer()
        assert opt.device.type in ("cpu", "cuda")
        assert opt.backend_name in ("cpu", "cuda", "triton")

    def test_init_cpu(self):
        """Test explicit CPU initialization."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        assert opt.device.type == "cpu"
        assert opt.backend_name == "cpu"

    def test_optimize_weight(self):
        """Test weight optimization."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        weight = torch.randn(4, 4) * 0.5
        optimized = opt.optimize_weight(weight)
        assert optimized.dtype == torch.int8
        assert optimized.is_contiguous()
        # All values should be in {-1, 0, 1}
        assert torch.all((optimized >= -1) & (optimized <= 1))

    def test_matmul(self):
        """Test matmul through optimizer."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        a = torch.randint(-1, 2, (4, 8), dtype=torch.int8)
        b = torch.randint(-1, 2, (8, 4), dtype=torch.int8)
        result = opt.matmul(a, b)
        expected = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        assert torch.equal(result, expected)

    def test_batch_matmul(self):
        """Test batch matmul through optimizer."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        a_batch = [torch.randint(-1, 2, (4, 4), dtype=torch.int8) for _ in range(3)]
        b_batch = [torch.randint(-1, 2, (4, 4), dtype=torch.int8) for _ in range(3)]
        results = opt.batch_matmul(a_batch, b_batch)
        assert len(results) == 3

    def test_pack_unpack(self):
        """Test pack/unpack through optimizer."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        tensor = torch.randint(-1, 2, (16,), dtype=torch.int8)
        packed = opt.pack(tensor)
        unpacked = opt.unpack(packed, torch.Size([16]))
        assert torch.equal(tensor, unpacked)

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
    def test_init_cuda(self):
        """Test CUDA initialization."""
        opt = GPUOptimizer(device=torch.device("cuda"))
        assert opt.device.type == "cuda"

    def test_backend_name(self):
        """Test backend name reporting."""
        opt = GPUOptimizer(device=torch.device("cpu"))
        assert isinstance(opt.backend_name, str)
        assert opt.backend_name in ("cpu", "cuda", "triton")


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests verifying GPU optimizer works with ternary layers."""

    def test_ternary_linear_with_optimizer(self):
        """Test TernaryLinear works with GPU optimizer imported."""
        from backend.pytorch.ternary_tensor import TernaryLinear
        layer = TernaryLinear(8, 4)
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 4)

    def test_ternary_conv2d_with_optimizer(self):
        """Test TernaryConv2d works with GPU optimizer imported."""
        from backend.pytorch.ternary_tensor import TernaryConv2d
        layer = TernaryConv2d(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 8, 8)
        out = layer(x)
        assert out.shape == (1, 16, 8, 8)

    def test_ternary_matmul_function(self):
        """Test ternary_matmul function with GPU optimizer."""
        from backend.pytorch.ternary_tensor import ternary_matmul
        a = torch.randn(4, 8)
        b = torch.randn(8, 4)
        result = ternary_matmul(a, b)
        assert result.shape == (4, 4)

    def test_module_exports(self):
        """Test that GPU optimizer is accessible from backend module."""
        from backend.triton_gpu import (
            GPUOptimizer,
            gpu_ternary_matmul,
            batch_ternary_matmul,
            pack_ternary_vectorized,
            unpack_ternary_vectorized,
            optimize_memory_layout,
            ensure_contiguous_layout,
        )
        # All imports should work
        assert GPUOptimizer is not None
        assert callable(gpu_ternary_matmul)
        assert callable(batch_ternary_matmul)

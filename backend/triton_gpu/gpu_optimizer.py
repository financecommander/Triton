"""
GPU Optimization Module for Ternary Neural Networks.

Provides optimized GPU execution for ternary operations including:
- CUDA kernel dispatch with automatic fallback
- Batch processing optimization with efficient memory reuse
- Memory layout optimization for GPU-friendly access patterns
- Vectorized packing/unpacking for CPU fallback paths

This module bridges the gap between the low-level kernels in kernels/triton/
and the high-level model layers in backend/pytorch/.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

# Check for GPU kernel availability
try:
    from kernels.triton.ternary_ops import (
        ternary_matmul_triton,
        TernaryMatMulTriton,
    )
    from kernels.triton.ternary_packing import (
        pack_ternary_triton,
        unpack_ternary_triton,
    )
    TRITON_KERNELS_AVAILABLE = True
except ImportError:
    TRITON_KERNELS_AVAILABLE = False

HAS_CUDA = torch.cuda.is_available()


def pack_ternary_vectorized(tensor: torch.Tensor) -> torch.Tensor:
    """
    Vectorized CPU packing of ternary values into 2-bit representation.

    Replaces the element-by-element loop in pack_ternary_cpu with a fully
    vectorized implementation using torch operations for significantly
    better performance on large tensors.

    Args:
        tensor: Flat tensor with values in {-1, 0, 1}, dtype int8

    Returns:
        Packed tensor with dtype uint8, 4x smaller
    """
    flat = tensor.flatten().to(torch.int8)
    n_elements = flat.numel()

    # Pad to multiple of 4
    pad_size = (4 - n_elements % 4) % 4
    if pad_size > 0:
        flat = torch.cat([flat, torch.zeros(pad_size, dtype=torch.int8, device=flat.device)])

    # Map: -1 -> 0, 0 -> 1, 1 -> 2
    encoded = (flat + 1).to(torch.uint8)

    # Reshape to groups of 4 and pack
    encoded = encoded.view(-1, 4)
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    )

    return packed.to(torch.uint8)


def unpack_ternary_vectorized(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """
    Vectorized CPU unpacking of 2-bit packed ternary values.

    Args:
        packed: Packed tensor with dtype uint8
        n_elements: Number of original ternary elements

    Returns:
        Unpacked tensor with values in {-1, 0, 1}, dtype int8
    """
    packed_uint = packed.to(torch.uint8)

    # Extract each 2-bit value
    t0 = (packed_uint & 0x03).to(torch.int8)
    t1 = ((packed_uint >> 2) & 0x03).to(torch.int8)
    t2 = ((packed_uint >> 4) & 0x03).to(torch.int8)
    t3 = ((packed_uint >> 6) & 0x03).to(torch.int8)

    # Interleave and decode: 0 -> -1, 1 -> 0, 2 -> 1
    unpacked = torch.stack([t0, t1, t2, t3], dim=1).flatten()
    unpacked = unpacked[:n_elements] - 1

    return unpacked


def ensure_contiguous_layout(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor has contiguous memory layout optimal for GPU kernels.

    GPU kernels perform best with contiguous, row-major tensors.
    This function checks and fixes memory layout if needed.

    Args:
        tensor: Input tensor

    Returns:
        Contiguous tensor (may be the same object if already contiguous)
    """
    if not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor


def optimize_memory_layout(
    weight: torch.Tensor,
    layout: str = "contiguous",
) -> torch.Tensor:
    """
    Optimize tensor memory layout for GPU execution.

    For ternary operations, contiguous row-major layout ensures coalesced
    memory accesses on GPU, which is critical for throughput.

    Args:
        weight: Weight tensor to optimize
        layout: Layout strategy - "contiguous" for row-major (default)

    Returns:
        Tensor with optimized memory layout
    """
    weight = ensure_contiguous_layout(weight)

    # Ensure proper dtype for ternary operations
    if weight.dtype == torch.float32:
        # Ternarize: values below threshold become 0
        threshold = 0.05
        result = torch.sign(weight)
        result[torch.abs(weight) < threshold] = 0
        return result.to(torch.int8).contiguous()
    elif weight.dtype in (torch.int8, torch.int16, torch.int32):
        # Clamp to ternary range
        return torch.clamp(weight, -1, 1).to(torch.int8).contiguous()

    return weight


def gpu_ternary_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Perform ternary matrix multiplication with automatic GPU kernel dispatch.

    Selects the best available backend:
    1. Triton kernels (if available and on CUDA)
    2. Optimized int32 matmul fallback

    Args:
        a: Matrix A (M x K) with values in {-1, 0, 1}
        b: Matrix B (K x N) with values in {-1, 0, 1}

    Returns:
        Result matrix C (M x N)
    """
    a = ensure_contiguous_layout(a)
    b = ensure_contiguous_layout(b)

    if TRITON_KERNELS_AVAILABLE and a.is_cuda:
        a_int8 = a.to(torch.int8) if a.dtype != torch.int8 else a
        b_int8 = b.to(torch.int8) if b.dtype != torch.int8 else b
        return ternary_matmul_triton(a_int8, b_int8)

    # Fallback: standard matmul with int32
    return torch.matmul(a.to(torch.int32), b.to(torch.int32))


def batch_ternary_matmul(
    a_batch: List[torch.Tensor],
    b_batch: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Optimized batch processing for ternary matrix multiplications.

    Processes multiple matrix multiplications efficiently by:
    1. Stacking same-shaped inputs into a single batched operation
    2. Using torch.bmm for parallel execution on GPU
    3. Falling back to sequential processing for mixed shapes

    Args:
        a_batch: List of A matrices
        b_batch: List of B matrices

    Returns:
        List of result matrices
    """
    if len(a_batch) != len(b_batch):
        raise ValueError(
            f"Batch size mismatch: {len(a_batch)} vs {len(b_batch)}"
        )

    if len(a_batch) == 0:
        return []

    # Check if all matrices have the same shape for batched processing
    shapes_a = [a.shape for a in a_batch]
    shapes_b = [b.shape for b in b_batch]
    all_same_shape = (
        len(set(shapes_a)) == 1 and len(set(shapes_b)) == 1
    )

    if all_same_shape and len(a_batch) > 1:
        return _batched_matmul_uniform(a_batch, b_batch)
    else:
        return _batched_matmul_sequential(a_batch, b_batch)


def _batched_matmul_uniform(
    a_batch: List[torch.Tensor],
    b_batch: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Batch matmul for uniformly-shaped inputs using torch.bmm.

    Stacks all inputs into a single 3D tensor and uses batched
    matrix multiplication for parallel GPU execution.
    """
    device = a_batch[0].device

    # Stack into batch tensors: (batch, M, K) and (batch, K, N)
    a_stacked = torch.stack(
        [ensure_contiguous_layout(a) for a in a_batch]
    ).to(torch.int32)
    b_stacked = torch.stack(
        [ensure_contiguous_layout(b) for b in b_batch]
    ).to(torch.int32)

    # Use batched matrix multiply
    results = torch.bmm(a_stacked, b_stacked)

    return [results[i] for i in range(results.shape[0])]


def _batched_matmul_sequential(
    a_batch: List[torch.Tensor],
    b_batch: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Sequential fallback for mixed-shape batch processing."""
    return [
        gpu_ternary_matmul(a, b)
        for a, b in zip(a_batch, b_batch)
    ]


class GPUOptimizer:
    """
    GPU optimization coordinator for ternary neural network inference.

    Manages GPU kernel dispatch, memory layout optimization, and
    batch processing for optimal GPU utilization.

    Usage:
        optimizer = GPUOptimizer()
        optimized_weight = optimizer.optimize_weight(weight_tensor)
        result = optimizer.matmul(a, b)
        results = optimizer.batch_matmul(a_list, b_list)
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the GPU optimizer.

        Args:
            device: Target device. If None, uses CUDA if available, else CPU.
        """
        if device is not None:
            self.device = device
        elif HAS_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._triton_available = TRITON_KERNELS_AVAILABLE and self.device.type == "cuda"

    @property
    def backend_name(self) -> str:
        """Return the name of the active computation backend."""
        if self._triton_available:
            return "triton"
        elif self.device.type == "cuda":
            return "cuda"
        else:
            return "cpu"

    def optimize_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Optimize a weight tensor for GPU execution.

        Applies memory layout optimization and moves to target device.

        Args:
            weight: Weight tensor to optimize

        Returns:
            Optimized weight tensor on target device
        """
        optimized = optimize_memory_layout(weight)
        return optimized.to(self.device)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform optimized ternary matrix multiplication.

        Args:
            a: Matrix A (M x K)
            b: Matrix B (K x N)

        Returns:
            Result matrix C (M x N)
        """
        return gpu_ternary_matmul(a, b)

    def batch_matmul(
        self,
        a_batch: List[torch.Tensor],
        b_batch: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Perform optimized batch matrix multiplication.

        Args:
            a_batch: List of A matrices
            b_batch: List of B matrices

        Returns:
            List of result matrices
        """
        return batch_ternary_matmul(a_batch, b_batch)

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pack ternary tensor into 2-bit representation using best available method.

        Args:
            tensor: Tensor with values in {-1, 0, 1}

        Returns:
            Packed uint8 tensor
        """
        if self._triton_available and tensor.is_cuda:
            return pack_ternary_triton(tensor)
        return pack_ternary_vectorized(tensor)

    def unpack(
        self, packed: torch.Tensor, original_shape: torch.Size
    ) -> torch.Tensor:
        """
        Unpack 2-bit packed tensor back to ternary values.

        Args:
            packed: Packed uint8 tensor
            original_shape: Original tensor shape

        Returns:
            Unpacked tensor with values in {-1, 0, 1}
        """
        n_elements = math.prod(original_shape)

        if self._triton_available and packed.is_cuda:
            return unpack_ternary_triton(packed, original_shape)
        return unpack_ternary_vectorized(packed, n_elements).view(original_shape)

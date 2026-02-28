"""
Triton GPU backend for optimized ternary neural network operations.

Provides GPU-optimized batch processing, memory layout optimization,
and kernel dispatch for ternary matrix multiplication.
"""

from .gpu_optimizer import (
    GPUOptimizer,
    gpu_ternary_matmul,
    batch_ternary_matmul,
    pack_ternary_vectorized,
    unpack_ternary_vectorized,
    optimize_memory_layout,
    ensure_contiguous_layout,
)

__all__ = [
    "GPUOptimizer",
    "gpu_ternary_matmul",
    "batch_ternary_matmul",
    "pack_ternary_vectorized",
    "unpack_ternary_vectorized",
    "optimize_memory_layout",
    "ensure_contiguous_layout",
]
"""
Triton-based GPU kernels for ternary operations.

This package provides optimized Triton implementations for ternary matrix
multiplication with auto-tuning for high-performance GPUs like A100 and H100.
"""

from .ternary_ops import TernaryMatMulTriton, ternary_matmul, get_triton_matmul
from .ternary_packing import pack_ternary_triton, unpack_ternary_triton

__all__ = [
    'TernaryMatMulTriton',
    'ternary_matmul',
    'get_triton_matmul',
    'pack_ternary_triton',
    'unpack_ternary_triton',
]
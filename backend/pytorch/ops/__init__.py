"""
PyTorch operations for ternary neural networks.

This package provides quantization and activation functions optimized for
ternary neural networks with values in {-1, 0, 1}.
"""

from backend.pytorch.ops.activations import ternary_activation
from backend.pytorch.ops.quantize import quantize

__all__ = [
    "quantize",
    "ternary_activation",
]

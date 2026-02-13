"""PyTorch backend for Triton DSL."""

from .ternary_tensor import (
    TernaryTensor,
    ternary_matmul,
    TernaryLinear,
    TernaryConv2d,
)

from .ternary_models import (
    TernaryResNet18,
    TernaryMobileNetV2,
    TernaryBertTiny,
    TernaryMNISTNet,
    TernaryCIFAR10Net,
)

__all__ = [
    'TernaryTensor',
    'ternary_matmul',
    'TernaryLinear',
    'TernaryConv2d',
    'TernaryResNet18',
    'TernaryMobileNetV2',
    'TernaryBertTiny',
    'TernaryMNISTNet',
    'TernaryCIFAR10Net',
]

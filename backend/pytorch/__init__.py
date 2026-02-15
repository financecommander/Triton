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

# Import GPU optimizer with graceful fallback
try:
    from backend.triton_gpu import GPUOptimizer
    _GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    _GPU_OPTIMIZER_AVAILABLE = False

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

if _GPU_OPTIMIZER_AVAILABLE:
    __all__.append('GPUOptimizer')

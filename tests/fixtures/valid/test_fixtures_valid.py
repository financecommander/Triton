"""
Test fixtures for valid input data.

This module provides pre-defined valid test data for various test scenarios.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any


# Basic tensor fixtures
VALID_TENSORS = {
    "small_square": torch.randn(4, 4),
    "medium_square": torch.randn(16, 16),
    "large_square": torch.randn(64, 64),
    "rectangular": torch.randn(32, 16),
    "tall": torch.randn(128, 32),
    "wide": torch.randn(32, 128),
    "vector": torch.randn(256),
    "single_value": torch.randn(1, 1),
}

# Special value tensors
SPECIAL_VALUE_TENSORS = {
    "zeros": torch.zeros(32, 32),
    "ones": torch.ones(32, 32),
    "identity": torch.eye(32),
    "diagonal": torch.diag(torch.randn(32)),
    "upper_triangular": torch.triu(torch.randn(32, 32)),
    "lower_triangular": torch.tril(torch.randn(32, 32)),
    "symmetric": lambda: (torch.randn(32, 32) + torch.randn(32, 32).t()) / 2,
    "positive_definite": lambda: torch.randn(32, 32) @ torch.randn(32, 32).t() + torch.eye(32),
}

# Compatible tensor pairs for matrix multiplication
COMPATIBLE_PAIRS = [
    (torch.randn(8, 12), torch.randn(12, 16)),
    (torch.randn(16, 24), torch.randn(24, 8)),
    (torch.randn(32, 32), torch.randn(32, 32)),
    (torch.randn(64, 16), torch.randn(16, 48)),
    (torch.randn(4, 8), torch.randn(8, 4)),
]

# Batch tensor fixtures
BATCH_TENSORS = {
    "small_batch": torch.randn(2, 16, 16),
    "medium_batch": torch.randn(4, 32, 32),
    "large_batch": torch.randn(8, 64, 64),
    "mixed_batch": torch.randn(3, 24, 48),
}

# Quantized-friendly tensors (values that work well with ternary quantization)
QUANTIZED_FRIENDLY = {
    "ternary_like": torch.tensor([-1, 0, 1] * 341 + [-1], dtype=torch.float32).reshape(32, 32),
    "binary_like": torch.tensor([0, 1] * 512, dtype=torch.float32).reshape(32, 32),
    "small_integers": torch.randint(-10, 11, (32, 32), dtype=torch.float32),
    "powers_of_two": torch.tensor([2**i for i in range(-10, 22)] * 4, dtype=torch.float32).reshape(16, 8),
}

# Memory-efficient tensors for performance testing
MEMORY_EFFICIENT = {
    "sparse_like": torch.randn(64, 64) * (torch.rand(64, 64) > 0.8).float(),
    "low_rank_approx": torch.randn(32, 8) @ torch.randn(8, 32),
    "block_diagonal": torch.block_diag(*[torch.randn(8, 8) for _ in range(4)]),
}

# Cross-platform compatible tensors (work on CPU and GPU)
def get_cross_platform_tensor(shape: Tuple[int, ...], device: str = 'auto') -> torch.Tensor:
    """Get a tensor that works across different platforms."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(shape, device=device)

# Kernel code fixtures
VALID_KERNELS = {
    "simple_matmul": """
    @triton.jit
    def simple_matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * 32 + tl.arange(0, 32)
        offs_n = pid_n * 32 + tl.arange(0, 32)
        offs_k = tl.arange(0, 32)

        a_ptrs = a + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = b + offs_k[:, None] * N + offs_n[None, :]

        accumulator = tl.zeros((32, 32), dtype=tl.float32)

        for k in range(0, K, 32):
            a_block = tl.load(a_ptrs)
            b_block = tl.load(b_ptrs)
            accumulator += tl.dot(a_block, b_block)
            a_ptrs += 32
            b_ptrs += 32 * N

        tl.store(c + offs_m[:, None] * N + offs_n[None, :], accumulator)
    """,

    "elementwise": """
    @triton.jit
    def elementwise_kernel(x, y, output, n: tl.constexpr):
        pid = tl.program_id(0)
        block_size = 1024
        for i in range(block_size):
            if pid * block_size + i < n:
                a = tl.load(x + pid * block_size + i)
                b = tl.load(y + pid * block_size + i)
                tl.store(output + pid * block_size + i, a + b)
    """,

    "reduction": """
    @triton.jit
    def reduction_kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        block_size = 1024

        accumulator = 0.0
        for i in range(block_size):
            if pid * block_size + i < n:
                accumulator += tl.load(x + pid * block_size + i)

        tl.store(output + pid, accumulator)
    """
}

# Performance benchmark configurations
PERFORMANCE_CONFIGS = {
    "micro": {"sizes": [16, 32, 64], "iterations": 10},
    "small": {"sizes": [128, 256], "iterations": 5},
    "medium": {"sizes": [512, 1024], "iterations": 3},
    "large": {"sizes": [2048], "iterations": 1},
}

# Memory usage patterns for testing
MEMORY_PATTERNS = {
    "increasing": lambda: [torch.randn(64 * i, 64 * i) for i in range(1, 5)],
    "decreasing": lambda: [torch.randn(64 * (5-i), 64 * (5-i)) for i in range(1, 5)],
    "random": lambda: [torch.randn(np.random.randint(32, 256), np.random.randint(32, 256)) for _ in range(10)],
}

def get_fixture(name: str, **kwargs) -> Any:
    """Get a test fixture by name with optional parameters."""
    if name in VALID_TENSORS:
        return VALID_TENSORS[name]
    elif name in SPECIAL_VALUE_TENSORS:
        fixture = SPECIAL_VALUE_TENSORS[name]
        return fixture() if callable(fixture) else fixture
    elif name in QUANTIZED_FRIENDLY:
        return QUANTIZED_FRIENDLY[name]
    elif name in MEMORY_EFFICIENT:
        return MEMORY_EFFICIENT[name]
    elif name == "cross_platform":
        return get_cross_platform_tensor(**kwargs)
    else:
        raise ValueError(f"Unknown fixture: {name}")

def get_compatible_pair(size_category: str = "medium") -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a pair of compatible tensors for matrix multiplication."""
    if size_category == "small":
        return torch.randn(8, 12), torch.randn(12, 16)
    elif size_category == "medium":
        return torch.randn(32, 48), torch.randn(48, 64)
    elif size_category == "large":
        return torch.randn(128, 256), torch.randn(256, 512)
    else:
        return COMPATIBLE_PAIRS[np.random.randint(len(COMPATIBLE_PAIRS))]

def get_kernel_fixture(kernel_type: str) -> str:
    """Get a kernel code fixture."""
    if kernel_type in VALID_KERNELS:
        return VALID_KERNELS[kernel_type]
    else:
        return list(VALID_KERNELS.values())[0]  # Return first one as default
"""
Test fixtures for edge case input data.

This module provides pre-defined edge case test data for testing boundary conditions.
"""

import torch
import numpy as np
from typing import Any, List, Dict, Union, Tuple


# Boundary size tensors
BOUNDARY_SIZE_TENSORS = [
    torch.randn(1, 1),        # Minimum 2D size
    torch.randn(1, 32),       # Minimum rows
    torch.randn(32, 1),       # Minimum columns
    torch.randn(1024, 1024),  # Large square matrix
    torch.randn(1, 1024),     # Very wide matrix
    torch.randn(1024, 1),     # Very tall matrix
    torch.randn(16384, 1),    # Maximum practical size for some GPUs
    torch.randn(1, 16384),    # Maximum practical size for some GPUs
]

# Precision boundary tensors
PRECISION_BOUNDARY_TENSORS = [
    torch.tensor([[1.0, 1.0000000000000002]]),  # Machine epsilon boundary
    torch.tensor([[float('inf'), -float('inf')]]),  # Infinity boundaries
    torch.tensor([[1e-323, 1e308]]),  # Subnormal and overflow boundaries
    torch.tensor([[0.0, -0.0]]),      # Positive and negative zero
    torch.tensor([[np.pi, np.e]]),    # Mathematical constants
    torch.tensor([[2**53, 2**53 + 1]]),  # Float64 precision boundary
]

# Memory alignment boundary tensors
ALIGNMENT_BOUNDARY_TENSORS = [
    torch.randn(31, 32),      # Not aligned to 32
    torch.randn(32, 31),      # Not aligned to 32
    torch.randn(33, 32),      # Slightly over alignment
    torch.randn(32, 33),      # Slightly over alignment
    torch.randn(64, 64),      # Aligned to 64
    torch.randn(128, 128),    # Aligned to 128
]

# Sparse tensor patterns
SPARSE_PATTERNS = [
    torch.eye(32),            # Identity matrix
    torch.triu(torch.randn(32, 32)),  # Upper triangular
    torch.tril(torch.randn(32, 32)),  # Lower triangular
    torch.diag(torch.randn(32)),      # Diagonal matrix
    torch.zeros(32, 32).masked_fill_(torch.rand(32, 32) < 0.1, 1.0),  # 10% sparsity
    torch.zeros(32, 32).masked_fill_(torch.rand(32, 32) < 0.5, 1.0),  # 50% sparsity
    torch.zeros(32, 32).masked_fill_(torch.rand(32, 32) < 0.9, 1.0),  # 90% sparsity
]

# Special value combinations
SPECIAL_VALUE_COMBINATIONS = [
    torch.tensor([[0.0, 1.0, -1.0, 0.5, -0.5]]),  # Common values
    torch.tensor([[np.pi, np.e, np.sqrt(2), np.log(2)]]),  # Mathematical constants
    torch.tensor([[2**n for n in range(-10, 11)]]),  # Powers of 2
    torch.tensor([[1.0 / 2**n for n in range(1, 11)]]),  # Inverse powers of 2
]

# Data type boundary tensors
DTYPE_BOUNDARY_TENSORS = {
    torch.float16: [
        torch.tensor([[6.1e-5, 6.5e4]], dtype=torch.float16),  # Min/max normal
        torch.tensor([[0.0, float('inf'), -float('inf')]], dtype=torch.float16),
    ],
    torch.float32: [
        torch.tensor([[1.4e-45, 3.4e38]], dtype=torch.float32),  # Min/max normal
        torch.tensor([[0.0, float('inf'), -float('inf')]], dtype=torch.float32),
    ],
    torch.float64: [
        torch.tensor([[2.2e-308, 1.8e308]], dtype=torch.float64),  # Min/max normal
        torch.tensor([[0.0, float('inf'), -float('inf')]], dtype=torch.float64),
    ],
}

# Block size boundary configurations
BLOCK_SIZE_BOUNDARIES = [
    {"block_size": 1},         # Minimum block size
    {"block_size": 16},        # Small block
    {"block_size": 32},        # Standard block
    {"block_size": 64},        # Medium block
    {"block_size": 128},       # Large block
    {"block_size": 256},       # Very large block
    {"block_size": 512},       # Maximum practical block
    {"block_size": 1024},      # Extreme block size
]

# Warp count boundaries
WARP_COUNT_BOUNDARIES = [
    {"num_warps": 1},          # Minimum warps
    {"num_warps": 2},          # Small warp count
    {"num_warps": 4},          # Standard warp count
    {"num_warps": 8},          # Large warp count
    {"num_warps": 16},         # Very large warp count
    {"num_warps": 32},         # Maximum warps
]

# Pipeline stage boundaries
PIPELINE_STAGE_BOUNDARIES = [
    {"num_stages": 1},         # Minimum stages
    {"num_stages": 2},         # Basic pipelining
    {"num_stages": 3},         # Moderate pipelining
    {"num_stages": 4},         # Deep pipelining
    {"num_stages": 5},         # Very deep pipelining
]

# Memory layout boundaries
MEMORY_LAYOUT_BOUNDARIES = [
    torch.randn(32, 32).contiguous(),      # Contiguous memory
    torch.randn(32, 32).transpose(0, 1),  # Transposed layout
    torch.randn(32, 32).t(),               # Transposed
    torch.randn(32, 32).contiguous().view(-1),  # Flattened
    torch.randn(32, 32).contiguous().view(16, 64),  # Reshaped
]

# Concurrent execution boundaries
CONCURRENT_EXECUTION_BOUNDARIES = [
    {"streams": 1},            # Single stream
    {"streams": 2},            # Dual stream
    {"streams": 4},            # Quad stream
    {"streams": 8},            # Octo stream
    {"streams": 16},           # Many streams
]

# Time-based boundaries
TIME_BASED_BOUNDARIES = [
    {"timeout": 0.001},        # Very short timeout (1ms)
    {"timeout": 0.01},         # Short timeout (10ms)
    {"timeout": 0.1},          # Medium timeout (100ms)
    {"timeout": 1.0},          # Standard timeout (1s)
    {"timeout": 10.0},         # Long timeout (10s)
    {"timeout": 60.0},         # Very long timeout (1min)
]

# Resource allocation boundaries
RESOURCE_ALLOCATION_BOUNDARIES = [
    {"memory_limit": 1024},      # 1KB limit
    {"memory_limit": 1024*1024}, # 1MB limit
    {"memory_limit": 1024**3},   # 1GB limit
    {"memory_limit": 4*1024**3}, # 4GB limit
    {"cpu_limit": 1},            # Single CPU core
    {"cpu_limit": 4},            # Quad core
    {"cpu_limit": 8},            # Octo core
    {"cpu_limit": 16},           # Many cores
]

# Kernel launch boundaries
KERNEL_LAUNCH_BOUNDARIES = [
    {"grid": (1, 1, 1)},         # Single block
    {"grid": (32, 1, 1)},        # Linear grid
    {"grid": (32, 32, 1)},       # 2D grid
    {"grid": (16, 16, 16)},      # 3D grid
    {"grid": (1024, 1, 1)},      # Large linear grid
    {"grid": (64, 64, 1)},       # Large 2D grid
]

# Edge case kernel code
EDGE_CASE_KERNELS = [
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            # Boundary condition: pid == 0
            val = tl.load(x + pid) if pid > 0 else 0.0
            tl.store(output + pid, val)
    """,
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            # Boundary condition: pid == n-1
            val = tl.load(x + pid) if pid < n-1 else 1.0
            tl.store(output + pid, val)
    """,
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            # Edge case: single element
            val = tl.load(x + pid) if n > 1 else 42.0
            tl.store(output + pid, val)
    """,
]

def get_edge_case_fixture(category: str, index: int = 0) -> Any:
    """Get an edge case test fixture by category."""
    if category == "boundary_size":
        return BOUNDARY_SIZE_TENSORS[index % len(BOUNDARY_SIZE_TENSORS)]
    elif category == "precision":
        return PRECISION_BOUNDARY_TENSORS[index % len(PRECISION_BOUNDARY_TENSORS)]
    elif category == "alignment":
        return ALIGNMENT_BOUNDARY_TENSORS[index % len(ALIGNMENT_BOUNDARY_TENSORS)]
    elif category == "sparse":
        return SPARSE_PATTERNS[index % len(SPARSE_PATTERNS)]
    elif category == "special_values":
        return SPECIAL_VALUE_COMBINATIONS[index % len(SPECIAL_VALUE_COMBINATIONS)]
    elif category == "dtype":
        dtype = list(DTYPE_BOUNDARY_TENSORS.keys())[index % len(DTYPE_BOUNDARY_TENSORS)]
        dtype_index = index // len(DTYPE_BOUNDARY_TENSORS)
        return DTYPE_BOUNDARY_TENSORS[dtype][dtype_index % len(DTYPE_BOUNDARY_TENSORS[dtype])]
    elif category == "block_size":
        return BLOCK_SIZE_BOUNDARIES[index % len(BLOCK_SIZE_BOUNDARIES)]
    elif category == "warps":
        return WARP_COUNT_BOUNDARIES[index % len(WARP_COUNT_BOUNDARIES)]
    elif category == "stages":
        return PIPELINE_STAGE_BOUNDARIES[index % len(PIPELINE_STAGE_BOUNDARIES)]
    elif category == "memory_layout":
        return MEMORY_LAYOUT_BOUNDARIES[index % len(MEMORY_LAYOUT_BOUNDARIES)]
    elif category == "concurrent":
        return CONCURRENT_EXECUTION_BOUNDARIES[index % len(CONCURRENT_EXECUTION_BOUNDARIES)]
    elif category == "time":
        return TIME_BASED_BOUNDARIES[index % len(TIME_BASED_BOUNDARIES)]
    elif category == "resources":
        return RESOURCE_ALLOCATION_BOUNDARIES[index % len(RESOURCE_ALLOCATION_BOUNDARIES)]
    elif category == "kernel_launch":
        return KERNEL_LAUNCH_BOUNDARIES[index % len(KERNEL_LAUNCH_BOUNDARIES)]
    elif category == "kernel":
        return EDGE_CASE_KERNELS[index % len(EDGE_CASE_KERNELS)]
    else:
        raise ValueError(f"Unknown edge case fixture category: {category}")

def get_all_edge_case_fixtures(category: str) -> List[Any]:
    """Get all edge case fixtures for a category."""
    if category == "boundary_size":
        return BOUNDARY_SIZE_TENSORS
    elif category == "precision":
        return PRECISION_BOUNDARY_TENSORS
    elif category == "alignment":
        return ALIGNMENT_BOUNDARY_TENSORS
    elif category == "sparse":
        return SPARSE_PATTERNS
    elif category == "special_values":
        return SPECIAL_VALUE_COMBINATIONS
    elif category == "dtype":
        return DTYPE_BOUNDARY_TENSORS
    elif category == "block_size":
        return BLOCK_SIZE_BOUNDARIES
    elif category == "warps":
        return WARP_COUNT_BOUNDARIES
    elif category == "stages":
        return PIPELINE_STAGE_BOUNDARIES
    elif category == "memory_layout":
        return MEMORY_LAYOUT_BOUNDARIES
    elif category == "concurrent":
        return CONCURRENT_EXECUTION_BOUNDARIES
    elif category == "time":
        return TIME_BASED_BOUNDARIES
    elif category == "resources":
        return RESOURCE_ALLOCATION_BOUNDARIES
    elif category == "kernel_launch":
        return KERNEL_LAUNCH_BOUNDARIES
    elif category == "kernel":
        return EDGE_CASE_KERNELS
    else:
        raise ValueError(f"Unknown edge case fixture category: {category}")
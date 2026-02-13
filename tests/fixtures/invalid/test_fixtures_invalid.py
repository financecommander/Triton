"""
Test fixtures for invalid input data.

This module provides pre-defined invalid test data for testing error handling.
"""

import torch
import numpy as np
from typing import Any, List, Dict, Union


# Invalid tensor inputs
INVALID_TENSORS = [
    None,
    "string_input",
    42,
    [1, 2, 3, 4],
    {"key": "value"},
    lambda x: x * 2,
    np.array([1, 2, 3, 4]),
    torch.tensor([1, 2, 3, 4]).numpy(),  # NumPy array
]

# Malformed tensor shapes
INVALID_SHAPES = [
    torch.randn(0, 32),      # Zero dimension
    torch.randn(32, 0),      # Zero dimension
    torch.randn(0, 0),       # Both dimensions zero
    torch.randn(1),          # 1D tensor
    torch.randn(32),         # 1D tensor
    torch.randn(2, 3, 4, 5), # 4D tensor
    torch.randn(1, 1, 32),   # 3D tensor
]

# Incompatible shape pairs for matrix multiplication
INCOMPATIBLE_PAIRS = [
    (torch.randn(32, 32), torch.randn(16, 32)),  # Wrong K dimension
    (torch.randn(32, 16), torch.randn(32, 32)),  # Wrong K dimension
    (torch.randn(16, 32), torch.randn(64, 16)),  # Wrong K dimension
    (torch.randn(32), torch.randn(32, 32)),      # 1D vs 2D
    (torch.randn(32, 32), torch.randn(32)),      # 2D vs 1D
    (torch.randn(32, 32, 32), torch.randn(32)),  # 3D vs 1D
]

# Extreme value tensors
EXTREME_VALUE_TENSORS = [
    torch.full((32, 32), float('inf')),
    torch.full((32, 32), float('-inf')),
    torch.full((32, 32), float('nan')),
    torch.full((32, 32), torch.finfo(torch.float32).max),    # Max float value
    torch.full((32, 32), torch.finfo(torch.float32).min),   # Min float value
    torch.full((32, 32), 1e-323),   # Subnormal
    torch.full((32, 32), 0.0),      # All zeros
]

# Invalid data types
INVALID_DTYPES = [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
]

# Invalid kernel code
INVALID_KERNELS = [
    "",  # Empty string
    "@triton.jit",  # Incomplete decorator
    "@triton.jit\ndef kernel(): pass",  # Missing tl import
    "def kernel(x, output):\n    return x + output",  # Not a Triton kernel
    "@triton.jit\ndef kernel(x, output):\n    tl.store(output, x)",  # Missing program_id
    "@triton.jit\ndef kernel(x, output, n):\n    pass",  # Missing tl.constexpr
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            val = tl.load(x + pid)
            tl.store(output + pid, val
    """,  # Syntax error - missing closing paren
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            val = tl.load(x + pid)
            # Undefined variable
            tl.store(output + pid, undefined_var)
    """,  # Undefined variable
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        # Invalid program_id dimension
        pid = tl.program_id(2)
        if pid < n:
            val = tl.load(x + pid)
            tl.store(output + pid, val)
    """,  # Invalid program_id dimension
]

# Malicious kernel attempts
MALICIOUS_KERNELS = [
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            val = tl.load(x + pid)
            # Attempt code injection
            import os
            os.system("echo 'hacked'")
            tl.store(output + pid, val)
    """,
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            val = tl.load(x + pid)
            # Attempt file access
            with open("/etc/passwd", "r") as f:
                data = f.read()
            tl.store(output + pid, val)
    """,
    """
    @triton.jit
    def kernel(x, output, n: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            val = tl.load(x + pid)
            # Attempt arbitrary code execution
            exec("print('Code injection successful')")
            tl.store(output + pid, val)
    """,
]

# Invalid configuration parameters
INVALID_CONFIGS = [
    {"block_size": 0},        # Zero block size
    {"block_size": -1},       # Negative block size
    {"block_size": 1e6},      # Too large block size
    {"num_warps": 0},         # Zero warps
    {"num_warps": -1},        # Negative warps
    {"num_warps": 100},       # Too many warps
    {"num_stages": -1},       # Negative stages
    {"num_stages": 100},      # Too many stages
]

# Memory exhaustion attempts
MEMORY_EXHAUSTION_INPUTS = [
    # Very large tensors that may cause OOM
    lambda: torch.randn(10000, 10000) if torch.cuda.is_available() else torch.randn(5000, 5000),
    lambda: torch.randn(5000, 5000, 5000) if torch.cuda.is_available() else torch.randn(1000, 1000, 1000),
    # Many small allocations
    lambda: [torch.randn(1000, 1000) for _ in range(100)] if torch.cuda.is_available() else [torch.randn(500, 500) for _ in range(50)],
]

# Invalid file paths and URIs
INVALID_PATHS = [
    "",
    "/nonexistent/path/file.txt",
    "../../../../etc/passwd",
    "C:\\Windows\\System32\\config\\SAM",  # Windows SAM file
    "file:///etc/passwd",
    "http://malicious.com/malware.exe",
    "~/../../../../etc/shadow",
    "/dev/null",
    "/proc/self/mem",
]

# Invalid environment variables
INVALID_ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "999",  # Invalid GPU ID
    "CUDA_VISIBLE_DEVICES": "-1",   # Negative GPU ID
    "CUDA_VISIBLE_DEVICES": "all",  # Invalid value
    "OMP_NUM_THREADS": "-1",        # Negative thread count
    "OMP_NUM_THREADS": "0",         # Zero threads
    "PYTHONPATH": "/nonexistent/path",
    "LD_LIBRARY_PATH": "/malicious/libs",
}

def get_invalid_fixture(category: str, index: int = 0) -> Any:
    """Get an invalid test fixture by category."""
    if category == "tensor":
        return INVALID_TENSORS[index % len(INVALID_TENSORS)]
    elif category == "shape":
        return INVALID_SHAPES[index % len(INVALID_SHAPES)]
    elif category == "pair":
        return INCOMPATIBLE_PAIRS[index % len(INCOMPATIBLE_PAIRS)]
    elif category == "extreme":
        return EXTREME_VALUE_TENSORS[index % len(EXTREME_VALUE_TENSORS)]
    elif category == "dtype":
        return INVALID_DTYPES[index % len(INVALID_DTYPES)]
    elif category == "kernel":
        return INVALID_KERNELS[index % len(INVALID_KERNELS)]
    elif category == "malicious":
        return MALICIOUS_KERNELS[index % len(MALICIOUS_KERNELS)]
    elif category == "config":
        return INVALID_CONFIGS[index % len(INVALID_CONFIGS)]
    elif category == "memory":
        fixture = MEMORY_EXHAUSTION_INPUTS[index % len(MEMORY_EXHAUSTION_INPUTS)]
        return fixture() if callable(fixture) else fixture
    elif category == "path":
        return INVALID_PATHS[index % len(INVALID_PATHS)]
    else:
        raise ValueError(f"Unknown invalid fixture category: {category}")

def get_all_invalid_fixtures(category: str) -> List[Any]:
    """Get all invalid fixtures for a category."""
    if category == "tensor":
        return INVALID_TENSORS
    elif category == "shape":
        return INVALID_SHAPES
    elif category == "pair":
        return INCOMPATIBLE_PAIRS
    elif category == "extreme":
        return EXTREME_VALUE_TENSORS
    elif category == "dtype":
        return INVALID_DTYPES
    elif category == "kernel":
        return INVALID_KERNELS
    elif category == "malicious":
        return MALICIOUS_KERNELS
    elif category == "config":
        return INVALID_CONFIGS
    elif category == "path":
        return INVALID_PATHS
    else:
        raise ValueError(f"Unknown invalid fixture category: {category}")
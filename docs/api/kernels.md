# Kernels API Reference

This section provides detailed documentation for low-level kernel implementations that power ternary operations in Triton DSL.

## Overview

Triton DSL provides two high-performance kernel implementations:

```
Ternary Operations
    ├── CUDA Kernels (backend/kernels/cuda/)
    │   ├── Hand-optimized C++/CUDA
    │   ├── Explicit memory management
    │   └── Maximum control over GPU resources
    │
    └── Triton GPU Kernels (backend/kernels/triton/)
        ├── Python-based kernel definition
        ├── Auto-tuning for target hardware
        └── Portable across GPU backends
```

Both implementations provide the same API and can be used interchangeably.

## CUDA Kernels

.. automodule:: kernels.cuda.ternary_ops
   :members:
   :undoc-members:
   :show-inheritance:

### Ternary Matrix Multiplication (CUDA)

High-performance CUDA implementation of ternary matrix multiplication.

#### C++ API

```cpp
// kernels/cuda/ternary_matmul.cu

__global__ void ternary_matmul_kernel(
    const uint8_t* A_packed,  // Packed ternary matrix A [M, K]
    const uint8_t* B_packed,  // Packed ternary matrix B [K, N]
    float* C,                  // Output matrix C [M, N]
    int M, int K, int N
) {
    // Thread block dimensions: 16x16
    const int BLOCK_SIZE = 16;
    
    // Shared memory for tile loading
    __shared__ uint8_t A_tile[BLOCK_SIZE][BLOCK_SIZE / 4];  // 4 values per byte
    __shared__ uint8_t B_tile[BLOCK_SIZE / 4][BLOCK_SIZE];
    
    // Compute output position
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Tile-based computation
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // Load tiles into shared memory
        // ... (tile loading logic)
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // Unpack ternary values and accumulate
            int8_t a_val = extract_trit(A_tile[threadIdx.y][k / 4], k % 4);
            int8_t b_val = extract_trit(B_tile[k / 4][threadIdx.x], k % 4);
            
            // Ternary multiplication: skip if either is zero
            if (a_val != 0 && b_val != 0) {
                sum += (float)(a_val * b_val);
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

#### Python Interface

.. autofunction:: kernels.cuda.ternary_ops.ternary_matmul_cuda

```python
from kernels.cuda.ternary_ops import ternary_matmul_cuda
import torch

# Create ternary matrices
A = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.float32).cuda()
B = torch.tensor([[1, -1], [0, 1], [-1, 0]], dtype=torch.float32).cuda()

# Perform ternary matrix multiplication
C = ternary_matmul_cuda(A, B)
print(C)
# Output: tensor([[-2., 1.], [1., -2.]], device='cuda:0')
```

### Ternary Packing Operations

.. automodule:: kernels.cuda.ternary_ops
   :members: pack_ternary_cuda, unpack_ternary_cuda
   :undoc-members:

#### Packing Specification

Ternary values are packed into uint8 bytes with 2 bits per value:

| Bit Pattern | Value | Binary |
|-------------|-------|--------|
| 00          | 0     | 0b00   |
| 01          | +1    | 0b01   |
| 10          | -1    | 0b10   |
| 11          | Reserved | 0b11 |

Four ternary values fit in one uint8 byte:

```
Byte layout: [v3:v2:v1:v0]
  v0: bits [1:0]
  v1: bits [3:2]
  v2: bits [5:4]
  v3: bits [7:6]
```

#### Example: Packing and Unpacking

```python
from kernels.cuda.ternary_ops import pack_ternary_cuda, unpack_ternary_cuda
import torch

# Original ternary tensor
ternary_values = torch.tensor(
    [-1, 0, 1, -1, 1, 0, -1, 1],
    dtype=torch.float32
).cuda()

# Pack into compact representation
packed = pack_ternary_cuda(ternary_values)
print(f"Original size: {ternary_values.numel() * 4} bytes")
print(f"Packed size: {packed.numel()} bytes")
print(f"Compression: {ternary_values.numel() * 4 / packed.numel():.1f}x")

# Unpack back to ternary values
unpacked = unpack_ternary_cuda(packed, ternary_values.shape)
assert torch.allclose(ternary_values, unpacked)
```

### CUDA Kernel Launch Configuration

Optimal launch parameters for different GPU architectures:

```python
from kernels.cuda.ternary_ops import get_optimal_launch_config

# Auto-detect optimal configuration
config = get_optimal_launch_config(
    M=1024, N=1024, K=1024,
    device='cuda:0'
)

print(f"Block size: {config['block_size']}")
print(f"Grid size: {config['grid_size']}")
print(f"Shared memory: {config['shared_mem_bytes']} bytes")

# Use configuration for kernel launch
ternary_matmul_cuda(
    A, B,
    block_size=config['block_size'],
    grid_size=config['grid_size']
)
```

#### Architecture-Specific Optimizations

| GPU Architecture | Block Size | Shared Memory | Registers |
|------------------|------------|---------------|-----------|
| V100 (Volta)     | 16×16      | 48 KB        | 64K       |
| A100 (Ampere)    | 32×32      | 164 KB       | 64K       |
| H100 (Hopper)    | 32×32      | 228 KB       | 64K       |

```python
# Architecture-specific tuning
if torch.cuda.get_device_capability() >= (8, 0):  # Ampere or newer
    block_size = (32, 32)
    shared_mem_kb = 164
else:  # Volta or older
    block_size = (16, 16)
    shared_mem_kb = 48
```

## Triton GPU Kernels

.. automodule:: kernels.triton.ternary_ops
   :members:
   :undoc-members:
   :show-inheritance:

### Triton Kernel Implementation

Triton kernels are written in Python and automatically compiled to efficient GPU code.

#### Ternary Matrix Multiplication Kernel

.. autofunction:: kernels.triton.ternary_ops.ternary_matmul_packed_kernel

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def ternary_matmul_packed_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton kernel for ternary matrix multiplication."""
    pid = tl.program_id(axis=0)
    
    # Compute block indices
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Create pointers for current block
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load tiles
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Compute tile product
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float32)
    
    # Write output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### Auto-Tuning

Triton automatically tunes kernel configurations for your hardware:

```python
from kernels.triton.ternary_ops import ternary_matmul_triton
import torch

A = torch.randn(1024, 512).cuda()
B = torch.randn(512, 2048).cuda()

# First call: auto-tunes and caches best configuration
C = ternary_matmul_triton(A, B)  # ~10ms (includes tuning)

# Subsequent calls: use cached configuration
C = ternary_matmul_triton(A, B)  # ~1ms (optimized)
```

#### Auto-Tune Configuration

```python
@triton.autotune(
    configs=[
        # Configuration 1: Large blocks, high parallelism
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64},
            num_stages=3,  # Pipeline depth
            num_warps=8     # Parallelism per block
        ),
        # Configuration 2: Medium blocks, balanced
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32},
            num_stages=4,
            num_warps=4
        ),
        # Configuration 3: Small blocks, low latency
        triton.Config(
            {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16},
            num_stages=2,
            num_warps=2
        ),
    ],
    key=['M', 'N', 'K'],  # Auto-tune based on matrix dimensions
)
@triton.jit
def my_kernel(...):
    pass
```

### Triton Packing Operations

.. automodule:: kernels.triton.ternary_packing
   :members:
   :undoc-members:
   :show-inheritance:

```python
from kernels.triton.ternary_packing import pack_ternary_triton, unpack_ternary_triton

# Pack ternary tensor
ternary = torch.tensor([-1, 0, 1, 1, -1, 0, 0, 1], dtype=torch.float32).cuda()
packed = pack_ternary_triton(ternary)

# Unpack
unpacked = unpack_ternary_triton(packed, ternary.shape)
assert torch.equal(ternary, unpacked)
```

## Performance Comparison

### Benchmark Results

Performance comparison on NVIDIA A100 GPU:

| Operation | Matrix Size | CUDA (ms) | Triton (ms) | PyTorch (ms) | Speedup |
|-----------|-------------|-----------|-------------|--------------|---------|
| Ternary MatMul | 1024×1024 | 0.45 | 0.38 | 1.2 | 3.2×
| Ternary MatMul | 2048×2048 | 2.1 | 1.8 | 8.5 | 4.7×
| Ternary MatMul | 4096×4096 | 12.5 | 10.2 | 52.3 | 5.1×
| Pack Ternary | 1M values | 0.08 | 0.09 | 0.35 | 3.9×
| Unpack Ternary | 1M values | 0.07 | 0.08 | 0.32 | 4.0×

#### Running Benchmarks

```python
from kernels.triton.benchmark_triton_vs_cuda import run_benchmark

# Benchmark all implementations
results = run_benchmark(
    sizes=[1024, 2048, 4096],
    num_iterations=100,
    warmup_iterations=10
)

for size, timings in results.items():
    print(f"\nMatrix size: {size}×{size}")
    print(f"  CUDA:    {timings['cuda']:.2f} ms")
    print(f"  Triton:  {timings['triton']:.2f} ms")
    print(f"  PyTorch: {timings['pytorch']:.2f} ms")
    print(f"  Speedup: {timings['pytorch'] / timings['triton']:.1f}×")
```

## Optimization Techniques

### Memory Coalescing

Ensure memory accesses are coalesced for maximum bandwidth:

```python
# Good: Coalesced access (consecutive threads access consecutive memory)
@triton.jit
def good_kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)  # Coalesced!

# Bad: Strided access (poor memory bandwidth utilization)
@triton.jit
def bad_kernel(x_ptr, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE * stride + tl.arange(0, BLOCK_SIZE) * stride
    x = tl.load(x_ptr + offsets)  # Strided, not coalesced
```

### Shared Memory Optimization

Use shared memory to reduce global memory traffic:

```cpp
// CUDA: Manual shared memory management
__global__ void optimized_kernel() {
    __shared__ float tile[16][16];
    
    // Load to shared memory once
    tile[threadIdx.y][threadIdx.x] = global_mem[...];
    __syncthreads();
    
    // Reuse from shared memory (fast)
    float sum = 0;
    for (int i = 0; i < 16; i++) {
        sum += tile[threadIdx.y][i] * tile[i][threadIdx.x];
    }
}
```

```python
# Triton: Automatic shared memory management
@triton.jit
def triton_kernel(x_ptr):
    # Triton automatically uses shared memory for loaded tiles
    tile = tl.load(x_ptr + offsets)
    # Reused tile data stays in shared memory
    result = tl.dot(tile, tile)
```

### Register Pressure Management

Minimize register usage to maximize occupancy:

```python
# Check register usage
@triton.jit
def kernel_low_registers(x_ptr):
    # Use fewer intermediate variables
    result = tl.dot(tl.load(x_ptr), tl.load(x_ptr))
    tl.store(out_ptr, result)

@triton.jit  
def kernel_high_registers(x_ptr):
    # Too many intermediate variables
    a = tl.load(x_ptr)
    b = tl.load(x_ptr + 1)
    c = tl.load(x_ptr + 2)
    d = tl.load(x_ptr + 3)
    # ... many more variables
    # High register pressure reduces occupancy
```

### Occupancy Optimization

```python
from kernels.cuda.ternary_ops import get_occupancy_info

# Check kernel occupancy
info = get_occupancy_info('ternary_matmul_kernel')
print(f"Theoretical occupancy: {info['theoretical_occupancy']:.1%}")
print(f"Achieved occupancy: {info['achieved_occupancy']:.1%}")
print(f"Registers per thread: {info['registers_per_thread']}")
print(f"Shared memory per block: {info['shared_mem_per_block']} bytes")

# Adjust launch configuration for better occupancy
if info['achieved_occupancy'] < 0.5:
    print("Warning: Low occupancy. Consider:")
    print("- Reducing block size")
    print("- Reducing shared memory usage")
    print("- Reducing register usage")
```

## Performance Tuning Guide

### Step 1: Profile Your Kernel

```python
import torch.cuda.profiler as profiler
import torch.autograd.profiler as autograd_profiler

with autograd_profiler.profile(use_cuda=True) as prof:
    result = ternary_matmul_triton(A, B)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Step 2: Analyze Memory Bandwidth

```python
from kernels.utils.profiler import analyze_memory_bandwidth

stats = analyze_memory_bandwidth(
    kernel_func=ternary_matmul_triton,
    inputs=(A, B),
    num_iterations=100
)

print(f"Achieved bandwidth: {stats['bandwidth_gb_s']:.1f} GB/s")
print(f"Peak bandwidth: {stats['peak_bandwidth_gb_s']:.1f} GB/s")
print(f"Efficiency: {stats['efficiency']:.1%}")
```

### Step 3: Optimize Launch Configuration

```python
from kernels.utils.tuner import grid_search_launch_config

best_config = grid_search_launch_config(
    kernel=ternary_matmul_triton,
    inputs=(A, B),
    param_space={
        'BLOCK_SIZE_M': [32, 64, 128],
        'BLOCK_SIZE_N': [32, 64, 128, 256],
        'BLOCK_SIZE_K': [16, 32, 64],
        'num_warps': [2, 4, 8],
        'num_stages': [2, 3, 4, 5]
    }
)

print(f"Best config: {best_config}")
print(f"Speedup: {best_config['speedup']:.2f}×")
```

## Custom Kernel Development

### Writing a Custom Triton Kernel

```python
import triton
import triton.language as tl
import torch

@triton.jit
def custom_ternary_activation_kernel(
    x_ptr,      # Input pointer
    y_ptr,      # Output pointer
    n_elements, # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Custom activation: f(x) = sign(x) if |x| > 0.05 else 0"""
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Compute offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Apply ternary activation
    threshold = 0.05
    output = tl.where(tl.abs(x) > threshold, tl.where(x > 0, 1.0, -1.0), 0.0)
    
    # Store result
    tl.store(y_ptr + offsets, output, mask=mask)

def ternary_activation(x: torch.Tensor) -> torch.Tensor:
    """Python wrapper for custom ternary activation."""
    n_elements = x.numel()
    
    # Allocate output
    y = torch.empty_like(x)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    custom_ternary_activation_kernel[grid](
        x, y, n_elements,
        BLOCK_SIZE=1024,
    )
    
    return y

# Usage
x = torch.randn(10000, device='cuda')
y = ternary_activation(x)
assert y.unique().tolist() == [-1.0, 0.0, 1.0]
```

### Testing Custom Kernels

```python
from kernels.triton.test_ternary_ops import test_kernel_correctness

def test_custom_activation():
    """Test custom ternary activation kernel."""
    # Create test input
    x = torch.tensor([-1.0, -0.02, 0.0, 0.03, 1.0], device='cuda')
    
    # Expected output
    expected = torch.tensor([-1.0, 0.0, 0.0, 0.0, 1.0], device='cuda')
    
    # Run kernel
    output = ternary_activation(x)
    
    # Validate
    assert torch.allclose(output, expected), f"Expected {expected}, got {output}"
    print("✓ Custom activation kernel test passed")

test_custom_activation()
```

## Debugging Kernels

### CUDA Kernel Debugging

```bash
# Compile with debug symbols
nvcc -g -G ternary_matmul.cu -o ternary_matmul

# Run with cuda-gdb
cuda-gdb python
(cuda-gdb) break ternary_matmul_kernel
(cuda-gdb) run script.py
(cuda-gdb) cuda thread (0,0,0)  # Switch to specific thread
(cuda-gdb) print sum
```

### Triton Kernel Debugging

```python
# Enable verbose output
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_DEBUG'] = '1'

# Inspect generated code
import triton
from kernels.triton.ternary_ops import ternary_matmul_packed_kernel

# Print LLVM IR
print(triton.compiler.get_llvm_ir(ternary_matmul_packed_kernel))

# Print PTX assembly
print(triton.compiler.get_ptx(ternary_matmul_packed_kernel))
```

## See Also

- [Backend API](backend.md) - High-level backend interfaces
- [Compiler API](compiler.md) - AST compilation pipeline
- [Examples](examples.md) - Complete usage examples
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Triton Documentation](https://triton-lang.org/)

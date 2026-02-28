# CUDA Ternary Operations

This directory contains optimized CUDA kernels for ternary neural network operations.

## Overview

The ternary matrix multiplication kernel implements highly optimized computation for matrices containing only values in {-1, 0, 1}. This enables significant memory savings and computational efficiency for ternary neural networks.

## Features

### 2-bit Packing
- Values encoded as: -1 → 00, 0 → 01, 1 → 10
- 4 trits packed into each byte (4x memory compression)
- Efficient pack/unpack device functions

### Optimizations
1. **16x16 Thread Block Configuration**: Optimal for modern GPU architectures
2. **Shared Memory Tiling**: Reduces global memory accesses
3. **Zero-Skipping**: Avoids unnecessary multiplications when operands are zero
4. **Warp-Level Reduction**: Efficient accumulation using shuffle operations

## Files

- `ternary_matmul.cu`: CUDA kernel implementation
- `ternary_ops.py`: PyTorch C++ extension wrapper

## Usage

### Python Interface

```python
from kernels.cuda.ternary_ops import ternary_matmul

# Create ternary matrices
A = torch.randint(-1, 2, (128, 128), dtype=torch.int8).cuda()
B = torch.randint(-1, 2, (128, 128), dtype=torch.int8).cuda()

# Perform matrix multiplication
C = ternary_matmul(A, B)
```

### Advanced Usage

```python
from kernels.cuda.ternary_ops import get_ternary_matmul

# Get the operation handler
matmul_op = get_ternary_matmul()

# Pack matrices for storage efficiency
A_packed = matmul_op.pack_ternary(A)
B_packed = matmul_op.pack_ternary(B)

# Perform packed multiplication
C = matmul_op.matmul(A_packed, B_packed, M, N, K)

# Unpack if needed
A_unpacked = matmul_op.unpack_ternary(A_packed, M * K)
```

## Kernel Signature

```c
__global__ void ternary_matmul_kernel(
    const int8_t* __restrict__ A,  // Packed matrix A (M x K)
    const int8_t* __restrict__ B,  // Packed matrix B (K x N)
    int16_t* __restrict__ C,       // Output matrix C (M x N)
    int M,                         // Rows in A and C
    int N,                         // Columns in B and C
    int K                          // Columns in A, rows in B
)
```

## Device Functions

### extract_trit
```c
__device__ int8_t extract_trit(uint8_t packed, int index)
```
Extracts a single trit from a packed byte at the specified index (0-3).

### pack_4trits
```c
__device__ uint8_t pack_4trits(int8_t t0, int8_t t1, int8_t t2, int8_t t3)
```
Packs 4 trit values into a single byte.

## Performance

Typical performance characteristics:
- **Memory Compression**: 4x vs unpacked int8 storage
- **Speed**: 2-3x faster than naive PyTorch matmul for sparse ternary matrices
- **Accuracy**: Exact integer arithmetic (no floating-point errors)

## Benchmarks

Run the benchmark suite:
```bash
python tests/benchmarks/bench_matmul.py
```

## Testing

Run the test suite:
```bash
python tests/test_ternary_matmul.py
```

## Requirements

- CUDA Toolkit 11.0+
- PyTorch 2.0+
- GPU with compute capability 7.0+

## Implementation Details

### Memory Layout

**Matrix A (M x K)**:
- Stored row-major
- Each row padded to multiple of 4 elements
- Packed size: `M * ((K + 3) / 4)` bytes

**Matrix B (K x N)**:
- Stored row-major (transposed for coalescing)
- Each row padded to multiple of 4 elements
- Packed size: `((K + 3) / 4) * N` bytes

### Thread Mapping

- Each thread computes one element of the output matrix C
- Thread block size: 16x16 = 256 threads
- Grid dimensions: `ceil(N/16) x ceil(M/16)`

### Shared Memory Usage

- Two 16x16 tiles for matrices A and B
- Total shared memory per block: 2 * 16 * 16 = 512 bytes
- Unpacked on-the-fly during loading

## Future Improvements

- [ ] Batched matrix multiplication (batch dimension for transformer attention)
- [ ] Dynamic tile size selection — auto-select 8x8, 16x16, 32x32 based on matrix geometry
- [ ] Non-square matrix padding with minimal waste
- [ ] CPU fallback with SIMD (AVX2/NEON) ternary matmul
- [ ] Mixed precision output (int8, fp16, fp32) selectable at call site
- [ ] Fused ternary matmul + activation kernel (eliminate intermediate memory writes)
- [ ] Sparse block detection — skip entire zero tiles during matmul
- [ ] Kernel fusion for ternary conv2d (im2col + packed matmul in single launch)

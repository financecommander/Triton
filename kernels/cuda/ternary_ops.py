"""
PyTorch C++ Extension Wrapper for CUDA Ternary Operations

This module provides a Python interface to the optimized CUDA kernels
for ternary matrix multiplication with 2-bit packed storage.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel source code
CUDA_SOURCE = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_SIZE 16
#define WARP_SIZE 32

__device__ int8_t extract_trit(uint8_t packed, int index) {
    int shift = index * 2;
    uint8_t bits = (packed >> shift) & 0x03;
    if (bits == 0x00) return -1;
    if (bits == 0x01) return 0;
    if (bits == 0x02) return 1;
    return 0;
}

__device__ uint8_t pack_4trits(int8_t t0, int8_t t1, int8_t t2, int8_t t3) {
    auto encode = [](int8_t t) -> uint8_t {
        if (t == -1) return 0x00;
        if (t == 0) return 0x01;
        if (t == 1) return 0x02;
        return 0x01;
    };
    
    uint8_t packed = 0;
    packed |= (encode(t0) << 0);
    packed |= (encode(t1) << 2);
    packed |= (encode(t2) << 4);
    packed |= (encode(t3) << 6);
    return packed;
}

__device__ int16_t warp_reduce_sum(int16_t val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void ternary_matmul_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int16_t* __restrict__ C,
    int M, int N, int K
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __shared__ int8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ int8_t Bs[TILE_SIZE][TILE_SIZE];
    
    int16_t sum = 0;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;
        
        if (row < M && aCol < K) {
            int packedIdx = aCol / 4;
            int tritIdx = aCol % 4;
            int8_t packedByte = A[row * ((K + 3) / 4) + packedIdx];
            As[ty][tx] = extract_trit(packedByte, tritIdx);
        } else {
            As[ty][tx] = 0;
        }
        
        if (bRow < K && col < N) {
            int packedIdx = bRow / 4;
            int tritIdx = bRow % 4;
            int8_t packedByte = B[packedIdx * N + col];
            Bs[ty][tx] = extract_trit(packedByte, tritIdx);
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            int8_t a_val = As[ty][k];
            int8_t b_val = Bs[k][tx];
            if (a_val != 0 && b_val != 0) {
                sum += a_val * b_val;
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor ternary_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K
) {
    auto C = torch::zeros({M, N}, torch::dtype(torch::kInt16).device(A.device()));
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );
    
    ternary_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<int8_t>(),
        B.data_ptr<int8_t>(),
        C.data_ptr<int16_t>(),
        M, N, K
    );
    
    return C;
}
"""

# C++ binding code
CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor ternary_matmul_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K
);

torch::Tensor ternary_matmul(
    torch::Tensor A,
    torch::Tensor B,
    int M, int N, int K
) {
    if (A.is_cuda()) {
        return ternary_matmul_cuda(A, B, M, N, K);
    }
    // CPU fallback could be implemented here
    throw std::runtime_error("CPU version not implemented");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_matmul", &ternary_matmul, "Ternary Matrix Multiplication");
}
"""


class TernaryMatMul:
    """
    Wrapper class for ternary matrix multiplication operations.
    
    This class provides a high-level interface to the CUDA-accelerated
    ternary matrix multiplication kernel with 2-bit packed storage.
    """
    
    def __init__(self):
        self._module = None
        self._load_extension()
    
    def _load_extension(self):
        """Load the CUDA extension dynamically."""
        try:
            self._module = load_inline(
                name='ternary_matmul_ext',
                cpp_sources=CPP_SOURCE,
                cuda_sources=CUDA_SOURCE,
                functions=['ternary_matmul'],
                verbose=False,
                with_cuda=True,
            )
        except Exception as e:
            print(f"Warning: Failed to load CUDA extension: {e}")
            print("Ternary operations will not be available.")
            self._module = None
    
    def pack_ternary(self, tensor):
        """
        Pack a ternary tensor (-1, 0, 1) into 2-bit representation.
        
        Args:
            tensor: PyTorch tensor with values in {-1, 0, 1}
        
        Returns:
            Packed tensor with dtype int8
        """
        if not torch.all((tensor >= -1) & (tensor <= 1)):
            raise ValueError("Input tensor must contain only {-1, 0, 1}")
        
        # Flatten and pad to multiple of 4
        flat = tensor.flatten().to(torch.int8)
        pad_size = (4 - flat.size(0) % 4) % 4
        if pad_size > 0:
            flat = torch.cat([flat, torch.zeros(pad_size, dtype=torch.int8, device=flat.device)])
        
        # Pack 4 trits into 1 byte
        packed = torch.zeros(flat.size(0) // 4, dtype=torch.uint8, device=flat.device)
        
        for i in range(0, flat.size(0), 4):
            t0, t1, t2, t3 = flat[i:i+4]
            # Encode: -1→00, 0→01, 1→10
            encode = lambda t: 0 if t == -1 else (1 if t == 0 else 2)
            byte_val = (encode(t0.item()) | 
                       (encode(t1.item()) << 2) | 
                       (encode(t2.item()) << 4) | 
                       (encode(t3.item()) << 6))
            packed[i // 4] = byte_val
        
        return packed.to(torch.int8)
    
    def unpack_ternary(self, packed, original_size):
        """
        Unpack a 2-bit packed tensor back to ternary values.
        
        Args:
            packed: Packed tensor with dtype int8
            original_size: Original tensor size (number of elements)
        
        Returns:
            Unpacked tensor with values in {-1, 0, 1}
        """
        packed_uint = packed.to(torch.uint8)
        unpacked = []
        
        for byte_val in packed_uint:
            byte_val = byte_val.item()
            for i in range(4):
                bits = (byte_val >> (i * 2)) & 0x03
                if bits == 0:
                    unpacked.append(-1)
                elif bits == 1:
                    unpacked.append(0)
                elif bits == 2:
                    unpacked.append(1)
                else:
                    unpacked.append(0)
        
        result = torch.tensor(unpacked[:original_size], dtype=torch.int8, device=packed.device)
        return result
    
    def matmul(self, A_packed, B_packed, M, N, K):
        """
        Perform ternary matrix multiplication on packed tensors.
        
        Args:
            A_packed: Packed matrix A (M x K packed)
            B_packed: Packed matrix B (K x N packed)
            M: Number of rows in A
            N: Number of columns in B
            K: Number of columns in A / rows in B
        
        Returns:
            Result matrix C (M x N) with dtype int16
        """
        if self._module is None:
            raise RuntimeError("CUDA extension not loaded")
        
        if not A_packed.is_cuda or not B_packed.is_cuda:
            raise ValueError("Input tensors must be on CUDA device")
        
        return self._module.ternary_matmul(A_packed, B_packed, M, N, K)
    
    def matmul_unpacked(self, A, B):
        """
        Convenience method for matrix multiplication with unpacked inputs.
        
        Args:
            A: Ternary matrix A (M x K) with values in {-1, 0, 1}
            B: Ternary matrix B (K x N) with values in {-1, 0, 1}
        
        Returns:
            Result matrix C (M x N) with dtype int16
        """
        M, K = A.shape
        K2, N = B.shape
        
        if K != K2:
            raise ValueError(f"Incompatible dimensions: {A.shape} @ {B.shape}")
        
        # Pack inputs
        A_packed = self.pack_ternary(A)
        B_packed = self.pack_ternary(B)
        
        # Perform multiplication
        C = self.matmul(A_packed, B_packed, M, N, K)
        
        return C


# Global instance for easy access
_ternary_matmul = None


def get_ternary_matmul():
    """Get or create the global TernaryMatMul instance."""
    global _ternary_matmul
    if _ternary_matmul is None:
        _ternary_matmul = TernaryMatMul()
    return _ternary_matmul


def ternary_matmul(A, B):
    """
    Perform ternary matrix multiplication.
    
    Args:
        A: Ternary matrix A (M x K) with values in {-1, 0, 1}
        B: Ternary matrix B (K x N) with values in {-1, 0, 1}
    
    Returns:
        Result matrix C (M x N)
    """
    return get_ternary_matmul().matmul_unpacked(A, B)

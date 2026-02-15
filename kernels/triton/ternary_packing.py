"""
Triton utilities for ternary operations with 2-bit packed storage.

This module provides functions for packing and unpacking ternary values
(-1, 0, 1) into 2-bit representation, where each byte stores 4 trits.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def pack_4trits_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    Triton kernel to pack 4 ternary values into a single byte.

    Each trit is stored in 2 bits:
    -1 -> 00
     0 -> 01
     1 -> 10
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load 4 trits at a time
    trit_offsets = offsets * 4
    trit0 = tl.load(input_ptr + trit_offsets, mask=trit_offsets < n_elements, other=0)
    trit1 = tl.load(input_ptr + trit_offsets + 1, mask=trit_offsets + 1 < n_elements, other=0)
    trit2 = tl.load(input_ptr + trit_offsets + 2, mask=trit_offsets + 2 < n_elements, other=0)
    trit3 = tl.load(input_ptr + trit_offsets + 3, mask=trit_offsets + 3 < n_elements, other=0)

    # Convert trits to 2-bit values
    # -1 -> 0, 0 -> 1, 1 -> 2
    bit0 = (trit0 + 1).to(tl.uint8)
    bit1 = (trit1 + 1).to(tl.uint8)
    bit2 = (trit2 + 1).to(tl.uint8)
    bit3 = (trit3 + 1).to(tl.uint8)

    # Pack into byte: bit0(2bits) | bit1(2bits) | bit2(2bits) | bit3(2bits)
    packed = (bit0) | (bit1 << 2) | (bit2 << 4) | (bit3 << 6)

    # Store packed byte
    tl.store(output_ptr + offsets, packed, mask=offsets < tl.cdiv(n_elements, 4))


@triton.jit
def unpack_4trits_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    Triton kernel to unpack a byte containing 4 trits into 4 ternary values.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load packed byte
    packed = tl.load(input_ptr + offsets, mask=offsets < tl.cdiv(n_elements + 3, 4), other=0)

    # Extract individual 2-bit values
    bit0 = (packed & 0x03).to(tl.int8)
    bit1 = ((packed >> 2) & 0x03).to(tl.int8)
    bit2 = ((packed >> 4) & 0x03).to(tl.int8)
    bit3 = ((packed >> 6) & 0x03).to(tl.int8)

    # Convert back to trits: 0 -> -1, 1 -> 0, 2 -> 1
    trit0 = bit0 - 1
    trit1 = bit1 - 1
    trit2 = bit2 - 1
    trit3 = bit3 - 1

    # Store unpacked trits
    trit_offsets = offsets * 4
    tl.store(output_ptr + trit_offsets, trit0, mask=trit_offsets < n_elements)
    tl.store(output_ptr + trit_offsets + 1, trit1, mask=trit_offsets + 1 < n_elements)
    tl.store(output_ptr + trit_offsets + 2, trit2, mask=trit_offsets + 2 < n_elements)
    tl.store(output_ptr + trit_offsets + 3, trit3, mask=trit_offsets + 3 < n_elements)


def pack_ternary_triton(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack a ternary tensor into 2-bit representation using Triton.

    Args:
        tensor: PyTorch tensor with values in {-1, 0, 1}

    Returns:
        Packed tensor with dtype uint8, 4x smaller than input
    """
    if not torch.all((tensor >= -1) & (tensor <= 1)):
        raise ValueError("Input tensor must contain only {-1, 0, 1}")

    # Flatten the tensor for processing
    flat_tensor = tensor.flatten().to(torch.int8)
    n_elements = flat_tensor.numel()

    # If CUDA is not available, use CPU implementation
    if not torch.cuda.is_available():
        return pack_ternary_cpu(flat_tensor)

    # Allocate output buffer (4 trits per byte)
    packed_size = (n_elements + 3) // 4
    packed = torch.empty(packed_size, dtype=torch.uint8, device=tensor.device)

    # Launch kernel
    grid = lambda META: (triton.cdiv(packed_size, META['BLOCK_SIZE']),)
    pack_4trits_kernel[grid](
        flat_tensor, packed,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return packed


def pack_ternary_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    CPU implementation of ternary packing using vectorized operations.

    Args:
        tensor: Flat tensor with values in {-1, 0, 1}

    Returns:
        Packed tensor with dtype uint8
    """
    n_elements = tensor.numel()
    pad_size = (4 - n_elements % 4) % 4
    if pad_size > 0:
        tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=torch.int8, device=tensor.device)])

    # Map: -1 -> 0, 0 -> 1, 1 -> 2 (vectorized)
    encoded = (tensor + 1).to(torch.uint8)

    # Pack 4 trits per byte
    encoded = encoded.view(-1, 4)
    packed = (
        encoded[:, 0]
        | (encoded[:, 1] << 2)
        | (encoded[:, 2] << 4)
        | (encoded[:, 3] << 6)
    )

    return packed.to(torch.uint8)


def unpack_ternary_triton(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpack a 2-bit packed tensor back to ternary values using Triton.

    Args:
        packed: Packed tensor with dtype uint8
        original_shape: Original tensor shape

    Returns:
        Unpacked tensor with values in {-1, 0, 1}
    """
    n_elements = int(torch.prod(torch.tensor(original_shape)))

    # If CUDA is not available, use CPU implementation
    if not torch.cuda.is_available():
        return unpack_ternary_cpu(packed, n_elements).view(original_shape)

    # Allocate output buffer
    unpacked = torch.empty(n_elements, dtype=torch.int8, device=packed.device)

    # Launch kernel
    packed_size = packed.numel()
    grid = lambda META: (triton.cdiv(packed_size, META['BLOCK_SIZE']),)
    unpack_4trits_kernel[grid](
        packed, unpacked,
        n_elements,
        BLOCK_SIZE=1024,
    )

    return unpacked.view(original_shape)


def unpack_ternary_cpu(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """
    CPU implementation of ternary unpacking using vectorized operations.

    Args:
        packed: Packed tensor with dtype uint8
        n_elements: Number of elements to unpack

    Returns:
        Unpacked tensor with values in {-1, 0, 1}
    """
    packed_uint = packed.to(torch.uint8)

    # Extract each 2-bit value (vectorized)
    t0 = (packed_uint & 0x03).to(torch.int8)
    t1 = ((packed_uint >> 2) & 0x03).to(torch.int8)
    t2 = ((packed_uint >> 4) & 0x03).to(torch.int8)
    t3 = ((packed_uint >> 6) & 0x03).to(torch.int8)

    # Interleave and decode: 0 -> -1, 1 -> 0, 2 -> 1
    unpacked = torch.stack([t0, t1, t2, t3], dim=1).flatten()
    unpacked = unpacked[:n_elements] - 1

    return unpacked


@triton.jit
def extract_trit_kernel(
    packed_ptr, trit_ptr,
    n_elements: tl.constexpr,
    trit_index: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    """
    Extract a specific trit from packed representation for matrix multiplication.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load packed byte
    packed = tl.load(packed_ptr + offsets, mask=offsets < tl.cdiv(n_elements + 3, 4), other=0)

    # Extract the specific 2-bit trit
    shift = trit_index * 2
    trit_bits = ((packed >> shift) & 0x03).to(tl.int8)

    # Convert to ternary: 0 -> -1, 1 -> 0, 2 -> 1
    trit = trit_bits - 1

    # Store trit
    tl.store(trit_ptr + offsets * 4 + trit_index, trit, mask=offsets * 4 + trit_index < n_elements)


def extract_trits_from_packed(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """
    Extract all trits from packed representation into a flat tensor.

    Args:
        packed: Packed tensor with dtype uint8
        n_elements: Number of ternary elements

    Returns:
        Unpacked tensor with values in {-1, 0, 1}
    """
    unpacked = torch.empty(n_elements, dtype=torch.int8, device=packed.device)

    # Extract each trit position
    for trit_idx in range(4):
        grid = lambda META: (triton.cdiv(packed.numel(), META['BLOCK_SIZE']),)
        extract_trit_kernel[grid](
            packed, unpacked,
            n_elements,
            trit_idx,
            BLOCK_SIZE=1024,
        )

    return unpacked
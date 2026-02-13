"""
Ternary packing/unpacking operations for efficient storage.
Packs 4 ternary values (-1, 0, 1) into 1 byte using 2-bit encoding.
"""

import torch


def pack_ternary(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary values into 2-bit representation.
    
    Encoding:
    -1 -> 00 (0)
     0 -> 01 (1)
     1 -> 10 (2)
    
    Packs 4 ternary values into 1 byte.
    
    Args:
        tensor: Input tensor with values in {-1, 0, 1}
    
    Returns:
        Packed tensor with dtype uint8, shape (original_numel + 3) // 4
    """
    # Flatten tensor
    flat = tensor.flatten()
    
    # Map ternary values to 2-bit codes: -1->0, 0->1, 1->2
    encoded = torch.where(flat == -1, torch.tensor(0, dtype=torch.uint8),
                         torch.where(flat == 0, torch.tensor(1, dtype=torch.uint8),
                                   torch.tensor(2, dtype=torch.uint8)))
    
    # Pad to multiple of 4
    pad_size = (4 - len(encoded) % 4) % 4
    if pad_size > 0:
        encoded = torch.cat([encoded, torch.zeros(pad_size, dtype=torch.uint8)])
    
    # Pack 4 values into 1 byte
    encoded = encoded.reshape(-1, 4)
    packed = (encoded[:, 0] << 6) | (encoded[:, 1] << 4) | (encoded[:, 2] << 2) | encoded[:, 3]
    
    return packed.to(torch.uint8)


def unpack_ternary(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """
    Unpack 2-bit encoded ternary values.
    
    Args:
        packed: Packed tensor with dtype uint8
        numel: Number of original elements
    
    Returns:
        Unpacked tensor with values in {-1, 0, 1}
    """
    # Unpack 4 values from each byte
    unpacked = torch.stack([
        (packed >> 6) & 0x3,
        (packed >> 4) & 0x3,
        (packed >> 2) & 0x3,
        packed & 0x3
    ], dim=1).flatten()
    
    # Take only the required number of elements
    unpacked = unpacked[:numel]
    
    # Decode: 0->-1, 1->0, 2->1
    decoded = torch.where(unpacked == 0, torch.tensor(-1, dtype=torch.int8),
                         torch.where(unpacked == 1, torch.tensor(0, dtype=torch.int8),
                                   torch.tensor(1, dtype=torch.int8)))
    
    return decoded

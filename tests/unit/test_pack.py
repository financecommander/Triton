"""
Unit tests for ternary packing/unpacking operations.
"""

import pytest
import torch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.pytorch.ops.pack import pack_ternary, unpack_ternary


class TestTernaryPacking:
    """Test ternary value packing and unpacking."""
    
    def test_pack_single_value(self):
        """Test packing a single ternary value."""
        tensor = torch.tensor([-1], dtype=torch.int8)
        packed = pack_ternary(tensor)
        
        assert packed.dtype == torch.uint8
        assert len(packed) == 1  # Should be padded to 4 values = 1 byte
    
    def test_pack_four_values(self):
        """Test packing exactly 4 ternary values (1 byte)."""
        tensor = torch.tensor([-1, 0, 1, -1], dtype=torch.int8)
        packed = pack_ternary(tensor)
        
        assert packed.dtype == torch.uint8
        assert len(packed) == 1
    
    def test_pack_multiple_bytes(self):
        """Test packing values that span multiple bytes."""
        tensor = torch.tensor([-1, 0, 1, -1, 1, 0, -1, 1], dtype=torch.int8)
        packed = pack_ternary(tensor)
        
        assert packed.dtype == torch.uint8
        assert len(packed) == 2  # 8 values = 2 bytes
    
    def test_unpack_single_byte(self):
        """Test unpacking a single byte."""
        original = torch.tensor([-1, 0, 1, -1], dtype=torch.int8)
        packed = pack_ternary(original)
        unpacked = unpack_ternary(packed, 4)
        
        assert torch.equal(unpacked, original)
    
    def test_unpack_multiple_bytes(self):
        """Test unpacking multiple bytes."""
        original = torch.tensor([-1, 0, 1, -1, 1, 0, -1, 1], dtype=torch.int8)
        packed = pack_ternary(original)
        unpacked = unpack_ternary(packed, 8)
        
        assert torch.equal(unpacked, original)
    
    def test_pack_unpack_roundtrip(self):
        """Test that pack->unpack is lossless."""
        original = torch.tensor([1, -1, 0, 1, 0, -1, 1, -1, 0, 0, 1, 1], dtype=torch.int8)
        packed = pack_ternary(original)
        unpacked = unpack_ternary(packed, len(original))
        
        assert torch.equal(unpacked, original)
    
    def test_pack_unpack_with_padding(self):
        """Test roundtrip with values that need padding."""
        original = torch.tensor([1, -1, 0, 1, 1], dtype=torch.int8)  # 5 values, needs padding
        packed = pack_ternary(original)
        unpacked = unpack_ternary(packed, len(original))
        
        assert torch.equal(unpacked, original)
    
    def test_pack_all_minus_ones(self):
        """Test packing all -1 values."""
        tensor = torch.tensor([-1, -1, -1, -1], dtype=torch.int8)
        packed = pack_ternary(tensor)
        unpacked = unpack_ternary(packed, 4)
        
        assert torch.equal(unpacked, tensor)
    
    def test_pack_all_zeros(self):
        """Test packing all 0 values."""
        tensor = torch.tensor([0, 0, 0, 0], dtype=torch.int8)
        packed = pack_ternary(tensor)
        unpacked = unpack_ternary(packed, 4)
        
        assert torch.equal(unpacked, tensor)
    
    def test_pack_all_ones(self):
        """Test packing all 1 values."""
        tensor = torch.tensor([1, 1, 1, 1], dtype=torch.int8)
        packed = pack_ternary(tensor)
        unpacked = unpack_ternary(packed, 4)
        
        assert torch.equal(unpacked, tensor)
    
    def test_pack_2d_tensor(self):
        """Test packing a 2D tensor (should flatten first)."""
        tensor = torch.tensor([[-1, 0], [1, -1]], dtype=torch.int8)
        packed = pack_ternary(tensor)
        unpacked = unpack_ternary(packed, 4)
        
        assert torch.equal(unpacked, tensor.flatten())
    
    def test_pack_large_tensor(self):
        """Test packing a larger tensor."""
        tensor = torch.randint(-1, 2, (100,), dtype=torch.int8)
        packed = pack_ternary(tensor)
        unpacked = unpack_ternary(packed, 100)
        
        assert torch.equal(unpacked, tensor)
    
    def test_encoding_values(self):
        """Test that encoding follows -1->00, 0->01, 1->10."""
        tensor = torch.tensor([-1, 0, 1, 0], dtype=torch.int8)
        packed = pack_ternary(tensor)
        
        # Expected: 00 01 10 01 = 0x19 = 25
        expected_byte = (0 << 6) | (1 << 4) | (2 << 2) | 1
        assert packed[0].item() == expected_byte


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

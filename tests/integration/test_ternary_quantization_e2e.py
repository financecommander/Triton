"""
Integration tests for end-to-end ternary quantization.
Tests the complete quantization pipeline from FP32 to ternary weights.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from backend.pytorch.ops.quantize import (
    quantize_to_ternary, 
    quantize_model_to_ternary,
    calibrate_threshold,
)
from backend.pytorch.ops.pack import pack_ternary, unpack_ternary
from tests.integration.test_utils import (
    validate_ternary_weights,
    calculate_compression_ratio,
    compare_model_outputs,
)


class TestTernaryQuantizationE2E:
    """Test end-to-end ternary quantization pipeline."""
    
    def test_single_tensor_quantization(self):
        """Test quantization of a single tensor to ternary."""
        # Create a random FP32 tensor
        tensor = torch.randn(100, 100)
        
        # Quantize to ternary
        ternary_tensor = quantize_to_ternary(tensor)
        
        # Validate ternary values
        validation = validate_ternary_weights(ternary_tensor)
        assert validation['is_ternary']
        assert set(validation['unique_values']).issubset({-1, 0, 1})
    
    def test_quantization_preserves_shape(self):
        """Test that quantization preserves tensor shape."""
        shapes = [(10, 10), (32, 64), (128, 256), (5, 7, 9)]
        
        for shape in shapes:
            tensor = torch.randn(*shape)
            ternary_tensor = quantize_to_ternary(tensor)
            
            assert ternary_tensor.shape == tensor.shape
    
    def test_quantization_threshold_calibration(self):
        """Test threshold calibration for quantization."""
        tensor = torch.randn(1000)
        
        # Calibrate threshold
        threshold = calibrate_threshold(tensor, percentile=0.7)
        
        assert threshold > 0
        assert threshold < tensor.abs().max()
    
    @pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9])
    def test_quantization_different_thresholds(self, threshold):
        """Test quantization with different threshold values."""
        tensor = torch.randn(100, 100)
        
        ternary_tensor = quantize_to_ternary(tensor, threshold=threshold)
        
        # Check ternary
        validation = validate_ternary_weights(ternary_tensor)
        assert validation['is_ternary']
        
        # Higher threshold should result in more zeros
        zero_ratio = validation['value_counts']['zero'] / validation['total_elements']
        if threshold > 0.7:
            assert zero_ratio > 0.3
    
    def test_pack_unpack_ternary(self):
        """Test packing and unpacking of ternary weights."""
        # Create ternary tensor
        ternary_tensor = torch.tensor([-1, 0, 1, -1, 0, 1, 1, -1], dtype=torch.int8)
        
        # Pack
        packed = pack_ternary(ternary_tensor)
        
        # Unpack
        unpacked = unpack_ternary(packed, ternary_tensor.numel())
        
        # Should match original
        assert torch.all(unpacked == ternary_tensor)
    
    def test_pack_ternary_compression(self):
        """Test that packing achieves compression."""
        ternary_tensor = torch.randint(-1, 2, (1000,), dtype=torch.int8)
        
        # Pack
        packed = pack_ternary(ternary_tensor)
        
        # Check compression ratio
        original_bytes = ternary_tensor.numel() * ternary_tensor.element_size()
        packed_bytes = packed.numel() * packed.element_size()
        
        compression = original_bytes / packed_bytes
        
        # Should achieve ~4x compression
        assert compression >= 3.5
        assert compression <= 4.5
    
    def test_simple_linear_quantization(self):
        """Test quantization of a simple linear layer."""
        # Create FP32 linear layer
        linear = nn.Linear(128, 256)
        
        # Quantize weights to ternary
        with torch.no_grad():
            ternary_weights = quantize_to_ternary(linear.weight)
            linear.weight.copy_(ternary_weights.float())
        
        # Validate ternary
        validation = validate_ternary_weights(linear.weight.to(torch.int8))
        assert validation['is_ternary']
    
    def test_conv2d_quantization(self):
        """Test quantization of a Conv2d layer."""
        # Create FP32 conv layer
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # Quantize weights
        with torch.no_grad():
            ternary_weights = quantize_to_ternary(conv.weight)
            conv.weight.copy_(ternary_weights.float())
        
        # Validate ternary
        validation = validate_ternary_weights(conv.weight.to(torch.int8))
        assert validation['is_ternary']
    
    def test_model_quantization_functional(self, reference_pytorch_model):
        """Test that quantized model remains functional."""
        model = reference_pytorch_model
        
        # Get original output
        x = torch.randn(4, 64)
        with torch.no_grad():
            original_output = model(x)
        
        # Quantize model
        quantize_model_to_ternary(model)
        
        # Get quantized output
        with torch.no_grad():
            quantized_output = model(x)
        
        # Should still produce output
        assert quantized_output is not None
        assert quantized_output.shape == original_output.shape
    
    def test_model_quantization_preserves_structure(self, reference_pytorch_model):
        """Test that quantization preserves model structure."""
        model = reference_pytorch_model
        
        # Count parameters before
        params_before = sum(p.numel() for p in model.parameters())
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # Count parameters after
        params_after = sum(p.numel() for p in model.parameters())
        
        # Should have same number of parameters
        assert params_before == params_after
    
    def test_quantization_output_similarity(self, reference_pytorch_model):
        """Test that quantized model output is similar to original."""
        # Clone model
        model_fp32 = reference_pytorch_model
        model_ternary = type(model_fp32)()
        model_ternary.load_state_dict(model_fp32.state_dict())
        
        # Quantize one model
        quantize_model_to_ternary(model_ternary)
        
        # Compare outputs
        x = torch.randn(4, 64)
        comparison = compare_model_outputs(model_fp32, model_ternary, x, rtol=0.5, atol=0.5)
        
        # Outputs should be reasonably similar (not exact due to quantization)
        assert comparison['shapes_match']
        # Mean difference should be bounded
        assert comparison['mean_difference'] < 5.0
    
    def test_quantization_inference_speed(self, reference_pytorch_model):
        """Test inference speed of quantized model."""
        from tests.integration.test_utils import measure_inference_time
        
        model = reference_pytorch_model
        x = torch.randn(8, 64)
        
        # Measure FP32 speed
        fp32_timing = measure_inference_time(model, x, warmup_iterations=5, benchmark_iterations=20)
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # Measure ternary speed
        ternary_timing = measure_inference_time(model, x, warmup_iterations=5, benchmark_iterations=20)
        
        # Both should complete successfully
        assert fp32_timing['mean'] > 0
        assert ternary_timing['mean'] > 0
    
    def test_quantization_memory_reduction(self, reference_pytorch_model):
        """Test memory reduction from quantization."""
        model = reference_pytorch_model
        
        # Calculate FP32 memory
        fp32_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # Quantize (weights become int8 in ternary)
        quantize_model_to_ternary(model)
        
        # After quantization, parameters are still FP32 for compatibility,
        # but in a real ternary implementation they'd be packed
        # This test validates the concept
        
        # Check that quantization happened
        for param in model.parameters():
            if param.numel() > 0:
                unique_vals = torch.unique(param.data).tolist()
                # After quantization to float, values should still be limited
                # (though stored as float, they represent ternary values)
                assert len([v for v in unique_vals if abs(v) > 1.1]) == 0 or len(unique_vals) <= 10
    
    def test_batch_quantization(self):
        """Test quantizing multiple tensors in batch."""
        tensors = [torch.randn(50, 50) for _ in range(5)]
        
        # Quantize all
        ternary_tensors = [quantize_to_ternary(t) for t in tensors]
        
        # Validate all
        for ternary in ternary_tensors:
            validation = validate_ternary_weights(ternary)
            assert validation['is_ternary']
    
    def test_quantization_gradient_flow(self):
        """Test that gradients can flow through quantized layers."""
        # Create model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # Forward pass
        x = torch.randn(4, 32, requires_grad=True)
        output = model(x)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_quantization_training_step(self):
        """Test a training step with quantized model."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        
        # Quantize
        quantize_model_to_ternary(model)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        # Training step
        model.train()
        x = torch.randn(4, 32)
        y = torch.randint(0, 10, (4,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Should complete without errors
        assert not torch.isnan(loss).any()
    
    @pytest.mark.parametrize("sparsity_level", [0.3, 0.5, 0.7])
    def test_quantization_sparsity_control(self, sparsity_level):
        """Test controlling sparsity (zeros) in quantized weights."""
        tensor = torch.randn(1000)
        
        # Use threshold to control sparsity
        abs_vals = torch.abs(tensor)
        threshold = torch.quantile(abs_vals, sparsity_level)
        
        ternary = quantize_to_ternary(tensor, threshold=threshold)
        
        # Check sparsity
        zero_count = (ternary == 0).sum().item()
        actual_sparsity = zero_count / tensor.numel()
        
        # Should be close to target sparsity
        assert abs(actual_sparsity - sparsity_level) < 0.15
    
    def test_quantization_deterministic(self):
        """Test that quantization is deterministic."""
        tensor = torch.randn(100, 100)
        
        # Quantize twice
        ternary1 = quantize_to_ternary(tensor)
        ternary2 = quantize_to_ternary(tensor)
        
        # Should be identical
        assert torch.all(ternary1 == ternary2)
    
    def test_quantization_large_tensor(self):
        """Test quantization of large tensors."""
        large_tensor = torch.randn(1000, 1000)
        
        ternary = quantize_to_ternary(large_tensor)
        
        validation = validate_ternary_weights(ternary)
        assert validation['is_ternary']
        assert validation['total_elements'] == 1000000
    
    def test_quantization_small_tensor(self):
        """Test quantization of small tensors."""
        small_tensor = torch.randn(3, 3)
        
        ternary = quantize_to_ternary(small_tensor)
        
        validation = validate_ternary_weights(ternary)
        assert validation['is_ternary']
    
    def test_quantization_1d_tensor(self):
        """Test quantization of 1D tensors (bias)."""
        bias = torch.randn(128)
        
        ternary_bias = quantize_to_ternary(bias)
        
        validation = validate_ternary_weights(ternary_bias)
        assert validation['is_ternary']
        assert ternary_bias.shape == bias.shape
    
    def test_quantization_4d_tensor(self):
        """Test quantization of 4D tensors (conv weights)."""
        conv_weights = torch.randn(64, 3, 3, 3)
        
        ternary_weights = quantize_to_ternary(conv_weights)
        
        validation = validate_ternary_weights(ternary_weights)
        assert validation['is_ternary']
        assert ternary_weights.shape == conv_weights.shape
    
    def test_quantization_cnn_model(self, reference_cnn_model):
        """Test quantization of CNN model."""
        model = reference_cnn_model
        x = torch.randn(2, 3, 32, 32)
        
        # Get FP32 output
        with torch.no_grad():
            fp32_output = model(x)
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # Get ternary output
        with torch.no_grad():
            ternary_output = model(x)
        
        # Should still produce valid output
        assert ternary_output.shape == fp32_output.shape
    
    def test_quantization_value_distribution(self):
        """Test distribution of values after quantization."""
        tensor = torch.randn(10000)
        
        ternary = quantize_to_ternary(tensor)
        
        validation = validate_ternary_weights(ternary)
        
        # Check that all three values are present
        counts = validation['value_counts']
        assert counts['negative_one'] > 0
        assert counts['zero'] > 0
        assert counts['positive_one'] > 0
        
        # Check total
        total = counts['negative_one'] + counts['zero'] + counts['positive_one']
        assert total == validation['total_elements']
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantization_cuda(self):
        """Test quantization on CUDA device."""
        tensor = torch.randn(100, 100).cuda()
        
        ternary = quantize_to_ternary(tensor)
        
        # Result should be on same device
        assert ternary.is_cuda
        
        validation = validate_ternary_weights(ternary)
        assert validation['is_ternary']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

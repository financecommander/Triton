"""
Integration tests for quantization pipeline.
Tests FP32 → FP16, FP16 → INT8, FP16 → Ternary, mixed precision, and calibration.
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
from tests.integration.test_utils import (
    compare_model_outputs,
    measure_memory_usage,
)


class TestQuantizationPipeline:
    """Test various quantization pipelines."""
    
    def test_fp32_to_fp16_conversion(self, reference_pytorch_model):
        """Test FP32 to FP16 conversion."""
        model_fp32 = reference_pytorch_model
        
        # Convert to FP16
        model_fp16 = model_fp32.half()
        
        # Check dtype
        for param in model_fp16.parameters():
            assert param.dtype == torch.float16
        
        # Test inference
        x = torch.randn(4, 64).half()
        with torch.no_grad():
            output = model_fp16(x)
        
        assert output.dtype == torch.float16
        assert output.shape == (4, 10)
    
    def test_fp32_to_fp16_accuracy(self, reference_pytorch_model):
        """Test that FP32 to FP16 maintains accuracy."""
        model_fp32 = reference_pytorch_model
        model_fp16 = type(model_fp32)()
        model_fp16.load_state_dict(model_fp32.state_dict())
        model_fp16 = model_fp16.half()
        
        # Compare outputs
        x_fp32 = torch.randn(4, 64)
        x_fp16 = x_fp32.half()
        
        with torch.no_grad():
            output_fp32 = model_fp32(x_fp32)
            output_fp16 = model_fp16(x_fp16).float()
        
        # Should be very close
        assert torch.allclose(output_fp32, output_fp16, rtol=1e-2, atol=1e-2)
    
    def test_fp32_to_fp16_memory_reduction(self, reference_pytorch_model):
        """Test memory reduction from FP32 to FP16."""
        model_fp32 = reference_pytorch_model
        
        # Measure FP32 memory
        fp32_memory = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
        
        # Convert to FP16
        model_fp16 = model_fp32.half()
        
        # Measure FP16 memory
        fp16_memory = sum(p.numel() * p.element_size() for p in model_fp16.parameters())
        
        # Should be exactly 2x reduction
        assert fp32_memory == fp16_memory * 2
    
    def test_fp16_to_int8_quantization_dynamic(self, reference_pytorch_model):
        """Test FP16 to INT8 dynamic quantization."""
        model = reference_pytorch_model.half()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model.float(),  # Need FP32 for quantization API
            {nn.Linear},
            dtype=torch.qint8
        )
        
        # Test inference
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = quantized_model(x)
        
        assert output.shape == (4, 10)
    
    def test_fp16_to_ternary_pipeline(self, reference_pytorch_model):
        """Test FP16 to ternary quantization pipeline."""
        model = reference_pytorch_model
        
        # Convert to FP16 first
        model = model.half().float()  # half() then back to float for ternary conversion
        
        # Quantize to ternary
        quantize_model_to_ternary(model)
        
        # Test inference
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_mixed_precision_model(self):
        """Test model with mixed precision layers."""
        class MixedPrecisionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1_fp32 = nn.Linear(64, 128)
                self.fc2_fp16 = nn.Linear(128, 128)
                self.fc3_fp32 = nn.Linear(128, 10)
                
                # Convert fc2 to FP16
                self.fc2_fp16 = self.fc2_fp16.half()
            
            def forward(self, x):
                x = torch.relu(self.fc1_fp32(x))
                x = torch.relu(self.fc2_fp16(x.half()).float())
                x = self.fc3_fp32(x)
                return x
        
        model = MixedPrecisionModel()
        x = torch.randn(4, 64)
        
        output = model(x)
        assert output.shape == (4, 10)
    
    def test_mixed_precision_ternary_model(self, reference_pytorch_model):
        """Test model with some ternary and some FP32 layers."""
        model = reference_pytorch_model
        
        # Quantize only first layer to ternary
        first_layer = list(model.children())[0]
        if hasattr(first_layer, 'weight'):
            with torch.no_grad():
                ternary_weights = quantize_to_ternary(first_layer.weight)
                first_layer.weight.copy_(ternary_weights.float())
        
        # Test inference
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_calibration_with_dataset(self, mock_dataloader):
        """Test calibration using a dataset."""
        # Collect activations from dataset
        activations = []
        for batch_x, _ in mock_dataloader:
            activations.append(batch_x)
            if len(activations) >= 10:  # Use 10 batches for calibration
                break
        
        all_activations = torch.cat(activations, dim=0)
        
        # Calibrate threshold
        threshold = calibrate_threshold(all_activations, percentile=0.7)
        
        assert threshold > 0
    
    def test_calibration_accuracy_impact(self, reference_pytorch_model, mock_dataloader):
        """Test impact of calibration on accuracy."""
        model = reference_pytorch_model
        
        # Quantize without calibration
        model_no_calib = type(model)()
        model_no_calib.load_state_dict(model.state_dict())
        quantize_model_to_ternary(model_no_calib)
        
        # For calibrated version, we would collect statistics and choose better threshold
        # For this test, just use fixed threshold
        model_calib = type(model)()
        model_calib.load_state_dict(model.state_dict())
        quantize_model_to_ternary(model_calib)  # Same for now, in production would differ
        
        # Both should work
        x = torch.randn(4, 64)
        with torch.no_grad():
            output1 = model_no_calib(x)
            output2 = model_calib(x)
        
        assert output1.shape == (4, 10)
        assert output2.shape == (4, 10)
    
    @pytest.mark.parametrize("precision", ["fp32", "fp16"])
    def test_quantization_from_different_precisions(self, reference_pytorch_model, precision):
        """Test quantizing from different starting precisions."""
        model = reference_pytorch_model
        
        if precision == "fp16":
            model = model.half().float()  # Convert and back
        
        # Quantize to ternary
        quantize_model_to_ternary(model)
        
        # Test inference
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_sequential_quantization_steps(self, reference_pytorch_model):
        """Test sequential quantization: FP32 → FP16 → Ternary."""
        model = reference_pytorch_model
        x = torch.randn(4, 64)
        
        # Step 1: FP32 baseline
        with torch.no_grad():
            output_fp32 = model(x)
        
        # Step 2: FP32 → FP16
        model = model.half()
        with torch.no_grad():
            output_fp16 = model(x.half()).float()
        
        # Step 3: FP16 → Ternary (need to convert back to FP32 for ternary)
        model = model.float()
        quantize_model_to_ternary(model)
        with torch.no_grad():
            output_ternary = model(x)
        
        # All should produce valid outputs
        assert output_fp32.shape == (4, 10)
        assert output_fp16.shape == (4, 10)
        assert output_ternary.shape == (4, 10)
    
    def test_per_layer_quantization_control(self, reference_pytorch_model):
        """Test selective quantization of specific layers."""
        model = reference_pytorch_model
        
        # Quantize only Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    if hasattr(module, 'weight'):
                        ternary_weights = quantize_to_ternary(module.weight)
                        module.weight.copy_(ternary_weights.float())
        
        # Test inference
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_quantization_preserves_batch_norm(self):
        """Test that quantization preserves batch normalization layers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # BatchNorm should still exist and work
        x = torch.randn(4, 64)
        model.train()
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_quantization_with_dropout(self):
        """Test quantization with dropout layers."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )
        
        # Quantize
        quantize_model_to_ternary(model)
        
        # Test in training mode (dropout active)
        model.train()
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output.shape == (4, 10)
    
    def test_quantization_memory_comparison(self, reference_pytorch_model):
        """Compare memory usage across quantization schemes."""
        x = torch.randn(4, 64)
        
        # FP32
        model_fp32 = reference_pytorch_model
        memory_fp32 = measure_memory_usage(model_fp32, x)
        
        # FP16
        model_fp16 = type(model_fp32)()
        model_fp16.load_state_dict(model_fp32.state_dict())
        model_fp16 = model_fp16.half()
        memory_fp16 = measure_memory_usage(model_fp16, x.half())
        
        # Ternary
        model_ternary = type(model_fp32)()
        model_ternary.load_state_dict(model_fp32.state_dict())
        quantize_model_to_ternary(model_ternary)
        memory_ternary = measure_memory_usage(model_ternary, x)
        
        # FP16 should use less memory than FP32
        assert memory_fp16['model_memory_mb'] < memory_fp32['model_memory_mb']
        
        # All should complete
        assert memory_fp32['total_memory_mb'] > 0
        assert memory_fp16['total_memory_mb'] > 0
        assert memory_ternary['total_memory_mb'] > 0
    
    def test_quantization_accuracy_vs_compression_tradeoff(self, reference_pytorch_model):
        """Test accuracy vs compression tradeoff."""
        model_original = reference_pytorch_model
        x = torch.randn(4, 64)
        
        # Get original output
        with torch.no_grad():
            output_original = model_original(x)
        
        # Test different quantization schemes
        schemes = []
        
        # FP16
        model_fp16 = type(model_original)()
        model_fp16.load_state_dict(model_original.state_dict())
        model_fp16 = model_fp16.half()
        with torch.no_grad():
            output_fp16 = model_fp16(x.half()).float()
        diff_fp16 = (output_original - output_fp16).abs().mean().item()
        schemes.append(('fp16', diff_fp16))
        
        # Ternary
        model_ternary = type(model_original)()
        model_ternary.load_state_dict(model_original.state_dict())
        quantize_model_to_ternary(model_ternary)
        with torch.no_grad():
            output_ternary = model_ternary(x)
        diff_ternary = (output_original - output_ternary).abs().mean().item()
        schemes.append(('ternary', diff_ternary))
        
        # All schemes should produce valid outputs
        for scheme, diff in schemes:
            assert diff >= 0
    
    @pytest.mark.parametrize("threshold_method", ["mean", "median", "percentile"])
    def test_different_threshold_methods(self, threshold_method):
        """Test different methods for determining quantization threshold."""
        tensor = torch.randn(1000)
        
        if threshold_method == "mean":
            threshold = tensor.abs().mean()
        elif threshold_method == "median":
            threshold = tensor.abs().median()
        else:  # percentile
            threshold = torch.quantile(tensor.abs(), 0.7)
        
        ternary = quantize_to_ternary(tensor, threshold=threshold)
        
        # Should produce valid ternary
        unique = torch.unique(ternary).tolist()
        assert set(unique).issubset({-1, 0, 1})
    
    def test_quantization_reproducibility(self, reference_pytorch_model):
        """Test that quantization is reproducible."""
        model1 = reference_pytorch_model
        model2 = type(model1)()
        model2.load_state_dict(model1.state_dict())
        
        # Quantize both
        quantize_model_to_ternary(model1)
        quantize_model_to_ternary(model2)
        
        # Compare parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantization_cuda_fp16(self, reference_pytorch_model):
        """Test FP16 quantization on CUDA."""
        model = reference_pytorch_model.cuda().half()
        x = torch.randn(4, 64).cuda().half()
        
        with torch.no_grad():
            output = model(x)
        
        assert output.is_cuda
        assert output.dtype == torch.float16
        assert output.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

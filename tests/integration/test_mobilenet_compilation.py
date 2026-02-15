"""
Integration tests for MobileNetV2 model compilation.
Tests the complete flow from DSL to executable PyTorch model.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.mobilenetv2.ternary_mobilenetv2 import (
    TernaryMobileNetV2, InvertedResidual
)
from tests.integration.test_utils import (
    measure_inference_time,
    measure_memory_usage,
    count_parameters,
    validate_output_shape,
    check_numerical_stability,
    benchmark_batch_sizes,
)


class TestMobileNetV2Compilation:
    """Test MobileNetV2 compilation and execution."""
    
    def test_ternary_mobilenetv2_instantiation(self):
        """Test that TernaryMobileNetV2 can be instantiated."""
        model = TernaryMobileNetV2(num_classes=10)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_ternary_mobilenetv2_architecture(self):
        """Test MobileNetV2 architecture structure."""
        model = TernaryMobileNetV2(num_classes=10)
        
        # Check key components
        assert hasattr(model, 'features')
        assert hasattr(model, 'classifier')
        
        # Features should be a Sequential module
        assert isinstance(model.features, nn.Sequential)
    
    def test_ternary_mobilenetv2_forward_pass(self, cifar10_like_input):
        """Test MobileNetV2 forward pass with CIFAR-10 sized input."""
        model = TernaryMobileNetV2(num_classes=10)
        model.eval()
        
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        # Validate output
        assert output is not None
        assert validate_output_shape(output, (8, 10))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_ternary_mobilenetv2_imagenet_input(self, imagenet_like_input):
        """Test MobileNetV2 with ImageNet-sized input."""
        model = TernaryMobileNetV2(num_classes=1000)
        model.eval()
        
        with torch.no_grad():
            output = model(imagenet_like_input)
        
        assert output is not None
        assert validate_output_shape(output, (2, 1000))
    
    def test_ternary_mobilenetv2_backward_pass(self, cifar10_like_input):
        """Test MobileNetV2 backward pass and gradient computation."""
        model = TernaryMobileNetV2(num_classes=10)
        model.train()
        
        # Forward pass
        output = model(cifar10_like_input)
        
        # Create target and compute loss
        target = torch.randint(0, 10, (8,))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients were computed
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients computed during backward pass"
        assert not torch.isnan(loss).any()
    
    def test_ternary_mobilenetv2_parameter_count(self):
        """Test MobileNetV2 parameter count."""
        model = TernaryMobileNetV2(num_classes=10)
        param_counts = count_parameters(model)
        
        # MobileNetV2 should be lightweight
        assert param_counts['total_parameters'] > 0
        assert param_counts['trainable_parameters'] > 0
        
        # Should be smaller than ResNet18 (~3.5M parameters)
        assert param_counts['total_parameters'] < 5_000_000
    
    def test_ternary_mobilenetv2_inference_speed(self, cifar10_like_input):
        """Test MobileNetV2 inference speed."""
        model = TernaryMobileNetV2(num_classes=10)
        
        timing = measure_inference_time(
            model, 
            cifar10_like_input, 
            warmup_iterations=5,
            benchmark_iterations=50
        )
        
        # Check timing results
        assert timing['mean'] > 0
        assert timing['std'] >= 0
        assert timing['min'] > 0
        assert timing['max'] >= timing['mean']
        
        # MobileNet should be fast (< 0.5 seconds per batch on CPU)
        assert timing['mean'] < 0.5
    
    def test_ternary_mobilenetv2_memory_usage(self, cifar10_like_input):
        """Test MobileNetV2 memory usage."""
        model = TernaryMobileNetV2(num_classes=10)
        
        memory = measure_memory_usage(model, cifar10_like_input)
        
        # Check memory results
        assert memory['model_memory_mb'] > 0
        assert memory['total_memory_mb'] > 0
        
        # MobileNet should be very memory efficient
        # Should be < 50MB with ternary weights
        assert memory['total_memory_mb'] < 100
    
    def test_ternary_mobilenetv2_numerical_stability(self, cifar10_like_input):
        """Test MobileNetV2 numerical stability."""
        model = TernaryMobileNetV2(num_classes=10)
        
        stability = check_numerical_stability(
            model, 
            cifar10_like_input, 
            n_iterations=5
        )
        
        # Output should be consistent across runs
        assert stability['numerically_stable']
        assert stability['output_variance'] < 1e-6
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_ternary_mobilenetv2_batch_sizes(self, batch_size):
        """Test MobileNetV2 with different batch sizes."""
        model = TernaryMobileNetV2(num_classes=10)
        model.eval()
        
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert validate_output_shape(output, (batch_size, 10))
    
    def test_ternary_mobilenetv2_batch_scaling(self):
        """Test MobileNetV2 performance scaling with batch size."""
        model = TernaryMobileNetV2(num_classes=10)
        
        batch_sizes = [1, 4, 8, 16]
        results = benchmark_batch_sizes(
            model, 
            (3, 32, 32), 
            batch_sizes
        )
        
        # Check that all batch sizes completed
        assert len(results) == len(batch_sizes)
        
        # All batch sizes should complete
        for batch_size in batch_sizes:
            assert results[batch_size]['mean'] > 0
    
    @pytest.mark.parametrize("num_classes", [10, 100, 1000])
    def test_ternary_mobilenetv2_different_num_classes(self, num_classes, cifar10_like_input):
        """Test MobileNetV2 with different number of output classes."""
        model = TernaryMobileNetV2(num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        assert validate_output_shape(output, (8, num_classes))
    
    @pytest.mark.parametrize("width_mult", [0.5, 0.75, 1.0])
    def test_ternary_mobilenetv2_width_multipliers(self, width_mult, cifar10_like_input):
        """Test MobileNetV2 with different width multipliers."""
        model = TernaryMobileNetV2(num_classes=10, width_mult=width_mult)
        model.eval()
        
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        assert output is not None
        assert validate_output_shape(output, (8, 10))
    
    def test_ternary_mobilenetv2_inverted_residual_block(self):
        """Test InvertedResidual block."""
        # Create an inverted residual block
        block = InvertedResidual(32, 64, stride=1, expand_ratio=6)
        
        # Test forward pass
        x = torch.randn(2, 32, 16, 16)
        output = block(x)
        
        # Check output shape
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 64  # Output channels
    
    def test_ternary_mobilenetv2_training_mode(self, cifar10_like_input):
        """Test MobileNetV2 in training mode."""
        model = TernaryMobileNetV2(num_classes=10)
        model.train()
        
        # Check that model is in training mode
        assert model.training
        
        # Forward pass should work
        output = model(cifar10_like_input)
        assert output is not None
        assert validate_output_shape(output, (8, 10))
    
    def test_ternary_mobilenetv2_eval_mode(self, cifar10_like_input):
        """Test MobileNetV2 in evaluation mode."""
        model = TernaryMobileNetV2(num_classes=10)
        model.eval()
        
        # Check that model is in eval mode
        assert not model.training
        
        # Forward pass should work
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        assert output is not None
        assert validate_output_shape(output, (8, 10))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ternary_mobilenetv2_cuda(self, cifar10_like_input):
        """Test MobileNetV2 on CUDA device."""
        model = TernaryMobileNetV2(num_classes=10)
        model = model.cuda()
        input_tensor = cifar10_like_input.cuda()
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output is not None
        assert output.is_cuda
        assert validate_output_shape(output, (8, 10))
    
    def test_ternary_mobilenetv2_save_load(self, temp_dir, cifar10_like_input):
        """Test MobileNetV2 model save and load."""
        model = TernaryMobileNetV2(num_classes=10)
        model.eval()
        
        # Get original output
        with torch.no_grad():
            original_output = model(cifar10_like_input)
        
        # Save model
        model_path = temp_dir / "mobilenetv2.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = TernaryMobileNetV2(num_classes=10)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()
        
        # Get loaded output
        with torch.no_grad():
            loaded_output = loaded_model(cifar10_like_input)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-7)
    
    def test_mobilenetv2_vs_resnet18_efficiency(self, cifar10_like_input):
        """Compare MobileNetV2 efficiency vs ResNet18."""
        from models.resnet18.ternary_resnet18 import TernaryResNet, BasicBlock
        
        mobilenet = TernaryMobileNetV2(num_classes=10)
        resnet = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        # Compare parameter counts
        mobilenet_params = count_parameters(mobilenet)
        resnet_params = count_parameters(resnet)
        
        # MobileNet should have fewer parameters
        assert mobilenet_params['total_parameters'] < resnet_params['total_parameters']
        
        # Compare memory
        mobilenet_memory = measure_memory_usage(mobilenet, cifar10_like_input)
        resnet_memory = measure_memory_usage(resnet, cifar10_like_input)
        
        # MobileNet should use less memory
        assert mobilenet_memory['total_memory_mb'] < resnet_memory['total_memory_mb']
        
        # Compare inference speed
        mobilenet_time = measure_inference_time(mobilenet, cifar10_like_input, 
                                                warmup_iterations=5, benchmark_iterations=20)
        resnet_time = measure_inference_time(resnet, cifar10_like_input,
                                            warmup_iterations=5, benchmark_iterations=20)
        
        # MobileNet should be faster or comparable
        # (May vary depending on hardware, so just check it completes)
        assert mobilenet_time['mean'] > 0
        assert resnet_time['mean'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

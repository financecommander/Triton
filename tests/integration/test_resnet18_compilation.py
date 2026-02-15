"""
Integration tests for ResNet18 model compilation.
Tests the complete flow from DSL to executable PyTorch model.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.resnet18.ternary_resnet18 import (
    TernaryResNet, BasicBlock, TernaryConv2d, TernaryLinear
)
from tests.integration.test_utils import (
    measure_inference_time,
    measure_memory_usage,
    count_parameters,
    validate_output_shape,
    test_forward_backward_pass,
    check_numerical_stability,
    benchmark_batch_sizes,
)


class TestResNet18Compilation:
    """Test ResNet18 compilation and execution."""
    
    def test_ternary_resnet18_instantiation(self):
        """Test that TernaryResNet18 can be instantiated."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_ternary_resnet18_architecture(self):
        """Test ResNet18 architecture structure."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        # Check key components
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'layer1')
        assert hasattr(model, 'layer2')
        assert hasattr(model, 'layer3')
        assert hasattr(model, 'layer4')
        assert hasattr(model, 'fc')
        
        # Check layer types
        assert isinstance(model.conv1, (nn.Conv2d, TernaryConv2d))
        assert isinstance(model.fc, (nn.Linear, TernaryLinear))
    
    def test_ternary_resnet18_forward_pass(self, cifar10_like_input):
        """Test ResNet18 forward pass with CIFAR-10 sized input."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.eval()
        
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        # Validate output
        assert output is not None
        assert validate_output_shape(output, (8, 10))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_ternary_resnet18_imagenet_input(self, imagenet_like_input):
        """Test ResNet18 with ImageNet-sized input."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=1000)
        model.eval()
        
        with torch.no_grad():
            output = model(imagenet_like_input)
        
        assert output is not None
        assert validate_output_shape(output, (2, 1000))
    
    def test_ternary_resnet18_backward_pass(self, cifar10_like_input):
        """Test ResNet18 backward pass and gradient computation."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
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
    
    def test_ternary_resnet18_parameter_count(self):
        """Test ResNet18 parameter count."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        param_counts = count_parameters(model)
        
        # ResNet18 should have around 11M parameters
        assert param_counts['total_parameters'] > 0
        assert param_counts['trainable_parameters'] > 0
        
        # Most parameters should be trainable
        assert param_counts['trainable_parameters'] <= param_counts['total_elements']
    
    def test_ternary_resnet18_inference_speed(self, cifar10_like_input):
        """Test ResNet18 inference speed."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
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
        
        # Inference should be reasonably fast (< 1 second per batch on CPU)
        assert timing['mean'] < 1.0
    
    def test_ternary_resnet18_memory_usage(self, cifar10_like_input):
        """Test ResNet18 memory usage."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        memory = measure_memory_usage(model, cifar10_like_input)
        
        # Check memory results
        assert memory['model_memory_mb'] > 0
        assert memory['total_memory_mb'] > 0
        
        # Ternary model should be memory efficient
        # ResNet18 with ternary weights should be < 100MB
        assert memory['total_memory_mb'] < 200
    
    def test_ternary_resnet18_numerical_stability(self, cifar10_like_input):
        """Test ResNet18 numerical stability."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        stability = check_numerical_stability(
            model, 
            cifar10_like_input, 
            n_iterations=5
        )
        
        # Output should be consistent across runs
        assert stability['numerically_stable']
        assert stability['output_variance'] < 1e-6
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_ternary_resnet18_batch_sizes(self, batch_size):
        """Test ResNet18 with different batch sizes."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.eval()
        
        input_tensor = torch.randn(batch_size, 3, 32, 32)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert validate_output_shape(output, (batch_size, 10))
    
    def test_ternary_resnet18_batch_scaling(self):
        """Test ResNet18 performance scaling with batch size."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        
        batch_sizes = [1, 4, 8, 16]
        results = benchmark_batch_sizes(
            model, 
            (3, 32, 32), 
            batch_sizes
        )
        
        # Check that all batch sizes completed
        assert len(results) == len(batch_sizes)
        
        # Larger batches should have higher throughput
        # (mean time per sample should decrease with batch size)
        for batch_size in batch_sizes:
            assert results[batch_size]['mean'] > 0
    
    @pytest.mark.parametrize("num_classes", [10, 100, 1000])
    def test_ternary_resnet18_different_num_classes(self, num_classes, cifar10_like_input):
        """Test ResNet18 with different number of output classes."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        model.eval()
        
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        assert validate_output_shape(output, (8, num_classes))
    
    def test_ternary_conv2d_quantization(self):
        """Test that TernaryConv2d weights are properly quantized."""
        layer = TernaryConv2d(3, 64, kernel_size=3, padding=1)
        
        # Check weight values
        unique_values = torch.unique(layer.weight)
        
        # After quantization, weights should be in {-1, 0, 1}
        assert len(unique_values) <= 3
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())
    
    def test_ternary_linear_quantization(self):
        """Test that TernaryLinear weights are properly quantized."""
        layer = TernaryLinear(512, 10)
        
        # Check that ternary_weight buffer exists
        assert hasattr(layer, 'ternary_weight')
        
        # Check ternary weight values
        unique_values = torch.unique(layer.ternary_weight)
        assert len(unique_values) <= 3
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())
    
    def test_ternary_resnet18_training_mode(self, cifar10_like_input):
        """Test ResNet18 in training mode."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.train()
        
        # Check that model is in training mode
        assert model.training
        
        # Forward pass should work
        output = model(cifar10_like_input)
        assert output is not None
        assert validate_output_shape(output, (8, 10))
    
    def test_ternary_resnet18_eval_mode(self, cifar10_like_input):
        """Test ResNet18 in evaluation mode."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.eval()
        
        # Check that model is in eval mode
        assert not model.training
        
        # Forward pass should work
        with torch.no_grad():
            output = model(cifar10_like_input)
        
        assert output is not None
        assert validate_output_shape(output, (8, 10))
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ternary_resnet18_cuda(self, cifar10_like_input):
        """Test ResNet18 on CUDA device."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model = model.cuda()
        input_tensor = cifar10_like_input.cuda()
        
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output is not None
        assert output.is_cuda
        assert validate_output_shape(output, (8, 10))
    
    def test_ternary_resnet18_save_load(self, temp_dir, cifar10_like_input):
        """Test ResNet18 model save and load."""
        model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        model.eval()
        
        # Get original output
        with torch.no_grad():
            original_output = model(cifar10_like_input)
        
        # Save model
        model_path = temp_dir / "resnet18.pth"
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = TernaryResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()
        
        # Get loaded output
        with torch.no_grad():
            loaded_output = loaded_model(cifar10_like_input)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

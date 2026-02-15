"""
Integration tests for custom model compilation from DSL.
Tests the complete flow from DSL definition to executable PyTorch model.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import generate_pytorch_code, PyTorchCodeGenerator
from tests.integration.test_utils import (
    measure_inference_time,
    measure_memory_usage,
    count_parameters,
    validate_output_shape,
    test_forward_backward_pass,
    validate_ternary_weights,
)


class TestCustomModelCompilation:
    """Test custom model compilation from DSL."""
    
    def test_simple_custom_layer_compilation(self):
        """Test compilation of a simple custom layer."""
        layer_def = LayerDef(
            name="SimpleCustomLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        # Verify code generation
        assert code is not None
        assert "class SimpleCustomLayer" in code
        assert "def __init__" in code
        assert "def forward" in code
    
    def test_custom_layer_instantiation(self, simple_layer_def):
        """Test instantiation of a custom compiled layer."""
        code = generate_pytorch_code(simple_layer_def)
        
        namespace = {}
        exec(code, namespace)
        
        model_class = namespace["SimpleTernaryLayer"]
        model = model_class()
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_custom_layer_forward_pass(self, compiled_simple_model):
        """Test forward pass of a custom compiled layer."""
        batch_size = 4
        input_dim = 64
        x = torch.randn(batch_size, input_dim)
        
        output = compiled_simple_model(x)
        
        assert output is not None
        assert validate_output_shape(output, (batch_size, 128))
    
    def test_custom_multi_layer_compilation(self, multi_layer_def):
        """Test compilation of a multi-parameter layer."""
        code = generate_pytorch_code(multi_layer_def)
        
        # Verify all parameters are included
        assert "weights1" in code
        assert "weights2" in code
        assert "weights3" in code
        assert "bias" in code
    
    def test_custom_multi_layer_instantiation(self, compiled_multi_model):
        """Test instantiation of multi-parameter layer."""
        assert compiled_multi_model is not None
        assert isinstance(compiled_multi_model, nn.Module)
        
        # Check that all weight buffers exist
        assert hasattr(compiled_multi_model, 'weights1_packed')
        assert hasattr(compiled_multi_model, 'weights2_packed')
        assert hasattr(compiled_multi_model, 'weights3_packed')
        assert hasattr(compiled_multi_model, 'bias_packed')
    
    def test_custom_layer_parameter_shapes(self, compiled_simple_model):
        """Test that compiled layer has correct parameter shapes."""
        # Check shape attributes
        assert hasattr(compiled_simple_model, '_weights_shape')
        assert hasattr(compiled_simple_model, '_bias_shape')
        
        assert compiled_simple_model._weights_shape == [64, 128]
        assert compiled_simple_model._bias_shape == [128]
    
    def test_custom_layer_ternary_packing(self, compiled_simple_model):
        """Test that weights are properly packed as ternary."""
        from backend.pytorch.ops.pack import unpack_ternary
        
        # Unpack weights
        unpacked_weights = unpack_ternary(
            compiled_simple_model.weights_packed,
            compiled_simple_model._weights_numel
        )
        
        # Validate ternary values
        validation = validate_ternary_weights(unpacked_weights)
        assert validation['is_ternary']
        assert set(validation['unique_values']).issubset({-1, 0, 1})
    
    def test_custom_layer_compression_ratio(self, compiled_simple_model):
        """Test compression ratio of ternary weights."""
        from backend.pytorch.ops.pack import unpack_ternary
        
        # Get packed and unpacked sizes
        packed_size = compiled_simple_model.weights_packed.numel()
        unpacked_size = compiled_simple_model._weights_numel
        
        compression_ratio = unpacked_size / packed_size
        
        # Should achieve ~4x compression (2 bits per weight, 8 bits per byte)
        assert compression_ratio >= 3.5
        assert compression_ratio <= 4.5
    
    @pytest.mark.parametrize("in_features,out_features", [
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
    ])
    def test_custom_layer_various_sizes(self, in_features, out_features, model_factory):
        """Test custom layers with various sizes."""
        model = model_factory(in_features, out_features, name=f"Layer_{in_features}_{out_features}")
        
        # Test forward pass
        x = torch.randn(2, in_features)
        output = model(x)
        
        assert validate_output_shape(output, (2, out_features))
    
    def test_custom_layer_backward_pass(self, compiled_simple_model):
        """Test backward pass through custom compiled layer."""
        x = torch.randn(4, 64, requires_grad=True)
        
        output = compiled_simple_model(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_custom_sequential_model(self):
        """Test sequential model with multiple custom layers."""
        # Create multiple layer definitions
        layer1_def = LayerDef(
            name="CustomLayer1",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[32, 64]),
            ],
            body=[]
        )
        
        layer2_def = LayerDef(
            name="CustomLayer2",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[64, 128]),
            ],
            body=[]
        )
        
        # Generate and instantiate both layers
        code1 = generate_pytorch_code(layer1_def)
        code2 = generate_pytorch_code(layer2_def)
        
        namespace = {}
        exec(code1, namespace)
        exec(code2, namespace)
        
        layer1 = namespace["CustomLayer1"]()
        layer2 = namespace["CustomLayer2"]()
        
        # Create sequential model
        model = nn.Sequential(layer1, nn.ReLU(), layer2)
        
        # Test forward pass
        x = torch.randn(2, 32)
        output = model(x)
        
        assert validate_output_shape(output, (2, 64))
    
    def test_custom_layer_with_activation(self, compiled_simple_model):
        """Test custom layer combined with activations."""
        model = nn.Sequential(
            compiled_simple_model,
            nn.ReLU(),
        )
        
        x = torch.randn(4, 64)
        output = model(x)
        
        # Output should be non-negative due to ReLU
        assert (output >= 0).all()
    
    def test_custom_layer_with_dropout(self, compiled_simple_model):
        """Test custom layer combined with dropout."""
        model = nn.Sequential(
            compiled_simple_model,
            nn.Dropout(0.5),
        )
        
        model.train()
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output is not None
        assert validate_output_shape(output, (4, 128))
    
    def test_custom_layer_with_batch_norm(self, compiled_simple_model):
        """Test custom layer combined with batch normalization."""
        model = nn.Sequential(
            compiled_simple_model,
            nn.BatchNorm1d(128),
        )
        
        model.train()
        x = torch.randn(4, 64)
        output = model(x)
        
        assert output is not None
        assert validate_output_shape(output, (4, 128))
    
    def test_custom_layer_training_loop(self, compiled_simple_model):
        """Test training loop with custom layer."""
        model = compiled_simple_model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        
        # Training loop
        for _ in range(5):
            x = torch.randn(4, 64)
            target = torch.randn(4, 128)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss).any()
    
    def test_custom_layer_inference_speed(self, compiled_simple_model):
        """Test inference speed of custom layer."""
        x = torch.randn(8, 64)
        
        timing = measure_inference_time(
            compiled_simple_model,
            x,
            warmup_iterations=10,
            benchmark_iterations=100
        )
        
        assert timing['mean'] > 0
        # Should be very fast for a single layer
        assert timing['mean'] < 0.1
    
    def test_custom_layer_memory_usage(self, compiled_simple_model):
        """Test memory usage of custom layer."""
        x = torch.randn(8, 64)
        
        memory = measure_memory_usage(compiled_simple_model, x)
        
        assert memory['model_memory_mb'] > 0
        # Single layer should use minimal memory
        assert memory['total_memory_mb'] < 10
    
    def test_custom_layer_parameter_count(self, compiled_simple_model):
        """Test parameter counting for custom layer."""
        params = count_parameters(compiled_simple_model)
        
        # Should have weights and bias parameters (as buffers)
        assert params['buffer_parameters'] > 0
    
    def test_code_generation_consistency(self, simple_layer_def):
        """Test that code generation is consistent across calls."""
        code1 = generate_pytorch_code(simple_layer_def)
        code2 = generate_pytorch_code(simple_layer_def)
        
        assert code1 == code2
    
    def test_generated_code_syntax_valid(self, simple_layer_def):
        """Test that generated code is syntactically valid Python."""
        code = generate_pytorch_code(simple_layer_def)
        
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")
    
    def test_custom_layer_device_transfer(self, compiled_simple_model):
        """Test transferring custom layer between devices."""
        x = torch.randn(4, 64)
        
        # CPU inference
        output_cpu = compiled_simple_model(x)
        
        # Transfer to CPU explicitly (already on CPU)
        model_cpu = compiled_simple_model.cpu()
        output_cpu2 = model_cpu(x)
        
        # Should produce same results
        assert torch.allclose(output_cpu, output_cpu2)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_custom_layer_cuda(self, compiled_simple_model):
        """Test custom layer on CUDA."""
        model = compiled_simple_model.cuda()
        x = torch.randn(4, 64).cuda()
        
        output = model(x)
        
        assert output.is_cuda
        assert validate_output_shape(output, (4, 128))
    
    def test_custom_layer_save_load(self, compiled_simple_model, temp_dir):
        """Test saving and loading custom layer."""
        x = torch.randn(4, 64)
        
        # Get original output
        original_output = compiled_simple_model(x)
        
        # Save
        save_path = temp_dir / "custom_layer.pth"
        torch.save(compiled_simple_model.state_dict(), save_path)
        
        # Create new instance and load
        from compiler.ast.nodes import LayerDef, Param
        from backend.pytorch.codegen import generate_pytorch_code
        
        layer_def = LayerDef(
            name="SimpleTernaryLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[64, 128]),
                Param(name="bias", param_type="TernaryTensor", shape=[128]),
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        namespace = {}
        exec(code, namespace)
        
        loaded_model = namespace["SimpleTernaryLayer"]()
        loaded_model.load_state_dict(torch.load(save_path))
        
        # Get loaded output
        loaded_output = loaded_model(x)
        
        # Should match
        assert torch.allclose(original_output, loaded_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

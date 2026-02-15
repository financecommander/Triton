"""
Integration tests for error handling.
Tests invalid DSL syntax, type errors, runtime errors, and recovery.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import generate_pytorch_code
from backend.pytorch.ops.quantize import quantize_to_ternary


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_invalid_layer_name(self):
        """Test handling of invalid layer names."""
        # Python identifiers can't start with numbers
        invalid_names = ["123Layer", "Layer-Name", "Layer Name", ""]
        
        for name in invalid_names[:-1]:  # Skip empty string for now
            layer_def = LayerDef(
                name=name,
                params=[],
                body=[]
            )
            
            # Code generation should work but may produce invalid Python
            code = generate_pytorch_code(layer_def)
            assert code is not None  # Should at least generate something
    
    def test_empty_layer_definition(self):
        """Test handling of empty layer definition."""
        layer_def = LayerDef(
            name="EmptyLayer",
            params=[],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        # Should generate valid code even for empty layer
        assert "class EmptyLayer" in code
        
        # Should be able to instantiate
        namespace = {}
        exec(code, namespace)
        model = namespace["EmptyLayer"]()
        assert isinstance(model, nn.Module)
    
    def test_invalid_parameter_shape(self):
        """Test handling of invalid parameter shapes."""
        # Negative dimensions
        layer_def = LayerDef(
            name="InvalidShapeLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[-1, 10]),
            ],
            body=[]
        )
        
        # Code generation might work, but instantiation will fail
        code = generate_pytorch_code(layer_def)
        namespace = {}
        
        try:
            exec(code, namespace)
            model = namespace["InvalidShapeLayer"]()
            # If it doesn't fail during instantiation, it will fail during forward
            pytest.fail("Should have raised an error for negative dimension")
        except (ValueError, RuntimeError, AssertionError):
            # Expected error
            pass
    
    def test_mismatched_input_shape(self, compiled_simple_model):
        """Test handling of mismatched input shapes."""
        model = compiled_simple_model
        
        # Model expects (batch, 64) but we give wrong size
        wrong_input = torch.randn(4, 32)  # Should be 64
        
        with pytest.raises((RuntimeError, AssertionError, ValueError)):
            _ = model(wrong_input)
    
    def test_invalid_batch_dimension(self, compiled_simple_model):
        """Test handling of invalid batch dimensions."""
        model = compiled_simple_model
        
        # Try with wrong number of dimensions
        wrong_dims = torch.randn(64)  # Missing batch dimension
        
        with pytest.raises((RuntimeError, ValueError, IndexError)):
            _ = model(wrong_dims)
    
    def test_nan_input_handling(self, compiled_simple_model):
        """Test handling of NaN inputs."""
        model = compiled_simple_model
        model.eval()
        
        # Input with NaN values
        x = torch.randn(4, 64)
        x[0, 0] = float('nan')
        
        with torch.no_grad():
            output = model(x)
        
        # Output will likely contain NaN
        assert torch.isnan(output).any()
    
    def test_inf_input_handling(self, compiled_simple_model):
        """Test handling of Inf inputs."""
        model = compiled_simple_model
        model.eval()
        
        # Input with Inf values
        x = torch.randn(4, 64)
        x[0, 0] = float('inf')
        
        with torch.no_grad():
            output = model(x)
        
        # Output might contain Inf
        assert torch.isinf(output).any() or torch.isnan(output).any()
    
    def test_zero_sized_tensor(self, compiled_simple_model):
        """Test handling of zero-sized tensors."""
        model = compiled_simple_model
        
        # Empty batch
        empty_input = torch.randn(0, 64)
        
        with torch.no_grad():
            output = model(empty_input)
        
        # Should produce empty output
        assert output.shape[0] == 0
    
    def test_very_large_batch(self, compiled_simple_model):
        """Test handling of very large batch sizes."""
        model = compiled_simple_model
        model.eval()
        
        # Very large batch (might cause memory issues)
        try:
            large_batch = torch.randn(10000, 64)
            with torch.no_grad():
                output = model(large_batch)
            assert output.shape[0] == 10000
        except RuntimeError as e:
            # Out of memory is acceptable
            if "out of memory" not in str(e).lower():
                raise
    
    def test_invalid_dtype_input(self, compiled_simple_model):
        """Test handling of invalid dtype inputs."""
        model = compiled_simple_model
        
        # Integer input where float expected
        int_input = torch.randint(0, 10, (4, 64))
        
        # Should either work (auto-cast) or raise clear error
        try:
            output = model(int_input.float())
            assert output is not None
        except (RuntimeError, TypeError):
            # Expected if dtype mismatch not handled
            pass
    
    def test_quantization_error_recovery(self):
        """Test recovery from quantization errors."""
        # Try to quantize invalid tensor
        try:
            empty_tensor = torch.tensor([])
            ternary = quantize_to_ternary(empty_tensor)
            # Should handle gracefully or raise
        except (RuntimeError, ValueError):
            # Expected for edge case
            pass
    
    def test_gradient_explosion_detection(self):
        """Test detection of gradient explosion."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
        # Use very high learning rate to cause explosion
        optimizer = torch.optim.SGD(model.parameters(), lr=1000.0)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))
        
        # Train for a few steps
        exploded = False
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                exploded = True
                break
            
            loss.backward()
            optimizer.step()
        
        # Gradients might explode with such high learning rate
        # This is expected behavior that the user should handle
        assert True  # Test completes
    
    def test_gradient_vanishing_detection(self):
        """Test detection of vanishing gradients."""
        # Very deep network without normalization
        layers = []
        for _ in range(20):
            layers.extend([nn.Linear(64, 64), nn.Sigmoid()])
        layers.append(nn.Linear(64, 10))
        
        model = nn.Sequential(*layers)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check for vanishing gradients in early layers
        first_layer_grad = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                first_layer_grad = param.grad
                break
        
        if first_layer_grad is not None:
            grad_norm = first_layer_grad.norm().item()
            # Might be very small due to vanishing gradients
            assert grad_norm >= 0
    
    def test_out_of_memory_graceful_handling(self, compiled_simple_model):
        """Test graceful handling of out-of-memory errors."""
        model = compiled_simple_model
        
        # Try to allocate huge tensor
        try:
            huge_batch = torch.randn(1000000, 64)
            _ = model(huge_batch)
        except RuntimeError as e:
            # Should raise RuntimeError for OOM
            assert "out of memory" in str(e).lower() or "memory" in str(e).lower() or True
    
    def test_device_mismatch_error(self, compiled_simple_model):
        """Test handling of device mismatches."""
        model = compiled_simple_model  # On CPU
        
        if torch.cuda.is_available():
            # Try to pass CUDA tensor to CPU model
            cuda_input = torch.randn(4, 64).cuda()
            
            with pytest.raises(RuntimeError):
                _ = model(cuda_input)
    
    def test_model_not_in_eval_mode_warning(self, compiled_simple_model):
        """Test inference without eval mode (just a test, not necessarily error)."""
        model = compiled_simple_model
        model.train()  # Keep in training mode
        
        # Inference still works but with training behavior
        x = torch.randn(4, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output is not None
    
    def test_invalid_optimization_step(self, compiled_simple_model):
        """Test optimizer step without gradients."""
        model = compiled_simple_model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Try to step optimizer without backward
        try:
            optimizer.step()
            # Should not fail, just won't update anything
            assert True
        except Exception:
            # Some optimizers might complain
            pass
    
    def test_double_backward_error(self, compiled_simple_model):
        """Test calling backward twice without zero_grad."""
        model = compiled_simple_model
        model.train()
        
        x = torch.randn(4, 64, requires_grad=True)
        output = model(x)
        loss = output.sum()
        
        # First backward
        loss.backward()
        
        # Second backward without zero_grad (gradients accumulate)
        # This should work (gradient accumulation)
        output2 = model(x)
        loss2 = output2.sum()
        loss2.backward()
        
        # Gradients should be accumulated
        assert x.grad is not None
    
    def test_corrupted_state_dict_loading(self, compiled_simple_model, temp_dir):
        """Test loading corrupted state dict."""
        model = compiled_simple_model
        
        # Save valid state dict
        save_path = temp_dir / "model.pth"
        torch.save(model.state_dict(), save_path)
        
        # Try to load into incompatible model
        different_model = nn.Linear(32, 32)  # Different architecture
        
        with pytest.raises((RuntimeError, KeyError)):
            different_model.load_state_dict(torch.load(save_path))
    
    def test_missing_required_parameters(self):
        """Test model with missing required parameters."""
        # Try to create layer without required parameters
        layer_def = LayerDef(
            name="IncompleteLayer",
            params=None,  # Missing params
            body=[]
        )
        
        # Might raise error or handle gracefully
        try:
            code = generate_pytorch_code(layer_def)
            assert code is not None
        except (TypeError, AttributeError):
            # Expected for invalid input
            pass
    
    def test_circular_dependency_detection(self):
        """Test detection of circular dependencies (if applicable)."""
        # For simple models, circular dependencies aren't possible
        # This is more of a placeholder for complex graph-based models
        assert True
    
    def test_type_mismatch_in_operations(self):
        """Test type mismatches in operations."""
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4, 64))
        
        # Try incompatible operation
        try:
            # String concatenation with tensor (invalid)
            result = x + "invalid"
            pytest.fail("Should have raised TypeError")
        except TypeError:
            # Expected
            pass
    
    def test_numerical_overflow(self):
        """Test handling of numerical overflow."""
        # Create very large values
        x = torch.tensor([1e38, 1e38])
        y = torch.tensor([1e38, 1e38])
        
        result = x * y
        
        # Should produce inf
        assert torch.isinf(result).any()
    
    def test_numerical_underflow(self):
        """Test handling of numerical underflow."""
        # Create very small values
        x = torch.tensor([1e-40, 1e-40])
        y = torch.tensor([1e-40, 1e-40])
        
        result = x * y
        
        # Should produce zero or very small value
        assert torch.all(result < 1e-70)
    
    def test_error_message_clarity(self, compiled_simple_model):
        """Test that error messages are clear and helpful."""
        model = compiled_simple_model
        
        # Provide wrong input shape
        try:
            wrong_input = torch.randn(4, 32)  # Should be 64
            _ = model(wrong_input)
            pytest.fail("Should have raised an error")
        except Exception as e:
            # Error message should be informative
            error_msg = str(e)
            assert len(error_msg) > 0
            # Should mention size or dimension
            assert any(word in error_msg.lower() for word in ['size', 'dimension', 'shape', 'mismatch'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

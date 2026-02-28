"""
Integration tests for PyTorch code generation.
Tests the complete flow from AST to executable PyTorch code.
"""

import pytest
import torch
import sys
import os
from io import StringIO

# Add project root to path for imports

from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import PyTorchCodeGenerator, generate_pytorch_code


class TestPyTorchCodeGenerator:
    """Test PyTorch code generation from AST."""
    
    def test_codegen_initialization(self):
        """Test that code generator initializes correctly."""
        generator = PyTorchCodeGenerator()
        assert generator is not None
        assert generator.env is not None
    
    def test_generate_simple_layer(self):
        """Test generating code for a simple layer definition."""
        # Create a simple LayerDef AST node
        layer_def = LayerDef(
            name="SimpleLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[10, 20]),
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[]
        )
        
        generator = PyTorchCodeGenerator()
        code = generator.generate_module(layer_def)
        
        # Verify generated code contains expected elements
        assert "class SimpleLayer(nn.Module)" in code
        assert "def __init__(self)" in code
        assert "def forward(self" in code
        assert "pack_ternary" in code
        assert "unpack_ternary" in code
        assert "import torch" in code
        assert "import torch.nn as nn" in code
    
    def test_generate_module_with_multiple_parameters(self):
        """Test generating module with multiple ternary parameters."""
        layer_def = LayerDef(
            name="MultiParamLayer",
            params=[
                Param(name="weights1", param_type="TernaryTensor", shape=[5, 10]),
                Param(name="weights2", param_type="TernaryTensor", shape=[10, 20]),
                Param(name="bias", param_type="TernaryTensor", shape=[20])
            ],
            body=[]
        )
        
        generator = PyTorchCodeGenerator()
        code = generator.generate_module(layer_def)
        
        # Check all parameters are included
        assert "weights1" in code
        assert "weights2" in code
        assert "bias" in code
        assert "_weights1_shape = [5, 10]" in code
        assert "_weights2_shape = [10, 20]" in code
        assert "_bias_shape = [20]" in code
    
    def test_generated_code_is_valid_python(self):
        """Test that generated code is syntactically valid Python."""
        layer_def = LayerDef(
            name="ValidLayer",
            params=[
                Param(name="w", param_type="TernaryTensor", shape=[3, 3])
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
            assert True
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")
    
    def test_generated_code_can_be_imported(self):
        """Test that generated code can be executed and module instantiated."""
        layer_def = LayerDef(
            name="ExecutableLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[4, 4])
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        # Execute the code in a namespace
        namespace = {}
        try:
            exec(code, namespace)
            
            # Check that the class was created
            assert "ExecutableLayer" in namespace
            
            # Try to instantiate the module
            module_class = namespace["ExecutableLayer"]
            module = module_class()
            
            assert isinstance(module, torch.nn.Module)
            
        except Exception as e:
            pytest.fail(f"Failed to execute generated code: {e}")
    
    def test_generated_module_forward_pass(self):
        """Test that generated module can perform forward pass."""
        layer_def = LayerDef(
            name="ForwardLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[8, 8])
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        # Execute the code
        namespace = {}
        exec(code, namespace)
        
        # Instantiate and run forward pass
        module_class = namespace["ForwardLayer"]
        module = module_class()
        
        # Create dummy input
        x = torch.randn(2, 8)
        
        try:
            output = module.forward(x)
            assert output is not None
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    def test_convenience_function(self):
        """Test the convenience function for code generation."""
        layer_def = LayerDef(
            name="ConvenienceLayer",
            params=[
                Param(name="w", param_type="TernaryTensor", shape=[2, 2])
            ],
            body=[]
        )
        
        code = generate_pytorch_code(layer_def)
        
        assert code is not None
        assert "class ConvenienceLayer" in code
    
    def test_parameter_extraction(self):
        """Test parameter extraction from LayerDef."""
        layer_def = LayerDef(
            name="ParamLayer",
            params=[
                Param(name="w1", param_type="TernaryTensor", shape=[3, 4]),
                Param(name="w2", param_type="TernaryTensor", shape=[4, 5]),
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[]
        )
        
        generator = PyTorchCodeGenerator()
        params = generator._extract_parameters(layer_def)
        
        # Should only extract TernaryTensor parameters
        assert len(params) == 2
        assert params[0]['name'] == 'w1'
        assert params[0]['shape'] == [3, 4]
        assert params[0]['numel'] == 12
        assert params[1]['name'] == 'w2'
        assert params[1]['shape'] == [4, 5]
        assert params[1]['numel'] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

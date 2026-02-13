"""
PyTorch Backend Code Generator
Converts Triton AST to executable PyTorch code.
"""

import os
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader
from compiler.ast.nodes import LayerDef, Param, TernaryTensor


class PyTorchCodeGenerator:
    """Generates PyTorch code from Triton AST."""
    
    def __init__(self):
        """Initialize code generator with Jinja2 templates."""
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def generate_module(self, layer_def: LayerDef) -> str:
        """
        Generate a complete PyTorch module from LayerDef AST node.
        
        Args:
            layer_def: LayerDef AST node containing layer definition
        
        Returns:
            Complete Python code as string with torch.nn.Module definition
        """
        template = self.env.get_template('module.py.jinja')
        
        # Extract parameters information
        parameters = self._extract_parameters(layer_def)
        
        # Generate forward method arguments (exclude ternary tensor parameters)
        forward_args = self._generate_forward_args(layer_def)
        
        # Generate forward body
        forward_body, return_value = self._generate_forward_body(layer_def)
        
        # Render template
        code = template.render(
            class_name=layer_def.name,
            parameters=parameters,
            forward_args=forward_args,
            forward_body=forward_body,
            return_value=return_value
        )
        
        return code
    
    def _extract_parameters(self, layer_def: LayerDef) -> List[Dict[str, Any]]:
        """
        Extract parameter information from layer definition.
        
        Args:
            layer_def: LayerDef AST node
        
        Returns:
            List of parameter dictionaries with name, shape, and numel
        """
        parameters = []
        
        for param in layer_def.params:
            if param.param_type == "TernaryTensor" and param.shape:
                numel = 1
                for dim in param.shape:
                    numel *= dim
                
                parameters.append({
                    'name': param.name,
                    'shape': param.shape,
                    'numel': numel
                })
        
        return parameters
    
    def _generate_forward_args(self, layer_def: LayerDef) -> str:
        """
        Generate forward method arguments.
        
        Args:
            layer_def: LayerDef AST node
        
        Returns:
            String of forward method arguments
        """
        # For now, use simple 'x' as input
        # In a full implementation, this would parse from layer_def.params
        args = []
        for param in layer_def.params:
            if param.param_type != "TernaryTensor":
                args.append(param.name)
        
        return ", ".join(args) if args else "x"
    
    def _generate_forward_body(self, layer_def: LayerDef) -> tuple[str, str]:
        """
        Generate forward method body from layer definition.
        
        Args:
            layer_def: LayerDef AST node
        
        Returns:
            Tuple of (forward_body, return_value)
        """
        # Simple default implementation
        # In a full implementation, this would traverse layer_def.body
        
        # For demonstration, generate a simple matrix multiplication
        if layer_def.body:
            # TODO: Implement proper AST traversal for body statements
            forward_body = "# TODO: Implement forward pass logic"
            return_value = "x"
        else:
            # Default simple forward pass
            forward_body = "# Simple forward pass\n        output = x"
            return_value = "output"
        
        return forward_body, return_value


def generate_pytorch_code(layer_def: LayerDef) -> str:
    """
    Convenience function to generate PyTorch code from LayerDef.
    
    Args:
        layer_def: LayerDef AST node
    
    Returns:
        Complete Python code as string
    """
    generator = PyTorchCodeGenerator()
    return generator.generate_module(layer_def)

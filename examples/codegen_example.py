"""
Example: PyTorch Backend Code Generation
Demonstrates the complete flow from Triton AST to executable PyTorch code.
"""

import sys
sys.path.insert(0, '.')

from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import generate_pytorch_code
import torch


def main():
    print("=" * 80)
    print("TRITON DSL: PyTorch Backend Code Generator Example")
    print("=" * 80)
    print()
    
    # Define a ternary neural network layer using AST nodes
    print("1. Creating LayerDef AST node for a ternary linear layer...")
    layer_def = LayerDef(
        name="TernaryLinearLayer",
        params=[
            Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
            Param(name="bias", param_type="TernaryTensor", shape=[256]),
            Param(name="x", param_type="Tensor", shape=None)
        ],
        body=[]
    )
    print("✓ LayerDef created")
    print(f"  - Layer name: {layer_def.name}")
    print(f"  - Parameters: {len(layer_def.params)}")
    for param in layer_def.params:
        print(f"    - {param.name}: {param.param_type} {param.shape if param.shape else ''}")
    print()
    
    # Generate PyTorch code
    print("2. Generating PyTorch code from AST...")
    code = generate_pytorch_code(layer_def)
    print("✓ PyTorch code generated")
    print(f"  - Code length: {len(code)} characters")
    print()
    
    # Display generated code
    print("3. Generated PyTorch Module Code:")
    print("-" * 80)
    print(code)
    print("-" * 80)
    print()
    
    # Execute the generated code
    print("4. Executing generated code...")
    namespace = {}
    exec(code, namespace)
    print("✓ Code executed successfully")
    print()
    
    # Instantiate the module
    print("5. Instantiating the generated module...")
    TernaryLinearLayer = namespace["TernaryLinearLayer"]
    layer = TernaryLinearLayer()
    print("✓ Module instantiated")
    print(f"  - Type: {type(layer)}")
    print(f"  - Is nn.Module: {isinstance(layer, torch.nn.Module)}")
    print(f"  - Parameters: {sum(p.numel() for p in layer.buffers())}")
    print()
    
    # Test forward pass
    print("6. Testing forward pass...")
    batch_size = 8
    input_dim = 128
    x = torch.randn(batch_size, input_dim)
    output = layer(x)
    print("✓ Forward pass successful")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Output shape: {output.shape}")
    print()
    
    # Verify ternary packing
    print("7. Verifying ternary weight packing...")
    from backend.pytorch.ops.pack import unpack_ternary
    
    # Check weights
    unpacked_weights = unpack_ternary(layer.weights_packed, layer._weights_numel)
    weights_reshaped = unpacked_weights.reshape(layer._weights_shape)
    print("✓ Weights unpacked successfully")
    print(f"  - Packed size: {layer.weights_packed.shape} (uint8)")
    print(f"  - Unpacked size: {weights_reshaped.shape} (int8)")
    print(f"  - Compression ratio: {weights_reshaped.numel() / layer.weights_packed.numel():.2f}x")
    print(f"  - All values in {{-1, 0, 1}}: {torch.all(torch.isin(unpacked_weights, torch.tensor([-1, 0, 1])))}")
    
    # Check bias
    unpacked_bias = unpack_ternary(layer.bias_packed, layer._bias_numel)
    bias_reshaped = unpacked_bias.reshape(layer._bias_shape)
    print("✓ Bias unpacked successfully")
    print(f"  - Packed size: {layer.bias_packed.shape} (uint8)")
    print(f"  - Unpacked size: {bias_reshaped.shape} (int8)")
    print(f"  - Compression ratio: {bias_reshaped.numel() / layer.bias_packed.numel():.2f}x")
    print(f"  - All values in {{-1, 0, 1}}: {torch.all(torch.isin(unpacked_bias, torch.tensor([-1, 0, 1])))}")
    print()
    
    print("=" * 80)
    print("✓ ALL TESTS PASSED - PyTorch Backend Code Generator Working!")
    print("=" * 80)


if __name__ == "__main__":
    main()

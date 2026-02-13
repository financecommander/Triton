# PyTorch Backend Code Generator - Quick Start Guide

This guide demonstrates how to use the PyTorch backend code generator to convert Triton AST nodes into executable PyTorch code.

## Installation

```bash
# Install the package with dependencies
pip install -e .
```

## Basic Usage

### Step 1: Create AST Nodes

```python
from compiler.ast.nodes import LayerDef, Param

# Define a ternary layer with weights
layer_def = LayerDef(
    name="MyTernaryLayer",
    params=[
        Param(name="weights", param_type="TernaryTensor", shape=[256, 512]),
        Param(name="bias", param_type="TernaryTensor", shape=[512]),
        Param(name="x", param_type="Tensor", shape=None)
    ],
    body=[]
)
```

### Step 2: Generate PyTorch Code

```python
from backend.pytorch.codegen import generate_pytorch_code

# Generate the code
pytorch_code = generate_pytorch_code(layer_def)

# Save to file or execute directly
with open('generated_layer.py', 'w') as f:
    f.write(pytorch_code)
```

### Step 3: Use Generated Module

```python
import torch

# Execute the generated code
namespace = {}
exec(pytorch_code, namespace)

# Instantiate the module
MyTernaryLayer = namespace["MyTernaryLayer"]
layer = MyTernaryLayer()

# Use like any PyTorch module
x = torch.randn(32, 256)  # Batch of 32, input dim 256
output = layer(x)
```

## Ternary Packing/Unpacking

The backend automatically handles efficient ternary storage:

```python
import torch
from backend.pytorch.ops.pack import pack_ternary, unpack_ternary

# Create ternary weights
weights = torch.tensor([1, -1, 0, 1, 0, -1, 1, 1], dtype=torch.int8)

# Pack: 8 values → 2 bytes (4x compression)
packed = pack_ternary(weights)
print(f"Original size: {weights.nbytes} bytes")
print(f"Packed size: {packed.nbytes} bytes")

# Unpack: lossless reconstruction
unpacked = unpack_ternary(packed, len(weights))
assert torch.equal(unpacked, weights)
```

## Features

### Generated Module Structure

```python
class GeneratedLayer(nn.Module):
    def __init__(self):
        # Packed ternary weights stored as buffers
        # Shape metadata stored as attributes
        
    def forward(self, x):
        # Weights unpacked on-demand
        # Ternary operations applied
        # Returns output
```

### Key Benefits

1. **4x Memory Compression**: 2-bit storage vs 8-bit
2. **Lossless**: Perfect reconstruction of ternary values
3. **PyTorch Native**: Standard nn.Module interface
4. **Auto-Generated**: No manual implementation needed

## Running Examples

```bash
# Run the full example
python examples/codegen_example.py

# Run tests
pytest tests/unit/test_pack.py -v
pytest tests/integration/test_codegen.py -v
```

## Advanced Usage

### Custom Parameter Initialization

Modify the template or generated code to customize initialization:

```python
# After generation, you can load pre-trained weights
layer = MyTernaryLayer()

# Load custom ternary weights
custom_weights = torch.tensor([...])  # Your ternary values
packed_custom = pack_ternary(custom_weights)
layer.weights_packed = packed_custom
```

### Integration with Training

```python
# Generated modules work with PyTorch training loops
optimizer = torch.optim.Adam(layer.parameters())

for epoch in range(num_epochs):
    output = layer(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## API Reference

### Code Generation

- **`PyTorchCodeGenerator()`**: Main code generator class
- **`generate_module(layer_def: LayerDef) -> str`**: Generate PyTorch code
- **`generate_pytorch_code(layer_def: LayerDef) -> str`**: Convenience function

### Packing Operations

- **`pack_ternary(tensor: torch.Tensor) -> torch.Tensor`**: Pack ternary values
- **`unpack_ternary(packed: torch.Tensor, numel: int) -> torch.Tensor`**: Unpack values

## Performance Notes

- **Memory**: 4x smaller than int8, 16x smaller than float32
- **Compute**: On-demand unpacking adds minimal overhead
- **Storage**: Efficient for model checkpoints and deployment

## Troubleshooting

### Import Errors

Make sure the package is installed:
```bash
pip install -e .
```

### Generated Code Errors

Verify your AST nodes are correctly formed:
```python
print(layer_def.to_dict())  # Check structure
```

### Packing Errors

Ensure values are in the ternary set {-1, 0, 1}:
```python
assert torch.all(torch.isin(tensor, torch.tensor([-1, 0, 1])))
```

## Next Steps

- Read the [Technical Documentation](../backend/pytorch/README.md)
- Explore [Test Examples](../tests/integration/test_codegen.py)
- Check out the [Complete Example](../examples/codegen_example.py)

## Support

For issues or questions, refer to the main repository README or open an issue on GitHub.

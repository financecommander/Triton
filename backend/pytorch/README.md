# PyTorch Backend Code Generator

This directory contains the PyTorch backend for the Triton DSL compiler, which translates Triton AST nodes into executable PyTorch code with optimized ternary weight storage.

## Components

### 1. Code Generator (`codegen.py`)

The `PyTorchCodeGenerator` class converts Triton AST nodes into PyTorch `nn.Module` code:

- **`generate_module(layer_def: LayerDef) -> str`**: Main entry point that takes a `LayerDef` AST node and generates complete PyTorch code
- Uses Jinja2 templates for clean code generation
- Automatically handles ternary weight packing and unpacking
- Generates valid, executable Python code

**Example:**
```python
from compiler.ast.nodes import LayerDef, Param
from backend.pytorch.codegen import generate_pytorch_code

layer_def = LayerDef(
    name="TernaryLayer",
    params=[
        Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
    ],
    body=[]
)

code = generate_pytorch_code(layer_def)
# Generates complete torch.nn.Module code
```

### 2. Packing Operations (`ops/pack.py`)

Efficient 2-bit encoding for ternary values:

- **`pack_ternary(tensor: torch.Tensor) -> torch.Tensor`**: Packs 4 ternary values into 1 byte
- **`unpack_ternary(packed: torch.Tensor, numel: int) -> torch.Tensor`**: Unpacks packed values

**Encoding scheme:**
- `-1` → `00` (0)
- `0` → `01` (1)
- `1` → `10` (2)

**Compression:** 4x memory reduction (2-bit vs 8-bit storage)

**Example:**
```python
import torch
from backend.pytorch.ops.pack import pack_ternary, unpack_ternary

# Create ternary tensor
tensor = torch.tensor([1, -1, 0, 1, 0, -1], dtype=torch.int8)

# Pack (6 values → 2 bytes)
packed = pack_ternary(tensor)  # shape: [2], dtype: uint8

# Unpack
unpacked = unpack_ternary(packed, 6)  # shape: [6], dtype: int8
assert torch.equal(unpacked, tensor)  # Lossless
```

### 3. Templates (`templates/`)

Jinja2 templates for code generation:

- **`module.py.jinja`**: Template for torch.nn.Module generation
  - Class definition with proper inheritance
  - `__init__` method with packed ternary weights
  - `forward()` method with unpacking logic
  - Parameter registration for training

## Generated Code Structure

Generated PyTorch modules follow this structure:

```python
import torch
import torch.nn as nn
from backend.pytorch.ops.pack import pack_ternary, unpack_ternary

class GeneratedLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Store shapes and sizes
        self._weights_shape = [128, 256]
        self._weights_numel = 32768
        
        # Initialize and pack ternary weights
        init_tensor = torch.randint(-1, 2, (32768,), dtype=torch.int8)
        packed = pack_ternary(init_tensor)
        self.register_buffer('weights_packed', packed)
    
    def forward(self, x):
        # Unpack weights on-demand
        weights = unpack_ternary(
            self.weights_packed, 
            self._weights_numel
        ).reshape(self._weights_shape)
        
        # Forward computation
        # ...
        return output
```

## Key Features

1. **Memory Efficiency**: 4x compression through 2-bit packing
2. **Valid Python Output**: Generated code is syntactically correct and executable
3. **PyTorch Integration**: Seamless nn.Module integration with buffer registration
4. **Template-Based**: Clean, maintainable code generation using Jinja2
5. **Type Safety**: Enforces ternary constraints {-1, 0, 1}
6. **Lossless**: Pack/unpack operations preserve values exactly

## Testing

Tests are located in:
- `tests/unit/test_pack.py` - Pack/unpack operations
- `tests/integration/test_codegen.py` - End-to-end code generation

Run tests:
```bash
pytest tests/unit/test_pack.py -v
pytest tests/integration/test_codegen.py -v
```

## Example

See `examples/codegen_example.py` for a complete demonstration:

```bash
python examples/codegen_example.py
```

## Dependencies

- `torch>=2.1.0` - PyTorch for tensors and nn.Module
- `jinja2>=3.0.0` - Template engine for code generation
- `numpy>=1.24.0` - Numerical operations

## Performance

Compared to FP32 storage:
- **Memory**: 4x smaller (2-bit vs 32-bit)
- **Storage**: Efficient uint8 packing
- **Runtime**: On-demand unpacking for forward pass

## Future Enhancements

- [x] CUDA kernel integration for GPU acceleration
- [ ] Sparse computation optimization (skip zeros)
- [ ] Training support with ternary gradients
- [ ] Quantization-aware training hooks

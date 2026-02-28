# Backend API Reference

This section provides detailed API documentation for the Triton DSL backend components that generate executable code from compiled AST.

## Overview

The Triton backend system supports multiple target platforms:

```
Typed AST
    ↓
Backend Selection
    ├── PyTorch Backend → Python/PyTorch Module
    ├── ONNX Export → .onnx Model File
    ├── TensorFlow Lite → .tflite Model
    └── Custom Backend → User-defined Target
```

Each backend transforms the validated AST into executable code optimized for its target platform.

## PyTorch Backend

.. automodule:: backend.pytorch.codegen
   :members:
   :undoc-members:
   :show-inheritance:

### PyTorchCodeGenerator

.. autoclass:: backend.pytorch.codegen.PyTorchCodeGenerator
   :members:
   :undoc-members:

The code generator converts Triton DSL layer definitions into executable PyTorch modules.

#### Example: Basic Code Generation

```python
from compiler.parser.triton_parser import TritonParser
from compiler.type_checker import TypeChecker
from backend.pytorch.codegen import PyTorchCodeGenerator

# Parse Triton DSL source
source = """
layer LinearClassifier {
    param weights: TernaryTensor[784, 10]
    param bias: Tensor[10]
    
    fn forward(x: Tensor[N, 784]) -> Tensor[N, 10] {
        return x @ weights + bias
    }
}
"""

parser = TritonParser()
ast = parser.parse(source)
type_checker = TypeChecker()
typed_ast = type_checker.check(ast)

# Generate PyTorch code
codegen = PyTorchCodeGenerator()
pytorch_code = codegen.generate_module(typed_ast.layers[0])
print(pytorch_code)
```

#### Generated PyTorch Module Structure

```python
import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Ternary parameters stored as packed uint8
        self.register_buffer('weights', torch.zeros(1960, dtype=torch.uint8))
        self.bias = nn.Parameter(torch.zeros(10))
    
    def forward(self, x):
        # Unpack and apply ternary operations
        weights_unpacked = unpack_ternary(self.weights, (784, 10))
        return torch.matmul(x, weights_unpacked) + self.bias
```

### Ternary Tensor Operations

.. automodule:: backend.pytorch.ternary_tensor
   :members:
   :undoc-members:
   :show-inheritance:

#### TernaryTensor Class

.. autoclass:: backend.pytorch.ternary_tensor.TernaryTensor
   :members:
   :undoc-members:

Efficient storage and operations for ternary-valued tensors.

#### Memory Layout

Ternary values {-1, 0, 1} are encoded using 2 bits per value:
- `00`: 0
- `01`: +1
- `10`: -1
- `11`: Reserved

Four values are packed into each uint8 byte, achieving 4x memory compression.

#### Example: Ternary Tensor Creation

```python
from backend.pytorch.ternary_tensor import TernaryTensor
import torch

# Create regular tensor
weights = torch.randn(256, 128)

# Convert to ternary
ternary_weights = TernaryTensor(weights)

print(f"Original size: {weights.numel() * 4} bytes")  # float32
print(f"Ternary size: {ternary_weights.numel() // 4} bytes")  # 2-bit packed
print(f"Compression ratio: {weights.numel() * 4 / (ternary_weights.numel() // 4):.1f}x")

# Convert back to float for computation
float_weights = ternary_weights.to_float()
```

### Quantization Operations

.. automodule:: backend.pytorch.ops
   :members:
   :undoc-members:
   :show-inheritance:

#### Ternary Quantization Functions

```python
from backend.pytorch.ops.quantization import ternarize, ternarize_stochastic

# Deterministic ternarization
tensor = torch.randn(100, 100)
ternary = ternarize(tensor, threshold=0.05)

# Stochastic ternarization (better gradient flow)
ternary_stochastic = ternarize_stochastic(tensor)
```

#### Ternary Matrix Multiplication

.. autofunction:: backend.pytorch.ternary_tensor.ternary_matmul

Optimized matrix multiplication for ternary tensors with zero-skipping.

```python
from backend.pytorch.ternary_tensor import ternary_matmul
import torch

A = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.float32)
B = torch.tensor([[1, -1], [0, 1], [-1, 0]], dtype=torch.float32)

# Optimized ternary multiplication
result = ternary_matmul(A, B, a_ternary=True, b_ternary=True)
print(result)
# Output: [[-2, 1], [1, -2]]
```

### Ternary Models

.. automodule:: backend.pytorch.ternary_models
   :members:
   :undoc-members:
   :show-inheritance:

Pre-built ternary model architectures.

#### Example: TernaryLinear Layer

```python
from backend.pytorch.ternary_models import TernaryLinear
import torch

layer = TernaryLinear(in_features=784, out_features=256)
x = torch.randn(32, 784)  # batch_size=32
output = layer(x)
print(output.shape)  # torch.Size([32, 256])

# Access quantized weights
ternary_weights = layer.get_ternary_weights()
print(f"Unique values: {ternary_weights.unique()}")  # [-1, 0, 1]
```

## ONNX Export

.. automodule:: backend.pytorch.export.onnx_exporter
   :members:
   :undoc-members:
   :show-inheritance:

### Exporting to ONNX

.. autofunction:: backend.pytorch.export.onnx_exporter.export_to_onnx

Export ternary models to ONNX format for deployment on edge devices.

#### Example: Export ResNet-18 with Ternary Quantization

```python
from backend.pytorch.export.onnx_exporter import export_to_onnx
from models.resnet18.ternary_resnet18 import ternary_resnet18
from pathlib import Path

# Create and load trained model
model = ternary_resnet18(num_classes=10)
model.load_state_dict(torch.load('resnet18_ternary.pth'))
model.eval()

# Export to ONNX
success = export_to_onnx(
    model=model,
    output_path=Path("resnet18_ternary.onnx"),
    input_shape=(1, 3, 32, 32),
    input_names=['image'],
    output_names=['logits'],
    opset_version=13,
    dynamic_axes={
        'image': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    },
    verbose=True
)

if success:
    print("ONNX export successful!")
```

#### ONNX Model Verification

```python
import onnx
import onnxruntime as ort
import numpy as np

# Load and verify ONNX model
onnx_model = onnx.load("resnet18_ternary.onnx")
onnx.checker.check_model(onnx_model)

# Run inference
session = ort.InferenceSession("resnet18_ternary.onnx")
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
outputs = session.run(None, {'image': input_data})
print(f"Output shape: {outputs[0].shape}")
```

### ONNX Optimization

.. autofunction:: backend.pytorch.export.onnx_exporter.optimize_onnx_model

Apply graph optimizations to exported ONNX models.

```python
from backend.pytorch.export.onnx_exporter import optimize_onnx_model

# Optimize ONNX model for inference
optimize_onnx_model(
    input_path="resnet18_ternary.onnx",
    output_path="resnet18_ternary_optimized.onnx",
    optimization_level="all"  # basic, extended, all
)
```

## TensorFlow Lite Backend

Export models to TensorFlow Lite for mobile and embedded deployment.

### Example: TFLite Conversion

```python
from backend.tensorflow.tflite_exporter import export_to_tflite
import torch

# Load PyTorch model
model = torch.load('model.pth')
model.eval()

# Convert to TFLite
export_to_tflite(
    model=model,
    output_path="model.tflite",
    input_shape=(1, 3, 224, 224),
    quantize=True,  # Apply int8 quantization
    optimize_for_size=True
)
```

### TFLite Quantization Options

```python
from backend.tensorflow.tflite_exporter import TFLiteQuantizationConfig

config = TFLiteQuantizationConfig(
    mode='int8',  # int8, float16, or dynamic
    representative_dataset=train_loader,  # For calibration
    inference_input_type='uint8',
    inference_output_type='uint8'
)

export_to_tflite(model, "model_int8.tflite", quantization_config=config)
```

## HuggingFace Hub Integration

.. automodule:: backend.pytorch.export.huggingface_hub
   :members:
   :undoc-members:
   :show-inheritance:

### Publishing Models

.. autofunction:: backend.pytorch.export.huggingface_hub.push_to_hub

Share trained ternary models on HuggingFace Hub.

#### Example: Upload Model to HuggingFace

```python
from backend.pytorch.export.huggingface_hub import push_to_hub
import torch

# Load trained model
model = torch.load('mnist_ternary_best.pth')

# Upload to HuggingFace Hub
push_to_hub(
    model=model,
    repo_id="username/mnist-ternary",
    commit_message="Add ternary MNIST classifier",
    private=False,
    token="hf_...",  # HuggingFace API token
    tags=["ternary", "quantization", "mnist", "image-classification"]
)
```

### Loading from HuggingFace

```python
from backend.pytorch.export.huggingface_hub import load_from_hub

# Download and load model
model = load_from_hub(
    repo_id="username/mnist-ternary",
    revision="main"
)
model.eval()

# Use for inference
with torch.no_grad():
    output = model(input_image)
```

## GitHub Model Publishing

.. automodule:: backend.pytorch.export.github_publisher
   :members:
   :undoc-members:
   :show-inheritance:

### Publishing to GitHub Releases

```python
from backend.pytorch.export.github_publisher import publish_to_github

# Publish model as GitHub release
publish_to_github(
    model_path="resnet18_ternary.onnx",
    repo="username/triton-models",
    tag="v1.0.0",
    release_name="ResNet-18 Ternary Quantized",
    description="CIFAR-10 trained ResNet-18 with ternary quantization",
    token="ghp_..."  # GitHub personal access token
)
```

## Custom Backend Development

Create custom backends for specialized hardware or frameworks.

### Backend Interface

.. autoclass:: backend.base.BackendBase
   :members:
   :undoc-members:

All backends must implement the `BackendBase` interface:

```python
from backend.base import BackendBase
from compiler.ast.nodes import LayerDef
from typing import Any

class CustomBackend(BackendBase):
    """Custom backend for specialized hardware."""
    
    def generate_code(self, layer_def: LayerDef) -> str:
        """Generate code for target platform."""
        # Implement code generation logic
        pass
    
    def compile(self, code: str) -> Any:
        """Compile generated code to executable form."""
        # Implement compilation logic
        pass
    
    def optimize(self, compiled_model: Any) -> Any:
        """Apply backend-specific optimizations."""
        # Implement optimization passes
        pass
```

### Example: Custom FPGA Backend

```python
from backend.base import BackendBase
from compiler.ast.nodes import LayerDef

class FPGABackend(BackendBase):
    """Backend targeting FPGA synthesis."""
    
    def __init__(self, target_device: str = "xcvu9p"):
        self.target_device = target_device
    
    def generate_code(self, layer_def: LayerDef) -> str:
        """Generate Verilog/VHDL for FPGA."""
        code_lines = []
        code_lines.append(f"// Target: {self.target_device}")
        code_lines.append(f"module {layer_def.name} (")
        
        # Generate port declarations
        for param in layer_def.params:
            if param.param_type == "TernaryTensor":
                code_lines.append(f"    input wire [{self._bit_width(param)}:0] {param.name},")
        
        code_lines.append("    ...")
        code_lines.append("endmodule")
        
        return "\n".join(code_lines)
    
    def _bit_width(self, param) -> int:
        """Calculate bit width for ternary parameter."""
        if param.shape:
            numel = 1
            for dim in param.shape:
                numel *= dim
            # 2 bits per ternary value
            return numel * 2 - 1
        return 0
```

### Registering Custom Backends

```python
from backend.registry import register_backend

# Register custom backend
register_backend('fpga', FPGABackend)

# Use in compilation
from compiler.driver import compile_model

compiled = compile_model(
    source_file="model.tri",
    backend='fpga',
    backend_options={'target_device': 'xcvu9p'}
)
```

## Backend Utilities

### Model Size Analysis

```python
from backend.utils.analysis import analyze_model_size
import torch

model = torch.load('model.pth')

analysis = analyze_model_size(model)
print(f"Total parameters: {analysis['total_params']:,}")
print(f"Ternary parameters: {analysis['ternary_params']:,}")
print(f"Float parameters: {analysis['float_params']:,}")
print(f"Model size (float32): {analysis['size_float32_mb']:.2f} MB")
print(f"Model size (ternary): {analysis['size_ternary_mb']:.2f} MB")
print(f"Compression ratio: {analysis['compression_ratio']:.1f}x")
```

### Performance Profiling

```python
from backend.utils.profiler import profile_model
import torch

model = torch.load('model.pth')
input_tensor = torch.randn(1, 3, 224, 224)

# Profile model execution
profile = profile_model(model, input_tensor, num_iterations=100)

print(f"Average inference time: {profile['avg_time_ms']:.2f} ms")
print(f"Throughput: {profile['throughput_fps']:.1f} FPS")
print(f"Memory usage: {profile['memory_mb']:.2f} MB")

# Layer-by-layer breakdown
for layer_name, layer_stats in profile['layer_times'].items():
    print(f"{layer_name}: {layer_stats['time_ms']:.2f} ms")
```

## Best Practices

### Memory-Efficient Training

```python
import torch
from backend.pytorch.ternary_models import TernaryResNet18

# Use mixed precision training
model = TernaryResNet18(num_classes=10)
scaler = torch.cuda.amp.GradScaler()

for batch in train_loader:
    with torch.cuda.amp.autocast():
        output = model(batch['image'])
        loss = criterion(output, batch['label'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class TernaryResNetWithCheckpointing(TernaryResNet18):
    def forward(self, x):
        x = self.conv1(x)
        # Checkpoint memory-intensive blocks
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        x = checkpoint(self.layer3, x)
        x = checkpoint(self.layer4, x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

### Deployment Optimization

```python
from backend.pytorch.export import optimize_for_mobile
import torch

model = torch.load('model.pth')
model.eval()

# Optimize for mobile deployment
optimized_model = optimize_for_mobile(
    model,
    input_shape=(1, 3, 224, 224),
    backend='onnx',  # or 'tflite'
    optimization_options={
        'constant_folding': True,
        'dead_code_elimination': True,
        'operator_fusion': True,
        'quantization': 'dynamic'  # dynamic, static, or None
    }
)

# Save optimized model
torch.save(optimized_model, 'model_optimized.pth')
```

## See Also

- [Compiler API](compiler.md) - AST compilation pipeline
- [Kernels API](kernels.md) - Low-level kernel implementations
- [Examples](examples.md) - Complete usage examples

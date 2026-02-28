# Getting Started with Triton DSL

Welcome to Triton DSL! This guide will help you get started with building and deploying ternary neural networks.

## What is Triton DSL?

Triton DSL is a high-performance Domain-Specific Language designed specifically for Ternary Neural Networks (TNNs). It enforces ternary constraints (`{-1, 0, 1}`) at the syntax level, enabling:

- **20-40% memory density improvements** over standard FP32 representations
- **2-3x faster inference** through sparse computation and zero-skipping
- **Hardware optimization** with 2-bit packed storage and custom CUDA kernels
- **Seamless PyTorch integration** for easy adoption

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.1.0 or higher
- CUDA 11.8+ (optional, for GPU acceleration)

### Install via pip

```bash
pip install triton-dsl
```

### Install via conda

```bash
conda create -n triton python=3.10
conda activate triton
conda install pytorch torchvision -c pytorch
pip install triton-dsl
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/financecommander/Triton.git
cd Triton

# Install in development mode
pip install -e .

# Install optional dependencies
pip install -e ".[dev,cuda,export]"
```

## Your First Model in 5 Minutes

Let's create a simple ternary neural network for MNIST classification.

### 1. Write the Triton DSL Code

Create a file `mnist_model.tri`:

```triton
layer TernaryLinear(weights: TernaryTensor<trit, [784, 128]>, 
                     bias: TernaryTensor<trit, [128]>,
                     x: Tensor<float32, [?, 784]>) -> Tensor<float32, [?, 128]> {
    let result = x @ weights + bias
    return result
}

layer TernaryNet(w1: TernaryTensor<trit, [784, 128]>,
                 b1: TernaryTensor<trit, [128]>,
                 w2: TernaryTensor<trit, [128, 10]>,
                 b2: TernaryTensor<trit, [10]>,
                 x: Tensor<float32, [?, 784]>) -> Tensor<float32, [?, 10]> {
    let h1 = TernaryLinear(w1, b1, x)
    let a1 = relu(h1)
    let out = TernaryLinear(w2, b2, a1)
    return out
}
```

### 2. Compile to PyTorch

```python
from compiler.driver import compile_model

# Compile the Triton DSL code to PyTorch
model = compile_model(
    'mnist_model.tri',
    backend='pytorch',
    optimization_level='O2'
)

print(f"Model compiled successfully!")
print(f"Model type: {type(model)}")
```

### 3. Train the Model

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Set up training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(5):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten the input
        data = data.view(-1, 784)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    print(f'Epoch {epoch} completed, Average Loss: {total_loss/len(train_loader):.4f}')

print("Training complete!")
```

### 4. Export and Deploy

```python
from backend.pytorch.export import export_to_onnx

# Export to ONNX format
export_to_onnx(
    model,
    'mnist_ternary.onnx',
    input_shape=(1, 784),
    opset_version=14
)

print("Model exported to ONNX successfully!")
```

## Common Patterns

### Quantization Methods

Triton supports multiple quantization methods:

#### Deterministic Quantization

```python
from backend.pytorch.ops.quantize import quantize

# Threshold-based quantization
weights_ternary = quantize(weights, method='deterministic', threshold=0.33)
```

#### Stochastic Quantization

```python
# Probabilistic quantization for better gradient flow
weights_ternary = quantize(weights, method='stochastic', threshold=0.33)
```

### Custom Layers

Define custom ternary layers in the DSL:

```triton
layer TernaryConv2D(
    weights: TernaryTensor<trit, [32, 3, 3, 3]>,
    bias: TernaryTensor<trit, [32]>,
    x: Tensor<float32, [?, 3, 32, 32]>
) -> Tensor<float32, [?, 32, 30, 30]> {
    let conv_out = conv2d(x, weights, stride=1, padding=0)
    let result = conv_out + bias
    return result
}
```

### Mixed Precision Training

Combine ternary and full-precision layers:

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ternary_layers = compile_model('ternary_layers.tri')
        self.fp32_classifier = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.ternary_layers(x)
        x = self.fp32_classifier(x)
        return x
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'compiler'`

**Solution:** Make sure you've installed the package correctly:
```bash
pip install -e .
```

#### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:** Reduce batch size or use gradient accumulation:
```python
# Reduce batch size
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Or use gradient accumulation
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Type Checker Errors

**Problem:** `TypeError: Expected TernaryTensor but got Tensor`

**Solution:** Ensure your model definition matches the expected types:
```triton
# Correct: Use TernaryTensor for weights
layer MyLayer(weights: TernaryTensor<trit, [128, 256]>, ...)

# Incorrect: Using regular Tensor for ternary weights
layer MyLayer(weights: Tensor<float32, [128, 256]>, ...)
```

#### Compilation Errors

**Problem:** Compilation fails with syntax errors

**Solution:** Check your DSL syntax:
```bash
# Use the compiler with verbose mode
python -m compiler.driver --input model.tri --verbose
```

### Performance Issues

#### Slow Training

1. **Enable CUDA acceleration:**
   ```python
   model = model.cuda()
   data = data.cuda()
   ```

2. **Use optimized quantization:**
   ```python
   # Deterministic is faster than stochastic
   quantize(x, method='deterministic')
   ```

3. **Enable compilation optimization:**
   ```python
   compile_model('model.tri', optimization_level='O3')
   ```

#### High Memory Usage

1. **Use gradient checkpointing:**
   ```python
   from torch.utils.checkpoint import checkpoint
   
   def forward_with_checkpointing(self, x):
       return checkpoint(self.layer, x)
   ```

2. **Clear cache regularly:**
   ```python
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   ```

### Getting Help

- **Documentation:** Check the [full documentation](https://financecommander.github.io/Triton/)
- **Issues:** Report bugs on [GitHub Issues](https://github.com/financecommander/Triton/issues)
- **Examples:** See more examples in the `examples/` directory
- **Community:** Join discussions on GitHub Discussions

## Next Steps

Now that you've created your first ternary model, explore:

1. **[Tutorial Series](tutorials/01_basic_model.md)** - Comprehensive step-by-step tutorials
2. **[DSL Reference](dsl/language_spec.md)** - Complete language specification
3. **[API Documentation](api/compiler.md)** - Detailed API reference
4. **[Architecture Guide](architecture/compiler_pipeline.md)** - Understanding the internals

## Quick Reference

### Common Commands

```bash
# Compile a model
python -m compiler.driver --input model.tri --output model.py

# Run tests
pytest tests/

# Check model memory usage
python -m tools.memory_profiler model.py

# Export to ONNX
python -m backend.pytorch.export.onnx_exporter model.py output.onnx
```

### Useful Links

- [GitHub Repository](https://github.com/financecommander/Triton)
- [PyPI Package](https://pypi.org/project/triton-dsl/)
- [Documentation](https://financecommander.github.io/Triton/)
- [Examples](https://github.com/financecommander/Triton/tree/main/examples)

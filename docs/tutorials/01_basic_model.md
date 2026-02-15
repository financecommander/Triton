# Tutorial 1: Building Your First Ternary Model

In this tutorial, you'll learn how to build a basic ternary neural network using Triton DSL. We'll start with a simple feedforward network for MNIST digit classification.

## Learning Objectives

By the end of this tutorial, you will:
- Understand the basic syntax of Triton DSL
- Create a simple ternary neural network
- Compile and train your model
- Evaluate model performance

## Prerequisites

- Python 3.10+
- Triton DSL installed (see [Getting Started](../getting_started.md))
- Basic understanding of neural networks
- Familiarity with PyTorch (helpful but not required)

## Step 1: Understanding Ternary Neural Networks

### What are Ternary Neural Networks?

Ternary Neural Networks (TNNs) constrain weights to three values: **{-1, 0, 1}**. This offers several advantages:

- **Memory Efficiency**: 2 bits per weight vs. 32 bits for FP32 (16x compression)
- **Faster Inference**: Sparse computation and zero-skipping
- **Hardware Friendly**: Simple operations suitable for edge devices

### The Trade-off

While TNNs are efficient, they require careful training to maintain accuracy:

```
FP32 Model: High accuracy, High memory, Slow inference
Ternary Model: Good accuracy, Low memory, Fast inference
```

## Step 2: Writing Your First Triton DSL Program

Create a file named `simple_mnist.tri`:

```triton
# Simple MNIST classifier with ternary weights

layer TernaryFC(
    weights: TernaryTensor<trit, [784, 128]>,
    bias: TernaryTensor<trit, [128]>,
    x: Tensor<float32, [?, 784]>
) -> Tensor<float32, [?, 128]> {
    # Matrix multiplication with ternary weights
    let result = x @ weights
    let output = result + bias
    return output
}

layer SimpleMNIST(
    w1: TernaryTensor<trit, [784, 128]>,
    b1: TernaryTensor<trit, [128]>,
    w2: TernaryTensor<trit, [128, 64]>,
    b2: TernaryTensor<trit, [64]>,
    w3: TernaryTensor<trit, [64, 10]>,
    b3: TernaryTensor<trit, [10]>,
    x: Tensor<float32, [?, 784]>
) -> Tensor<float32, [?, 10]> {
    # Layer 1
    let h1 = TernaryFC(w1, b1, x)
    let a1 = relu(h1)
    
    # Layer 2
    let h2 = TernaryFC(w2, b2, a1)
    let a2 = relu(h2)
    
    # Layer 3 (output)
    let h3 = TernaryFC(w3, b3, a2)
    return h3
}
```

### Understanding the Code

Let's break down the key components:

#### Layer Definition

```triton
layer TernaryFC(...)  -> Tensor<...> {
    ...
}
```

The `layer` keyword defines a reusable neural network layer.

#### Type Annotations

```triton
weights: TernaryTensor<trit, [784, 128]>
```

This declares:
- `weights`: parameter name
- `TernaryTensor<trit, ...>`: a tensor with ternary values
- `[784, 128]`: shape specification

#### Operations

```triton
let result = x @ weights  # Matrix multiplication
let output = result + bias  # Element-wise addition
```

Standard mathematical operations work as expected.

## Step 3: Compiling to PyTorch

Create a Python script `train_simple_mnist.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from compiler.driver import compile_model

# Compile the Triton DSL model to PyTorch
print("Compiling model...")
model = compile_model(
    'simple_mnist.tri',
    backend='pytorch',
    optimization_level='O2'
)

print(f"Model compiled successfully!")
print(f"Model architecture:\n{model}")

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")
```

### Compilation Options

- `backend='pytorch'`: Generate PyTorch code
- `optimization_level='O2'`: Balanced optimization
  - `O0`: No optimization (debug)
  - `O1`: Basic optimization
  - `O2`: Recommended (balanced)
  - `O3`: Aggressive optimization

## Step 4: Preparing the Data

Add data loading to your script:

```python
# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Load datasets
print("Loading MNIST dataset...")
train_dataset = datasets.MNIST(
    './data', 
    train=True, 
    download=True, 
    transform=transform
)

test_dataset = datasets.MNIST(
    './data', 
    train=False, 
    transform=transform
)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=128, 
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1000, 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
```

## Step 5: Training the Model

Add the training loop:

```python
# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Flatten images: [batch, 1, 28, 28] -> [batch, 784]
        data = data.view(-1, 784)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f'Epoch {epoch} - Training Loss: {avg_loss:.4f}, '
          f'Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# Train for multiple epochs
num_epochs = 10
print(f"\nStarting training for {num_epochs} epochs...\n")

for epoch in range(1, num_epochs + 1):
    train_epoch(model, device, train_loader, optimizer, criterion, epoch)

print("\nTraining complete!")
```

## Step 6: Evaluating the Model

Add evaluation code:

```python
def evaluate(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest Results:')
    print(f'Average Loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy:.2f}% ({correct}/{len(test_loader.dataset)})')
    
    return test_loss, accuracy

# Evaluate the trained model
evaluate(model, device, test_loader, criterion)
```

## Step 7: Running the Complete Example

Save and run your script:

```bash
python train_simple_mnist.py
```

Expected output:
```
Compiling model...
Model compiled successfully!
Using device: cuda

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Starting training for 10 epochs...

Epoch 1, Batch 0/469, Loss: 2.3156
Epoch 1, Batch 100/469, Loss: 0.8234
...
Epoch 1 - Training Loss: 0.6234, Accuracy: 82.45%

...

Epoch 10 - Training Loss: 0.1234, Accuracy: 96.78%

Test Results:
Average Loss: 0.1456
Accuracy: 95.23% (9523/10000)
```

## Understanding Model Performance

### Memory Comparison

```python
def compare_memory_usage():
    import sys
    
    # FP32 model
    fp32_params = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    
    # Ternary model (2 bits per weight)
    ternary_params = sum(p.numel() * 0.25 for p in model.parameters())  # 0.25 bytes (2 bits)
    
    print(f"\nMemory Usage Comparison:")
    print(f"FP32 Model: {fp32_params / 1024:.2f} KB")
    print(f"Ternary Model: {ternary_params / 1024:.2f} KB")
    print(f"Compression Ratio: {fp32_params / ternary_params:.1f}x")

compare_memory_usage()
```

### Accuracy vs. Efficiency Trade-off

| Model Type | Accuracy | Memory | Inference Speed |
|------------|----------|--------|-----------------|
| FP32       | ~98%     | 1.0x   | 1.0x            |
| Ternary    | ~95%     | 0.06x  | 2-3x            |

## Common Issues and Solutions

### Issue 1: Low Initial Accuracy

**Problem**: Model accuracy is very low in early epochs.

**Solution**: Use a warm-up learning rate schedule:

```python
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(epoch):
    if epoch < 3:
        return (epoch + 1) / 3  # Warm-up
    return 1.0

scheduler = LambdaLR(optimizer, lr_lambda)

# In training loop
for epoch in range(1, num_epochs + 1):
    train_epoch(model, device, train_loader, optimizer, criterion, epoch)
    scheduler.step()
```

### Issue 2: Gradient Instability

**Problem**: Loss fluctuates wildly during training.

**Solution**: Use gradient clipping:

```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Exercise: Improve the Model

Try these improvements:

1. **Add batch normalization** between layers
2. **Experiment with different architectures** (e.g., more layers, different sizes)
3. **Try different optimizers** (SGD with momentum, AdamW)
4. **Implement learning rate scheduling**

## Next Steps

Congratulations! You've built your first ternary neural network. Continue to:

- [Tutorial 2: Quantization Techniques](02_quantization.md) - Learn advanced quantization methods
- [Tutorial 3: Custom Layers](03_custom_layers.md) - Build custom ternary layers
- [DSL Reference](../dsl/language_spec.md) - Comprehensive language guide

## Summary

In this tutorial, you learned:
- ✓ Basic Triton DSL syntax
- ✓ How to define ternary layers
- ✓ Compiling DSL to PyTorch
- ✓ Training and evaluating ternary models
- ✓ Understanding the accuracy/efficiency trade-off

## Resources

- [Example Code](https://github.com/financecommander/Triton/tree/main/examples)
- [API Reference](../api/compiler.md)
- [Troubleshooting Guide](../getting_started.md#troubleshooting)

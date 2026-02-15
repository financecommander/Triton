# Examples API Reference

This section provides detailed documentation for example models, utilities, and training scripts in the Triton DSL repository.

## Overview

The examples module provides:

```
examples/
    ├── mnist_ternary.py          # Complete MNIST training example
    ├── cifar10_training_examples.sh  # CIFAR-10 training scripts
    ├── codegen_example.py        # Code generation demonstration
    ├── export_and_publish_example.py  # Model export workflows
    └── test_mnist_ternary.py     # Unit tests for MNIST example
```

## MNIST Ternary Example

.. automodule:: examples.mnist_ternary
   :members:
   :undoc-members:
   :show-inheritance:

### Complete Training Pipeline

The MNIST example demonstrates end-to-end ternary neural network training.

#### Model Architecture

```python
from examples.mnist_ternary import TernaryMNISTNet
import torch

# Create model
model = TernaryMNISTNet(
    input_size=784,
    hidden_sizes=[256, 128],
    num_classes=10,
    dropout=0.2
)

print(model)
# TernaryMNISTNet(
#   (layers): ModuleList(
#     (0): TernaryLinear(in_features=784, out_features=256)
#     (1): TernaryLinear(in_features=256, out_features=128)
#   )
#   (output): Linear(in_features=128, out_features=10)
# )
```

#### TernaryMNISTNet Class

.. autoclass:: examples.mnist_ternary.TernaryMNISTNet
   :members:
   :undoc-members:

The network uses ternary quantization for all hidden layers:

```python
class TernaryMNISTNet(nn.Module):
    """Ternary quantized network for MNIST classification."""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], 
                 num_classes=10, dropout=0.2):
        super().__init__()
        
        # Build ternary hidden layers
        layers = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(TernaryLinear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_size
        
        self.layers = nn.ModuleList(layers)
        
        # Final classification layer (float32 for numerical stability)
        self.output = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
```

### Training Function

.. autofunction:: examples.mnist_ternary.train_epoch

```python
from examples.mnist_ternary import train_epoch
import torch.optim as optim

# Setup
model = TernaryMNISTNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train one epoch
train_stats = train_epoch(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    epoch=1
)

print(f"Loss: {train_stats['loss']:.4f}")
print(f"Accuracy: {train_stats['accuracy']:.2%}")
print(f"Time: {train_stats['time']:.2f}s")
```

### Evaluation Function

.. autofunction:: examples.mnist_ternary.evaluate

```python
from examples.mnist_ternary import evaluate

# Evaluate model
test_stats = evaluate(
    model=model,
    test_loader=test_loader,
    criterion=criterion,
    device='cuda'
)

print(f"Test Loss: {test_stats['loss']:.4f}")
print(f"Test Accuracy: {test_stats['accuracy']:.2%}")
print(f"Inference time: {test_stats['inference_time_ms']:.2f}ms per batch")
```

### Command-Line Interface

```bash
# Basic training
python examples/mnist_ternary.py

# Custom hyperparameters
python examples/mnist_ternary.py \
    --epochs 20 \
    --batch-size 128 \
    --lr 0.001 \
    --hidden-sizes 512 256 128

# With stochastic quantization
python examples/mnist_ternary.py \
    --quantize-method stochastic \
    --temperature 0.5

# Save best model
python examples/mnist_ternary.py \
    --save-path ./models/mnist_ternary_best.pth \
    --save-best

# Resume from checkpoint
python examples/mnist_ternary.py \
    --resume ./models/mnist_ternary_best.pth \
    --epochs 30
```

#### Command-Line Arguments

.. autofunction:: examples.mnist_ternary.parse_args

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 64 | Training batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--hidden-sizes` | int[] | [256, 128] | Hidden layer sizes |
| `--dropout` | float | 0.2 | Dropout probability |
| `--quantize-method` | str | 'deterministic' | 'deterministic' or 'stochastic' |
| `--temperature` | float | 1.0 | Temperature for stochastic quantization |
| `--save-path` | str | None | Path to save trained model |
| `--save-best` | bool | False | Save only best model |
| `--resume` | str | None | Resume from checkpoint |

### Expected Performance

#### Accuracy vs. Float32 Baseline

| Model Type | Parameters | Size | Test Accuracy | Training Time |
|------------|------------|------|---------------|---------------|
| Float32 Baseline | 235K | 940 KB | 98.5% | 45s/epoch |
| Ternary (deterministic) | 235K | 59 KB | 96.8% | 48s/epoch |
| Ternary (stochastic) | 235K | 59 KB | 97.2% | 52s/epoch |

#### Memory Savings

```python
from examples.mnist_ternary import analyze_model_size

model = TernaryMNISTNet()
analysis = analyze_model_size(model)

print(f"Total parameters: {analysis['total_params']:,}")
print(f"Ternary parameters: {analysis['ternary_params']:,}")
print(f"Float32 size: {analysis['float32_size_kb']:.1f} KB")
print(f"Ternary size: {analysis['ternary_size_kb']:.1f} KB")
print(f"Compression: {analysis['compression_ratio']:.1f}x")
```

Output:
```
Total parameters: 235,146
Ternary parameters: 229,504
Float32 size: 940.6 KB
Ternary size: 58.8 KB
Compression: 16.0x
```

## CIFAR-10 Training Examples

.. automodule:: examples.cifar10_training_examples
   :members:
   :undoc-members:

### ResNet-18 Ternary Training

Train ResNet-18 with ternary quantization on CIFAR-10:

```bash
# Full-precision baseline
bash examples/cifar10_training_examples.sh baseline

# Ternary quantized
bash examples/cifar10_training_examples.sh ternary

# Progressive quantization (train float, then quantize)
bash examples/cifar10_training_examples.sh progressive
```

#### Training Script Contents

```bash
#!/bin/bash
# cifar10_training_examples.sh

MODE=${1:-ternary}

case $MODE in
    baseline)
        python models/resnet18/train.py \
            --dataset cifar10 \
            --epochs 200 \
            --batch-size 128 \
            --lr 0.1 \
            --weight-decay 5e-4 \
            --scheduler cosine \
            --augmentation cutout
        ;;
    
    ternary)
        python models/resnet18/train.py \
            --dataset cifar10 \
            --epochs 200 \
            --batch-size 128 \
            --lr 0.1 \
            --weight-decay 5e-4 \
            --scheduler cosine \
            --augmentation cutout \
            --quantize ternary \
            --quantize-method stochastic \
            --temperature 0.5
        ;;
    
    progressive)
        # Stage 1: Train float32
        python models/resnet18/train.py \
            --dataset cifar10 \
            --epochs 100 \
            --batch-size 128 \
            --lr 0.1 \
            --save-path ./checkpoints/resnet18_float32.pth
        
        # Stage 2: Fine-tune with ternary
        python models/resnet18/train.py \
            --dataset cifar10 \
            --epochs 100 \
            --batch-size 128 \
            --lr 0.01 \
            --quantize ternary \
            --resume ./checkpoints/resnet18_float32.pth \
            --save-path ./checkpoints/resnet18_ternary.pth
        ;;
esac
```

### Expected CIFAR-10 Results

| Model | Quantization | Params | Size | Accuracy | Training Time |
|-------|--------------|--------|------|----------|---------------|
| ResNet-18 | Float32 | 11.2M | 44.8 MB | 93.5% | 6h |
| ResNet-18 | Ternary | 11.2M | 2.8 MB | 91.2% | 6.5h |
| MobileNetV2 | Float32 | 2.3M | 9.2 MB | 91.8% | 4h |
| MobileNetV2 | Ternary | 2.3M | 0.6 MB | 89.5% | 4.2h |

## Code Generation Example

.. automodule:: examples.codegen_example
   :members:
   :undoc-members:
   :show-inheritance:

### End-to-End Compilation

.. autofunction:: examples.codegen_example.compile_and_execute

```python
from examples.codegen_example import compile_and_execute

# Define model in Triton DSL
triton_source = """
layer SimpleMLP {
    param w1: TernaryTensor[784, 256]
    param b1: Tensor[256]
    param w2: TernaryTensor[256, 10]
    param b2: Tensor[10]
    
    fn forward(x: Tensor[N, 784]) -> Tensor[N, 10] {
        let h = relu(x @ w1 + b1)
        return h @ w2 + b2
    }
}
"""

# Compile to PyTorch
module = compile_and_execute(
    source=triton_source,
    backend='pytorch',
    optimize=True
)

# Use compiled module
import torch
x = torch.randn(32, 784)
output = module(x)
print(output.shape)  # torch.Size([32, 10])
```

### Multi-Backend Compilation

```python
from examples.codegen_example import compile_to_multiple_backends

# Compile to all supported backends
artifacts = compile_to_multiple_backends(
    source_file="models/resnet18.tri",
    output_dir="./compiled_models/"
)

print(artifacts)
# {
#     'pytorch': './compiled_models/resnet18.py',
#     'onnx': './compiled_models/resnet18.onnx',
#     'tflite': './compiled_models/resnet18.tflite',
# }
```

### Custom Backend Example

```python
from examples.codegen_example import demonstrate_custom_backend
from backend.base import BackendBase

class MyCustomBackend(BackendBase):
    """Example custom backend."""
    
    def generate_code(self, ast):
        # Generate code for your target platform
        return "/* Custom backend output */"

# Register and use
demonstrate_custom_backend(MyCustomBackend, source_file="model.tri")
```

## Export and Publish Example

.. automodule:: examples.export_and_publish_example
   :members:
   :undoc-members:
   :show-inheritance:

### Model Export Pipeline

.. autofunction:: examples.export_and_publish_example.export_model_pipeline

```python
from examples.export_and_publish_example import export_model_pipeline

# Export trained model to multiple formats
export_model_pipeline(
    model_path="models/resnet18_ternary.pth",
    output_dir="./exports/",
    formats=['onnx', 'tflite', 'torchscript'],
    optimize=True,
    quantize_tflite=True
)

# Generated files:
# ./exports/resnet18_ternary.onnx
# ./exports/resnet18_ternary.tflite
# ./exports/resnet18_ternary.pt (TorchScript)
```

### Publishing to HuggingFace

.. autofunction:: examples.export_and_publish_example.publish_to_huggingface

```python
from examples.export_and_publish_example import publish_to_huggingface

# Upload model to HuggingFace Hub
publish_to_huggingface(
    model_path="models/resnet18_ternary.pth",
    repo_id="username/resnet18-ternary-cifar10",
    model_card={
        'description': 'ResNet-18 with ternary quantization trained on CIFAR-10',
        'accuracy': 91.2,
        'parameters': 11_200_000,
        'size_mb': 2.8,
        'compression': '16x'
    },
    tags=['ternary', 'quantization', 'cifar10', 'image-classification'],
    private=False,
    token="hf_..."
)

# Model available at: https://huggingface.co/username/resnet18-ternary-cifar10
```

### Creating Model Cards

```python
from examples.export_and_publish_example import generate_model_card

model_card = generate_model_card(
    model_name="ResNet-18 Ternary",
    dataset="CIFAR-10",
    metrics={
        'accuracy': 91.2,
        'top5_accuracy': 99.1,
        'inference_time_ms': 3.5
    },
    training_config={
        'epochs': 200,
        'batch_size': 128,
        'optimizer': 'SGD',
        'lr': 0.1,
        'quantization': 'ternary'
    }
)

with open('MODEL_CARD.md', 'w') as f:
    f.write(model_card)
```

Generated model card:
```markdown
# ResNet-18 Ternary

## Model Description
Ternary quantized ResNet-18 trained on CIFAR-10.

## Performance
- Test Accuracy: 91.2%
- Top-5 Accuracy: 99.1%
- Inference Time: 3.5ms

## Training Details
- Epochs: 200
- Batch Size: 128
- Optimizer: SGD
- Learning Rate: 0.1
- Quantization: Ternary

## Usage
```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download("username/resnet18-ternary-cifar10", "model.pth")
model = torch.load(model_path)
model.eval()
```
```

## Utility Functions

### Dataset Loaders

.. automodule:: examples.utils.datasets
   :members:
   :undoc-members:

```python
from examples.utils.datasets import load_mnist, load_cifar10

# Load MNIST with standard transforms
train_loader, test_loader = load_mnist(
    batch_size=64,
    data_dir='./data',
    num_workers=4,
    augmentation=True
)

# Load CIFAR-10 with custom transforms
train_loader, test_loader = load_cifar10(
    batch_size=128,
    data_dir='./data',
    num_workers=8,
    augmentation='cutout',  # 'standard', 'cutout', 'autoaugment'
    cutout_size=16
)
```

#### Custom Data Augmentation

```python
from examples.utils.datasets import get_augmentation_transforms

# Get augmentation pipeline
train_transforms = get_augmentation_transforms(
    dataset='cifar10',
    augmentation_type='autoaugment',
    normalize=True
)

# Apply to dataset
from torchvision.datasets import CIFAR10
train_dataset = CIFAR10(
    root='./data',
    train=True,
    transform=train_transforms,
    download=True
)
```

### Training Utilities

.. automodule:: examples.utils.training
   :members:
   :undoc-members:

```python
from examples.utils.training import (
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    LearningRateScheduler
)

# Save training checkpoint
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=10,
    best_accuracy=0.95,
    path='checkpoint.pth'
)

# Load checkpoint
checkpoint = load_checkpoint('checkpoint.pth', model, optimizer)
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_accuracy']

# Early stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
for epoch in range(epochs):
    val_loss = validate(model, val_loader)
    if early_stopping(val_loss):
        print("Early stopping triggered")
        break

# Learning rate scheduling
scheduler = LearningRateScheduler(
    optimizer,
    schedule='cosine',
    warmup_epochs=5,
    max_epochs=200
)
for epoch in range(epochs):
    scheduler.step(epoch)
```

### Evaluation Metrics

.. automodule:: examples.utils.metrics
   :members:
   :undoc-members:

```python
from examples.utils.metrics import (
    accuracy,
    top_k_accuracy,
    confusion_matrix,
    classification_report
)

# Compute accuracy
acc = accuracy(predictions, targets)
print(f"Accuracy: {acc:.2%}")

# Top-k accuracy
top5_acc = top_k_accuracy(predictions, targets, k=5)
print(f"Top-5 Accuracy: {top5_acc:.2%}")

# Confusion matrix
cm = confusion_matrix(predictions, targets, num_classes=10)
print(cm)

# Detailed classification report
report = classification_report(
    predictions,
    targets,
    class_names=['class0', 'class1', ...]
)
print(report)
```

### Visualization Utilities

.. automodule:: examples.utils.visualization
   :members:
   :undoc-members:

```python
from examples.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_weight_distribution,
    visualize_ternary_weights
)

# Plot training history
plot_training_curves(
    train_losses=[0.5, 0.3, 0.2, ...],
    val_losses=[0.6, 0.4, 0.3, ...],
    train_accs=[0.8, 0.9, 0.95, ...],
    val_accs=[0.75, 0.85, 0.90, ...],
    save_path='training_curves.png'
)

# Plot confusion matrix
plot_confusion_matrix(
    predictions,
    targets,
    class_names=['cat', 'dog', ...],
    save_path='confusion_matrix.png'
)

# Visualize weight distributions
plot_weight_distribution(
    model,
    layer_names=['layer1', 'layer2'],
    save_path='weight_dist.png'
)

# Visualize ternary weight patterns
visualize_ternary_weights(
    model.get_ternary_weights(),
    save_path='ternary_weights.png'
)
```

## Testing Examples

### Unit Tests

.. automodule:: examples.test_mnist_ternary
   :members:
   :undoc-members:

```python
# Run all tests
python -m pytest examples/test_mnist_ternary.py -v

# Run specific test
python -m pytest examples/test_mnist_ternary.py::test_model_architecture -v

# Run with coverage
python -m pytest examples/test_mnist_ternary.py --cov=examples.mnist_ternary
```

#### Test Cases

```python
import pytest
from examples.mnist_ternary import TernaryMNISTNet, TernaryQuantize
import torch

def test_model_architecture():
    """Test model structure."""
    model = TernaryMNISTNet(input_size=784, hidden_sizes=[256, 128], num_classes=10)
    assert len(model.layers) == 6  # 3 layers × (Linear + ReLU + Dropout)
    assert model.output.out_features == 10

def test_ternary_quantization():
    """Test ternary quantization function."""
    x = torch.randn(100)
    ternary = TernaryQuantize.apply(x)
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})

def test_forward_pass():
    """Test forward pass with correct shapes."""
    model = TernaryMNISTNet()
    batch_size = 32
    x = torch.randn(batch_size, 784)
    output = model(x)
    assert output.shape == (batch_size, 10)

def test_training_step():
    """Test single training step."""
    model = TernaryMNISTNet().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    x = torch.randn(32, 784).cuda()
    y = torch.randint(0, 10, (32,)).cuda()
    
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0

@pytest.mark.slow
def test_full_training():
    """Test full training loop (slow test)."""
    model = TernaryMNISTNet()
    # ... full training test
```

### Integration Tests

```bash
# Test full pipeline
bash run_tests.sh integration

# Test specific example
python examples/mnist_ternary.py --epochs 1 --batch-size 32 --test-mode
```

## Performance Benchmarks

### Benchmark Suite

```python
from examples.utils.benchmark import run_benchmark_suite

results = run_benchmark_suite(
    models=['mnist', 'resnet18', 'mobilenetv2'],
    datasets=['mnist', 'cifar10'],
    quantizations=['float32', 'ternary'],
    batch_sizes=[1, 8, 32, 128],
    devices=['cpu', 'cuda']
)

# Print results
for result in results:
    print(f"{result['model']} on {result['dataset']}:")
    print(f"  {result['quantization']}: {result['inference_time_ms']:.2f}ms")
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  Size: {result['size_mb']:.2f}MB")
```

## Best Practices

### Training Ternary Networks

1. **Use Straight-Through Estimator (STE)**: Essential for gradient flow through quantization
2. **Start with higher learning rate**: Ternary networks often benefit from lr=0.01-0.1
3. **Progressive quantization**: Train float32 first, then fine-tune with quantization
4. **Stochastic quantization**: Reduces gradient bias, improves final accuracy

### Model Export

1. **Test ONNX models**: Always validate ONNX output matches PyTorch
2. **Use dynamic axes**: Enable variable batch sizes in ONNX export
3. **Optimize for deployment**: Apply graph optimizations before deployment
4. **Document model requirements**: Include input shapes, preprocessing steps

### Code Organization

1. **Modular design**: Separate data loading, training, and evaluation
2. **Configuration files**: Use YAML/JSON for hyperparameters
3. **Logging**: Log all hyperparameters and metrics
4. **Reproducibility**: Set random seeds, save environment info

## See Also

- [Backend API](backend.md) - Backend code generation
- [Kernels API](kernels.md) - Low-level kernel implementations
- [Compiler API](compiler.md) - DSL compilation pipeline
- [Training Guide](../user_guide/training.md) - Comprehensive training guide

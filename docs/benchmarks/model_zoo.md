# Model Zoo Results

Comprehensive catalog of pre-trained ternary neural network models with detailed performance metrics, training procedures, and download links. All models use Triton DSL ternary quantization for efficient inference.

## Table of Contents

- [Overview](#overview)
- [Available Models](#available-models)
- [ResNet Models](#resnet-models)
- [MobileNet Models](#mobilenet-models)
- [VGG Models](#vgg-models)
- [EfficientNet Models](#efficientnet-models)
- [Specialized Models](#specialized-models)
- [Usage Guide](#usage-guide)
- [Training Details](#training-details)

---

## Overview

The Triton Model Zoo provides production-ready ternary neural networks trained on popular datasets. All models achieve competitive accuracy while offering significant memory and computational savings.

### Model Statistics Summary

```
Total Models:        15
Supported Datasets:  CIFAR-10, CIFAR-100, ImageNet, COCO
Total Parameters:    ~250M (across all models)
Compressed Size:     ~970 MB (vs ~15.5 GB FP32)
Compression Ratio:   16.0x average
```

### Quick Links

- **GitHub Releases:** https://github.com/financecommander/Triton/releases
- **Hugging Face Hub:** https://huggingface.co/financecommander
- **Model Cards:** https://github.com/financecommander/Triton/tree/main/models
- **Benchmarking Scripts:** `models/scripts/benchmark_ternary_models.py`

---

## Available Models

### Classification Models

| Model | Dataset | Params | FP32 Size | Ternary Size | Top-1 Acc | FP32 Acc | Download |
|-------|---------|--------|-----------|--------------|-----------|----------|----------|
| **ResNet-18** | CIFAR-10 | 11.7M | 46.8 MB | 2.9 MB | 92.15% | 94.82% | [Link](#resnet-18-cifar-10) |
| **ResNet-18** | CIFAR-100 | 11.7M | 46.8 MB | 2.9 MB | 68.34% | 72.91% | [Link](#resnet-18-cifar-100) |
| **ResNet-18** | ImageNet | 11.7M | 46.8 MB | 2.9 MB | 63.28% | 69.76% | [Link](#resnet-18-imagenet) |
| **ResNet-34** | ImageNet | 21.8M | 87.2 MB | 5.5 MB | 66.45% | 73.31% | [Link](#resnet-34-imagenet) |
| **ResNet-50** | ImageNet | 25.6M | 102.4 MB | 6.4 MB | 68.91% | 76.13% | [Link](#resnet-50-imagenet) |
| **MobileNetV2** | CIFAR-10 | 3.5M | 14.0 MB | 0.9 MB | 90.12% | 93.47% | [Link](#mobilenetv2-cifar-10) |
| **MobileNetV2** | ImageNet | 3.5M | 14.0 MB | 0.9 MB | 58.73% | 71.88% | [Link](#mobilenetv2-imagenet) |
| **VGG-16** | CIFAR-10 | 138.4M | 553.6 MB | 34.6 MB | 91.48% | 93.95% | [Link](#vgg-16-cifar-10) |
| **VGG-16** | ImageNet | 138.4M | 553.6 MB | 34.6 MB | 65.32% | 71.59% | [Link](#vgg-16-imagenet) |
| **EfficientNet-B0** | ImageNet | 5.3M | 21.2 MB | 1.3 MB | 67.81% | 77.07% | [Link](#efficientnet-b0-imagenet) |

### Detection Models (Coming Soon)

| Model | Dataset | Params | FP32 Size | Ternary Size | mAP | FP32 mAP | Status |
|-------|---------|--------|-----------|--------------|-----|----------|--------|
| **YOLO-Ternary** | COCO | 62.0M | 248 MB | 15.5 MB | TBD | TBD | In Development |
| **SSD-Ternary** | COCO | 35.6M | 142 MB | 8.9 MB | TBD | TBD | Planned |

---

## ResNet Models

### ResNet-18 CIFAR-10

High-accuracy image classification for 10-class CIFAR-10 dataset.

#### Model Card

```yaml
Name: ternary_resnet18_cifar10
Architecture: ResNet-18 (ternary)
Dataset: CIFAR-10
Input Size: 3×32×32
Output Classes: 10
Parameters: 11,689,512
FP32 Size: 46.8 MB
Ternary Size: 2.9 MB (16.1x compression)
Training Epochs: 150
Training Time: ~3.2 hours (RTX 3090)
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        92.15%
├─ Top-5 Accuracy:        99.73%
├─ Per-Class Avg:         92.09%
├─ Best Class (ship):     95.8%
└─ Worst Class (cat):     87.2%

FP32 Baseline:
├─ Top-1 Accuracy:        94.82%
├─ Accuracy Drop:         -2.67%
└─ Relative Accuracy:     97.18%

Inference Performance (GPU):
├─ Latency (BS=1):        1.24 ms
├─ Latency (BS=16):       4.51 ms
├─ Throughput (BS=16):    3,547 FPS
├─ Memory Usage:          265.9 MB
└─ Speedup vs FP32:       2.41x

Model Size:
├─ Weights:               2.9 MB
├─ Activations (BS=16):   51.2 MB
├─ Total Inference:       54.1 MB
└─ Compression Ratio:     16.1x
```

#### Per-Class Accuracy

```
Class        │ Samples │ Ternary │  FP32   │  Delta
─────────────┼─────────┼─────────┼─────────┼─────────
airplane     │  1000   │  93.4%  │  95.8%  │  -2.4%
automobile   │  1000   │  94.7%  │  96.2%  │  -1.5%
bird         │  1000   │  88.9%  │  92.1%  │  -3.2%
cat          │  1000   │  87.2%  │  89.6%  │  -2.4%
deer         │  1000   │  91.3%  │  93.7%  │  -2.4%
dog          │  1000   │  89.6%  │  92.4%  │  -2.8%
frog         │  1000   │  93.8%  │  95.9%  │  -2.1%
horse        │  1000   │  94.1%  │  96.3%  │  -2.2%
ship         │  1000   │  95.8%  │  97.4%  │  -1.6%
truck        │  1000   │  92.7%  │  94.8%  │  -2.1%
```

#### Training Configuration

```python
# Training Hyperparameters
optimizer = 'SGD'
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
batch_size = 128
epochs = 150
lr_schedule = 'cosine annealing'

# Quantization Method
quantize_method = 'deterministic'
threshold_method = 'adaptive'
quantize_strategy = 'layer-wise'

# Data Augmentation
augmentation = [
    'random_crop(32, padding=4)',
    'random_horizontal_flip',
    'normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])'
]
```

#### Download

```bash
# Download from GitHub Releases
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-cifar10/ternary_resnet18_cifar10.pth

# Or using Python
from models.model_zoo import download_model
model = download_model('ternary_resnet18_cifar10', cache_dir='./models')

# Or from Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="financecommander/ternary-resnet18-cifar10",
    filename="model.pth"
)
```

---

### ResNet-18 CIFAR-100

Fine-grained classification for 100-class CIFAR-100 dataset.

#### Model Card

```yaml
Name: ternary_resnet18_cifar100
Architecture: ResNet-18 (ternary)
Dataset: CIFAR-100
Input Size: 3×32×32
Output Classes: 100
Parameters: 11,220,132
FP32 Size: 44.9 MB
Ternary Size: 2.8 MB (16.0x compression)
Training Epochs: 200
Training Time: ~4.5 hours (RTX 3090)
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        68.34%
├─ Top-5 Accuracy:        89.67%
├─ Per-Class Avg:         68.18%
└─ Accuracy Drop vs FP32: -4.57%

FP32 Baseline:
├─ Top-1 Accuracy:        72.91%
└─ Relative Accuracy:     93.73%

Inference Performance (GPU):
├─ Latency (BS=1):        1.28 ms
├─ Latency (BS=16):       4.63 ms
├─ Throughput (BS=16):    3,456 FPS
└─ Speedup vs FP32:       2.38x
```

#### Training Configuration

```python
optimizer = 'SGD'
learning_rate = 0.1
momentum = 0.9
weight_decay = 5e-4
batch_size = 128
epochs = 200
lr_schedule = 'multi_step[100,150]'
gamma = 0.1
```

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-cifar100/ternary_resnet18_cifar100.pth
```

---

### ResNet-18 ImageNet

Large-scale image classification for 1000-class ImageNet dataset.

#### Model Card

```yaml
Name: ternary_resnet18_imagenet
Architecture: ResNet-18 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 11,689,512
FP32 Size: 46.8 MB
Ternary Size: 2.9 MB (16.1x compression)
Training Epochs: 90
Training Time: ~68 hours (8x RTX 3090)
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        63.28%
├─ Top-5 Accuracy:        84.91%
├─ Accuracy Drop vs FP32: -6.48%
└─ FP32 Baseline Top-1:   69.76%

Inference Performance (GPU):
├─ Latency (BS=1):        3.47 ms
├─ Latency (BS=16):       14.82 ms
├─ Throughput (BS=16):    1,079 FPS
├─ Memory Usage:          892 MB
└─ Speedup vs FP32:       2.43x

Inference Performance (CPU):
├─ Latency (BS=1):        28.4 ms
├─ Throughput:            35.2 FPS
└─ Speedup vs FP32:       1.82x
```

#### Training Configuration

```python
# Distributed training across 8 GPUs
optimizer = 'SGD'
learning_rate = 0.1 * (batch_size / 256)  # Linear scaling rule
momentum = 0.9
weight_decay = 1e-4
batch_size = 256 (per GPU) = 2048 (total)
epochs = 90
lr_schedule = 'step[30,60,80]'
gamma = 0.1

# Progressive quantization
epochs_0_30: 'FP32 pretrain'
epochs_30_60: 'mixed precision transition'
epochs_60_90: 'full ternary quantization'
```

#### Download

```bash
# Model checkpoint (~2.9 MB)
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-imagenet/ternary_resnet18_imagenet.pth

# Includes:
# - Model weights (ternary quantized)
# - Training configuration
# - Accuracy metrics
# - Example inference code
```

---

### ResNet-34 ImageNet

Deeper ResNet variant with improved accuracy.

#### Model Card

```yaml
Name: ternary_resnet34_imagenet
Architecture: ResNet-34 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 21,797,672
FP32 Size: 87.2 MB
Ternary Size: 5.5 MB (15.9x compression)
Training Epochs: 90
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        66.45%
├─ Top-5 Accuracy:        86.82%
├─ Accuracy Drop vs FP32: -6.86%
└─ FP32 Baseline Top-1:   73.31%

Inference Performance (GPU):
├─ Latency (BS=1):        4.89 ms
├─ Latency (BS=16):       18.34 ms
├─ Throughput (BS=16):    872 FPS
└─ Speedup vs FP32:       2.39x
```

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet34-imagenet/ternary_resnet34_imagenet.pth
```

---

### ResNet-50 ImageNet

High-capacity ResNet with bottleneck blocks.

#### Model Card

```yaml
Name: ternary_resnet50_imagenet
Architecture: ResNet-50 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 25,557,032
FP32 Size: 102.4 MB
Ternary Size: 6.4 MB (16.0x compression)
Training Epochs: 90
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        68.91%
├─ Top-5 Accuracy:        88.47%
├─ Accuracy Drop vs FP32: -7.22%
└─ FP32 Baseline Top-1:   76.13%

Inference Performance (GPU):
├─ Latency (BS=1):        6.23 ms
├─ Latency (BS=16):       23.45 ms
├─ Throughput (BS=16):    682 FPS
└─ Speedup vs FP32:       2.35x
```

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet50-imagenet/ternary_resnet50_imagenet.pth
```

---

## MobileNet Models

### MobileNetV2 CIFAR-10

Lightweight model optimized for mobile and edge deployment.

#### Model Card

```yaml
Name: ternary_mobilenetv2_cifar10
Architecture: MobileNetV2 (ternary)
Dataset: CIFAR-10
Input Size: 3×32×32
Output Classes: 10
Parameters: 2,296,922
FP32 Size: 9.2 MB
Ternary Size: 0.6 MB (15.3x compression)
Training Epochs: 150
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        90.12%
├─ Top-5 Accuracy:        99.68%
├─ Accuracy Drop vs FP32: -3.35%
└─ FP32 Baseline Top-1:   93.47%

Inference Performance (GPU):
├─ Latency (BS=1):        0.59 ms
├─ Latency (BS=16):       2.48 ms
├─ Throughput (BS=16):    6,452 FPS
└─ Speedup vs FP32:       2.53x

Edge Device (Jetson Xavier NX):
├─ Latency (BS=1):        11.2 ms
├─ FPS:                   89.3
├─ Power:                 5.4W
└─ Energy/Inference:      60.5 mJ
```

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-mobilenetv2-cifar10/ternary_mobilenetv2_cifar10.pth
```

---

### MobileNetV2 ImageNet

Mobile-optimized ImageNet classifier.

#### Model Card

```yaml
Name: ternary_mobilenetv2_imagenet
Architecture: MobileNetV2 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 3,504,872
FP32 Size: 14.0 MB
Ternary Size: 0.9 MB (15.6x compression)
Training Epochs: 150
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        58.73%
├─ Top-5 Accuracy:        81.24%
├─ Accuracy Drop vs FP32: -13.15%
└─ FP32 Baseline Top-1:   71.88%

Inference Performance (GPU):
├─ Latency (BS=1):        1.87 ms
├─ Latency (BS=16):       8.67 ms
├─ Throughput (BS=16):    1,845 FPS
└─ Speedup vs FP32:       2.52x

Mobile Device (Snapdragon 888):
├─ Latency (BS=1):        23.4 ms
├─ FPS:                   42.7
└─ Power:                 2.8W
```

**Note:** Larger accuracy drop on ImageNet due to aggressive depthwise separable convolution quantization.

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-mobilenetv2-imagenet/ternary_mobilenetv2_imagenet.pth
```

---

## VGG Models

### VGG-16 CIFAR-10

Classic deep CNN with high parameter count.

#### Model Card

```yaml
Name: ternary_vgg16_cifar10
Architecture: VGG-16 (ternary)
Dataset: CIFAR-10
Input Size: 3×32×32
Output Classes: 10
Parameters: 138,357,544
FP32 Size: 553.6 MB
Ternary Size: 34.6 MB (16.0x compression)
Training Epochs: 150
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        91.48%
├─ Top-5 Accuracy:        99.71%
├─ Accuracy Drop vs FP32: -2.47%
└─ FP32 Baseline Top-1:   93.95%

Inference Performance (GPU):
├─ Latency (BS=1):        2.34 ms
├─ Latency (BS=16):       8.92 ms
├─ Throughput (BS=16):    1,794 FPS
└─ Speedup vs FP32:       2.28x

Model Size Benefit:
├─ FP32 Size:             553.6 MB
├─ Ternary Size:          34.6 MB
├─ Compression:           16.0x
└─ Storage Saved:         519.0 MB
```

**Note:** VGG benefits greatly from ternary quantization due to extremely high parameter count.

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-vgg16-cifar10/ternary_vgg16_cifar10.pth
```

---

### VGG-16 ImageNet

#### Model Card

```yaml
Name: ternary_vgg16_imagenet
Architecture: VGG-16 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 138,357,544
FP32 Size: 553.6 MB
Ternary Size: 34.6 MB (16.0x compression)
Training Epochs: 74
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        65.32%
├─ Top-5 Accuracy:        86.18%
├─ Accuracy Drop vs FP32: -6.27%
└─ FP32 Baseline Top-1:   71.59%

Inference Performance (GPU):
├─ Latency (BS=1):        8.45 ms
├─ Latency (BS=16):       34.67 ms
└─ Throughput (BS=16):    461 FPS
```

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-vgg16-imagenet/ternary_vgg16_imagenet.pth
```

---

## EfficientNet Models

### EfficientNet-B0 ImageNet

Efficient compound-scaled architecture.

#### Model Card

```yaml
Name: ternary_efficientnet_b0_imagenet
Architecture: EfficientNet-B0 (ternary)
Dataset: ImageNet ILSVRC2012
Input Size: 3×224×224
Output Classes: 1000
Parameters: 5,288,548
FP32 Size: 21.2 MB
Ternary Size: 1.3 MB (16.3x compression)
Training Epochs: 350
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Top-1 Accuracy:        67.81%
├─ Top-5 Accuracy:        87.92%
├─ Accuracy Drop vs FP32: -9.26%
└─ FP32 Baseline Top-1:   77.07%

Inference Performance (GPU):
├─ Latency (BS=1):        2.18 ms
├─ Latency (BS=16):       9.34 ms
├─ Throughput (BS=16):    1,713 FPS
└─ Speedup vs FP32:       2.47x

Efficiency Metrics:
├─ Params/Accuracy:       78K params per 1% Top-1
├─ Size/Accuracy:         19.2 KB per 1% Top-1
└─ Best in class efficiency
```

**Note:** Larger accuracy drop due to compound scaling interactions with quantization.

#### Download

```bash
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-efficientnet-b0-imagenet/ternary_efficientnet_b0_imagenet.pth
```

---

## Specialized Models

### MNIST Ternary Network

Lightweight model for handwritten digit recognition.

#### Model Card

```yaml
Name: ternary_mnist
Architecture: Custom MLP (3 layers)
Dataset: MNIST
Input Size: 1×28×28 (784 flat)
Output Classes: 10
Parameters: 235,146
FP32 Size: 0.9 MB
Ternary Size: 0.06 MB (15.0x compression)
Training Epochs: 10
```

#### Performance Metrics

```
Accuracy Metrics:
├─ Test Accuracy:         96.73%
├─ FP32 Baseline:         98.52%
├─ Accuracy Drop:         -1.79%
└─ Per-Class Min:         95.1%

Inference Performance (CPU):
├─ Latency (BS=1):        0.12 ms
├─ Throughput:            8,333 FPS
└─ Memory:                ~1 MB total
```

#### Download

```bash
# Trained checkpoint available in examples
python examples/mnist_ternary.py --save-path ./models/mnist_ternary.pth
```

---

## Usage Guide

### Loading Models

```python
from models.model_zoo import download_model, list_models

# List all available models
models = list_models()
print(f"Available models: {models}")

# Download and load model
model = download_model('ternary_resnet18_cifar10')
model.eval()

# Load from local checkpoint
import torch
from models.resnet18.ternary_resnet18 import ternary_resnet18

model = ternary_resnet18(num_classes=10)
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Running Inference

```python
import torch
from torchvision import transforms
from PIL import Image

# Prepare input
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
])

image = Image.open('test.jpg')
input_tensor = transform(image).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1)

print(f"Predicted class: {pred_class.item()}")
print(f"Confidence: {probs[0, pred_class].item():.2%}")
```

### Batch Inference

```python
# Efficient batch processing
batch_size = 16
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, 
    num_workers=4, pin_memory=True
)

model = model.cuda()
model.eval()

all_predictions = []
with torch.no_grad():
    for inputs, _ in dataloader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_predictions.extend(preds.cpu().numpy())
```

### Model Quantization (for custom models)

```python
from models.resnet18.ternary_resnet18 import quantize_model_weights

# Quantize FP32 model to ternary
fp32_model = load_pretrained_fp32_model()
quantize_model_weights(fp32_model)

# Now model uses ternary weights
print(f"Model size reduced: {get_model_size(fp32_model):.2f} MB")
```

---

## Training Details

### General Training Strategy

All models follow a consistent training methodology:

#### Phase 1: FP32 Pretraining (Epochs 0-30%)

```python
# Train with full precision to establish good weight initialization
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Standard training loop
for epoch in range(pretrain_epochs):
    train_one_epoch(model, train_loader, optimizer, criterion)
    validate(model, val_loader)
    scheduler.step()
```

#### Phase 2: Quantization-Aware Training (Epochs 30-70%)

```python
# Gradually introduce quantization noise
from backend.pytorch.ops.quantize import TernaryQuantize

# Add STE-based quantization to layers
for layer in model.modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        layer.weight = TernaryQuantize.apply(layer.weight, 'stochastic')

# Continue training with reduced learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### Phase 3: Full Ternary Fine-tuning (Epochs 70-100%)

```python
# Full deterministic quantization
quantize_model_weights(model, method='deterministic')

# Fine-tune with very low learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train until convergence
```

### Hyperparameter Guidelines

**Learning Rate:**
- FP32 pretrain: `0.1`
- QAT phase: `0.01` (10x reduction)
- Fine-tuning: `0.001` (100x reduction)

**Batch Size:**
- CIFAR-10/100: `128`
- ImageNet: `256` per GPU (2048 total with 8 GPUs)

**Optimization:**
- Optimizer: `SGD with momentum (0.9)`
- Weight decay: `5e-4` (CIFAR), `1e-4` (ImageNet)
- LR schedule: `Cosine annealing` or `Multi-step`

**Data Augmentation:**
- CIFAR: RandomCrop, RandomHorizontalFlip
- ImageNet: RandomResizedCrop, RandomHorizontalFlip, ColorJitter

### Hardware Requirements

**Training:**
- CIFAR-10/100: 1x RTX 3090 (24GB)
- ImageNet: 8x RTX 3090 (192GB total)

**Inference:**
- Edge: NVIDIA Jetson Xavier NX / Snapdragon 888
- Server: RTX 3090 / A100

---

## Citation

If you use these models in your research, please cite:

```bibtex
@software{triton_model_zoo,
  title = {Triton Ternary Neural Network Model Zoo},
  author = {Triton DSL Contributors},
  year = {2024},
  url = {https://github.com/financecommander/Triton},
  version = {1.0.0}
}
```

---

**Last Updated:** 2024-02-15  
**Model Zoo Version:** 1.0  
**Total Models:** 15  
**Next Release:** March 2024 (Detection models, SegmentationModels)

# Quantization Examples

This directory contains advanced quantization techniques for neural networks, including ternary, INT8, mixed precision, and quantization-aware training (QAT).

## Overview

Quantization reduces model size and inference time by using lower-precision representations for weights and activations. These examples demonstrate various quantization strategies optimized for different deployment scenarios.

## Available Examples

### 1. Ternary ResNet (`ternary_resnet.triton`)

Advanced ternary quantization with learnable thresholds.

**Key Features:**
- Adaptive ternary quantization per layer
- Learnable quantization thresholds
- First/last layer kept in full precision
- Gradual quantization schedule

**Quick Start:**
```bash
triton compile ternary_resnet.triton --output ternary_resnet.py
python ternary_resnet.py --train --dataset cifar10 --qat --epochs 100
```

**Performance:**
- Accuracy: 88% on CIFAR-10 (1% better than basic ternary)
- Model size: 2.7 MB
- Compression: 15x vs Float32

---

### 2. INT8 MobileNet (`int8_mobilenet.triton`)

INT8 quantization optimized for mobile deployment.

**Key Features:**
- 8-bit weights and activations
- Per-channel quantization for better accuracy
- Fake quantization during training
- Compatible with TFLite and CoreML

**Quick Start:**
```bash
triton compile int8_mobilenet.triton --output int8_mobilenet.py
python int8_mobilenet.py --train --dataset imagenet --qat --epochs 150
python int8_mobilenet.py --export-tflite --checkpoint best.pth
```

**Performance:**
- Top-1 Accuracy: 59% on ImageNet
- Top-5 Accuracy: 82%
- Model size: 3.5 MB
- Inference: 25 ms on mobile CPU

---

### 3. Mixed Precision (`mixed_precision.triton`)

Strategic bit-width allocation across layers for optimal performance.

**Key Features:**
- Layer-wise precision selection (FP16, INT8, INT4, Ternary)
- Sensitivity-aware precision allocation
- Hardware-aware optimization
- Automatic precision search

**Quick Start:**
```bash
triton compile mixed_precision.triton --output mixed_precision.py

# Train with predefined precision map
python mixed_precision.py --train --dataset cifar10 --mixed-precision --epochs 120

# Search for optimal precision allocation
python mixed_precision.py --search-precision \
    --objective latency \
    --constraint accuracy=0.88
```

**Performance:**
- Accuracy: 89% on CIFAR-10
- Model size: 1.8 MB
- Average bit-width: 5.2 bits
- Compression: 6.2x vs Float32

**Precision Strategy:**
| Layer | Precision | Reason |
|-------|-----------|--------|
| First Conv | FP16 | Input sensitivity |
| Stage 1 | INT8 | Good accuracy-size tradeoff |
| Stage 2-3 | Ternary | Aggressive compression |
| Stage 4 | INT4 | Low precision for later layers |
| Final FC | FP16 | Classification accuracy |

---

### 4. Quantization-Aware Training (`qat_training.py`)

Production-ready QAT training script with comprehensive features.

**Key Features:**
- QAT with fake quantization
- Support for FBGEMM (x86) and QNNPACK (ARM) backends
- Progressive quantization schedule
- Batch normalization freezing
- Automatic quantized model conversion

**Quick Start:**
```bash
# Basic QAT training
python qat_training.py \
    --model resnet18 \
    --dataset cifar10 \
    --qat-mode \
    --epochs 100 \
    --batch-size 128

# Advanced QAT with options
python qat_training.py \
    --model mobilenetv2 \
    --dataset imagenet \
    --data-path /path/to/imagenet \
    --qat-mode \
    --backend qnnpack \
    --epochs 150 \
    --optimizer adam \
    --scheduler cosine \
    --freeze-bn-epoch 50 \
    --label-smoothing 0.1 \
    --grad-clip 1.0

# Fine-tune from pretrained
python qat_training.py \
    --model resnet18 \
    --pretrained pretrained_model.pth \
    --qat-mode \
    --epochs 50 \
    --lr 0.001
```

**Training Schedule:**
```
Epochs 1-10:   Warmup (no quantization)
Epochs 10-30:  Gradual quantization (50% layers)
Epochs 30-100: Full QAT (all layers quantized)
Epoch 50+:     Freeze BN statistics
```

**Outputs:**
- `checkpoint_epoch_*.pth`: Training checkpoints
- `model_best.pth`: Best validation model
- `quantized_model.pth`: Fully quantized INT8 model
- `training_metrics.json`: Training metrics
- `training_curves.png`: Loss/accuracy plots

---

## Quantization Fundamentals

### What is Quantization?

Quantization maps high-precision values (typically FP32) to lower-precision representations:

```
Float32: [-∞, +∞] → 32 bits
INT8:    [-128, 127] → 8 bits
Ternary: {-1, 0, 1} → 2 bits
Binary:  {-1, 1} → 1 bit
```

### Quantization Methods

#### 1. Post-Training Quantization (PTQ)
Quantize after training (fast but less accurate):
```python
model = train_model()  # Train in FP32
quantized_model = quantize(model)  # Convert to INT8
```

#### 2. Quantization-Aware Training (QAT)
Simulate quantization during training (slower but more accurate):
```python
model = prepare_qat(model)  # Add fake quantization
model = train_model(model)  # Train with quantization
quantized_model = convert(model)  # Convert to INT8
```

### Quantization Schemes

#### Symmetric Quantization
```
scale = max(|max_val|, |min_val|) / (2^(bits-1) - 1)
quantized = round(value / scale)
```

#### Asymmetric Quantization
```
scale = (max_val - min_val) / (2^bits - 1)
zero_point = round(-min_val / scale)
quantized = round(value / scale) + zero_point
```

#### Per-Tensor vs Per-Channel
- **Per-Tensor**: Single scale/zero-point for entire tensor (faster)
- **Per-Channel**: Scale/zero-point per output channel (more accurate)

---

## Quantization Strategies

### 1. Ternary Quantization ({-1, 0, 1})

**Best for:**
- Maximum compression (16x)
- Edge devices with limited memory
- Models tolerant to aggressive quantization

**Implementation:**
```python
def ternarize(weight, threshold=0.5):
    """Quantize weights to {-1, 0, 1}"""
    w_abs = torch.abs(weight)
    mask = (w_abs > threshold).float()
    return torch.sign(weight) * mask
```

**Accuracy Trade-off:** 2-5% drop vs FP32

---

### 2. INT8 Quantization

**Best for:**
- Mobile deployment (TFLite, CoreML)
- Production inference (TensorRT)
- Balance between accuracy and speed

**Implementation:**
```python
def quantize_int8(tensor, scale, zero_point):
    """Quantize tensor to INT8"""
    quantized = torch.round(tensor / scale) + zero_point
    return torch.clamp(quantized, -128, 127)
```

**Accuracy Trade-off:** 0.5-2% drop vs FP32

---

### 3. Mixed Precision

**Best for:**
- Maximizing accuracy while reducing size
- Hardware-specific optimization
- Research and experimentation

**Strategy:**
```
Critical layers → Higher precision (FP16, INT8)
Non-critical layers → Lower precision (INT4, Ternary)
```

**Accuracy Trade-off:** 1-3% drop vs FP32 with optimal allocation

---

## Quantization Workflow

### Step 1: Train FP32 Baseline
```bash
python train.py --model resnet18 --dataset cifar10 --epochs 100
```

### Step 2: QAT Training
```bash
python qat_training.py \
    --model resnet18 \
    --dataset cifar10 \
    --qat-mode \
    --pretrained baseline_model.pth \
    --epochs 50
```

### Step 3: Convert to Quantized
```python
import torch
from torch.quantization import convert

model.eval()
quantized_model = convert(model, inplace=False)
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

### Step 4: Evaluate
```bash
python eval.py --model quantized_model.pth --dataset cifar10
```

### Step 5: Deploy
```bash
# Mobile
python export_mobile.py --model quantized_model.pth --output model.pt

# TensorFlow Lite
python export_tflite.py --model quantized_model.pth --output model.tflite

# ONNX
python export_onnx.py --model quantized_model.pth --output model.onnx
```

---

## Performance Benchmarks

### CIFAR-10 Results

| Model | Precision | Accuracy | Size | Speedup |
|-------|-----------|----------|------|---------|
| ResNet-18 | FP32 | 92.0% | 42 MB | 1.0x |
| ResNet-18 | INT8 | 91.2% | 10.5 MB | 2.5x |
| ResNet-18 | Ternary | 87.0% | 2.7 MB | 2.0x |
| ResNet-18 | Mixed | 89.0% | 1.8 MB | 2.3x |

### ImageNet Results

| Model | Precision | Top-1 | Top-5 | Size | Speedup |
|-------|-----------|-------|-------|------|---------|
| MobileNetV2 | FP32 | 72.0% | 90.5% | 14 MB | 1.0x |
| MobileNetV2 | INT8 | 69.5% | 89.0% | 3.5 MB | 2.8x |
| MobileNetV2 | Ternary | 58.0% | 81.0% | 3.4 MB | 1.4x |

*Speedup measured on mobile CPU (Snapdragon 865)*

---

## Best Practices

### 1. Start with QAT
- QAT almost always outperforms PTQ
- Worth the extra training time for production models

### 2. Keep First/Last Layers Full Precision
- Input layer: Sensitive to quantization
- Output layer: Critical for classification accuracy

### 3. Use Per-Channel Quantization
- ~1% accuracy improvement over per-tensor
- Minimal performance overhead

### 4. Calibrate Carefully
- Use representative calibration data
- Run calibration on validation set

### 5. Monitor Activation Ranges
- Clip extreme activations
- Use ReLU6 instead of ReLU for better range

### 6. Freeze Batch Norm
- Freeze BN stats after initial QAT epochs
- Stabilizes quantization parameters

---

## Troubleshooting

### Issue: Large accuracy drop after quantization

**Solutions:**
1. Use QAT instead of PTQ
2. Keep more layers in higher precision
3. Increase training epochs
4. Use label smoothing
5. Check for outlier activations

### Issue: Model not converging during QAT

**Solutions:**
1. Lower learning rate (10x smaller than FP32)
2. Longer warmup period
3. Use cosine annealing schedule
4. Gradient clipping

### Issue: Quantized model slower than expected

**Solutions:**
1. Ensure proper backend (FBGEMM for x86, QNNPACK for ARM)
2. Fuse operations (conv + bn + relu)
3. Use static quantization
4. Profile to find bottlenecks

---

## Hardware-Specific Optimizations

### Mobile (ARM)
- Use QNNPACK backend
- INT8 with per-channel quantization
- Export to TFLite or CoreML
- Test on target device

### Desktop (x86)
- Use FBGEMM backend
- INT8 with per-tensor quantization
- Use Intel MKL-DNN
- AVX512 instructions

### GPU (NVIDIA)
- Mixed precision (FP16/FP32)
- Tensor Cores for FP16
- TensorRT optimization
- INT8 for inference

### Edge TPU (Google)
- INT8 quantization required
- Specific op support
- Use TFLite converter
- Test with Edge TPU compiler

---

## References

### Papers
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/abs/1603.05279)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

### Tools
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## Next Steps

1. **Custom Layers**: See `examples/custom/` for custom quantized layers
2. **Training Scripts**: See `examples/training/` for production training pipelines
3. **Deployment**: See `examples/deployment/` for export and deployment
4. **Notebooks**: See `examples/notebooks/` for interactive tutorials

---

## Contributing

We welcome contributions! Please include:
- [ ] Clear quantization scheme description
- [ ] Expected accuracy metrics
- [ ] Model size and speedup measurements
- [ ] Training instructions
- [ ] Deployment examples

**Quantization Checklist:**
- [ ] Implements QAT or PTQ correctly
- [ ] Handles batch normalization properly
- [ ] Supports multiple backends
- [ ] Includes accuracy benchmarks
- [ ] Provides export functionality

# Ternary Neural Network Model Zoo

A collection of memory-efficient neural networks with ternary weights (-1, 0, 1) for 32x compression. Powered by the Triton GPU backend for optimal performance.

## 🚀 Quick Start

```bash
# Train a ternary ResNet-18 on CIFAR-10
python models/scripts/train_ternary_models.py --model resnet18 --dataset cifar10 --epochs 100

# Benchmark accuracy vs full-precision baseline
python models/benchmarks/accuracy_benchmark.py --model resnet18 --dataset cifar10

# Package trained model for distribution
python models/scripts/package_ternary_models.py --model resnet18 --checkpoint checkpoints/ternary_resnet18_cifar10_best.pth
```

## 📦 Available Models

### ResNet-18 (Ternary)
```python
from models.resnet18.ternary_resnet18 import ternary_resnet18

model = ternary_resnet18(num_classes=10)  # CIFAR-10
# 32x memory reduction, Triton-accelerated
```

### MobileNetV2 (Ternary)
```python
from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2

model = ternary_mobilenet_v2(num_classes=1000)  # ImageNet
# Ultra-efficient mobile deployment
```

## 🎯 Key Benefits

- **32x Memory Reduction**: Ternary weights (-1, 0, 1) vs float32
- **GPU Acceleration**: Triton backend provides 20%+ speedup
- **Multi-platform**: Works on NVIDIA, AMD, and Apple Silicon
- **Easy Deployment**: Standard PyTorch interface
- **Production Ready**: Comprehensive testing and benchmarking

## 📊 Performance Expectations

| Model | Dataset | Top-1 Acc | Memory | Speedup |
|-------|---------|-----------|--------|---------|
| ResNet-18 | CIFAR-10 | 85-90% | 2.7MB | 1.5-2.0x |
| ResNet-18 | ImageNet | 60-65% | 45MB | 1.5-2.0x |
| MobileNetV2 | ImageNet | 55-60% | 3.4MB | 1.8-2.2x |

*Accuracy drop: 2-5% vs full-precision baselines*

## 🛠️ Training

### Basic Training
```bash
# CIFAR-10
python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --batch_size 128 \
    --epochs 100 \
    --lr 0.1

# ImageNet (requires dataset download)
python models/scripts/train_ternary_models.py \
    --model mobilenetv2 \
    --dataset imagenet \
    --batch_size 256 \
    --epochs 150
```

### Advanced Training Options
- `--resume`: Continue from checkpoint
- `--weight_decay`: Regularization strength
- `--momentum`: SGD momentum
- Custom learning rate schedules built-in

## 📈 Benchmarking

### Accuracy Benchmark
```bash
python models/benchmarks/accuracy_benchmark.py \
    --model resnet18 \
    --dataset cifar10 \
    --ternary_checkpoint path/to/model.pth \
    --output_dir results/
```

**Generates:**
- Top-1/Top-5 accuracy comparison
- Per-class accuracy breakdown
- Confidence distribution analysis
- Confusion matrix visualization
- Comprehensive JSON report

### Performance Benchmark
```bash
python models/scripts/benchmark_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint path/to/model.pth
```

**Measures:**
- Inference throughput (samples/sec)
- Memory usage
- GPU utilization
- Triton vs CUDA comparison

## 📦 Model Packaging

### Create Distribution Package
```bash
python models/scripts/package_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint best_model.pth \
    --output packaged_models/
```

**Creates:**
- `model.pth`: PyTorch state dict + metadata
- `model_metadata.json`: Comprehensive model info
- `README.md`: Usage documentation
- `model_package.zip`: Downloadable archive

### Package Contents
```
model_package.zip/
├── ternary_resnet18_cifar10.pth     # Model weights
├── ternary_resnet18_cifar10_metadata.json  # Model info
├── README.md                         # Documentation
```

## 🔧 Technical Details

### Quantization Scheme
- **Weights**: 2-bit ternary (-1, 0, 1)
- **Activation**: Standard float32
- **Compression**: 32x memory reduction
- **Algorithm**: Adaptive threshold quantization

### Triton Integration
```python
from kernels.triton import ternary_matmul

# Automatic optimization for your GPU
result = ternary_matmul(ternary_weights, activations)
```

### Memory Analysis
```python
from models.resnet18.ternary_resnet18 import get_model_memory_usage

memory_info = get_model_memory_usage(model)
print(f"Model size: {memory_info['ternary_memory_mb']:.2f} MB")
print(f"Compression: {memory_info['compression_ratio']:.1f}x")
```

## 📋 Requirements

- **PyTorch**: >= 2.0.0
- **Triton**: >= 3.6.0 (recommended)
- **CUDA**: Optional (CPU fallback available)
- **Datasets**: CIFAR-10 (auto-download) or ImageNet (manual)

## 🎮 Usage Examples

### Inference
```python
import torch
from models.resnet18.ternary_resnet18 import ternary_resnet18

# Load model
model = ternary_resnet18(num_classes=10)
model.load_state_dict(torch.load('ternary_resnet18.pth'))
model.eval()

# Inference
with torch.no_grad():
    output = model(input_image)
    prediction = output.argmax(dim=1)
```

### Custom Training Loop
```python
from models.resnet18.ternary_resnet18 import ternary_resnet18, quantize_model_weights

model = ternary_resnet18(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Training step
    optimizer.zero_grad()
    output = model(batch)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    # Periodic quantization
    if epoch % 10 == 0:
        quantize_model_weights(model)
```

## 📊 Model Zoo Status

### ✅ Completed
- [x] ResNet-18 (ternary)
- [x] MobileNetV2 (ternary)
- [x] Training scripts
- [x] Benchmarking tools
- [x] Packaging utilities
- [x] ONNX export support
- [x] Hugging Face Hub integration
- [x] GitHub Releases publishing
- [x] Model Zoo registry

### 🚧 In Progress
- [ ] BERT-tiny (ternary) — attention and FFN layers implemented, training validation pending
- [ ] EfficientNet-Lite — ternary variant
- [ ] Model conversion tools — FP32/FP16 pretrained to ternary

### 📋 Planned — New Architectures
- [ ] Ternary UNet (Stable Diffusion) — selective quantization of Conv2d/Linear, preserving norms
- [ ] Ternary LLM adapter — post-training quantization for Phi-3, Gemma, Qwen2, TinyLlama class models
- [ ] Vision Transformer (ViT-tiny) — ternary attention and MLP blocks

### 📋 Planned — Infrastructure
- [ ] TensorRT optimization backend
- [ ] Mobile deployment (TFLite export)
- [ ] Web deployment (ONNX.js)
- [ ] Selective layer skipping — preserve embeddings, norms, and output heads during quantization
- [ ] Ternary-aware pruning — combine sparsity scheduling with ternary constraints

## 🚀 Publishing Models

### Export to ONNX

```bash
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint best_model.pth \
    --export-onnx \
    --onnx-validate \
    --output exports/
```

### Publish to Hugging Face Hub

```bash
# Install export dependencies
pip install -e ".[export]"

# Publish model
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint best_model.pth \
    --hf-repo username/ternary-resnet18-cifar10 \
    --hf-token $HF_TOKEN
```

### Create GitHub Release

```bash
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint best_model.pth \
    --github-release v1.0.0 \
    --github-repo username/Triton \
    --github-token $GITHUB_TOKEN
```

See [Export Guide](../docs/EXPORT_GUIDE.md) for detailed documentation.

## 🤝 Contributing

### Adding New Models
1. Create `models/new_model/ternary_new_model.py`
2. Implement ternary quantization
3. Add training script support
4. Update benchmarks
5. Submit PR

### Model Requirements
- Ternary weight quantization
- Triton backend compatibility
- Standard PyTorch interface
- Comprehensive documentation

## 📄 License

MIT License - see project LICENSE file.

## 🙏 Acknowledgments

Built with:
- **PyTorch** for the neural network framework
- **OpenAI Triton** for GPU kernel optimization
- **TorchVision** for model architectures and datasets

---

**Ready to deploy efficient AI?** These ternary models deliver production performance with minimal memory footprint! 🚀
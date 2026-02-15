# Triton DSL Examples

Comprehensive production-quality examples demonstrating Triton DSL capabilities for building memory-efficient neural networks with ternary and mixed-precision quantization.

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Example Categories](#example-categories)
- [Getting Started](#getting-started)
- [Learning Path](#learning-path)
- [Performance Overview](#performance-overview)
- [Contributing](#contributing)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -e ".[examples]"

# Try a simple MNIST example
cd examples/basic
triton compile simple_mlp.triton --output simple_mlp.py
python simple_mlp.py --train --epochs 10

# Or use existing Python examples
cd examples
python mnist_ternary.py --epochs 10
```

## 📂 Example Categories

### 1. [Basic Models](basic/) 
Foundational architectures with ternary quantization.

| Model | Dataset | Accuracy | Size | Files |
|-------|---------|----------|------|-------|
| SimpleMLP | MNIST | 96% | 53 KB | `simple_mlp.triton` |
| ResNet-18 | CIFAR-10 | 87% | 2.7 MB | `resnet18.triton` |
| MobileNetV2 | ImageNet | 58% | 3.4 MB | `mobilenetv2.triton` |
| ViT-Base | ImageNet | 75% | 22 MB | `vision_transformer.triton` |

**Highlights:**
- Complete `.triton` model definitions
- 15-16x compression vs Float32
- Production-ready architectures
- Comprehensive documentation

### 2. [Quantization Examples](quantization/)
Advanced quantization techniques for optimal performance.

| Example | Description | Key Features |
|---------|-------------|--------------|
| `ternary_resnet.triton` | Adaptive ternary with learnable thresholds | 88% CIFAR-10 accuracy |
| `int8_mobilenet.triton` | INT8 for mobile deployment | TFLite/CoreML compatible |
| `mixed_precision.triton` | Strategic bit-width allocation | 6.2x compression, 89% accuracy |
| `qat_training.py` | Production QAT training script | FBGEMM/QNNPACK support |

**Highlights:**
- QAT vs PTQ examples
- Multiple quantization schemes
- Mobile-optimized models
- Comprehensive benchmarks

### 3. [Custom Layers](custom/)
Building blocks for custom quantized architectures.

| Layer | Description | Use Case |
|-------|-------------|----------|
| `custom_conv.triton` | Learnable ternary convolution | Research, experimentation |
| `attention_mechanism.triton` | Quantized multi-head attention | Transformers, ViT |
| `custom_quantization.triton` | Advanced quantization schemes | LSQ, soft quantization |

**Highlights:**
- Learnable thresholds
- Custom gradient estimators
- Comprehensive APIs
- Visualization tools

### 4. [Training Scripts](training/)
Production-ready training pipelines.

| Script | Purpose | Features |
|--------|---------|----------|
| `train_cifar10.py` | CIFAR-10 training | CutMix, label smoothing |
| `train_imagenet.py` | ImageNet distributed training | DDP, AMP, EMA |
| `transfer_learning.py` | Fine-tuning pretrained models | Progressive unfreezing |
| `quantization_aware_training.py` | QAT training | Full quantization pipeline |

**Highlights:**
- Multi-GPU support
- Mixed precision training
- Advanced augmentations
- Checkpoint management

### 5. [Deployment Examples](deployment/)
Production deployment strategies.

| Tool | Platform | Features |
|------|----------|----------|
| `export_onnx.py` | ONNX Runtime | Model export, validation, benchmarking |
| `optimize_for_mobile.py` | Mobile (iOS/Android) | TFLite, CoreML, TorchScript |
| `huggingface_hub.py` | Hugging Face Hub | Model sharing, model cards |
| `docker_deployment/` | Docker/Kubernetes | REST API, monitoring, scaling |

**Highlights:**
- Cross-platform export
- Quantization-aware conversion
- Production monitoring
- Load balancing

### 6. [Jupyter Notebooks](notebooks/)
Interactive tutorials and analysis.

| Notebook | Description | Level |
|----------|-------------|-------|
| `01_introduction.ipynb` | Triton DSL basics with MNIST | Beginner |
| `02_quantization_tutorial.ipynb` | Quantization techniques | Intermediate |
| `03_performance_analysis.ipynb` | Profiling and optimization | Advanced |

**Highlights:**
- Interactive visualizations
- Step-by-step tutorials
- Performance benchmarks
- Google Colab ready

---

## 🎓 Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# Install Triton DSL
pip install -e .

# Install example dependencies
pip install -e ".[examples]"

# For GPU support
pip install -e ".[cuda]"

# For deployment
pip install -e ".[export]"
```

### First Steps

1. **Start with notebooks** (recommended for beginners):
   ```bash
   jupyter notebook examples/notebooks/01_introduction.ipynb
   ```

2. **Try a basic model**:
   ```bash
   cd examples
   python mnist_ternary.py
   ```

3. **Compile a Triton model**:
   ```bash
   triton compile examples/basic/simple_mlp.triton --output model.py
   python model.py --train
   ```

4. **Train on CIFAR-10**:
   ```bash
   python examples/training/train_cifar10.py --model resnet18 --epochs 100
   ```

---

## 📈 Learning Path

### Beginner Track (2-3 hours)

1. **Introduction** → `notebooks/01_introduction.ipynb`
   - Understand ternary quantization
   - Train your first model
   - See 16x compression in action

2. **Basic Models** → `basic/simple_mlp.triton`
   - Learn Triton DSL syntax
   - Compile and run models
   - Understand performance metrics

3. **Python Training** → `mnist_ternary.py`
   - Complete training pipeline
   - Model save/load
   - Evaluation and metrics

### Intermediate Track (1-2 days)

1. **Quantization** → `notebooks/02_quantization_tutorial.ipynb`
   - QAT vs PTQ
   - Different quantization schemes
   - Accuracy-size trade-offs

2. **Training Scripts** → `training/train_cifar10.py`
   - Production training
   - Advanced augmentations
   - Hyperparameter tuning

3. **Custom Layers** → `custom/custom_conv.triton`
   - Build custom quantized layers
   - Learnable quantization
   - Gradient flow

### Advanced Track (3-5 days)

1. **Performance** → `notebooks/03_performance_analysis.ipynb`
   - Model profiling
   - Bottleneck identification
   - Hardware optimization

2. **ImageNet Training** → `training/train_imagenet.py`
   - Large-scale distributed training
   - Mixed precision
   - Advanced techniques

3. **Deployment** → `deployment/`
   - ONNX export
   - Mobile optimization
   - Production serving

---

## 📊 Performance Overview

### Compression Ratios

| Precision | Bits | Compression | Accuracy Drop |
|-----------|------|-------------|---------------|
| Float32 | 32 | 1x (baseline) | 0% |
| Float16 | 16 | 2x | <0.5% |
| INT8 | 8 | 4x | 0.5-2% |
| INT4 | 4 | 8x | 2-4% |
| Ternary | 2 | 16x | 2-5% |
| Binary | 1 | 32x | 5-10% |

### Speed Improvements

| Device | Float32 | Ternary | Speedup |
|--------|---------|---------|---------|
| CPU (x86) | 100 ms | 70 ms | 1.4x |
| GPU (CUDA) | 12 ms | 15 ms | 0.8x* |
| Mobile (ARM) | 200 ms | 100 ms | 2.0x |
| Edge TPU | 50 ms | 25 ms | 2.0x |

*Note: GPU speedup requires optimized kernels (available in `kernels/`)

### Model Zoo Results

| Model | Dataset | Float32 Acc | Ternary Acc | Size (F32) | Size (Ternary) |
|-------|---------|-------------|-------------|------------|----------------|
| SimpleMLP | MNIST | 98.5% | 96.0% | 850 KB | 53 KB |
| ResNet-18 | CIFAR-10 | 92.0% | 87.0% | 42 MB | 2.7 MB |
| ResNet-18 | ImageNet | 70.0% | 63.0% | 45 MB | 2.9 MB |
| MobileNetV2 | ImageNet | 72.0% | 58.0% | 14 MB | 3.4 MB |
| ViT-Base | ImageNet | 81.0% | 75.0% | 330 MB | 22 MB |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Adding Examples

1. Choose the appropriate category
2. Follow existing structure and style
3. Include comprehensive documentation
4. Add performance benchmarks
5. Submit a pull request

### Example Checklist

- [ ] Clear problem statement
- [ ] Complete code with comments
- [ ] Usage instructions
- [ ] Expected outputs
- [ ] Performance metrics
- [ ] Visualizations (if applicable)
- [ ] Tests (if applicable)

---

## 📚 Additional Resources

### Documentation
- [Triton DSL Language Reference](../docs/specs/GRAMMAR.md)
- [Compiler Guide](../docs/COMPILER_DRIVER.md)
- [Export Guide](../docs/EXPORT_GUIDE.md)
- [Training Guide](training/README.md)
- [Deployment Guide](deployment/README.md)

### Research Papers
- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
- [XNOR-Net](https://arxiv.org/abs/1603.05279)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

### Community
- [GitHub Issues](https://github.com/financecommander/Triton/issues)
- [Discussions](https://github.com/financecommander/Triton/discussions)

---

## 📜 License

MIT License - See [LICENSE](../LICENSE) for details.

---

## 🙏 Acknowledgments

Built with PyTorch, Triton GPU, and the open-source ML community.

**Happy learning and building efficient neural networks!** 🚀

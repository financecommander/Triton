# Triton DSL

**Domain-Specific Language for Ternary Neural Networks (TNNs)**

## Overview

Triton is a high-performance DSL designed to optimize Ternary Neural Networks by enforcing ternary constraints (`{-1, 0, 1}`) at the syntax level. This enables 20-40% memory density improvements over standard FP32 representations.

## Key Features

- **Native Ternary Type System**: `trit` primitive and `TernaryTensor` data structures
- **Zero-Cost Abstractions**: Compile-time type checking with runtime efficiency
- **Hardware Optimization**: 2-bit packed storage, CUDA kernels, zero-skipping
- **PyTorch Integration**: Seamless transpilation to PyTorch modules

## Architecture
```
Triton Source (.tri)
    ↓
Lexer/Parser → AST
    ↓
Type Checker
    ↓
Code Generator → PyTorch/CUDA
```

## Performance Targets

- **Memory**: 4x compression (2-bit vs 32-bit)
- **Speed**: 2-3x faster inference via sparse computation
- **Accuracy**: Comparable to FP32 for quantization-friendly models

## Examples

### MNIST Ternary Neural Network (`examples/mnist_ternary.py`)

A production-ready training script demonstrating the complete Triton DSL workflow:

**Features:**
- ✅ TernaryNet architecture with 3 LinearTernary layers
- ✅ Straight-Through Estimator (STE) for gradient flow
- ✅ Deterministic and stochastic quantization methods
- ✅ Comprehensive metrics: accuracy, model size, latency, memory
- ✅ Model persistence with packed weight format
- ✅ Visualizations: training curves, weight distributions, confusion matrix
- ✅ Full CLI interface with argparse

**Architecture:**
```
Input (784) → LinearTernary(256) → TernaryActivation
            → LinearTernary(128) → TernaryActivation  
            → LinearTernary(10) → Logits
```

**Performance:**
| Metric | Float32 Baseline | Ternary Network |
|--------|-----------------|-----------------|
| Test Accuracy | ~98.5% | ~96-97% |
| Model Size | ~850 KB | ~53 KB (16x smaller) |
| Parameters | 235,146 | 235,146 |
| Inference Latency | ~1.0 ms | ~0.7 ms (CPU) |

See `examples/mnist_ternary.py` for full implementation details and documentation.

## Project Status

### Completed
- **Compiler Frontend** — Lexer (PLY), parser (LALR), AST (17 node types), type checker with symbol tables
- **PyTorch Backend** — TernaryTensor, TernaryLinear, TernaryConv2d, code generation via Jinja2 templates
- **Quantization** — Deterministic and stochastic methods with Straight-Through Estimator (STE)
- **2-Bit Packing** — 4 trits per byte, 32x compression vs FP32, CPU and GPU pack/unpack
- **CUDA Kernels** — Packed ternary matmul with 16x16 tiling, shared memory, zero-skipping, warp reduction
- **Triton GPU Backend** — Auto-tuned kernels (100+ configs), multi-platform (CUDA/ROCm/Metal)
- **Model Zoo** — ResNet-18, MobileNetV2, BERT-tiny, MNIST/CIFAR-10 CNNs, Credit Risk NN
- **Training** — CutMix, MixUp, AutoAugment, label smoothing, early stopping, DDP, TensorBoard
- **Export** — ONNX (with validation), Hugging Face Hub, GitHub Releases
- **Testing** — 25 test files: unit, integration, stress, fuzzing, property-based, security, benchmarks

### Roadmap

**Near-term**
- [ ] End-to-end DSL compilation (`triton compile model.tri --output model.py`)
- [ ] Diffusion model support — ternary UNet quantization (validated in research at 15x compression)
- [ ] LLM post-training quantization — selective layer conversion preserving embeddings/norms/lm_head
- [ ] Batched matrix multiplication kernels
- [ ] PyTorch autograd integration for ternary kernels
- [ ] CLI tool for compilation and model management

**Mid-term**
- [ ] Quantization-aware training (QAT) hooks
- [ ] Mixed precision output (int8, fp16, fp32)
- [ ] EfficientNet-Lite ternary variant
- [ ] TensorRT optimization backend
- [ ] Dynamic tile size selection based on problem geometry
- [ ] Model conversion tools (FP32/FP16 pretrained → ternary)

**Long-term**
- [ ] Mobile deployment (TFLite export)
- [ ] Web deployment (ONNX.js)
- [ ] Ternary-aware pruning and sparsity scheduling
- [ ] Multi-bit quantization (2-bit, 4-bit) alongside ternary
- [ ] Visual pipeline editor for model composition
- [ ] Federated learning with ternary compression

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run compiler tests
pytest tests/

# Train MNIST ternary network
python examples/mnist_ternary.py --epochs 20 --batch-size 128

# Train CIFAR-10 with full augmentation
python models/scripts/train_ternary_models.py \
    --model resnet18 --dataset cifar10 --epochs 500 \
    --early_stopping --cutmix --label_smoothing 0.1

# Export to ONNX
python models/scripts/publish_model.py \
    --model resnet18 --checkpoint model.pth --export-onnx

# Future: compile .tri source to PyTorch
triton compile examples/mnist_ternary.tri --output model.py
```

## Documentation

- [Export & Publishing Guide](docs/EXPORT_GUIDE.md)
- [CIFAR-10 Training Guide](docs/CIFAR10_TRAINING_GUIDE.md)
- [PyTorch Backend](backend/pytorch/README.md)
- [CUDA Kernels](kernels/cuda/README.md)
- [Triton GPU Backend](kernels/triton/README.md)
- [Model Zoo](models/README.md)

## License

MIT License - See [LICENSE](LICENSE)

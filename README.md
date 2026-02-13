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

🚧 **Active Development** - Phase 1: Compiler Frontend

## Quick Start

### Running the MNIST Example

The project includes a complete MNIST training example demonstrating ternary neural networks:

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib seaborn scikit-learn

# Run MNIST training with default settings (10 epochs)
python examples/mnist_ternary.py

# Custom training configuration
python examples/mnist_ternary.py --epochs 20 --batch-size 128 --lr 0.001

# Use stochastic quantization
python examples/mnist_ternary.py --quantize-method stochastic

# Save trained model
python examples/mnist_ternary.py --save-path ./models/mnist_ternary.pth

# Run unit tests
python examples/test_mnist_ternary.py
```

**Expected Results:**
- Test accuracy: ~96-97% (compared to ~98.5% for float32 baseline)
- Model size: ~0.06 MB (16x smaller than float32)
- Inference: 2-3x faster on optimized hardware

### Model Export & Publishing

Export and publish your trained ternary models:

```bash
# Install export dependencies
pip install -e ".[export]"

# Export to ONNX
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --export-onnx

# Publish to Hugging Face Hub
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --hf-repo username/ternary-resnet18

# Create GitHub Release
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --github-release v1.0.0 \
    --github-repo username/Triton
```

See [Export Guide](docs/EXPORT_GUIDE.md) for detailed documentation.

### Compiler Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run compiler tests
pytest tests/

# Future: Compile Triton DSL to PyTorch
triton compile examples/mnist_ternary.tri --output model.py
```

## Documentation

- [Technical Specification](docs/specs/TECHNICAL_SPEC.md)
- [Grammar Reference](docs/specs/GRAMMAR.md)
- [Export & Publishing Guide](docs/EXPORT_GUIDE.md)
- [API Documentation](docs/api/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - See [LICENSE](LICENSE)
 Principal Triton Language Designer &amp; ML Architect

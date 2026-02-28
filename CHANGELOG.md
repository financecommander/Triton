## [Unreleased]

### Planned
- End-to-end DSL compilation (`triton compile model.tri --output model.py`)
- Diffusion model support — ternary UNet quantization
- LLM post-training quantization with selective layer preservation
- Batched ternary matmul kernels
- PyTorch autograd integration for ternary ops
- Fused kernel chains (matmul + activation in single launch)
- CPU fallback with SIMD acceleration (AVX2/NEON)
- TensorRT / TFLite / ONNX.js export targets

## [0.3.0] - 2026-02-27

### Added
- **Training Infrastructure**
  - CIFAR-10 and MNIST full training pipelines
  - CutMix, MixUp, AutoAugment, RandAugment data augmentation
  - Label smoothing cross-entropy loss
  - Early stopping with configurable patience
  - Checkpoint resume with full state restoration
  - TensorBoard and CSV logging
  - DDP distributed training support
- **Model Zoo**
  - ResNet-18 ternary (CIFAR-10: 90-92% accuracy target)
  - MobileNetV2 ternary with inverted residuals
  - BERT-tiny ternary (attention + feed-forward)
  - Credit Risk NN with custom tokenizer
- **Export Pipeline**
  - ONNX export with validation and optimization
  - Hugging Face Hub publishing with model cards
  - GitHub Releases with automated packaging
- **Testing**
  - 25 test files across unit, integration, stress, fuzzing, property-based, and security
  - Benchmark suite for inference, matmul, and memory

### Changed
- Project status updated from "Phase 1: Compiler Frontend" to reflect completed state

## [0.2.0] - 2026-02-13

### Added
- **Triton GPU Backend** - Complete replacement for hand-written CUDA kernels
  - Auto-tuning framework with 100+ configurations
  - Multi-platform support (CUDA/ROCm/Metal)
  - A100/H100 GPU optimization
  - CPU fallback mode
- **Performance Improvements**
  - 20%+ speedup vs CUDA implementation
  - Auto-tuned block sizes (16×16 to 256×256)
  - Optimized pipeline stages (2-5)
  - Configurable warp counts (2-8)
- **New Testing Infrastructure**
  - CPU validation tests
  - GPU performance tests
  - Benchmark comparison tools
  - Integration demos
- **Documentation**
  - Comprehensive Triton backend README
  - Migration guide from CUDA
  - Performance benchmarking guide

### Changed
- Default backend remains CUDA for compatibility
- Triton backend available as opt-in upgrade

### Performance
- Matrix multiplication: 20%+ faster on A100/H100
- Memory efficiency: Maintained 4x compression (2-bit packing)
- Portability: Now works on NVIDIA/AMD/Apple GPUs

### Deprecated
- None - CUDA backend still fully supported

### Migration
- Drop-in replacement: `from kernels.triton import ternary_matmul`
- Zero code changes required for existing users
- Benchmark before switching: `python kernels/triton/benchmark_triton_vs_cuda.py`

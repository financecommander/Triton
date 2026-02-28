# Changelog

All notable changes to Triton DSL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-15

### Added
- **v1.0 Stable Release** - Production-ready release of Triton DSL
- **Complete Compiler Pipeline**
  - Lexer, Parser, AST, Type Checker, Code Generator
  - Compiler driver with CLI and Python API
  - 4 optimization levels (O0-O3)
  - Multiple backends (PyTorch, ONNX, TFLite, Python)
- **Type System**
  - Production-quality type checker with inference and unification
  - Effect tracking and comprehensive error reporting
  - Performance caching
- **Triton GPU Backend**
  - Auto-tuning framework with 100+ configurations
  - Multi-platform support (CUDA/ROCm/Metal)
  - A100/H100 GPU optimization
  - CPU fallback mode
- **Model Export**
  - ONNX export with custom ternary operations
  - HuggingFace Hub publishing
  - GitHub Releases integration
- **Documentation**
  - Complete API reference
  - Quick start guides
  - CIFAR-10 training guide
  - Export and deployment guides
  - Migration guide from v0.x

### Changed
- Version bumped from 0.2.0 to 1.0.0 (production stable)
- Development Status classifier updated to "Production/Stable"
- Added Python 3.12 support to classifiers
- MANIFEST.in added for proper source distributions

### Performance
- Matrix multiplication: 20%+ faster on A100/H100
- Memory efficiency: 4x compression (2-bit packing)
- Portability: NVIDIA/AMD/Apple GPU support
- Inference: 2-3x faster via sparse computation

### Migration
- See [MIGRATION.md](MIGRATION.md) for upgrade guide from v0.x to v1.0

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

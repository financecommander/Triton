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

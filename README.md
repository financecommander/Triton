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

## Project Status

🚧 **Active Development** - Phase 1: Compiler Frontend

## Quick Start
```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Example usage
triton compile examples/mnist_ternary.tri --output model.py
```

## Documentation

- [Technical Specification](docs/specs/TECHNICAL_SPEC.md)
- [Grammar Reference](docs/specs/GRAMMAR.md)
- [API Documentation](docs/api/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - See [LICENSE](LICENSE)
 Principal Triton Language Designer &amp; ML Architect

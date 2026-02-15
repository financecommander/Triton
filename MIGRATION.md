# Migration Guide: v0.x → v1.0

## Overview

Triton DSL v1.0.0 is the first stable release. This guide covers upgrading from
v0.1.0 or v0.2.0 to v1.0.0.

## Version Changes

| Component | v0.x | v1.0 |
|-----------|------|------|
| Python | ≥3.10 | ≥3.10 |
| PyTorch | ≥2.1.0 | ≥2.1.0 |
| Package status | Alpha | Stable |
| ONNX (export) | ≥1.17.0 | ≥1.17.0 |

## Breaking Changes

There are **no breaking API changes** in v1.0.0. The release focuses on
stabilizing existing functionality.

## Upgrade Steps

### 1. Update the Package

```bash
pip install --upgrade triton-dsl==1.0.0
```

### 2. Verify Installation

```python
from compiler import __version__
print(__version__)  # Should print "1.0.0"
```

### 3. Triton GPU Backend (from v0.2.0)

If you adopted the Triton GPU backend in v0.2.0, no changes are needed:

```python
# This still works exactly the same
from kernels.triton import ternary_matmul
```

### 4. ONNX Export (Security Update)

Ensure you are using ONNX ≥1.17.0 to avoid known security vulnerabilities:

```bash
pip install "onnx>=1.17.0"
```

## New Features in v1.0

### Compiler Driver

The compiler driver provides CLI and Python API access:

```bash
# CLI usage
triton compile model.tri --backend pytorch --optimize O2

# Cache management
triton cache info
triton cache clear
```

```python
# Python API
from compiler.driver import compile_model

result = compile_model("model.tri", backend="pytorch", optimize="O2")
```

### Type Checker Improvements

The type checker now includes:
- Full type inference with unification
- Effect tracking
- Comprehensive error messages with suggestions
- Performance caching

## Troubleshooting

### Import Errors

If you see import errors after upgrading, ensure your installation is clean:

```bash
pip uninstall triton-dsl
pip install triton-dsl==1.0.0
```

### Compatibility

v1.0.0 maintains full backward compatibility with v0.2.0 APIs. If you
encounter any issues, please file a bug at:
https://github.com/financecommander/Triton/issues

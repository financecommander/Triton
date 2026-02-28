# Triton Compiler Quick Reference

## Installation

```bash
pip install triton-dsl
```

## CLI Commands

### Compile

```bash
# Basic
triton compile model.triton

# With options
triton compile model.triton --O2 --target pytorch -o output.py

# Verbose with statistics
triton compile model.triton -v --statistics
```

### Cache Management

```bash
triton cache clear    # Clear cache
triton cache info     # Show cache info
```

### Version

```bash
triton version
```

## Python API

### Basic Usage

```python
from triton.compiler.driver import compile_model

result = compile_model('model.triton')

if result.success:
    print(f"Output: {result.output_file}")
```

### With Options

```python
result = compile_model(
    'model.triton',
    output_file='model.py',
    optimization_level=2,
    target='pytorch',
    verbose=True
)
```

## Options Quick Reference

| Option | Values | Default |
|--------|--------|---------|
| `--O0`, `--O1`, `--O2`, `--O3` | - | `--O1` |
| `--target` | `pytorch`, `onnx`, `tflite`, `python` | `pytorch` |
| `--format` | `py`, `onnx`, `json`, `bin` | `py` |
| `-o, --output` | file path | auto |
| `-v, --verbose` | flag | off |
| `--debug` | flag | off |
| `-q, --quiet` | flag | off |
| `--Werror` | flag | off |
| `--no-cache` | flag | cache on |
| `--force` | flag | use cache |
| `--statistics` | flag | off |
| `--optimization-report` | flag | off |

## Optimization Levels

- **O0**: No optimization (fastest compile)
- **O1**: Basic optimization (default)
- **O2**: Aggressive optimization
- **O3**: Maximum optimization (slowest compile)

## Target Backends

- **pytorch**: PyTorch code with nn.Module
- **onnx**: ONNX export code
- **tflite**: TensorFlow Lite code
- **python**: Pure Python with NumPy

## Error Format

```
<file>:<line>:<col>: error: [<stage>] <message>
```

## Common Workflows

### Development

```bash
# Fast iteration with cache
triton compile model.triton --O1
```

### Production Build

```bash
# Optimized build without cache
triton compile model.triton --O2 --force --statistics
```

### Debug Build

```bash
# Full diagnostics
triton compile model.triton --O0 --debug --statistics
```

### Batch Compilation

```python
import glob
from triton.compiler.driver import compile_model

for f in glob.glob('models/*.triton'):
    result = compile_model(f, optimization_level=2)
    print(f"{'✓' if result.success else '✗'} {f}")
```

## Cache Management

```bash
# Clear cache
triton cache clear

# Disable cache
triton compile model.triton --no-cache

# Force recompile
triton compile model.triton --force

# Custom cache location
triton compile model.triton --cache-dir /tmp/cache
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Slow compilation | Use `--O0` or `--O1` |
| Cache not working | Run `triton cache clear` |
| Out of memory | Lower optimization level |
| Syntax errors | Check source file syntax |

## Links

- [Full Documentation](COMPILER_DRIVER.md)
- [Repository](https://github.com/financecommander/Triton)
- [Issues](https://github.com/financecommander/Triton/issues)

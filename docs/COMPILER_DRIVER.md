# Triton Compiler Driver Documentation

## Overview

The Triton Compiler Driver (`triton.compiler.driver`) is the main compilation orchestrator for Triton DSL programs. It manages the entire compilation pipeline from source code to executable output, with support for multiple target backends, optimization levels, and comprehensive diagnostics.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Interface](#cli-interface)
- [Python API](#python-api)
- [Compilation Pipeline](#compilation-pipeline)
- [Optimization Levels](#optimization-levels)
- [Target Backends](#target-backends)
- [Caching System](#caching-system)
- [Diagnostics](#diagnostics)
- [Error Handling](#error-handling)
- [Advanced Usage](#advanced-usage)

## Installation

The compiler driver is part of the Triton DSL package:

```bash
pip install triton-dsl
```

For development with all features:

```bash
pip install triton-dsl[dev]
```

## Quick Start

### CLI Usage

```bash
# Compile a Triton source file
triton compile model.triton

# With optimization
triton compile model.triton --O2

# Specify output and target
triton compile model.triton -o model.py --target pytorch

# Enable verbose mode with statistics
triton compile model.triton -v --statistics
```

### Python API

```python
from triton.compiler.driver import compile_model

# Basic compilation
result = compile_model('model.triton')

if result.success:
    print(f"Compiled successfully to: {result.output_file}")
else:
    for error in result.errors:
        print(error)
```

## CLI Interface

### Commands

#### `triton compile`

Compile a Triton DSL source file.

**Syntax:**
```bash
triton compile <source> [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `-o, --output FILE` | Specify output file path |
| `--format FORMAT` | Output format: `py`, `onnx`, `json`, `bin` (default: `py`) |
| `--O0` | No optimization |
| `--O1` | Basic optimization (default) |
| `--O2` | Aggressive optimization |
| `--O3` | Maximum optimization |
| `--target TARGET` | Target backend: `pytorch`, `onnx`, `tflite`, `python` (default: `pytorch`) |
| `-v, --verbose` | Enable verbose output |
| `--debug` | Enable debug mode |
| `-q, --quiet` | Suppress progress output |
| `--Werror` | Treat warnings as errors |
| `--no-cache` | Disable compilation cache |
| `--force` | Force recompilation (ignore cache) |
| `--cache-dir DIR` | Custom cache directory |
| `--statistics` | Show compilation statistics |
| `--optimization-report` | Show optimization report |
| `--profile` | Enable performance profiling |

**Examples:**

```bash
# Basic compilation
triton compile model.triton

# Aggressive optimization for PyTorch
triton compile model.triton --O2 --target pytorch

# Compile for ONNX with custom output
triton compile model.triton -o exported_model.onnx --target onnx --format onnx

# Debug mode with full statistics
triton compile model.triton --debug --statistics --optimization-report

# Force recompilation without cache
triton compile model.triton --force --no-cache

# Quiet mode (for scripts)
triton compile model.triton -q
```

#### `triton cache`

Manage compilation cache.

**Subcommands:**

- `triton cache clear` - Clear all cached compilations
- `triton cache info` - Show cache information

**Examples:**

```bash
# Clear cache
triton cache clear

# Show cache info
triton cache info
```

#### `triton version`

Show version information.

```bash
triton version
```

## Python API

### `compile_model()`

Main function to compile Triton DSL models.

```python
def compile_model(
    source_file: str,
    output_file: Optional[str] = None,
    optimization_level: int = 1,
    target: str = "pytorch",
    verbose: bool = False,
    use_cache: bool = True,
    **kwargs
) -> CompilationResult
```

**Parameters:**

- `source_file` (str): Path to the source `.triton` file
- `output_file` (str, optional): Output file path. If not specified, derived from source filename
- `optimization_level` (int): Optimization level (0-3, default: 1)
- `target` (str): Target backend (`'pytorch'`, `'onnx'`, `'tflite'`, `'python'`)
- `verbose` (bool): Enable verbose output
- `use_cache` (bool): Enable compilation caching
- `**kwargs`: Additional compilation options

**Returns:**

`CompilationResult` object with:
- `success` (bool): Whether compilation succeeded
- `output_file` (str): Path to generated output file
- `ast` (Node): Abstract Syntax Tree
- `ir` (Any): Intermediate Representation
- `errors` (List[CompilationError]): Compilation errors
- `warnings` (List[CompilationError]): Compilation warnings
- `statistics` (CompilationStatistics): Compilation statistics
- `optimization_report` (dict): Optimization report

**Examples:**

```python
from triton.compiler.driver import compile_model

# Basic usage
result = compile_model('model.triton')

# With options
result = compile_model(
    'model.triton',
    output_file='model.py',
    optimization_level=2,
    target='pytorch',
    verbose=True
)

# Check result
if result.success:
    print(f"Success! Output: {result.output_file}")
    print(f"Compiled {result.statistics.lines_of_code} lines")
    print(f"Generated {result.statistics.ast_nodes} AST nodes")
else:
    print("Compilation failed:")
    for error in result.errors:
        print(f"  {error}")

# Access generated code
with open(result.output_file, 'r') as f:
    generated_code = f.read()
```

### `TritonCompiler` Class

For advanced usage, you can instantiate the compiler directly:

```python
from triton.compiler.driver import TritonCompiler, CompilationOptions, OptimizationLevel, TargetBackend

# Create compilation options
options = CompilationOptions(
    source_file='model.triton',
    output_file='model.py',
    optimization_level=OptimizationLevel.O2,
    target_backend=TargetBackend.PYTORCH,
    verbose=True,
    show_statistics=True
)

# Create and run compiler
compiler = TritonCompiler(options)
result = compiler.compile()
```

## Compilation Pipeline

The compiler executes the following stages in order:

1. **Lexical Analysis (Lexer)**
   - Tokenizes source code
   - Identifies keywords, operators, literals, identifiers
   
2. **Syntax Analysis (Parser)**
   - Generates Abstract Syntax Tree (AST)
   - Validates syntax structure
   
3. **Type Checking**
   - Validates type consistency
   - Checks function signatures
   - Verifies tensor dimensions
   
4. **Semantic Analysis**
   - Symbol resolution
   - Scope checking
   - Control flow analysis
   - Dead code detection
   
5. **IR Generation**
   - Converts AST to Intermediate Representation
   - Prepares for optimization
   
6. **Optimization**
   - Runs optimization passes based on level
   - Constant folding
   - Dead code elimination
   - Common subexpression elimination
   - Loop optimization
   
7. **Code Generation**
   - Generates target-specific code
   - Produces PyTorch/ONNX/TFLite/Python output
   
8. **Output Writing**
   - Writes generated code to file
   - Updates cache

### Pipeline Stages Timing

You can see how long each stage takes by using `--statistics`:

```bash
triton compile model.triton --statistics
```

Output:
```
=== Compilation Statistics ===
Total Time: 0.036s
  - Lexer: 0.001s
  - Parser: 0.003s
  - Type Checker: 0.002s
  - Semantic Analyzer: 0.001s
  - IR Generation: 0.005s
  - Optimization: 0.010s (4 passes)
  - Code Generation: 0.008s
  - Output Writing: 0.006s

Lines of Code: 50
AST Nodes: 234
Peak Memory: 45.2 MB
Cache Hit: No
```

## Optimization Levels

The compiler supports four optimization levels:

### O0 - No Optimization

- Fastest compilation
- Largest output
- Easiest to debug
- No optimization passes

```bash
triton compile model.triton --O0
```

### O1 - Basic Optimization (Default)

- Fast compilation
- Moderate optimizations
- Good balance
- Includes:
  - Constant folding
  - Dead code elimination

```bash
triton compile model.triton --O1
```

### O2 - Aggressive Optimization

- Slower compilation
- Better runtime performance
- Includes O1 plus:
  - Common subexpression elimination
  - Inline expansion

```bash
triton compile model.triton --O2
```

### O3 - Maximum Optimization

- Slowest compilation
- Best runtime performance
- Includes O2 plus:
  - Loop optimization
  - Aggressive inlining

```bash
triton compile model.triton --O3
```

### Optimization Report

View detailed optimization information:

```bash
triton compile model.triton --O2 --optimization-report
```

Output:
```json
{
  "level": "O2",
  "passes": [
    "constant_folding",
    "dead_code_elimination",
    "common_subexpression_elimination",
    "inline_expansion"
  ],
  "transformations": []
}
```

## Target Backends

### PyTorch (Default)

Generates PyTorch code with `nn.Module` classes.

```bash
triton compile model.triton --target pytorch
```

**Output example:**
```python
import torch
import torch.nn as nn
from backend.pytorch.ternary_tensor import TernaryTensor

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model initialization
        pass
    
    def forward(self, x):
        # Forward pass implementation
        return x

model = GeneratedModel()
```

### ONNX

Generates ONNX export code.

```bash
triton compile model.triton --target onnx
```

### TFLite

Generates TensorFlow Lite compatible code.

```bash
triton compile model.triton --target tflite
```

### Python

Generates pure Python code with NumPy.

```bash
triton compile model.triton --target python
```

## Caching System

The compiler caches compilation results to speed up repeated builds.

### How Caching Works

1. **Cache Key Generation**: Hash of source file content + compilation options
2. **Cache Storage**: Serialized `CompilationResult` objects stored in `~/.triton/cache/`
3. **Cache Validation**: Checks source file modification time
4. **Cache Invalidation**: Automatic on source file changes

### Cache Management

```bash
# Clear all cached compilations
triton cache clear

# Show cache information
triton cache info
```

**Output:**
```
Cache directory: /home/user/.triton/cache
Cache entries: 15
Total size: 23.45 MB
```

### Controlling Cache Behavior

```bash
# Disable cache for a single compilation
triton compile model.triton --no-cache

# Force recompilation (ignore cache)
triton compile model.triton --force

# Use custom cache directory
triton compile model.triton --cache-dir /tmp/triton-cache
```

### Python API

```python
from triton.compiler.driver import CompilationCache

# Create cache manager
cache = CompilationCache('/custom/cache/dir')

# Clear cache
cache.clear()

# Invalidate specific file
cache.invalidate('model.triton')
```

## Diagnostics

### Compilation Statistics

View detailed compilation statistics:

```bash
triton compile model.triton --statistics
```

### Performance Profiling

Enable performance profiling:

```bash
triton compile model.triton --profile
```

### Memory Tracking

Memory usage is automatically tracked and reported in statistics.

### Debug Mode

Enable debug mode for detailed logging:

```bash
triton compile model.triton --debug
```

**Output includes:**
- Stack traces for errors
- Detailed stage information
- Internal compiler state

## Error Handling

### Error Types

The compiler reports errors at different stages:

1. **Lexer Errors**: Invalid tokens
2. **Parser Errors**: Syntax errors
3. **Type Errors**: Type mismatches
4. **Semantic Errors**: Undefined symbols, scope errors
5. **Code Generation Errors**: Backend-specific issues

### Error Format

```
<file>:<line>:<column>: error: [<stage>] <message>
```

**Example:**
```
model.triton:10:5: error: [type_checker] Type mismatch: expected int8, got float32
```

### Warnings

Warnings are reported but don't stop compilation:

```
model.triton:15:8: warning: [semantic_analyzer] Unused variable 'x'
```

### Warnings as Errors

Treat warnings as errors:

```bash
triton compile model.triton --Werror
```

### Error Recovery

The compiler attempts to continue after errors when possible to report multiple issues in one pass.

## Advanced Usage

### Jupyter Notebook Integration

```python
# In a Jupyter notebook
from triton.compiler.driver import compile_model

%%capture --no-display
result = compile_model('model.triton', verbose=False)

if result.success:
    # Execute generated code
    exec(open(result.output_file).read())
    
    # Use the model
    model = GeneratedModel()
    output = model(torch.randn(1, 10))
```

### Build System Integration

**Makefile example:**
```makefile
TRITON_SOURCES = $(wildcard models/*.triton)
TRITON_OUTPUTS = $(TRITON_SOURCES:.triton=.py)

%.py: %.triton
	triton compile $< -o $@ --O2 --target pytorch

all: $(TRITON_OUTPUTS)

clean:
	rm -f $(TRITON_OUTPUTS)
	triton cache clear
```

### Custom Compilation Pipeline

```python
from triton.compiler.driver import TritonCompiler, CompilationOptions

class CustomCompiler(TritonCompiler):
    def _optimize_ir(self, ir):
        # Custom optimization logic
        ir = super()._optimize_ir(ir)
        # Additional passes
        return ir

options = CompilationOptions(source_file='model.triton')
compiler = CustomCompiler(options)
result = compiler.compile()
```

### Batch Compilation

```python
import glob
from triton.compiler.driver import compile_model

# Compile all .triton files in a directory
for source_file in glob.glob('models/*.triton'):
    result = compile_model(
        source_file,
        optimization_level=2,
        verbose=False
    )
    
    if result.success:
        print(f"✓ {source_file}")
    else:
        print(f"✗ {source_file}: {result.errors[0]}")
```

### VS Code Extension Integration

The compiler provides hooks for IDE integration:

```python
from triton.compiler.driver import compile_model

# Compile with IDE-friendly output
result = compile_model(
    'model.triton',
    verbose=False,
    show_progress=False
)

# Extract diagnostics for IDE
diagnostics = []
for error in result.errors + result.warnings:
    diagnostics.append({
        'file': error.source_file,
        'line': error.lineno,
        'column': error.col_offset,
        'severity': 'error' if not error.is_warning else 'warning',
        'message': error.message
    })
```

## Best Practices

1. **Use Caching**: Enable caching for faster iterative development
2. **Start with O1**: Use O1 during development, O2 for production
3. **Check Statistics**: Monitor compilation times and optimize hot paths
4. **Handle Errors**: Always check `result.success` before using output
5. **Clean Cache**: Periodically clear cache to free disk space
6. **Use Quiet Mode**: Use `-q` in scripts to avoid progress output
7. **Version Control**: Commit generated files separately or add to `.gitignore`

## Troubleshooting

### Compilation Takes Too Long

- Lower optimization level (`--O0` or `--O1`)
- Check if cache is being used (`--statistics`)
- Profile compilation (`--profile`)

### Cache Not Working

- Check cache directory exists and is writable
- Verify source file hasn't been modified
- Try clearing cache (`triton cache clear`)

### Out of Memory

- Process large files in batches
- Lower optimization level
- Increase system memory or use swap

### Generated Code Errors

- Try different target backend
- Check source code for errors
- Use `--debug` for detailed information

## API Reference

See the Python API documentation for complete reference:

```python
from triton.compiler import driver
help(driver)
```

## Contributing

To contribute to the compiler driver:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest tests/unit/test_driver.py`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

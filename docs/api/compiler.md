# Compiler API Reference

This section provides detailed API documentation for the Triton DSL compiler components.

## Overview

The Triton compiler transforms DSL source code through multiple stages:

```
Source Code (.tri)
    ↓
Lexer → Tokens
    ↓
Parser → AST
    ↓
Type Checker → Typed AST
    ↓
Code Generator → Target Code (PyTorch, ONNX, etc.)
```

## Compiler Driver

.. automodule:: compiler.driver
   :members:
   :undoc-members:
   :show-inheritance:

### CompileOptions

.. autoclass:: compiler.driver.CompileOptions
   :members:
   :undoc-members:

### Main Functions

.. autofunction:: compiler.driver.compile_model

.. autofunction:: compiler.driver.compile_from_source

## Lexer

.. automodule:: compiler.lexer.triton_lexer
   :members:
   :undoc-members:
   :show-inheritance:

### Token Types

The lexer recognizes the following token types:

- **Keywords**: `layer`, `fn`, `let`, `return`, etc.
- **Identifiers**: Variable and function names
- **Literals**: Integer, float, ternary values
- **Operators**: `+`, `-`, `*`, `@`, etc.
- **Delimiters**: `(`, `)`, `{`, `}`, `[`, `]`, etc.

## Parser

.. automodule:: compiler.parser.triton_parser
   :members:
   :undoc-members:
   :show-inheritance:

### Parse Tree Structure

The parser builds an Abstract Syntax Tree (AST) representing the source code structure.

## AST Nodes

.. automodule:: compiler.ast.nodes
   :members:
   :undoc-members:
   :show-inheritance:

### Node Types

#### Program Node

.. autoclass:: compiler.ast.nodes.Program
   :members:
   :undoc-members:

#### Layer Definition

.. autoclass:: compiler.ast.nodes.LayerDef
   :members:
   :undoc-members:

#### Function Definition

.. autoclass:: compiler.ast.nodes.FunctionDef
   :members:
   :undoc-members:

#### Expression Nodes

.. autoclass:: compiler.ast.nodes.BinaryOp
   :members:
   :undoc-members:

.. autoclass:: compiler.ast.nodes.UnaryOp
   :members:
   :undoc-members:

.. autoclass:: compiler.ast.nodes.FunctionCall
   :members:
   :undoc-members:

#### Type Nodes

.. autoclass:: compiler.ast.nodes.Type
   :members:
   :undoc-members:

.. autoclass:: compiler.ast.nodes.TritType
   :members:
   :undoc-members:

.. autoclass:: compiler.ast.nodes.TensorType
   :members:
   :undoc-members:

## Type Checker

.. automodule:: compiler.typechecker.type_checker
   :members:
   :undoc-members:
   :show-inheritance:

### Type Checking Rules

The type checker enforces:

1. **Type Safety**: All operations must be type-compatible
2. **Shape Compatibility**: Tensor shapes must match for operations
3. **Ternary Constraints**: Ternary tensors maintain their constraints
4. **Effect Tracking**: Side effects are properly tracked

### Type Inference

.. autofunction:: compiler.typechecker.type_checker.infer_type

### Type Validation

.. autofunction:: compiler.typechecker.type_checker.validate_types

## Code Generator

.. automodule:: compiler.codegen
   :members:
   :undoc-members:
   :show-inheritance:

### Backend Interface

.. autoclass:: compiler.codegen.Backend
   :members:
   :undoc-members:

### PyTorch Backend

.. autoclass:: compiler.codegen.PyTorchBackend
   :members:
   :undoc-members:

## Usage Examples

### Basic Compilation

```python
from compiler.driver import compile_model

# Compile a Triton DSL file to PyTorch
model = compile_model(
    'model.tri',
    backend='pytorch',
    optimization_level='O2'
)

# Use the model
import torch
x = torch.randn(32, 784)
output = model(x)
```

### Custom Compilation Options

```python
from compiler.driver import CompileOptions, compile_from_source

# Create custom options
options = CompileOptions(
    backend='pytorch',
    optimization_level='O3',
    enable_caching=True,
    output_dir='./compiled_models',
    verbose=True
)

# Compile from source string
source_code = """
layer SimpleLayer(...) -> ... {
    ...
}
"""

model = compile_from_source(source_code, options)
```

### Working with AST

```python
from compiler.parser.triton_parser import parse
from compiler.ast.nodes import LayerDef, BinaryOp

# Parse source code
source = open('model.tri').read()
ast = parse(source)

# Traverse AST
for node in ast.body:
    if isinstance(node, LayerDef):
        print(f"Layer: {node.name}")
        print(f"Parameters: {len(node.params)}")
```

### Type Checking

```python
from compiler.typechecker.type_checker import TypeChecker

# Create type checker
checker = TypeChecker()

# Type check AST
typed_ast = checker.check(ast)

# Get type information
for node in typed_ast.body:
    if hasattr(node, 'type'):
        print(f"Node type: {node.type}")
```

## Error Handling

### Compilation Errors

```python
from compiler.driver import compile_model, CompilationError

try:
    model = compile_model('model.tri')
except CompilationError as e:
    print(f"Compilation failed: {e}")
    print(f"Error location: {e.line}:{e.column}")
    print(f"Suggestion: {e.suggestion}")
```

### Type Errors

```python
from compiler.typechecker.type_checker import TypeChecker, TypeError

checker = TypeChecker()

try:
    typed_ast = checker.check(ast)
except TypeError as e:
    print(f"Type error: {e.message}")
    print(f"Expected: {e.expected_type}")
    print(f"Got: {e.actual_type}")
```

## Advanced Features

### Custom Backends

Implement a custom code generation backend:

```python
from compiler.codegen import Backend

class MyCustomBackend(Backend):
    def __init__(self):
        super().__init__('my_backend')
    
    def generate_layer(self, layer_def):
        # Generate code for layer
        pass
    
    def generate_function(self, func_def):
        # Generate code for function
        pass
    
    def emit_code(self):
        # Return generated code
        return self.code

# Register custom backend
from compiler.codegen import register_backend
register_backend('my_backend', MyCustomBackend)

# Use custom backend
model = compile_model('model.tri', backend='my_backend')
```

### Optimization Passes

Add custom optimization passes:

```python
from compiler.optimization import OptimizationPass

class MyOptimization(OptimizationPass):
    def visit_BinaryOp(self, node):
        # Optimize binary operations
        if node.op == '+' and is_zero(node.right):
            return node.left  # x + 0 = x
        return node

# Apply optimization
from compiler.driver import compile_model

model = compile_model(
    'model.tri',
    custom_passes=[MyOptimization()]
)
```

## Compiler Configuration

### Configuration File

Create a `triton.toml` configuration file:

```toml
[compiler]
backend = "pytorch"
optimization_level = "O2"
cache_dir = ".triton_cache"
verbose = false

[type_checker]
strict_mode = true
allow_implicit_conversions = false

[codegen]
emit_comments = true
inline_small_functions = true
```

Load configuration:

```python
from compiler.driver import load_config, compile_model

config = load_config('triton.toml')
model = compile_model('model.tri', config=config)
```

## Command-Line Interface

The compiler provides a CLI for batch compilation:

```bash
# Basic compilation
triton compile model.tri -o model.py

# With options
triton compile model.tri \
    --backend pytorch \
    --optimization O3 \
    --output model.py \
    --verbose

# Show AST
triton ast model.tri

# Type check only
triton check model.tri

# Show help
triton --help
```

## See Also

- [Backend API Reference](backend.md) - PyTorch and other backends
- [Kernel API Reference](kernels.md) - CUDA and Triton GPU kernels
- [DSL Language Spec](../dsl/language_spec.md) - Language specification
- [Architecture Guide](../architecture/compiler_pipeline.md) - Compiler internals

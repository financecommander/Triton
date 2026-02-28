# Code Style Guide

This guide defines coding standards and best practices for contributing to Triton DSL.

## Table of Contents

- [Python Style](#python-style)
- [DSL Code Style](#dsl-code-style)
- [Documentation Standards](#documentation-standards)
- [Naming Conventions](#naming-conventions)
- [Type Annotations](#type-annotations)
- [Examples and Anti-patterns](#examples-and-anti-patterns)

## Python Style

### General Principles

Triton DSL follows **PEP 8** with specific customizations:

- **Line length**: 100 characters (not 79)
- **Formatter**: Black (v23.0+)
- **Linter**: Ruff
- **Type checker**: mypy
- **Python version**: 3.10+ (use modern features)

### Formatting with Black

Black is the authoritative code formatter. All Python code must be formatted with Black:

```bash
# Format entire project
black .

# Format specific files
black compiler/ backend/ kernels/

# Check without modifying
black --check .

# Show diff
black --diff .
```

**Black Configuration** (`pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py310']
```

### Linting with Ruff

Ruff provides fast linting with auto-fix capabilities:

```bash
# Lint project
ruff check .

# Auto-fix issues
ruff check --fix .

# Lint specific files
ruff check compiler/ backend/
```

**Ruff Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N"]
ignore = ["E501"]  # Line length handled by Black
```

**Enabled Rules**:
- `E`: pycodestyle errors
- `F`: Pyflakes
- `W`: pycodestyle warnings
- `I`: isort (import sorting)
- `N`: pep8-naming

### Imports

**Import Order** (enforced by isort via Ruff):

1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# ✅ Good
import os
import sys
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import numpy as np

from compiler.ast.nodes import Node, Program
from compiler.lexer.triton_lexer import TritonLexer
from backend.pytorch_backend import PyTorchBackend

# ❌ Bad (mixed order)
from compiler.ast.nodes import Node
import torch
import os
from backend.pytorch_backend import PyTorchBackend
import sys
```

**Import Style**:

```python
# ✅ Preferred: Explicit imports
from typing import List, Optional, Dict, Any
from compiler.ast.nodes import Node, Program, Declaration

# ✅ Acceptable: Module import for clarity
import compiler.ast.nodes as ast_nodes

# ❌ Avoid: Star imports (except in __init__.py)
from compiler.ast.nodes import *

# ✅ Exception: Type checking imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compiler.parser.triton_parser import TritonParser
```

### Code Structure

**Module Docstrings**:

```python
"""
Module name and brief description.

More detailed description of the module's purpose,
key classes, and usage examples if applicable.
"""

import os
import sys
```

**Class Structure**:

```python
class TernaryLayer:
    """
    A ternary neural network layer with constrained weights.
    
    This class implements a linear layer where weights are constrained
    to ternary values {-1, 0, 1} for memory-efficient inference.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        bias: Whether to include bias term
        
    Attributes:
        weight: Ternary weight tensor
        bias: Optional bias tensor
        
    Example:
        >>> layer = TernaryLayer(128, 256)
        >>> output = layer(input_tensor)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with ternary quantization."""
        # Implementation
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return torch.matmul(x, self.weight)
```

**Function Structure**:

```python
def quantize_to_ternary(
    tensor: torch.Tensor,
    method: str = "deterministic",
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Quantize a tensor to ternary values {-1, 0, 1}.
    
    Args:
        tensor: Input tensor to quantize
        method: Quantization method ("deterministic" or "stochastic")
        threshold: Threshold for zero values (0 to 1)
        
    Returns:
        Quantized tensor with values in {-1, 0, 1}
        
    Raises:
        ValueError: If method is not recognized
        
    Example:
        >>> tensor = torch.randn(10, 10)
        >>> quantized = quantize_to_ternary(tensor)
        >>> assert set(quantized.unique().tolist()).issubset({-1, 0, 1})
    """
    if method not in ("deterministic", "stochastic"):
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Implementation
    pass
```

### Error Handling

**Explicit Error Messages**:

```python
# ✅ Good: Descriptive error messages
def parse_layer_def(self, tokens: List[Token]) -> LayerDef:
    if not tokens:
        raise ValueError("Cannot parse layer definition from empty token list")
    
    if tokens[0].type != "LAYER":
        raise SyntaxError(
            f"Expected LAYER keyword at line {tokens[0].lineno}, "
            f"got {tokens[0].type} instead"
        )

# ❌ Bad: Generic error messages
def parse_layer_def(self, tokens: List[Token]) -> LayerDef:
    if not tokens:
        raise ValueError("Invalid input")
    
    if tokens[0].type != "LAYER":
        raise SyntaxError("Syntax error")
```

**Exception Hierarchy**:

```python
# ✅ Good: Use appropriate exception types
class TritonCompilerError(Exception):
    """Base exception for Triton compiler errors."""
    pass

class TritonSyntaxError(TritonCompilerError):
    """Raised when DSL syntax is invalid."""
    pass

class TritonTypeError(TritonCompilerError):
    """Raised when type checking fails."""
    pass

# Usage
def check_type(node: Node, expected_type: Type) -> None:
    if node.type != expected_type:
        raise TritonTypeError(
            f"Type mismatch at line {node.lineno}: "
            f"expected {expected_type}, got {node.type}"
        )
```

## DSL Code Style

### Triton DSL Syntax

**File Extension**: `.tri`

**Basic Structure**:

```triton
// Module-level comment explaining the network architecture

// Layer definitions
layer conv1: TernaryConv2d {
    in_channels: 3,
    out_channels: 32,
    kernel_size: 3,
    stride: 1
}

layer fc1: TernaryLinear {
    in_features: 512,
    out_features: 128
}

// Variable declarations with type annotations
let input: TernaryTensor[batch, 3, 32, 32] = load("data/mnist_input.bin")
let weights: TernaryTensor[32, 3, 3, 3] = quantize(conv1.weight)

// Operations
let features: TernaryTensor = conv1(input)
let output: TernaryTensor = fc1(features)
```

**Naming Conventions**:

```triton
// ✅ Good: Descriptive layer names
layer conv1_input: TernaryConv2d { ... }
layer conv2_features: TernaryConv2d { ... }
layer fc_classifier: TernaryLinear { ... }

// ✅ Good: Clear variable names
let mnist_batch: TernaryTensor = ...
let quantized_weights: TernaryTensor = ...
let classification_logits: TernaryTensor = ...

// ❌ Bad: Single letters (except for common math variables)
layer a: TernaryConv2d { ... }
let x: TernaryTensor = ...
```

**Comments**:

```triton
// Single-line comments for brief explanations

// Multi-line explanation:
// This layer implements the first convolutional block
// with ternary weights for 4x memory reduction
layer conv1: TernaryConv2d { ... }

// TODO: Implement batch normalization
// FIXME: Zero-skipping optimization not working
// NOTE: This threshold is tuned for MNIST
```

## Documentation Standards

### Docstring Format

Use **Google-style docstrings** for consistency:

```python
def train_ternary_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Train a ternary neural network model.
    
    This function trains the model using the Straight-Through Estimator
    (STE) for gradient computation through quantization layers.
    
    Args:
        model: The ternary neural network model to train
        train_loader: DataLoader providing training batches
        epochs: Number of training epochs (default: 10)
        lr: Learning rate for optimizer (default: 0.001)
        device: Device to train on ("cpu" or "cuda", default: "cpu")
        
    Returns:
        Dictionary containing training metrics:
            - "train_loss": List of epoch losses
            - "train_accuracy": List of epoch accuracies
            
    Raises:
        ValueError: If epochs < 1 or lr <= 0
        RuntimeError: If CUDA is specified but unavailable
        
    Example:
        >>> model = TernaryNet()
        >>> train_loader = DataLoader(train_dataset, batch_size=32)
        >>> metrics = train_ternary_model(model, train_loader, epochs=5)
        >>> print(f"Final accuracy: {metrics['train_accuracy'][-1]:.2f}%")
        
    Note:
        The model is modified in-place. Clone the model before training
        if you need to preserve the original state.
    """
    # Implementation
    pass
```

### Module Documentation

**README in each package**:

```python
# compiler/__init__.py
"""
Triton DSL Compiler Package.

This package contains the complete compiler pipeline for Triton DSL:

Modules:
    lexer: Tokenization of .tri source files
    parser: Parsing tokens into Abstract Syntax Tree
    ast: AST node definitions and utilities
    typechecker: Type checking and inference
    codegen: Code generation to PyTorch/CUDA

Example:
    >>> from compiler import compile_triton_source
    >>> compiled = compile_triton_source("model.tri")
    >>> model = compiled.to_pytorch()
"""
```

### Inline Comments

**When to Comment**:

```python
# ✅ Good: Explain non-obvious logic
# Use Straight-Through Estimator for backpropagation through quantization
def ternary_quantize_ste(x: torch.Tensor) -> torch.Tensor:
    # Forward: quantize to {-1, 0, 1}
    quantized = torch.sign(x) * (torch.abs(x) > threshold).float()
    # Backward: pass gradients unchanged (STE)
    return quantized + (x - x.detach())

# ✅ Good: Document complex algorithms
# Implements 2-bit packing: [a, b, c, d] -> [aa|bb|cc|dd] in one byte
def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    pass

# ❌ Bad: State the obvious
# Increment i by 1
i += 1

# ❌ Bad: Redundant with code
# Create a tensor
tensor = torch.zeros(10, 10)
```

### Documentation Files

**Structure**:

```markdown
# Document Title

Brief description (1-2 sentences).

## Table of Contents

- [Section 1](#section-1)
- [Section 2](#section-2)

## Section 1

Content with code examples...

## Examples

### Example 1: Basic Usage

\```python
# Code with comments
\```

## See Also

- [Related Doc 1](link1.md)
- [Related Doc 2](link2.md)
```

## Naming Conventions

### Python Naming

**Modules and Packages**:
```python
# ✅ Good: lowercase with underscores
compiler/
    lexer/
        triton_lexer.py
    parser/
        triton_parser.py
    ast/
        ast_nodes.py
        type_system.py

# ❌ Bad: CamelCase or hyphens
Compiler/
triton-parser.py
```

**Classes**:
```python
# ✅ Good: PascalCase
class TernaryLinear(nn.Module):
    pass

class PyTorchBackend:
    pass

class TritonCompilerError(Exception):
    pass

# ❌ Bad: snake_case or lowercase
class ternary_linear:
    pass

class pytorchbackend:
    pass
```

**Functions and Methods**:
```python
# ✅ Good: snake_case, verb-based
def parse_expression(tokens: List[Token]) -> Expression:
    pass

def quantize_to_ternary(tensor: torch.Tensor) -> torch.Tensor:
    pass

def _compute_gradient_internal(x: torch.Tensor) -> torch.Tensor:
    """Private method (single underscore)."""
    pass

# ❌ Bad: PascalCase or non-descriptive
def ParseExpression():
    pass

def qtz():
    pass
```

**Variables**:
```python
# ✅ Good: snake_case, descriptive
ternary_weights = quantize_weights(weights)
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# ✅ Good: Short names for common patterns
for i, (x, y) in enumerate(train_loader):
    pass

# ❌ Bad: Non-descriptive or confusing
tw = quantize_weights(weights)
bs = 32
l = []  # confusing with number 1
```

**Constants**:
```python
# ✅ Good: UPPER_CASE
TERNARY_VALUES = (-1, 0, 1)
MAX_LAYER_DEPTH = 100
DEFAULT_BATCH_SIZE = 32
CUDA_DEVICE_ID = 0

# Module-level "constants" (not truly constant in Python)
DEFAULT_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32
}
```

**Type Variables**:
```python
from typing import TypeVar, Generic

# ✅ Good: Single capital letter or descriptive PascalCase
T = TypeVar('T')
NodeType = TypeVar('NodeType', bound='Node')

class TreeNode(Generic[T]):
    value: T
```

### DSL Naming

**Layers**:
```triton
// ✅ Good: Descriptive with purpose
layer input_conv: TernaryConv2d { ... }
layer feature_extractor: TernaryLinear { ... }
layer output_classifier: TernaryLinear { ... }

// ❌ Bad: Generic or numbered without context
layer layer1: TernaryConv2d { ... }
layer l: TernaryLinear { ... }
```

**Variables**:
```triton
// ✅ Good: Descriptive with type hint from name
let input_batch: TernaryTensor = ...
let quantized_features: TernaryTensor = ...
let classification_scores: TernaryTensor = ...

// ❌ Bad: Single letters
let x: TernaryTensor = ...
let t: TernaryTensor = ...
```

## Type Annotations

### Required Type Hints

**All function signatures**:

```python
# ✅ Good: Full type annotations
def compile_source(
    source_path: str,
    output_path: Optional[str] = None,
    optimize: bool = True
) -> CompiledModule:
    pass

# ❌ Bad: Missing annotations
def compile_source(source_path, output_path=None, optimize=True):
    pass
```

**Class attributes** (when not obvious):

```python
class TritonCompiler:
    """Triton DSL compiler."""
    
    # ✅ Good: Annotated attributes
    lexer: TritonLexer
    parser: TritonParser
    type_checker: TypeChecker
    backend: PyTorchBackend
    _cache: Dict[str, CompiledModule]
    
    def __init__(self, backend: str = "pytorch"):
        self.lexer = TritonLexer()
        self.parser = TritonParser()
        self._cache = {}
```

**Complex return types**:

```python
from typing import Tuple, List, Dict, Optional, Union

# ✅ Good: Explicit return types
def parse_program(source: str) -> Tuple[Program, List[Error]]:
    pass

def get_layer_params(layer: LayerDef) -> Dict[str, Union[int, float, str]]:
    pass

def find_node(tree: Program, name: str) -> Optional[Node]:
    pass
```

### Type Aliases

```python
from typing import Dict, List, Tuple, Union

# ✅ Good: Define aliases for complex types
TensorShape = Tuple[int, ...]
LayerParams = Dict[str, Union[int, float, str, bool]]
TokenStream = List[Token]
ErrorList = List[Tuple[int, str]]  # (line_number, error_message)

def create_tensor(shape: TensorShape) -> torch.Tensor:
    pass

def parse_layer_params(params: LayerParams) -> LayerDef:
    pass
```

### Generic Types

```python
from typing import TypeVar, Generic, List, Optional

T = TypeVar('T')

class TreeNode(Generic[T]):
    """Generic tree node."""
    
    value: T
    children: List['TreeNode[T]']
    
    def __init__(self, value: T):
        self.value = value
        self.children = []
    
    def add_child(self, child: 'TreeNode[T]') -> None:
        self.children.append(child)
    
    def find(self, predicate: Callable[[T], bool]) -> Optional['TreeNode[T]']:
        pass
```

### Mypy Configuration

```bash
# Run type checking
mypy compiler/ backend/ kernels/

# Ignore specific line
result = compute()  # type: ignore[attr-defined]

# Ignore entire file (use sparingly)
# type: ignore
```

## Examples and Anti-patterns

### Good Practices

**✅ Clear Variable Names**:
```python
# Good
ternary_quantized_weights = quantize_to_ternary(float_weights)
training_accuracy_history = []
batch_start_index = 0

# Bad
tqw = quantize_to_ternary(fw)
a = []
i = 0  # confusing: index or iterator?
```

**✅ Early Returns**:
```python
# Good: Early returns reduce nesting
def validate_config(config: Dict[str, Any]) -> bool:
    if "model" not in config:
        return False
    if config["model"] not in SUPPORTED_MODELS:
        return False
    if config.get("batch_size", 0) <= 0:
        return False
    return True

# Bad: Deep nesting
def validate_config(config: Dict[str, Any]) -> bool:
    if "model" in config:
        if config["model"] in SUPPORTED_MODELS:
            if config.get("batch_size", 0) > 0:
                return True
    return False
```

**✅ List Comprehensions**:
```python
# Good: Readable comprehensions
quantized = [quantize_to_ternary(w) for w in weights]
errors = [err for err in errors if err.severity == "error"]

# Good: Generator for memory efficiency
total = sum(tensor.numel() for tensor in tensors)

# Bad: Complex nested comprehension (use regular loop)
result = [[f(x, y) for x in row if x > 0] for row in matrix if any(row)]
```

**✅ Context Managers**:
```python
# Good: Use context managers
with open("model.tri", "r") as f:
    source = f.read()

with torch.no_grad():
    output = model(input)

# Bad: Manual management
f = open("model.tri", "r")
source = f.read()
f.close()  # Easy to forget!
```

### Anti-patterns to Avoid

**❌ Mutable Default Arguments**:
```python
# Bad: Mutable default
def add_layer(layers=[]):
    layers.append(new_layer)
    return layers

# Good: Use None and create new
def add_layer(layers: Optional[List[Layer]] = None) -> List[Layer]:
    if layers is None:
        layers = []
    layers.append(new_layer)
    return layers
```

**❌ Bare Except**:
```python
# Bad: Catches everything including KeyboardInterrupt
try:
    result = risky_operation()
except:
    result = None

# Good: Catch specific exceptions
try:
    result = risky_operation()
except (ValueError, RuntimeError) as e:
    logger.error(f"Operation failed: {e}")
    result = None
```

**❌ String Type Checking**:
```python
# Bad: String comparison
if type(x) == type([]):
    pass

# Good: isinstance
if isinstance(x, list):
    pass

# Good: Duck typing when appropriate
try:
    for item in x:
        process(item)
except TypeError:
    process(x)
```

**❌ Modifying List While Iterating**:
```python
# Bad: Modifying during iteration
for item in items:
    if should_remove(item):
        items.remove(item)  # Skips elements!

# Good: Filter or iterate over copy
items = [item for item in items if not should_remove(item)]
# Or
for item in items.copy():
    if should_remove(item):
        items.remove(item)
```

## Quick Reference

```bash
# Format code
black .

# Lint code  
ruff check . --fix

# Type check
mypy compiler/ backend/ kernels/

# Full check
black --check . && ruff check . && mypy compiler/ backend/ kernels/
```

## See Also

- [Development Setup](development_setup.md)
- [Testing Guide](testing.md)
- [PR Process](pr_process.md)
- [PEP 8](https://peps.python.org/pep-0008/)
- [Black Documentation](https://black.readthedocs.io/)
- [Ruff Documentation](https://beta.ruff.rs/docs/)

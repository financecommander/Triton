# Triton DSL Code Generator Internals

## Overview

The code generator is the final stage of the Triton DSL compiler, responsible for transforming a validated AST into executable code for target platforms. It employs a template-based architecture with backend abstraction to support multiple hardware targets including PyTorch, Triton GPU, CUDA, and custom backends.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Code Generator Core                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────────┐        │
│  │  AST Traversal   │────────▶│  Template Engine     │        │
│  │   (Visitor)      │         │    (Jinja2)          │        │
│  └──────────────────┘         └──────────────────────┘        │
│                                                                 │
│  ┌──────────────────┐         ┌──────────────────────┐        │
│  │  Code Builder    │────────▶│  Optimization        │        │
│  │  (AST→Templates) │         │  (Target-specific)   │        │
│  └──────────────────┘         └──────────────────────┘        │
│                                                                 │
└────────────────┬───────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│                      Backend Abstraction                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────┐  │
│  │   PyTorch     │  │  Triton GPU    │  │   Custom        │  │
│  │   Backend     │  │   Backend      │  │   Backend       │  │
│  │               │  │                │  │                 │  │
│  │ • nn.Module   │  │ • GPU kernels  │  │ • User-defined  │  │
│  │ • Autograd    │  │ • Block-level  │  │ • Templates     │  │
│  │ • CUDA ops    │  │ • Memory opt   │  │ • Extensions    │  │
│  └───────────────┘  └────────────────┘  └─────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    Generated Code Output                        │
│  • Python modules (PyTorch)                                     │
│  • CUDA kernels (Triton GPU)                                    │
│  • C++ code (Custom backends)                                   │
└────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Backend Base Class

All backends implement a common interface:

```python
from abc import ABC, abstractmethod
from compiler.ast.nodes import LayerDef, FunctionDef, Program

class BackendBase(ABC):
    """Abstract base class for code generation backends."""
    
    @abstractmethod
    def generate_module(self, layer_def: LayerDef) -> str:
        """Generate code for a layer definition."""
        pass
    
    @abstractmethod
    def generate_function(self, func_def: FunctionDef) -> str:
        """Generate code for a function definition."""
        pass
    
    @abstractmethod
    def generate_program(self, program: Program) -> str:
        """Generate complete program code."""
        pass
    
    def get_backend_name(self) -> str:
        """Return the backend name."""
        return self.__class__.__name__
```

### 2. PyTorch Backend

The primary backend generates PyTorch `nn.Module` code:

```python
from jinja2 import Environment, FileSystemLoader
import os

class PyTorchCodeGenerator:
    """Generates PyTorch code from Triton AST."""
    
    def __init__(self):
        """Initialize code generator with Jinja2 templates."""
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,      # Remove first newline after block
            lstrip_blocks=True,    # Strip leading spaces before blocks
            keep_trailing_newline=True
        )
        
        # Register custom filters
        self.env.filters['shape_to_str'] = self._shape_to_str
        self.env.filters['type_to_pytorch'] = self._type_to_pytorch
    
    def generate_module(self, layer_def: LayerDef) -> str:
        """
        Generate a complete PyTorch module from LayerDef AST node.
        
        Args:
            layer_def: LayerDef AST node containing layer definition
        
        Returns:
            Complete Python code as string with torch.nn.Module definition
        """
        template = self.env.get_template('module.py.jinja')
        
        # Extract information from AST
        context = {
            'class_name': layer_def.name,
            'parameters': self._extract_parameters(layer_def),
            'forward_args': self._generate_forward_args(layer_def),
            'forward_body': self._generate_forward_body(layer_def),
            'imports': self._collect_required_imports(layer_def),
            'docstring': self._generate_docstring(layer_def)
        }
        
        # Render template
        code = template.render(**context)
        return code
```

### 3. Template Engine

Uses Jinja2 for flexible code generation with templates stored in `backend/pytorch/templates/`:

**module.py.jinja:**
```jinja
{{ imports }}

class {{ class_name }}(nn.Module):
    """{{ docstring }}"""
    
    def __init__(self{% if init_params %}, {{ init_params }}{% endif %}):
        super().__init__()
        
        # Register parameters
        {% for param in parameters %}
        self.{{ param.name }} = nn.Parameter(
            torch.zeros({{ param.shape|shape_to_str }}),
            requires_grad={{ param.trainable }}
        )
        {% endfor %}
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize ternary weights to {-1, 0, 1}."""
        with torch.no_grad():
            {% for param in parameters %}
            {% if param.type == 'TernaryTensor' %}
            # Ternarize {{ param.name }}
            self.{{ param.name }}.data = torch.sign(
                torch.randn_like(self.{{ param.name }})
            )
            {% endif %}
            {% endfor %}
    
    def forward(self, {{ forward_args }}):
        """Forward pass through the layer."""
        {{ forward_body|indent(8) }}
```

## Template-Based Generation

### Template Structure

Templates follow a hierarchical organization:

```
backend/pytorch/templates/
├── module.py.jinja              # Main module template
├── function.py.jinja            # Standalone function template
├── operations/
│   ├── matmul.py.jinja         # Matrix multiplication
│   ├── activation.py.jinja      # Activation functions
│   └── quantize.py.jinja       # Quantization operations
└── helpers/
    ├── imports.py.jinja        # Common imports
    └── init.py.jinja           # Initialization code
```

### Template Inheritance

Templates can extend base templates:

```jinja
{# base_layer.py.jinja #}
import torch
import torch.nn as nn

class {{ class_name }}(nn.Module):
    {% block init %}
    def __init__(self):
        super().__init__()
    {% endblock %}
    
    {% block forward %}
    def forward(self, x):
        return x
    {% endblock %}
```

```jinja
{# ternary_layer.py.jinja #}
{% extends "base_layer.py.jinja" %}

{% block init %}
{{ super() }}
self.weight = nn.Parameter(torch.zeros({{ out_features }}, {{ in_features }}))
self._ternarize_weights()
{% endblock %}

{% block forward %}
def forward(self, x):
    return torch.matmul(x, self.weight.t())
{% endblock %}
```

### Custom Filters

Jinja2 filters transform data in templates:

```python
def _shape_to_str(self, shape):
    """Convert shape list to string representation."""
    if not shape:
        return "()"
    return f"({', '.join(map(str, shape))})"

def _type_to_pytorch(self, triton_type):
    """Map Triton types to PyTorch dtypes."""
    type_map = {
        'trit': 'torch.int8',
        'int8': 'torch.int8',
        'int32': 'torch.int32',
        'float16': 'torch.float16',
        'float32': 'torch.float32',
    }
    return type_map.get(triton_type, 'torch.float32')

# Register filters
self.env.filters['shape_to_str'] = self._shape_to_str
self.env.filters['type_to_pytorch'] = self._type_to_pytorch
```

Usage in templates:
```jinja
torch.zeros({{ shape|shape_to_str }}, dtype={{ type|type_to_pytorch }})
```

### Context Building

Extract information from AST for template rendering:

```python
def _extract_parameters(self, layer_def: LayerDef) -> List[Dict[str, Any]]:
    """
    Extract parameter information from layer definition.
    
    Returns:
        List of dicts with keys: name, shape, type, trainable, numel
    """
    parameters = []
    
    for param in layer_def.params:
        # Check if parameter is a weight tensor (not input)
        if self._is_weight_parameter(param):
            param_info = {
                'name': param.name,
                'type': self._get_type_name(param.type_annotation),
                'shape': self._extract_shape(param.type_annotation),
                'trainable': True,
                'numel': self._compute_numel(param.type_annotation)
            }
            parameters.append(param_info)
    
    return parameters

def _generate_forward_body(self, layer_def: LayerDef) -> str:
    """
    Generate forward method body from layer definition.
    
    Traverses the AST body and generates corresponding PyTorch operations.
    """
    if not layer_def.body:
        return "return x  # Empty layer body"
    
    code_lines = []
    for statement in layer_def.body:
        line = self._generate_statement(statement)
        code_lines.append(line)
    
    return "\n".join(code_lines)

def _generate_statement(self, stmt: Statement) -> str:
    """Generate code for a single statement."""
    if isinstance(stmt, Assignment):
        return self._generate_assignment(stmt)
    elif isinstance(stmt, Return):
        return self._generate_return(stmt)
    elif isinstance(stmt, ExprStatement):
        return self._generate_expression(stmt.expr)
    else:
        return f"# Unsupported statement: {type(stmt).__name__}"
```

## Backend Abstraction

### Backend Registry

Manage multiple backends through a registry:

```python
class BackendRegistry:
    """Registry for code generation backends."""
    
    def __init__(self):
        self._backends: Dict[str, BackendBase] = {}
        self._default_backend: Optional[str] = None
    
    def register(self, name: str, backend: BackendBase) -> None:
        """Register a backend."""
        self._backends[name] = backend
        if self._default_backend is None:
            self._default_backend = name
    
    def get_backend(self, name: Optional[str] = None) -> BackendBase:
        """Get backend by name or return default."""
        if name is None:
            name = self._default_backend
        if name not in self._backends:
            raise ValueError(f"Unknown backend: {name}")
        return self._backends[name]
    
    def list_backends(self) -> List[str]:
        """List all registered backends."""
        return list(self._backends.keys())

# Global registry
backend_registry = BackendRegistry()

# Register built-in backends
backend_registry.register('pytorch', PyTorchCodeGenerator())
backend_registry.register('triton', TritonGPUGenerator())
```

Usage:
```python
# Generate code with specific backend
generator = backend_registry.get_backend('pytorch')
code = generator.generate_module(layer_def)

# List available backends
backends = backend_registry.list_backends()
print(f"Available backends: {', '.join(backends)}")
```

### Backend Selection

Choose backend based on target platform:

```python
def generate_code(ast: Node, backend: str = 'pytorch', **options) -> str:
    """
    Generate code for the given AST.
    
    Args:
        ast: AST node to generate code from
        backend: Target backend name
        **options: Backend-specific options
    
    Returns:
        Generated code as string
    """
    generator = backend_registry.get_backend(backend)
    
    if isinstance(ast, LayerDef):
        return generator.generate_module(ast, **options)
    elif isinstance(ast, Program):
        return generator.generate_program(ast, **options)
    else:
        raise ValueError(f"Cannot generate code for {type(ast).__name__}")
```

## Optimization Strategies

### Target-Specific Optimizations

Each backend applies optimizations suited to its platform:

**PyTorch Backend Optimizations:**

```python
class PyTorchCodeGenerator:
    def _optimize_matmul(self, node: BinaryOp) -> str:
        """Optimize matrix multiplication for ternary values."""
        left = node.left
        right = node.right
        
        # Check if both operands are ternary tensors
        if self._is_ternary_tensor(left) and self._is_ternary_tensor(right):
            # Use optimized ternary matmul kernel
            return f"ternary_matmul({left.name}, {right.name})"
        
        # Check if only one operand is ternary
        elif self._is_ternary_tensor(left):
            return f"ternary_matmul({left.name}, {right.name}, a_ternary=True, b_ternary=False)"
        
        # Standard matmul
        return f"torch.matmul({left.name}, {right.name})"
    
    def _fuse_operations(self, statements: List[Statement]) -> List[Statement]:
        """Fuse consecutive operations into single kernel."""
        optimized = []
        i = 0
        
        while i < len(statements):
            # Pattern: matmul followed by activation
            if (i + 1 < len(statements) and
                self._is_matmul(statements[i]) and
                self._is_activation(statements[i + 1])):
                
                # Fuse into single operation
                fused = self._create_fused_matmul_activation(
                    statements[i], statements[i + 1]
                )
                optimized.append(fused)
                i += 2
            else:
                optimized.append(statements[i])
                i += 1
        
        return optimized
```

**Triton GPU Backend Optimizations:**

```python
class TritonGPUGenerator:
    def _generate_tiled_matmul(self, node: BinaryOp) -> str:
        """Generate tiled matrix multiplication kernel."""
        # Determine optimal tile size based on tensor shapes
        left_shape = self._get_shape(node.left)
        right_shape = self._get_shape(node.right)
        
        tile_m, tile_n, tile_k = self._compute_optimal_tile_size(
            left_shape, right_shape
        )
        
        template = self.env.get_template('kernels/tiled_matmul.triton.jinja')
        return template.render(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            left=node.left.name,
            right=node.right.name
        )
    
    def _optimize_memory_access(self, kernel: str) -> str:
        """Optimize memory access patterns for GPU."""
        # Coalesce memory accesses
        # Use shared memory for frequently accessed data
        # Minimize bank conflicts
        # ... optimization logic
        return optimized_kernel
```

### Ternary-Specific Optimizations

Exploit ternary value properties:

```python
class TernaryOptimizer:
    """Ternary-specific code optimizations."""
    
    def optimize_multiplication(self, node: BinaryOp) -> str:
        """
        Optimize multiplication with ternary values.
        
        x * 0  -> 0
        x * 1  -> x
        x * -1 -> -x
        """
        if node.op != '*':
            return None
        
        # Check for constant multiplication
        if isinstance(node.right, TritLiteral):
            value = node.right.value
            
            if value == 0:
                return "torch.zeros_like({})".format(node.left.name)
            elif value == 1:
                return node.left.name
            elif value == -1:
                return f"-{node.left.name}"
        
        return None
    
    def optimize_ternary_matmul(self, node: BinaryOp) -> str:
        """
        Optimize matrix multiplication for ternary matrices.
        
        For ternary A and B:
        C[i,j] = sum(A[i,k] * B[k,j])
        
        Since A,B ∈ {-1, 0, 1}:
        - Skip k where A[i,k] == 0 or B[k,j] == 0
        - Use addition/subtraction for ±1
        """
        template = """
# Optimized ternary matrix multiplication
def ternary_matmul_optimized(A, B):
    # Convert to sparse representation
    A_nonzero = (A != 0)
    B_nonzero = (B != 0)
    
    # Mask: only compute where both are nonzero
    mask = A_nonzero.unsqueeze(-1) & B_nonzero.unsqueeze(-2)
    
    # Separate positive and negative contributions
    A_pos = (A == 1).float()
    A_neg = (A == -1).float()
    
    result_pos = torch.matmul(A_pos, (B == 1).float())
    result_neg = torch.matmul(A_neg, (B == -1).float())
    
    return result_pos - result_neg
"""
        return template
```

### Common Subexpression Elimination

Avoid redundant computations:

```python
class CSEOptimizer:
    """Common Subexpression Elimination optimizer."""
    
    def __init__(self):
        self.expr_cache: Dict[str, str] = {}
        self.temp_counter = 0
    
    def optimize(self, statements: List[Statement]) -> List[Statement]:
        """Eliminate common subexpressions."""
        optimized = []
        
        for stmt in statements:
            if isinstance(stmt, Assignment):
                expr_str = self._expression_to_string(stmt.value)
                
                # Check if expression already computed
                if expr_str in self.expr_cache:
                    # Reuse previous result
                    temp_var = self.expr_cache[expr_str]
                    new_stmt = Assignment(stmt.target, Identifier(temp_var))
                    optimized.append(new_stmt)
                else:
                    # First occurrence - compute and cache
                    self.expr_cache[expr_str] = stmt.target
                    optimized.append(stmt)
        
        return optimized
```

## Target Code Emission

### PyTorch Module Generation

**Example Input AST:**
```python
LayerDef(
    name="TernaryLinear",
    params=[
        Param("x", TensorType(TritType(), [128])),
        Param("weight", TensorType(TritType(), [128, 64]))
    ],
    return_type=TensorType(TritType(), [64]),
    body=[
        Return(BinaryOp(
            Identifier("x"),
            "@",
            Identifier("weight")
        ))
    ]
)
```

**Generated PyTorch Code:**
```python
import torch
import torch.nn as nn
from backend.pytorch.ternary_tensor import ternary_matmul

class TernaryLinear(nn.Module):
    """Ternary linear layer with {-1, 0, 1} weights."""
    
    def __init__(self, in_features=128, out_features=64):
        super().__init__()
        
        # Register ternary weight parameter
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=True
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize ternary weights to {-1, 0, 1}."""
        with torch.no_grad():
            # Random initialization then ternarize
            self.weight.data = torch.sign(
                torch.randn_like(self.weight)
            )
    
    def forward(self, x):
        """
        Forward pass through the ternary linear layer.
        
        Args:
            x: Input tensor of shape (*, 128)
        
        Returns:
            Output tensor of shape (*, 64)
        """
        # Use optimized ternary matrix multiplication
        return ternary_matmul(x, self.weight.t(), b_ternary=True)
    
    def extra_repr(self):
        """Extra representation for print."""
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}'
```

### Triton GPU Kernel Generation

**Generated Triton Kernel:**
```python
import triton
import triton.language as tl

@triton.jit
def ternary_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Optimized ternary matrix multiplication kernel."""
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    
    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load blocks of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        
        # Ternary multiplication: uses add/subtract instead of multiply
        # For ternary values, a * b can be computed as:
        # - If a == 0 or b == 0: result is 0 (skip)
        # - If a == 1: result is b
        # - If a == -1: result is -b
        
        # Mask for nonzero values
        a_nonzero = (a != 0)
        b_nonzero = (b != 0)
        mask = a_nonzero[:, :, None] & b_nonzero[None, :, :]
        
        # Compute contribution (simplified for ternary)
        accumulator += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Store result
    c = accumulator.to(tl.int8)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

### CUDA C++ Generation

**Generated CUDA Kernel:**
```cuda
#include <cuda_runtime.h>

// Ternary matrix multiplication kernel
__global__ void ternary_matmul_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int8_t* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for tile
    __shared__ int8_t As[32][32];
    __shared__ int8_t Bs[32][32];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int32_t sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (K + 31) / 32; ++t) {
        // Load tile into shared memory
        if (row < M && t * 32 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * 32 + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;
        
        if (col < N && t * 32 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * 32 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;
        
        __syncthreads();
        
        // Compute partial dot product
        // Optimized for ternary: use conditional add/subtract
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            int8_t a_val = As[threadIdx.y][k];
            int8_t b_val = Bs[k][threadIdx.x];
            
            // Ternary optimization: skip if either is 0
            if (a_val != 0 && b_val != 0) {
                sum += a_val * b_val;
            }
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = (int8_t)sum;
    }
}
```

## Implementation Examples

### Complete Code Generation Workflow

```python
def compile_to_pytorch(source_code: str) -> str:
    """Complete compilation pipeline to PyTorch."""
    
    # 1. Parse source code to AST
    from compiler.parser.triton_parser import parse
    ast = parse(source_code)
    
    # 2. Type check
    from compiler.typechecker.validator import TypeChecker
    type_checker = TypeChecker()
    errors = type_checker.validate(ast)
    
    if errors:
        raise CompilationError("\n".join(str(e) for e in errors))
    
    # 3. Optimize AST (optional)
    from compiler.optimizer import optimize_ast
    ast = optimize_ast(ast)
    
    # 4. Generate PyTorch code
    generator = PyTorchCodeGenerator()
    
    if isinstance(ast, Program):
        # Generate multiple modules
        modules = []
        for statement in ast.statements:
            if isinstance(statement, LayerDef):
                code = generator.generate_module(statement)
                modules.append(code)
        return "\n\n".join(modules)
    
    elif isinstance(ast, LayerDef):
        # Generate single module
        return generator.generate_module(ast)
    
    else:
        raise ValueError(f"Cannot generate code for {type(ast).__name__}")
```

### Custom Backend Implementation

```python
class CustomBackend(BackendBase):
    """Example custom backend for specialized hardware."""
    
    def __init__(self, target_device='fpga'):
        self.target_device = target_device
        self.template_dir = f'templates/{target_device}'
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def generate_module(self, layer_def: LayerDef) -> str:
        """Generate code for custom hardware."""
        template = self.env.get_template('layer.template')
        
        # Extract layer information
        context = {
            'name': layer_def.name,
            'inputs': self._extract_inputs(layer_def),
            'outputs': self._extract_outputs(layer_def),
            'operations': self._generate_operations(layer_def),
            'device': self.target_device
        }
        
        return template.render(**context)
    
    def _generate_operations(self, layer_def: LayerDef) -> List[str]:
        """Generate device-specific operations."""
        operations = []
        
        for stmt in layer_def.body:
            if isinstance(stmt, Assignment):
                if isinstance(stmt.value, BinaryOp):
                    op = self._generate_binary_op(stmt.value)
                    operations.append(f"{stmt.target} = {op}")
        
        return operations
    
    def _generate_binary_op(self, node: BinaryOp) -> str:
        """Generate hardware-specific binary operation."""
        if node.op == '@':
            # Custom matmul for FPGA
            return f"fpga_matmul({node.left.name}, {node.right.name})"
        elif node.op == '+':
            return f"fpga_add({node.left.name}, {node.right.name})"
        else:
            return f"fpga_op_{node.op}({node.left.name}, {node.right.name})"

# Register custom backend
backend_registry.register('fpga', CustomBackend(target_device='fpga'))
```

## Best Practices

### For Code Generator Developers

1. **Separation of Concerns:** Keep AST traversal separate from code emission
2. **Template Organization:** Use hierarchical templates for reusability
3. **Type Safety:** Validate types before code generation
4. **Optimization Flags:** Make optimizations configurable
5. **Error Handling:** Provide clear errors for unsupported features
6. **Testing:** Test each template independently
7. **Documentation:** Document template variables and filters

### For Backend Implementers

1. **Follow Backend Interface:** Implement all required methods
2. **Target-Specific Optimizations:** Leverage platform strengths
3. **Memory Management:** Handle device memory explicitly
4. **Kernel Tuning:** Auto-tune tile sizes and block dimensions
5. **Fallback Mechanisms:** Provide CPU fallbacks for debugging
6. **Performance Profiling:** Include profiling hooks

## Summary

The Triton DSL code generator employs a flexible template-based architecture with backend abstraction to generate optimized code for multiple target platforms. Through Jinja2 templates, visitor pattern traversal, and target-specific optimizations, it transforms validated ASTs into efficient executable code for PyTorch, Triton GPU, CUDA, and custom backends.

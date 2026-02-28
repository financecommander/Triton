# Triton Compiler Code Generation Documentation

## Overview

The Triton compiler code generation module (`triton/compiler/codegen.py`) provides a complete production-quality pipeline for converting Triton DSL programs to executable PyTorch code.

## Architecture

```
Triton DSL Source Code
         ↓
    AST (nodes.py)
         ↓
  IR (Intermediate Representation)
         ↓
   Optimization Passes
         ↓
  Optimized IR
         ↓
   PyTorch Code Generation
         ↓
    Formatted Python Code
         ↓
  Executable PyTorch nn.Module
```

## Features

### 1. Intermediate Representation (IR)

The IR is a low-level representation in Static Single Assignment (SSA) form:

- **IRValue**: Represents values (variables, constants, temporaries)
- **IRInstruction**: Single operations (add, mul, matmul, etc.)
- **IRBasicBlock**: Sequence of instructions
- **IRFunction**: Collection of basic blocks
- **IRModule**: Collection of functions

#### Example IR

```
%x = load(param_x)
%y = load(param_y)
%t0 = add(%x, %y)
%t1 = mul(%t0, 2)
return %t1
```

### 2. AST to IR Conversion

The `ASTToIRConverter` class converts Triton AST nodes to IR:

```python
from triton.compiler.codegen import ASTToIRConverter

converter = ASTToIRConverter()
ir_module = converter.convert_program(ast_program)
```

**Supported Conversions:**
- LayerDef → IRFunction
- BinaryOp → IR arithmetic instructions
- Assignment → STORE instructions
- Return → RETURN instructions
- FunctionCall → CALL instructions
- Literals → Constant values

### 3. Optimization Passes

Four main optimization passes improve generated code:

#### Constant Folding
```python
# Before
%t0 = add(2, 3)
%t1 = mul(%t0, 4)

# After
%t0 = const(5)
%t1 = const(20)
```

#### Dead Code Elimination
```python
# Before
%unused = add(%x, %y)  # Never used
%result = mul(%a, %b)
return %result

# After
%result = mul(%a, %b)
return %result
```

#### Common Subexpression Elimination
```python
# Before
%t0 = add(%x, %y)
%t1 = add(%x, %y)  # Duplicate
%result = mul(%t0, %t1)

# After
%t0 = add(%x, %y)
%result = mul(%t0, %t0)
```

#### Quantization Fusion
```python
# Before
%quant = quantize_ternary(%x)
%dequant = dequantize(%quant)

# After
# (both operations removed - no-op)
```

### 4. PyTorch Code Generation

The `PyTorchCodeGenerator` converts IR to PyTorch code:

```python
from triton.compiler.codegen import PyTorchCodeGenerator

generator = PyTorchCodeGenerator()
pytorch_code = generator.generate(ir_module)
```

**Generated Code Includes:**
- `nn.Module` class definitions
- `__init__` method with parameter initialization
- `forward` method with computation logic
- Automatic import management
- Ternary weight packing/unpacking
- Type conversions
- Inline comments

#### Example Generated Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.pytorch.ops.pack import pack_ternary, unpack_ternary


class TernaryLinearLayer(nn.Module):
    """Generated PyTorch module: TernaryLinearLayer"""

    def __init__(self):
        """Initialize module parameters."""
        super().__init__()

        # Ternary parameter: weights
        self._weights_shape = [128, 256]
        self._weights_numel = 32768
        init_tensor = torch.randint(-1, 2, (32768,), dtype=torch.int8)
        packed = pack_ternary(init_tensor)
        self.register_buffer('weights_packed', packed)

    def forward(self, x):
        """Forward pass."""
        # Unpack ternary tensor: weights
        weights = unpack_ternary(
            self.weights_packed,
            self._weights_numel
        ).reshape(self._weights_shape).float()

        result = torch.matmul(x, weights)
        return result
```

### 5. Quantization Code Generation

Specialized code generators for different quantization schemes:

#### Ternary Quantization
```python
from triton.compiler.codegen import QuantizationCodeGenerator

code = QuantizationCodeGenerator.generate_ternary_quantize("input", "output")
```

Generates:
```python
# Ternary quantization: map to {-1, 0, 1}
output_float = input
output_sign = torch.sign(output_float)
output_threshold = 0.3 * torch.max(torch.abs(output_float))
output_mask = torch.abs(output_float) > output_threshold
output = output_sign * output_mask.float()
```

#### INT8 Quantization
```python
code = QuantizationCodeGenerator.generate_int8_quantize("input", "output")
```

#### INT4 Quantization
```python
code = QuantizationCodeGenerator.generate_int4_quantize("input", "output")
```

#### Per-Channel Quantization
```python
code = QuantizationCodeGenerator.generate_per_channel_quantize("input", "output", axis=0)
```

### 6. Advanced Features

#### CUDA Kernel Generation

Generate optimized Triton CUDA kernels:

```python
from triton.compiler.codegen import CUDAKernelGenerator

kernel_code = CUDAKernelGenerator.generate_ternary_matmul_kernel()
```

Generates a `@triton.jit` decorated kernel for efficient ternary matrix multiplication.

#### Custom Autograd Functions

Generate custom backward passes:

```python
from triton.compiler.codegen import AutogradFunctionGenerator

backward_code = AutogradFunctionGenerator.generate_ternary_backward()
```

Implements straight-through estimator for ternary operations.

### 7. Code Quality

#### Black Formatting

Automatically format generated code with Black (if available):

```python
from triton.compiler.codegen import CodeFormatter

formatted_code = CodeFormatter.format_code(code)
```

#### Syntax Validation

Validate generated code before returning:

```python
valid, error = CodeFormatter.validate_syntax(code)
if not valid:
    raise SyntaxError(f"Generated code has errors: {error}")
```

## Complete Pipeline

### Basic Usage

```python
from compiler.ast.nodes import Program, LayerDef, Param
from triton.compiler.codegen import generate_pytorch_code

# Create AST
layer = LayerDef(
    name="MyLayer",
    params=[
        Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
        Param(name="x", param_type="Tensor", shape=None)
    ],
    body=[]
)
program = Program(statements=[layer])

# Generate PyTorch code
pytorch_code = generate_pytorch_code(program, optimize=True)

# Execute generated code
exec(pytorch_code)
```

### Advanced Usage with Metadata

```python
from triton.compiler.codegen import CodeGenerationPipeline

pipeline = CodeGenerationPipeline(optimize=True)

# Generate with metadata
result = pipeline.generate_with_metadata(program)

print("Generated Code:", result["code"])
print("IR Structure:", result["ir"])
print("Optimizations Applied:", result["optimizations_applied"])
print("Imports:", result["imports"])
print("Functions:", result["functions"])
```

### Pipeline Configuration

```python
pipeline = CodeGenerationPipeline(optimize=True)

# Add custom optimization passes
from triton.compiler.codegen import MemoryLayoutOptimizationPass
pipeline.optimization_passes.append(MemoryLayoutOptimizationPass())

# Generate code
code = pipeline.generate(program)
```

## API Reference

### Public Functions

#### `generate_pytorch_code(program, optimize=True)`

Generate PyTorch code from Triton AST.

**Parameters:**
- `program` (Program): Triton program AST node
- `optimize` (bool): Whether to run optimization passes (default: True)

**Returns:**
- `str`: Generated PyTorch code

#### `generate_with_ir(program)`

Generate PyTorch code and return IR for inspection.

**Parameters:**
- `program` (Program): Triton program AST

**Returns:**
- `tuple`: (generated_code: str, ir_module: IRModule)

#### `compile_and_execute(program, namespace=None)`

Compile and execute Triton program.

**Parameters:**
- `program` (Program): Triton program AST
- `namespace` (dict, optional): Namespace for execution

**Returns:**
- `dict`: Namespace with executed code

### Classes

#### `CodeGenerationPipeline`

Complete code generation pipeline.

**Methods:**
- `generate(program, optimize=None)`: Generate code
- `optimize_ir(module)`: Run optimization passes
- `generate_with_metadata(program)`: Generate with metadata

#### `ASTToIRConverter`

Convert AST to IR.

**Methods:**
- `convert_program(program)`: Convert Program to IRModule
- `convert_layer_def(layer_def)`: Convert LayerDef to IRFunction
- `convert_expr(expr)`: Convert expression to IRValue

#### `PyTorchCodeGenerator`

Generate PyTorch code from IR.

**Methods:**
- `generate(module)`: Generate code from IRModule
- `generate_function(func)`: Generate module class from IRFunction

### Optimization Passes

- `ConstantFoldingPass`: Fold constant expressions
- `DeadCodeEliminationPass`: Remove unused code
- `CommonSubexpressionEliminationPass`: Eliminate duplicates
- `QuantizationFusionPass`: Fuse quantization ops
- `MemoryLayoutOptimizationPass`: Optimize tensor layouts

## Testing

### Unit Tests

Run comprehensive unit tests:

```bash
pytest tests/unit/test_codegen_comprehensive.py -v
```

49 test methods covering:
- IR data structures
- AST to IR conversion
- Optimization passes
- PyTorch code generation
- Complete pipeline
- Quantization
- Advanced features
- Error handling
- Integration tests

### Benchmarks

Run performance benchmarks:

```bash
pytest tests/benchmarks/bench_codegen.py --benchmark-only
```

Benchmarks:
- AST to IR conversion speed
- Optimization pass performance
- Code generation throughput
- Scalability tests
- Memory usage

### Validation

Validate generated code:

```bash
python tests/validation/validate_codegen_output.py
```

Validates:
- Syntax correctness
- Executability
- Module creation
- Forward pass execution
- Ternary weight handling

## Performance Characteristics

### Compilation Speed

- Simple layer: < 10ms
- Complex layer (50 ops): < 20ms
- Ternary layer (10 params): < 15ms
- Multiple layers (10): < 30ms

### Throughput

- ~50-100 programs/second on typical hardware

### Memory Usage

- < 1 MB per compilation on average
- Linear scaling with program size

### Optimization Impact

- 10-30% reduction in IR instructions
- Minimal overhead (< 2x compilation time)
- Improved runtime performance of generated code

## Best Practices

### 1. Use Optimization

Always enable optimization for production code:

```python
code = generate_pytorch_code(program, optimize=True)
```

### 2. Validate Generated Code

Always validate before deployment:

```python
from triton.compiler.codegen import CodeFormatter

valid, error = CodeFormatter.validate_syntax(code)
assert valid, f"Invalid code: {error}"
```

### 3. Inspect IR for Debugging

Use IR inspection to understand compilation:

```python
code, ir_module = generate_with_ir(program)

# Print IR
for func in ir_module.functions.values():
    print(f"Function: {func.name}")
    for block in func.blocks.values():
        for inst in block.instructions:
            print(f"  {inst}")
```

### 4. Profile Compilation

Profile for large programs:

```python
import time

start = time.time()
code = generate_pytorch_code(program)
elapsed = time.time() - start

print(f"Compilation time: {elapsed*1000:.2f}ms")
```

### 5. Cache Compiled Code

Cache generated code for reuse:

```python
import hashlib
import json

def get_program_hash(program):
    """Get unique hash for program."""
    program_str = json.dumps(program.to_dict(), sort_keys=True)
    return hashlib.md5(program_str.encode()).hexdigest()

# Cache lookup
cache = {}
program_hash = get_program_hash(program)

if program_hash in cache:
    code = cache[program_hash]
else:
    code = generate_pytorch_code(program)
    cache[program_hash] = code
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

If you see import errors, ensure all paths are correct:

```python
import sys
sys.path.insert(0, '/path/to/triton')
```

#### 2. Syntax Errors in Generated Code

Check that AST is well-formed:

```python
# Validate AST before compilation
from compiler.typechecker.validator import TypeChecker

checker = TypeChecker()
errors = checker.check(program)
if errors:
    print("AST errors:", errors)
```

#### 3. Runtime Errors in Generated Code

Enable metadata to debug:

```python
result = pipeline.generate_with_metadata(program)
print("IR:", result["ir"])
print("Optimizations:", result["optimizations_applied"])
```

#### 4. Performance Issues

Profile each stage:

```python
import time

# AST to IR
start = time.time()
ir_module = converter.convert_program(program)
print(f"AST→IR: {(time.time()-start)*1000:.2f}ms")

# Optimization
start = time.time()
ir_module = pipeline.optimize_ir(ir_module)
print(f"Optimization: {(time.time()-start)*1000:.2f}ms")

# Code generation
start = time.time()
code = generator.generate(ir_module)
print(f"IR→PyTorch: {(time.time()-start)*1000:.2f}ms")
```

## Future Enhancements

### Planned Features

1. **Loop Fusion**: Merge adjacent loops
2. **Auto-tuning**: Automatic hyperparameter tuning
3. **Multi-device**: Multi-GPU code generation
4. **Mixed Precision**: FP16/BF16 support
5. **Export**: ONNX/TensorRT export
6. **Profiling**: Built-in performance profiling
7. **Debugging**: Source map generation
8. **Incremental Compilation**: Fast recompilation

### Contributing

To add new features:

1. Add IR opcodes in `IROpcode` enum
2. Update `ASTToIRConverter` for new AST nodes
3. Add optimization passes as needed
4. Update `PyTorchCodeGenerator` for new opcodes
5. Add comprehensive tests
6. Update documentation

## License

MIT License - see LICENSE file for details.

## Contact

For issues and questions:
- GitHub Issues: https://github.com/financecommander/Triton/issues
- Email: dev@financecommander.com

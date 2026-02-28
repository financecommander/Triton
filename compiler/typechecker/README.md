# Type Checker Module

Production-quality type checker for the Triton DSL with comprehensive error reporting and advanced features.

## Features

### Type Inference
- **Forward Type Propagation**: Infers types from literals and expressions
- **Backward Type Propagation**: Uses unification to propagate constraints backwards
- **Unification Algorithm**: Implements Hindley-Milner style type unification
- **Generic Type Handling**: Support for type variables and polymorphic types
- **Tensor Shape Inference**: Validates and infers tensor shapes in operations

### Type Validation
- **Function Signature Checking**: Validates function calls against definitions
- **Operator Type Compatibility**: Ensures operators are used with compatible types
- **Quantization Type Rules**: Supports FP32, FP16, INT8, and Ternary quantization levels
- **Matrix Multiplication**: Special dimension checking for @ operator
- **Type Coercion Rules**: Controls implicit type conversions

### Error Reporting
- **Precise Locations**: Line and column numbers for all errors
- **Helpful Messages**: Clear, actionable error descriptions
- **Suggested Fixes**: Provides suggestions for fixing type errors
- **Type Context**: Shows expected vs actual types
- **Inference Stack**: Debugging information showing type inference path

### Advanced Features
- **Effect System**: Tracks side effects (PURE, READ, WRITE, IO)
- **Purity Analysis**: Identifies pure functions (no side effects)
- **Type Cache**: Optional caching for improved performance
- **Quantization Constraints**: Infrastructure for quantization validation
- **Forward Declarations**: Functions can call other functions defined later

### Performance
- **Type Caching**: Reuses computed types with hit/miss statistics
- **Linear Complexity**: O(n) type checking for most programs
- **Efficient Unification**: Optimized unification algorithm
- **Incremental Support**: Ready for incremental type checking

## Usage

### Basic Usage

```python
from compiler.typechecker import TypeChecker
from compiler.ast.nodes import Program

# Create type checker instance
checker = TypeChecker()

# Validate an AST
errors = checker.validate(ast)

# Check for errors
if errors:
    for error in errors:
        print(error)  # Formatted error with line, col, suggestions
else:
    print("Type checking passed!")
```

### Advanced Usage

```python
# Enable caching for performance
checker = TypeChecker(enable_cache=True)
errors = checker.validate(ast)

# Get cache statistics
stats = checker.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")

# Enable effect tracking
checker = TypeChecker(enable_effects=True)
errors = checker.validate(ast)

# Check function purity
for func_name, signature in checker.function_table.items():
    if signature.is_pure:
        print(f"{func_name} is pure")
```

### Error Handling

```python
errors = checker.validate(ast)

for error in errors:
    # Access error details
    print(f"Line {error.lineno}, Col {error.col_offset}")
    print(f"Message: {error.message}")
    
    if error.suggested_fix:
        print(f"Fix: {error.suggested_fix}")
    
    if error.context:
        print(f"Context: {error.context}")
    
    # Formatted output
    print(str(error))  # All information in one string
```

## Type System

### Supported Types

- **TritType**: Ternary values {-1, 0, 1}
- **IntType**: Integer types with configurable bit width
- **FloatType**: Floating point types (FP32, FP16)
- **TensorType**: Multi-dimensional tensors with element type and shape

### Quantization Hierarchy

```
FP32 (FloatType(bits=32))
  ↓
FP16 (FloatType(bits=16))
  ↓
INT8 (IntType(bits=8))
  ↓
Ternary (TritType)
```

Quantization can only go down the hierarchy (higher precision to lower precision).
Explicit conversion operators should be used for quantization.

### Type Compatibility

- Same types are compatible
- IntType and TritType are compatible (in non-strict mode)
- Different bit widths are compatible in non-strict mode
- Tensor shapes must match for element-wise operations
- Matrix multiplication has special dimension rules: (m,n) @ (n,p) -> (m,p)

## Architecture

### Core Components

1. **TypeChecker**: Main visitor class that traverses AST
2. **TypeUnifier**: Implements unification algorithm
3. **TypeCache**: Caches inferred types for performance
4. **FunctionSignature**: Stores function type information
5. **TypeError**: Rich error information with suggestions

### Visitor Pattern

The type checker uses the visitor pattern to traverse the AST:

```python
def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
    # Infer types of operands
    left_type = self._infer_type(node.left)
    right_type = self._infer_type(node.right)
    
    # Validate compatibility
    if not self._types_compatible(left_type, right_type):
        self._add_error("Incompatible types", node)
    
    # Return result type
    return left_type
```

### Two-Pass Processing

For forward declarations, the program visitor uses two passes:

1. **First pass**: Register all function signatures
2. **Second pass**: Validate all function bodies

This allows functions to call other functions defined later in the program.

## Testing

The module includes 66 comprehensive test cases:

- **45 new tests** covering advanced features
- **21 legacy tests** for backward compatibility

### Test Coverage

- Basic type validation (trits, ints, floats)
- Tensor shape validation
- Binary operations and type compatibility
- Matrix multiplication with dimension checking
- Function signature validation
- Variable scoping
- Quantization rules
- Error reporting quality
- Type unification
- Performance benchmarks
- Real DSL examples
- Effect system

### Running Tests

```bash
# Run all type checker tests
pytest tests/unit/test_type_checker_comprehensive.py -v

# Run legacy tests
pytest tests/unit/test_typechecker.py -v

# Run with coverage
pytest tests/unit/test_type_checker_comprehensive.py --cov=compiler.typechecker
```

## Performance

The type checker is designed for performance:

- **Linear complexity**: O(n) for most programs
- **Type caching**: Avoids redundant type inference
- **Efficient unification**: Quick constraint solving
- **Minimal allocations**: Reuses type objects where possible

### Benchmarks

From the test suite:

- 100 trit literals: < 0.1s
- 100x100 tensor validation: < 1.0s
- 10-level function nesting: < 1.0s
- Cache hit rate: typically > 80% on real code

## Future Enhancements

- Dependent types for tensor shapes (infrastructure in place)
- More sophisticated constraint solving
- Parallel type inference for large programs
- Incremental type checking for IDEs
- Better type error recovery
- Type-directed code generation hints

## API Reference

### TypeChecker

```python
class TypeChecker(Visitor):
    def __init__(self, enable_cache=True, enable_effects=True):
        """Initialize type checker with optional features."""
    
    def validate(self, ast: Node) -> List[TypeError]:
        """Validate an AST and return errors."""
    
    def get_cache_stats(self) -> Optional[Dict[str, int]]:
        """Get type cache statistics."""
```

### TypeError

```python
@dataclass
class TypeError:
    message: str
    lineno: int = 0
    col_offset: int = 0
    suggested_fix: Optional[str] = None
    context: Optional[str] = None
    inference_stack: List[str] = field(default_factory=list)
```

### TypeUnifier

```python
class TypeUnifier:
    def unify(self, type1: Type, type2: Type) -> bool:
        """Unify two types, returns True if successful."""
```

### FunctionSignature

```python
@dataclass
class FunctionSignature:
    name: str
    param_types: List[Type]
    return_type: Optional[Type]
    effects: Set[EffectType] = field(default_factory=...)
    is_pure: bool = True
```

## Contributing

When adding new type checking features:

1. Add visitor method for new AST node types
2. Implement type inference logic
3. Add comprehensive tests
4. Update this documentation
5. Ensure backward compatibility

## License

Part of the Triton DSL project. See LICENSE file in the root directory.

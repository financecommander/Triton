# Triton DSL Type Checker Design

## Overview

The type checker is a critical component of the Triton DSL compiler that ensures semantic correctness of programs through static type analysis. It validates that operations are performed on compatible types, variables are properly declared, and ternary value constraints are satisfied.

## Design Philosophy

The Triton type system is designed with three key principles:

1. **Safety First:** Catch type errors at compile time, not runtime
2. **Ternary-Aware:** Special handling for {-1, 0, 1} value constraints
3. **Inference-Capable:** Minimize explicit type annotations where possible

## Type System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         Type Checker                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │   Symbol     │  │   Function   │  │   Constraint       │  │
│  │   Table      │  │   Table      │  │   Solver           │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │    Type      │  │  Inference   │  │    Error           │  │
│  │  Validator   │  │   Engine     │  │  Reporter          │  │
│  └──────────────┘  └──────────────┘  └────────────────────┘  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Typed AST +    │
                    │   Error List     │
                    └──────────────────┘
```

## Core Components

### 1. Type Representations

All types inherit from the `Type` base class:

```python
@dataclass
class Type(Node):
    """Base class for type representations."""
    name: str = ""
```

#### Primitive Types

**TritType - Ternary Values**
```python
@dataclass
class TritType(Type):
    """Ternary type: {-1, 0, 1}."""
    
    def __post_init__(self):
        self.name = "trit"
```

Represents the fundamental ternary value type. Only accepts values in the set {-1, 0, 1}.

**IntType - Integer Values**
```python
@dataclass
class IntType(Type):
    """Integer type with bit width."""
    bits: int = 32  # 8, 16, 32, 64
    
    def __post_init__(self):
        self.name = f"int{self.bits}"
```

**FloatType - Floating Point Values**
```python
@dataclass
class FloatType(Type):
    """Floating point type with bit width."""
    bits: int = 32  # 16, 32, 64
    
    def __post_init__(self):
        self.name = f"float{self.bits}"
```

#### Composite Types

**TensorType - Multi-dimensional Arrays**
```python
@dataclass
class TensorType(Type):
    """Tensor type with element type and shape."""
    element_type: Optional[Type] = None
    shape: Optional[List[int]] = None
    
    def __post_init__(self):
        self.name = "tensor"
```

Example tensor types:
```triton
tensor<trit, [128, 64]>      # 128x64 ternary matrix
tensor<float32, [10]>        # 10-element float vector
TernaryTensor                # Shorthand for tensor<trit, [...]>
```

### 2. TypeChecker Class

The main type checking engine uses the Visitor pattern to traverse the AST:

```python
class TypeChecker(Visitor):
    """
    Type checker that validates AST nodes using the visitor pattern.
    """
    
    def __init__(self) -> None:
        self.errors: List[TypeError] = []
        self.symbol_table: Dict[str, Type] = {}
        self.function_table: Dict[str, Tuple[List[Type], Optional[Type]]] = {}
        self.current_function_return_type: Optional[Type] = None
```

#### Core Methods

**validate() - Main Entry Point**
```python
def validate(self, ast: Node) -> List[TypeError]:
    """
    Validate an AST node and return list of errors.
    
    Returns:
        List of type errors found (empty if validation succeeds)
    """
    self.errors = []
    self.symbol_table = {}
    self.function_table = {}
    self.current_function_return_type = None
    
    ast.accept(self)
    return self.errors
```

**infer_type() - Type Inference**
```python
def infer_type(self, node: Expr) -> Optional[Type]:
    """
    Infer the type of an expression through visitor traversal.
    Returns None if type cannot be determined.
    """
    return node.accept(self)
```

**types_compatible() - Compatibility Checking**
```python
def types_compatible(self, type1: Type, type2: Type) -> bool:
    """
    Check if two types are compatible for operations.
    
    Rules:
    - Same type is always compatible
    - Trit and Int are compatible (trit can be promoted to int)
    - Tensors must have compatible element types and shapes
    """
    # Exact type match
    if type(type1) == type(type2):
        if isinstance(type1, IntType):
            return type1.bits == type2.bits
        if isinstance(type1, TensorType):
            return (self.types_compatible(type1.element_type, type2.element_type) and
                    type1.shape == type2.shape)
        return True
    
    # Trit-Int compatibility
    if isinstance(type1, (TritType, IntType)) and isinstance(type2, (TritType, IntType)):
        return True
    
    return False
```

### 3. Symbol Table Management

The type checker maintains a symbol table mapping identifiers to their types:

```python
self.symbol_table: Dict[str, Type] = {
    'x': TensorType(element_type=TritType(), shape=[128]),
    'y': TritType(),
    'weight': TensorType(element_type=TritType(), shape=[128, 64])
}
```

**Scope Handling:**
```python
def visit_function_def(self, node: FunctionDef) -> Any:
    # Save current scope
    old_symbol_table = self.symbol_table.copy()
    old_return_type = self.current_function_return_type
    
    # Create new scope for function body
    self.symbol_table = {}
    for param in node.params:
        self.symbol_table[param.name] = param.type_annotation
    self.current_function_return_type = node.return_type
    
    # Validate function body
    for statement in node.body:
        statement.accept(self)
    
    # Restore outer scope
    self.symbol_table = old_symbol_table
    self.current_function_return_type = old_return_type
```

### 4. Function Table

Stores function signatures for call validation:

```python
self.function_table: Dict[str, Tuple[List[Type], Optional[Type]]] = {
    'matmul': ([TensorType(...), TensorType(...)], TensorType(...)),
    'activation': ([TensorType(...)], TensorType(...))
}
```

## Type Checking Algorithm

### Algorithm Overview

```
function CHECK_TYPES(ast_node):
    match ast_node.type:
        case PROGRAM:
            for each statement in ast_node.statements:
                CHECK_TYPES(statement)
        
        case DECLARATION:
            value_type = INFER_TYPE(ast_node.value)
            if not COMPATIBLE(ast_node.declared_type, value_type):
                REPORT_ERROR("Type mismatch in declaration")
            SYMBOL_TABLE[ast_node.name] = ast_node.declared_type
        
        case ASSIGNMENT:
            var_type = SYMBOL_TABLE[ast_node.target]
            value_type = INFER_TYPE(ast_node.value)
            if not COMPATIBLE(var_type, value_type):
                REPORT_ERROR("Type mismatch in assignment")
        
        case BINARY_OP:
            left_type = INFER_TYPE(ast_node.left)
            right_type = INFER_TYPE(ast_node.right)
            if ast_node.op == '@':  # Matrix multiplication
                CHECK_MATMUL_DIMENSIONS(left_type, right_type)
            else:
                if not COMPATIBLE(left_type, right_type):
                    REPORT_ERROR("Incompatible operand types")
        
        case FUNCTION_CALL:
            (param_types, return_type) = FUNCTION_TABLE[ast_node.name]
            for arg, param_type in zip(ast_node.args, param_types):
                arg_type = INFER_TYPE(arg)
                if not COMPATIBLE(arg_type, param_type):
                    REPORT_ERROR("Argument type mismatch")
```

### Detailed Validation Rules

#### Rule 1: Trit Value Constraints

All trit literals must be in {-1, 0, 1}:

```python
def visit_trit_literal(self, node: TritLiteral) -> Type:
    """Validate trit literal value."""
    if node.value not in {-1, 0, 1}:
        self.add_error(
            f"Trit literal must be -1, 0, or 1, got {node.value}",
            node
        )
    return TritType()
```

Example:
```triton
let x: trit = 1   # ✓ Valid
let y: trit = 2   # ✗ Error: Trit literal must be -1, 0, or 1
```

#### Rule 2: TernaryTensor Shape Validation

Tensor shape must match the number of values provided:

```python
def visit_ternary_tensor(self, node: TernaryTensor) -> Type:
    """Validate ternary tensor shape and values."""
    # Check all values are trits
    for i, value in enumerate(node.values):
        if value not in {-1, 0, 1}:
            self.add_error(
                f"TernaryTensor value at index {i} must be -1, 0, or 1",
                node
            )
    
    # Validate shape matches value count
    if node.shape:
        expected_count = 1
        for dim in node.shape:
            expected_count *= dim
        
        if len(node.values) != expected_count:
            self.add_error(
                f"TernaryTensor shape {node.shape} expects {expected_count} values, "
                f"got {len(node.values)}",
                node
            )
    
    return TensorType(element_type=TritType(), shape=node.shape)
```

Example:
```triton
# ✓ Valid: 2x2 = 4 values
let matrix: TernaryTensor = TernaryTensor[2, 2](1, 0, -1, 1)

# ✗ Error: 2x2 = 4 values expected, got 3
let matrix: TernaryTensor = TernaryTensor[2, 2](1, 0, -1)
```

#### Rule 3: Variable Must Be Declared

References to undefined variables are errors:

```python
def visit_identifier(self, node: Identifier) -> Optional[Type]:
    """Check identifier is defined in symbol table."""
    if node.name not in self.symbol_table:
        self.add_error(f"Undefined variable '{node.name}'", node)
        return None
    return self.symbol_table[node.name]
```

Example:
```triton
let x: trit = 1
let y: trit = x    # ✓ Valid: x is defined
let z: trit = w    # ✗ Error: Undefined variable 'w'
```

#### Rule 4: Binary Operation Type Compatibility

Operands of binary operations must have compatible types:

```python
def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
    """Validate binary operation type compatibility."""
    left_type = self.infer_type(node.left)
    right_type = self.infer_type(node.right)
    
    if left_type is None or right_type is None:
        return None
    
    # Check compatibility
    if not self.types_compatible(left_type, right_type):
        self.add_error(
            f"Binary operation '{node.op}' requires compatible types, "
            f"got {type(left_type).__name__} and {type(right_type).__name__}",
            node
        )
        return None
    
    # Result type depends on operation
    if node.op in {'+', '-', '*', '/'}:
        return left_type  # Arithmetic preserves type
    elif node.op in {'==', '!=', '<', '>', '<=', '>='}:
        return IntType(bits=8)  # Comparisons return bool/int
```

Example:
```triton
let x: trit = 1
let y: trit = 0
let z: trit = x + y        # ✓ Valid: trit + trit

let a: float32 = 1.0
let b: trit = 1
let c: float32 = a + b     # ✗ Error: Incompatible types
```

#### Rule 5: Matrix Multiplication Dimension Compatibility

Matrix multiplication requires compatible inner dimensions:

```python
def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
    if node.op == '@':
        if not isinstance(left_type, TensorType) or not isinstance(right_type, TensorType):
            self.add_error("Matrix multiplication requires tensor operands", node)
            return None
        
        # Check dimension compatibility
        if left_type.shape and right_type.shape:
            if len(left_type.shape) < 2 or len(right_type.shape) < 2:
                self.add_error("Matrix multiplication requires at least 2D tensors", node)
                return None
            
            # For (m, n) @ (n, p) -> (m, p)
            if left_type.shape[-1] != right_type.shape[-2]:
                self.add_error(
                    f"Matrix multiplication dimension mismatch: "
                    f"{left_type.shape} cannot multiply with {right_type.shape}. "
                    f"Inner dimensions must match",
                    node
                )
                return None
            
            # Result shape
            result_shape = left_type.shape[:-1] + [right_type.shape[-1]]
            return TensorType(element_type=left_type.element_type, shape=result_shape)
```

Example:
```triton
# ✓ Valid: (128, 64) @ (64, 32) -> (128, 32)
let A: tensor<trit, [128, 64]> = ...
let B: tensor<trit, [64, 32]> = ...
let C: tensor<trit, [128, 32]> = A @ B

# ✗ Error: Dimension mismatch (128 != 32)
let D: tensor<trit, [128, 64]> = ...
let E: tensor<trit, [32, 16]> = ...
let F = D @ E  # Error: 64 != 32
```

#### Rule 6: Function Call Signature Matching

Function calls must match declared signatures:

```python
def visit_function_call(self, node: FunctionCall) -> Optional[Type]:
    """Validate function call against signature."""
    if node.name not in self.function_table:
        self.add_error(f"Undefined function '{node.name}'", node)
        return None
    
    param_types, return_type = self.function_table[node.name]
    
    # Check argument count
    if len(node.args) != len(param_types):
        self.add_error(
            f"Function '{node.name}' expects {len(param_types)} arguments, "
            f"got {len(node.args)}",
            node
        )
        return return_type
    
    # Check argument types
    for i, (arg, expected_type) in enumerate(zip(node.args, param_types)):
        arg_type = self.infer_type(arg)
        if arg_type and not self.types_compatible(arg_type, expected_type):
            self.add_error(
                f"Function '{node.name}' argument {i} expects "
                f"{type(expected_type).__name__}, got {type(arg_type).__name__}",
                node
            )
    
    return return_type
```

Example:
```triton
layer matmul(x: tensor<trit, [128]>, w: tensor<trit, [128, 64]>) -> tensor<trit, [64]> {
    ...
}

let x: tensor<trit, [128]> = ...
let w: tensor<trit, [128, 64]> = ...
let y = matmul(x, w)        # ✓ Valid

let bad = matmul(x)         # ✗ Error: Expected 2 arguments, got 1
```

#### Rule 7: Return Type Matching

Return statements must match function return type:

```python
def visit_return(self, node: Return) -> Any:
    """Validate return type matches function signature."""
    if node.value:
        return_type = self.infer_type(node.value)
        if self.current_function_return_type and return_type:
            if not self.types_compatible(self.current_function_return_type, return_type):
                self.add_error(
                    f"Return type {type(return_type).__name__} does not match "
                    f"function return type {type(self.current_function_return_type).__name__}",
                    node
                )
    elif self.current_function_return_type:
        self.add_error("Function must return a value", node)
```

Example:
```triton
layer compute(x: trit) -> trit {
    return x        # ✓ Valid: returns trit
}

layer bad(x: trit) -> int32 {
    return x        # ✗ Error: Returns trit, expected int32
}
```

## Type Inference Engine

The type inference engine determines expression types through bottom-up traversal:

### Inference Rules

**Literals:**
```python
INFER(TritLiteral(v))     → TritType()
INFER(IntLiteral(v))      → IntType(bits=32)
INFER(FloatLiteral(v))    → FloatType(bits=32)
```

**Variables:**
```python
INFER(Identifier(name))   → SYMBOL_TABLE[name]
```

**Binary Operations:**
```python
INFER(BinaryOp(left, op, right)):
    left_type = INFER(left)
    right_type = INFER(right)
    
    if op in {'+', '-', '*', '/'}:
        return left_type
    elif op == '@':
        return matmul_result_type(left_type, right_type)
```

**Function Calls:**
```python
INFER(FunctionCall(name, args)):
    (_, return_type) = FUNCTION_TABLE[name]
    return return_type
```

### Type Promotion

Triton supports limited type promotion:

```python
def promote_type(type1: Type, type2: Type) -> Type:
    """
    Find common type for mixed operations.
    
    Promotion hierarchy:
    trit -> int8 -> int16 -> int32 -> int64
          -> float16 -> float32 -> float64
    """
    if isinstance(type1, TritType) and isinstance(type2, IntType):
        return type2  # Promote trit to int
    
    if isinstance(type1, IntType) and isinstance(type2, FloatType):
        return type2  # Promote int to float
    
    # Same type - no promotion
    return type1
```

Example:
```triton
let x: trit = 1
let y: int32 = 10
let z = x + y          # z has type int32 (trit promoted to int32)
```

## Constraint Solving

The type checker solves constraints through iterative refinement:

### Constraint Collection

```python
# During type checking, collect constraints:
constraints = [
    (x, TritType()),                    # x must be trit
    (y, TensorType(element_type=?)),   # y is tensor of unknown element type
    (z, type_of(x @ y))                # z type depends on x and y
]
```

### Constraint Solving Algorithm

```python
def solve_constraints(constraints):
    """
    Solve type constraints using unification.
    
    1. Propagate known types
    2. Infer unknown types from usage
    3. Check for contradictions
    """
    changed = True
    while changed:
        changed = False
        for constraint in constraints:
            if refine_constraint(constraint):
                changed = True
    
    # Check for unsolved constraints
    for constraint in constraints:
        if not is_solved(constraint):
            report_error("Cannot infer type for variable")
```

### Example: Constraint Solving

```triton
layer compute(x) {
    let y = x + 1        # Constraint: type(x) compatible with int
    let z = y @ y        # Constraint: type(y) is 2D tensor
    return z
}
```

Constraint solving:
1. `x + 1` → `x` must be numeric type
2. `y @ y` → `y` must be 2D tensor
3. `y = x + 1` → `x` must be tensor (to produce tensor `y`)
4. Solve: `x: tensor<int32, [n, m]>`, `y: tensor<int32, [n, m]>`, `z: tensor<int32, [n, m]>`

## Error Reporting

### Error Structure

```python
@dataclass
class TypeError:
    """Type error with precise location information."""
    message: str
    lineno: int = 0
    col_offset: int = 0
    
    def __str__(self) -> str:
        if self.lineno > 0:
            return f"Line {self.lineno}, Col {self.col_offset}: {self.message}"
        return self.message
```

### Error Categories

**1. Type Mismatch Errors**
```
Line 5, Col 10: Binary operation '+' requires compatible types, 
                got TritType and FloatType
```

**2. Undefined Variable Errors**
```
Line 3, Col 14: Undefined variable 'weight'
```

**3. Shape Mismatch Errors**
```
Line 8, Col 5: Matrix multiplication dimension mismatch: 
               shape [128, 64] cannot multiply with shape [32, 16]. 
               Inner dimensions must match (64 != 32)
```

**4. Value Constraint Errors**
```
Line 2, Col 15: Trit literal must be -1, 0, or 1, got 5
```

### Error Recovery

The type checker continues after errors to report multiple issues:

```python
def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
    left_type = self.infer_type(node.left)
    right_type = self.infer_type(node.right)
    
    # Even if left has error, check right
    if left_type is None or right_type is None:
        return None  # Propagate error but continue checking
    
    # Continue validation
    ...
```

## Implementation Details

### Visitor Pattern

The type checker implements the Visitor pattern for AST traversal:

```python
class Visitor(ABC):
    """Base visitor interface."""
    
    def visit_program(self, node: Program) -> Any: pass
    def visit_trit_type(self, node: TritType) -> Any: pass
    def visit_int_type(self, node: IntType) -> Any: pass
    def visit_binary_op(self, node: BinaryOp) -> Any: pass
    # ... more visit methods
```

Each AST node accepts a visitor:

```python
@dataclass
class BinaryOp(Expr):
    left: Expr
    op: str
    right: Expr
    
    def accept(self, visitor: Visitor) -> Any:
        return visitor.visit_binary_op(self)
```

### Performance Optimization

**Single-Pass Type Checking:**
```python
# One traversal for both type inference and validation
def validate(self, ast: Node) -> List[TypeError]:
    self.errors = []
    ast.accept(self)  # Single pass
    return self.errors
```

**Early Exit on Critical Errors:**
```python
# Stop checking function body if return type is invalid
if not self.validate_return_type(node.return_type):
    return None  # Skip body validation
```

**Memoization:**
```python
# Cache inferred types to avoid recomputation
self.type_cache: Dict[Node, Type] = {}

def infer_type(self, node: Expr) -> Optional[Type]:
    if node in self.type_cache:
        return self.type_cache[node]
    
    result = node.accept(self)
    self.type_cache[node] = result
    return result
```

## Testing Strategy

### Unit Tests

Test each validation rule independently:

```python
def test_trit_literal_validation():
    checker = TypeChecker()
    
    # Valid trits
    assert len(checker.validate(TritLiteral(1))) == 0
    assert len(checker.validate(TritLiteral(0))) == 0
    assert len(checker.validate(TritLiteral(-1))) == 0
    
    # Invalid trits
    errors = checker.validate(TritLiteral(2))
    assert len(errors) == 1
    assert "must be -1, 0, or 1" in errors[0].message
```

### Integration Tests

Test complete programs:

```python
def test_full_program_type_checking():
    source = """
    layer linear(x: tensor<trit, [128]>) -> tensor<trit, [64]> {
        let w: tensor<trit, [128, 64]> = ...
        return x @ w
    }
    """
    
    ast = parse(source)
    checker = TypeChecker()
    errors = checker.validate(ast)
    assert len(errors) == 0
```

## Best Practices

### For Type Checker Developers

1. **Comprehensive Error Messages:** Include type names, shapes, and suggestions
2. **Location Tracking:** Always include line and column numbers
3. **Continue After Errors:** Don't stop at first error
4. **Cache Type Information:** Avoid redundant inference
5. **Test Edge Cases:** Nested types, complex expressions, etc.

### For DSL Users

1. **Explicit Type Annotations:** Help catch errors early
2. **Descriptive Variable Names:** Improve error readability
3. **Break Complex Expressions:** Easier to debug type errors
4. **Use Type Aliases:** Define common tensor types once

## Summary

The Triton DSL type checker provides robust static type validation through a visitor-based architecture with symbol table management, type inference, constraint solving, and comprehensive error reporting. It enforces ternary value constraints, tensor shape compatibility, and type safety throughout the compilation pipeline.

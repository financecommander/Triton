# Triton DSL Language Specification

This document provides the complete specification of the Triton Domain-Specific Language (DSL) for Ternary Neural Networks.

## Overview

Triton DSL is a statically-typed, domain-specific language designed specifically for defining and compiling ternary neural networks. It enforces ternary constraints at the language level, ensuring correctness and optimization opportunities.

### Design Goals

1. **Type Safety**: Catch ternary constraint violations at compile time
2. **Expressiveness**: Natural syntax for neural network operations
3. **Performance**: Enable aggressive optimizations for ternary operations
4. **Interoperability**: Seamless integration with PyTorch and other frameworks

### Language Features

- Static type system with type inference
- First-class support for ternary tensors
- Built-in neural network primitives
- Automatic differentiation support
- Shape checking and broadcasting
- Effect tracking for side effects

## Lexical Structure

### Keywords

Reserved words in Triton DSL:

```
layer       # Layer definition
fn          # Function definition
let         # Variable binding
return      # Return statement
if          # Conditional
else        # Alternative branch
for         # Loop construct
while       # While loop
```

### Type Keywords

```
trit            # Ternary type {-1, 0, 1}
int8, int32     # Integer types
float16, float32  # Floating-point types
tensor          # Generic tensor type
TernaryTensor   # Ternary tensor type
```

### Operators

#### Arithmetic Operators

```
+   # Addition
-   # Subtraction / Negation
*   # Element-wise multiplication
@   # Matrix multiplication
/   # Division
%   # Modulo
```

#### Comparison Operators

```
==  # Equality
!=  # Inequality
<   # Less than
<=  # Less than or equal
>   # Greater than
>=  # Greater than or equal
```

#### Logical Operators

```
&&  # Logical AND
||  # Logical OR
!   # Logical NOT
```

### Literals

#### Integer Literals

```triton
42          # Decimal
0x2A        # Hexadecimal
0b101010    # Binary
```

#### Float Literals

```triton
3.14
1.0e-5
-2.5
```

#### Ternary Literals

```triton
trit(-1)    # Negative ternary value
trit(0)     # Zero ternary value
trit(1)     # Positive ternary value
```

### Comments

```triton
# Single-line comment

/*
 * Multi-line comment
 * Spans multiple lines
 */
```

### Identifiers

Valid identifier names:

- Start with letter or underscore: `[a-zA-Z_]`
- Followed by letters, digits, underscores: `[a-zA-Z0-9_]*`
- Case-sensitive

```triton
layer_name      # Valid
_private        # Valid
MyLayer         # Valid
layer123        # Valid
123layer        # Invalid (starts with digit)
```

## Type System

### Primitive Types

#### Ternary Type

```triton
trit            # Single ternary value: {-1, 0, 1}
```

#### Integer Types

```triton
int8            # 8-bit signed integer
int32           # 32-bit signed integer
```

#### Floating-Point Types

```triton
float16         # 16-bit floating-point (half precision)
float32         # 32-bit floating-point (single precision)
```

### Tensor Types

#### Generic Tensor

```triton
Tensor<dtype, shape>
```

Examples:

```triton
Tensor<float32, [784, 128]>         # 2D tensor
Tensor<float32, [?, 128]>           # Dynamic first dimension
Tensor<float32, [batch, seq, 512]>  # Named dimensions
```

#### Ternary Tensor

```triton
TernaryTensor<trit, shape>
```

Examples:

```triton
TernaryTensor<trit, [128, 256]>     # Ternary weight matrix
TernaryTensor<trit, [?, 10]>        # Dynamic batch size
```

### Type Inference

The compiler can infer types in many contexts:

```triton
let x = 42              # Inferred as int32
let y = 3.14            # Inferred as float32
let z = x + y           # Inferred as float32 (promotion)
```

### Type Compatibility

#### Implicit Conversions

```triton
int8 → int32            # Integer widening
int32 → float32         # Integer to float
float16 → float32       # Float widening
```

#### Explicit Conversions

```triton
let x: float32 = 42.0
let y: int32 = int32(x)     # Explicit cast
```

### Shape Constraints

Shapes must be compatible for operations:

```triton
let a: Tensor<float32, [128, 256]>
let b: Tensor<float32, [256, 512]>
let c = a @ b           # Valid: [128, 256] @ [256, 512] = [128, 512]

let d: Tensor<float32, [128, 100]>
let e = a @ d           # Error: incompatible shapes
```

## Syntax

### Layer Definitions

```triton
layer LayerName(
    param1: Type1,
    param2: Type2,
    ...
) -> ReturnType {
    # Layer body
    return expression
}
```

Example:

```triton
layer TernaryLinear(
    weights: TernaryTensor<trit, [in_features, out_features]>,
    bias: TernaryTensor<trit, [out_features]>,
    x: Tensor<float32, [?, in_features]>
) -> Tensor<float32, [?, out_features]> {
    let result = x @ weights + bias
    return result
}
```

### Function Definitions

```triton
fn function_name(param1: Type1, param2: Type2) -> ReturnType {
    # Function body
    return expression
}
```

Example:

```triton
fn compute_scale(tensor: Tensor<float32, [?, ?]>) -> float32 {
    let max_val = max(abs(tensor))
    return max_val
}
```

### Variable Bindings

```triton
let identifier = expression             # Type inferred
let identifier: Type = expression       # Explicit type
```

### Control Flow

#### If-Else

```triton
if condition {
    # True branch
} else {
    # False branch
}
```

#### For Loop

```triton
for i in range(n) {
    # Loop body
}
```

#### While Loop

```triton
while condition {
    # Loop body
}
```

### Expressions

#### Binary Operations

```triton
a + b           # Addition
a - b           # Subtraction
a * b           # Element-wise multiplication
a @ b           # Matrix multiplication
a / b           # Division
```

#### Unary Operations

```triton
-x              # Negation
!x              # Logical NOT
```

#### Function Calls

```triton
function_name(arg1, arg2, ...)
```

#### Indexing

```triton
tensor[i]           # Single index
tensor[i, j]        # Multiple indices
tensor[i:j]         # Slicing
```

## Semantics

### Evaluation Order

Operations are evaluated left-to-right with standard precedence:

1. Function calls, indexing
2. Unary operators (-, !)
3. Multiplicative (*, /, %)
4. Additive (+, -)
5. Matrix multiplication (@)
6. Comparison (<, >, <=, >=)
7. Equality (==, !=)
8. Logical AND (&&)
9. Logical OR (||)

### Broadcasting

Tensors are automatically broadcast following NumPy/PyTorch rules:

```triton
let a: Tensor<float32, [3, 1]>
let b: Tensor<float32, [1, 4]>
let c = a + b           # Result: [3, 4]
```

### Quantization Semantics

Ternary tensors maintain their quantization through operations:

```triton
let w: TernaryTensor<trit, [128, 256]>  # Values in {-1, 0, 1}
let x: Tensor<float32, [?, 128]>
let y = x @ w           # Matrix multiply preserves ternary constraints
```

## Memory Model

### Pass-by-Value

Primitive types are passed by value:

```triton
fn increment(x: int32) -> int32 {
    return x + 1
}

let a = 5
let b = increment(a)    # a is copied
# a is still 5
```

### Pass-by-Reference

Tensors are passed by reference:

```triton
fn modify_tensor(t: Tensor<float32, [?]>) -> Tensor<float32, [?]> {
    # Operations on t affect the original
    return t * 2.0
}
```

## Scoping Rules

### Lexical Scoping

Variables follow lexical scoping:

```triton
let x = 10

fn outer() {
    let y = 20
    
    fn inner() {
        let z = 30
        # Can access x, y, z
        return x + y + z
    }
    
    return inner()
}
```

### Shadowing

Inner scopes can shadow outer variables:

```triton
let x = 10

fn test() {
    let x = 20      # Shadows outer x
    return x        # Returns 20
}
```

## Error Handling

### Compile-Time Errors

#### Type Errors

```triton
let x: int32 = 3.14         # Error: type mismatch

let a: Tensor<float32, [10]>
let b: Tensor<float32, [20]>
let c = a + b               # Error: incompatible shapes
```

#### Ternary Constraint Violations

```triton
let w: TernaryTensor<trit, [10]> = Tensor<float32, [10]>
# Error: cannot assign non-ternary to ternary tensor
```

### Runtime Errors

Runtime errors are propagated to the host framework (PyTorch):

- Shape mismatches with dynamic dimensions
- Out-of-bounds indexing
- Division by zero

## Complete Example

```triton
# Ternary ResNet Block

layer TernaryConvBlock(
    conv_weights: TernaryTensor<trit, [out_ch, in_ch, 3, 3]>,
    conv_bias: TernaryTensor<trit, [out_ch]>,
    bn_weight: Tensor<float32, [out_ch]>,
    bn_bias: Tensor<float32, [out_ch]>,
    x: Tensor<float32, [?, in_ch, h, w]>
) -> Tensor<float32, [?, out_ch, h, w]> {
    # Convolution with ternary weights
    let conv_out = conv2d(x, conv_weights, padding=1)
    let conv_result = conv_out + conv_bias
    
    # Batch normalization
    let bn_out = batch_norm(conv_result, bn_weight, bn_bias)
    
    # Activation
    let activated = relu(bn_out)
    
    return activated
}

layer TernaryResidualBlock(
    conv1_w: TernaryTensor<trit, [channels, channels, 3, 3]>,
    conv1_b: TernaryTensor<trit, [channels]>,
    bn1_w: Tensor<float32, [channels]>,
    bn1_b: Tensor<float32, [channels]>,
    conv2_w: TernaryTensor<trit, [channels, channels, 3, 3]>,
    conv2_b: TernaryTensor<trit, [channels]>,
    bn2_w: Tensor<float32, [channels]>,
    bn2_b: Tensor<float32, [channels]>,
    x: Tensor<float32, [?, channels, h, w]>
) -> Tensor<float32, [?, channels, h, w]> {
    # Save input for residual connection
    let residual = x
    
    # First convolution block
    let out = TernaryConvBlock(conv1_w, conv1_b, bn1_w, bn1_b, x)
    
    # Second convolution block (no activation)
    let conv_out = conv2d(out, conv2_w, padding=1)
    let conv_result = conv_out + conv2_b
    let bn_out = batch_norm(conv_result, bn2_w, bn2_b)
    
    # Residual connection
    let sum = bn_out + residual
    let result = relu(sum)
    
    return result
}
```

## Language Evolution

### Version History

- **v0.1.0** (Current): Initial release
  - Basic type system
  - Layer definitions
  - Built-in operations
  - PyTorch backend

### Future Additions

Planned features for future versions:

- Generic type parameters
- Higher-order functions
- Pattern matching
- Module system
- Effects and regions
- Compile-time evaluation
- Hardware-specific intrinsics

## Best Practices

### 1. Use Type Annotations

```triton
# Good: Clear types
let x: float32 = 3.14

# Less clear: Inferred type
let x = 3.14
```

### 2. Name Dimensions

```triton
# Good: Named dimensions
Tensor<float32, [batch, seq_len, hidden_dim]>

# Less clear: Anonymous dimensions
Tensor<float32, [?, ?, ?]>
```

### 3. Layer Composition

```triton
# Good: Reusable layers
layer FullyConnected(...)
layer Activation(...)

layer MLP(...) {
    let h1 = FullyConnected(...)
    let a1 = Activation(h1)
    # ...
}
```

### 4. Explicit Quantization Boundaries

```triton
# Good: Clear where quantization happens
let weights_fp32: Tensor<float32, [128, 256]>
let weights_ternary: TernaryTensor<trit, [128, 256]> = quantize(weights_fp32)
```

## References

- [Syntax Guide](syntax_guide.md) - Detailed syntax reference
- [Type System](type_system.md) - Complete type system documentation
- [Built-in Functions](builtin_functions.md) - All built-in functions
- [Quantization Primitives](quantization_primitives.md) - Quantization operations

## Formal Grammar

See [GRAMMAR.md](../../compiler/parser/GRAMMAR.md) for the complete formal grammar specification.

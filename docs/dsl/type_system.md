# Triton DSL Type System

This document provides comprehensive documentation of the Triton DSL type system, including the type hierarchy, checking rules, inference mechanisms, generics, and shape polymorphism.

## Table of Contents

- [Overview](#overview)
- [Type Hierarchy](#type-hierarchy)
- [Primitive Types](#primitive-types)
- [Tensor Types](#tensor-types)
- [Type Checking Rules](#type-checking-rules)
- [Type Inference](#type-inference)
- [Generics and Constraints](#generics-and-constraints)
- [Shape Polymorphism](#shape-polymorphism)
- [Advanced Topics](#advanced-topics)

## Overview

Triton DSL features a static type system with the following key characteristics:

- **Static typing**: All types are checked at compile time
- **Type inference**: Types can often be inferred from context
- **Nominal typing**: Types are distinguished by name and structure
- **Gradual shape checking**: Shapes can be concrete or abstract
- **Generic types**: Support for parameterized types
- **Effect tracking**: Side effects are tracked in function signatures

### Design Principles

1. **Safety First**: Catch type errors at compile time
2. **Ternary Constraints**: Enforce ternary values through the type system
3. **Shape Awareness**: Track tensor shapes to prevent dimension mismatches
4. **Inference**: Minimize explicit type annotations
5. **Expressiveness**: Support complex neural network patterns

## Type Hierarchy

The Triton DSL type system is organized hierarchically:

```
Type
├── Scalar
│   ├── Numeric
│   │   ├── Integer
│   │   │   ├── int8
│   │   │   ├── int16
│   │   │   ├── int32
│   │   │   └── int64
│   │   ├── Float
│   │   │   ├── float16
│   │   │   ├── float32
│   │   │   └── float64
│   │   └── Ternary
│   │       └── trit
│   └── Boolean
│       └── bool
├── Tensor<dtype, shape>
│   ├── TernaryTensor<trit, shape>
│   └── FloatTensor<float32, shape>
├── Collection
│   ├── List<T>
│   ├── Tuple<T1, T2, ...>
│   └── Dict<K, V>
├── Function<Args, Return>
└── Layer<Params, Input, Output>
```

### Subtyping Relationships

```triton
# Integer subtyping (widening conversions)
int8 <: int16 <: int32 <: int64

# Float subtyping (widening conversions)
float16 <: float32 <: float64

# Ternary is NOT a subtype of integer
# (explicit conversion required)
trit ≠ int8

# Tensor subtyping (contravariant in dtype)
TernaryTensor<trit, [m, n]> <: Tensor<trit, [m, n]>
```

## Primitive Types

### Integer Types

```triton
int8        # 8-bit signed integer: -128 to 127
int16       # 16-bit signed integer: -32,768 to 32,767
int32       # 32-bit signed integer: -2^31 to 2^31-1
int64       # 64-bit signed integer: -2^63 to 2^63-1
```

**Default**: `int32` for integer literals

**Operations**: All standard arithmetic and bitwise operations

```triton
let a: int8 = 42
let b: int32 = 1000000
let c = a + b  # Type: int32 (widening conversion)
```

### Floating-Point Types

```triton
float16     # 16-bit half-precision (IEEE 754)
float32     # 32-bit single-precision (IEEE 754)
float64     # 64-bit double-precision (IEEE 754)
```

**Default**: `float32` for floating-point literals

**Special values**: `inf`, `-inf`, `nan`

```triton
let pi: float32 = 3.14159
let e: float64 = 2.718281828459045

# Special values
let infinity = float32.inf
let not_a_number = float32.nan
```

### Ternary Type

The core type for ternary neural networks:

```triton
trit        # Ternary value: {-1, 0, 1}
```

**Properties**:
- Restricted to exactly three values: -1, 0, 1
- Not implicitly convertible to other types
- Specialized arithmetic rules

```triton
let neg: trit = trit(-1)
let zero: trit = trit(0)
let pos: trit = trit(1)

# Ternary arithmetic
let result = neg * pos  # trit(-1) * trit(1) = trit(-1)
```

### Boolean Type

```triton
bool        # Boolean: true or false
```

**Literals**: `true`, `false`

```triton
let flag: bool = true
let condition = (x > 0) && (y < 10)
```

### Unit Type

```triton
()          # Unit type (empty tuple)
void        # Alias for unit type in function returns
```

Used for functions that return no value:

```triton
fn print_message(msg: str) -> void {
    print(msg)
}
```

## Tensor Types

### Generic Tensor Type

```triton
Tensor<dtype, shape>
```

**Type Parameters**:
- `dtype`: Element data type (trit, int8, int32, float32, etc.)
- `shape`: Shape specification (tuple of dimensions)

**Examples**:

```triton
# Concrete shapes
Tensor<float32, [784, 128]>              # 2D tensor: 784×128
Tensor<int32, [10, 20, 30]>              # 3D tensor: 10×20×30

# Dynamic dimensions
Tensor<float32, [?, 128]>                # Batch size unknown
Tensor<float32, [batch, features]>       # Named dimensions

# Mixed concrete and dynamic
Tensor<float32, [?, 28, 28, 3]>          # Variable batch, fixed image size
```

### TernaryTensor Type

Specialized tensor for ternary values:

```triton
TernaryTensor<trit, shape>
```

**Properties**:
- All elements guaranteed to be in {-1, 0, 1}
- Optimized storage and operations
- Type-safe quantization

```triton
# Weight tensors
let weights: TernaryTensor<trit, [784, 10]> = initialize_ternary()

# Activations
let activations: TernaryTensor<trit, [batch, 128]> = quantize(x)
```

### Shape Specifications

#### Concrete Shapes

```triton
[128, 256]              # Exact shape
[10, 10, 3]            # 3D with all dimensions known
```

#### Dynamic Shapes

```triton
[?, 128]               # First dimension unknown
[batch, ?]             # Second dimension unknown
[?, ?, 3]              # First two dimensions unknown
```

#### Named Dimensions

```triton
[batch, seq_len, hidden]           # Named dimensions
[batch, channels, height, width]   # NCHW format
```

#### Shape Variables

```triton
# Generic over shapes
fn identity<S>(x: Tensor<float32, S>) -> Tensor<float32, S> {
    return x
}
```

### Tensor Type Properties

```triton
# Access shape at compile time
type InputTensor = Tensor<float32, [784, 128]>
let input_shape = InputTensor.shape  # [784, 128]

# Access dtype
let dtype = InputTensor.dtype  # float32

# Rank (number of dimensions)
let rank = InputTensor.rank  # 2
```

## Type Checking Rules

### Type Compatibility

#### Exact Matching

```triton
let x: int32 = 42       # OK: exact match
let y: int32 = 3.14     # Error: float32 not compatible with int32
```

#### Implicit Conversions

Widening conversions are allowed:

```triton
# Integer widening
let a: int8 = 42
let b: int32 = a        # OK: int8 widens to int32

# Float widening
let c: float16 = 1.0
let d: float32 = c      # OK: float16 widens to float32

# Integer to float
let e: int32 = 42
let f: float32 = e      # OK: int32 converts to float32
```

#### Prohibited Conversions

```triton
# Narrowing conversions require explicit cast
let a: int32 = 1000
let b: int8 = a         # Error: may overflow
let c: int8 = int8(a)   # OK: explicit cast

# Ternary conversions always explicit
let t: trit = 1         # Error: int32 not compatible with trit
let u: trit = trit(1)   # OK: explicit conversion
```

### Tensor Compatibility

#### Shape Matching

```triton
# Exact shape match required for assignment
let a: Tensor<float32, [10, 20]> = ...
let b: Tensor<float32, [10, 20]> = a    # OK

let c: Tensor<float32, [20, 10]> = a    # Error: shape mismatch
```

#### Broadcasting Rules

Operations follow NumPy-style broadcasting:

```triton
# Compatible for broadcasting
Tensor<float32, [5, 1]> + Tensor<float32, [1, 3]>  # Result: [5, 3]
Tensor<float32, [10, 1, 5]> + Tensor<float32, [1, 7, 5]>  # Result: [10, 7, 5]

# Incompatible
Tensor<float32, [5, 3]> + Tensor<float32, [5, 2]>  # Error: dimension 2 vs 3
```

#### Matrix Multiplication Rules

```triton
# Valid matrix multiplication
[m, k] @ [k, n] -> [m, n]

# Examples
Tensor<float32, [128, 256]> @ Tensor<float32, [256, 512]>  # OK: [128, 512]
Tensor<float32, [128, 256]> @ Tensor<float32, [128, 512]>  # Error: 256 != 128
```

#### Batch Matrix Multiplication

```triton
# Batch dimensions must match or broadcast
[b, m, k] @ [b, k, n] -> [b, m, n]
[b1, m, k] @ [b2, k, n] -> error if b1 != b2 and neither is 1

# Examples
Tensor<float32, [10, 128, 256]> @ Tensor<float32, [10, 256, 512]>  # OK
Tensor<float32, [1, 128, 256]> @ Tensor<float32, [10, 256, 512]>   # OK (broadcast)
```

### Function Type Checking

```triton
# Function signature
fn add(x: int32, y: int32) -> int32 {
    return x + y
}

# Call site type checking
let result = add(1, 2)      # OK: arguments match
let error = add(1, 2.0)     # Error: float32 not compatible with int32
let error2 = add(1)         # Error: missing argument
```

### Layer Type Checking

```triton
layer Linear(
    weights: TernaryTensor<trit, [in_features, out_features]>,
    input: Tensor<float32, [batch, in_features]>
) -> Tensor<float32, [batch, out_features]> {
    return input @ weights
}

# Instantiation
let layer = Linear(weights: my_weights)

# Type checking at call site
let output = layer(input: my_input)  # Shape must match [?, in_features]
```

## Type Inference

### Variable Type Inference

```triton
# Infer from literal
let x = 42              # int32
let y = 3.14            # float32
let z = true            # bool

# Infer from expression
let sum = x + y         # float32 (promotion)
let product = x * 2     # int32

# Infer from function return
let sin_x = sin(y)      # float32 (from sin signature)
```

### Function Return Type Inference

```triton
# Explicit return type
fn add(x: int32, y: int32) -> int32 {
    return x + y
}

# Inferred return type
fn add_inferred(x: int32, y: int32) {
    return x + y        # Return type inferred as int32
}

# Multiple return paths must agree
fn conditional(flag: bool) {
    if flag {
        return 42       # int32
    } else {
        return 0        # int32
    }
    # Inferred: -> int32
}
```

### Tensor Shape Inference

```triton
# Shape inferred from operations
let a: Tensor<float32, [10, 20]> = ...
let b = a.transpose()   # Inferred: Tensor<float32, [20, 10]>
let c = a @ b           # Inferred: Tensor<float32, [10, 10]>

# Shape propagation through layers
layer MyLayer(input: Tensor<float32, [?, 784]>) {
    let hidden = input @ weights  # Shape: [?, 128] if weights is [784, 128]
    return hidden
}
```

### Generic Type Inference

```triton
# Generic function
fn identity<T>(x: T) -> T {
    return x
}

# Type parameter inferred at call site
let a = identity(42)        # T = int32
let b = identity(3.14)      # T = float32
let c = identity(trit(1))   # T = trit
```

### Constraint-Based Inference

```triton
# Infer with constraints
fn process<T: Numeric>(x: T, y: T) -> T {
    return x + y
}

let result = process(1, 2)      # T = int32
let error = process(1, 2.0)     # Error: T cannot be both int32 and float32
```

## Generics and Constraints

### Generic Type Parameters

```triton
# Single type parameter
fn first<T>(list: List<T>) -> T {
    return list[0]
}

# Multiple type parameters
fn pair<A, B>(a: A, b: B) -> Tuple<A, B> {
    return (a, b)
}

# Generic layer
layer GenericLinear<T>(
    weights: Tensor<T, [in_features, out_features]>,
    input: Tensor<T, [batch, in_features]>
) -> Tensor<T, [batch, out_features]> {
    return input @ weights
}
```

### Type Constraints

#### Trait Bounds

```triton
# Numeric constraint
fn add<T: Numeric>(x: T, y: T) -> T {
    return x + y
}

# Multiple constraints
fn compare_and_add<T: Numeric + Comparable>(x: T, y: T) -> T {
    return if x > y { x + y } else { y }
}
```

#### Built-in Traits

```triton
Numeric         # Supports arithmetic operations
Comparable      # Supports comparison operations
Quantizable     # Can be quantized to ternary
Differentiable  # Supports automatic differentiation
Hashable        # Can be used as dictionary key
```

**Examples**:

```triton
# Numeric: int8, int32, float32, float64
fn multiply<T: Numeric>(x: T, y: T) -> T {
    return x * y
}

# Comparable: int8, int32, float32, float64, bool
fn max<T: Comparable>(x: T, y: T) -> T {
    return if x > y { x } else { y }
}

# Quantizable: float32, float64
fn quantize<T: Quantizable>(x: Tensor<T, [?, ?]>) -> TernaryTensor<trit, [?, ?]> {
    return ternary_quantize(x)
}
```

### Shape Generics

```triton
# Generic over entire shape
fn reshape_flatten<S>(x: Tensor<float32, S>) -> Tensor<float32, [?]> {
    return x.reshape([-1])
}

# Generic over specific dimensions
fn transpose_batch<B, M, N>(x: Tensor<float32, [B, M, N]>) -> Tensor<float32, [B, N, M]> {
    return x.transpose(1, 2)
}

# Constraint on dimensions
fn square_matmul<N>(x: Tensor<float32, [N, N]>, y: Tensor<float32, [N, N]>) 
    -> Tensor<float32, [N, N]> {
    return x @ y
}
```

### Where Clauses

For complex constraints:

```triton
fn complex_operation<T, S1, S2>(x: Tensor<T, S1>, y: Tensor<T, S2>) -> Tensor<T, ?>
where
    T: Numeric + Differentiable,
    S1: Shape,
    S2: Shape,
    S1.dims == 2,
    S2.dims == 2,
    S1[1] == S2[0]  # Compatible for matrix multiplication
{
    return x @ y
}
```

## Shape Polymorphism

### Dynamic Dimensions

```triton
# Unknown batch size
fn process_batch(x: Tensor<float32, [?, 784]>) -> Tensor<float32, [?, 10]> {
    return x @ weights  # Batch dimension preserved
}

# Multiple dynamic dimensions
fn dynamic_conv(
    input: Tensor<float32, [?, ?, ?, 3]>  # [batch, height, width, channels]
) -> Tensor<float32, [?, ?, ?, 64]> {
    return conv2d(input, kernel)
}
```

### Shape Variables

```triton
# Named shape variables
fn matmul<M, K, N>(
    a: Tensor<float32, [M, K]>,
    b: Tensor<float32, [K, N]>
) -> Tensor<float32, [M, N]> {
    return a @ b
}

# Preserving relationships
fn broadcast_add<B, M, N>(
    x: Tensor<float32, [B, M, N]>,
    y: Tensor<float32, [M, N]>
) -> Tensor<float32, [B, M, N]> {
    return x + y
}
```

### Shape Constraints

```triton
# Require specific relationships
fn concat_channels<B, H, W, C1, C2>(
    x: Tensor<float32, [B, H, W, C1]>,
    y: Tensor<float32, [B, H, W, C2]>
) -> Tensor<float32, [B, H, W, C1 + C2]> {
    return concat([x, y], axis=3)
}

# Square tensor constraint
fn diagonal<N>(x: Tensor<float32, [N, N]>) -> Tensor<float32, [N]> {
    return x.diagonal()
}
```

### Shape Arithmetic

```triton
# Compute output shapes
fn conv2d_output_shape<B, H, W, C>(
    input: Tensor<float32, [B, H, W, C]>,
    kernel_size: int32,
    stride: int32,
    padding: int32
) -> Tensor<float32, [B, (H + 2*padding - kernel_size)//stride + 1, 
                         (W + 2*padding - kernel_size)//stride + 1, C]> {
    return conv2d(input, ...)
}
```

## Advanced Topics

### Dependent Types

Limited support for types that depend on values:

```triton
# Vector type parameterized by length
type Vec<n: int32> = Tensor<float32, [n]>

fn dot<n: int32>(x: Vec<n>, y: Vec<n>) -> float32 {
    return (x * y).sum()
}
```

### Higher-Kinded Types

```triton
# Type constructor
type Container<F<_>> = {
    fn map<A, B>(self, f: fn(A) -> B) -> F<B>
}

# Example: List is a type constructor
impl Container<List> for List<A> {
    fn map<A, B>(self, f: fn(A) -> B) -> List<B> {
        return [f(x) for x in self]
    }
}
```

### Phantom Types

Types used for compile-time checking only:

```triton
# Phantom type parameter for units
type Meters<T> = Tensor<T, [?]>
type Feet<T> = Tensor<T, [?]>

fn add_meters<T: Numeric>(x: Meters<T>, y: Meters<T>) -> Meters<T> {
    return x + y
}

# Type error: cannot add different units
let m: Meters<float32> = ...
let f: Feet<float32> = ...
let error = add_meters(m, f)  # Error: Feet<float32> not compatible with Meters<float32>
```

### Effect Types

Track side effects in function signatures:

```triton
# Pure function (no effects)
fn pure_add(x: int32, y: int32) -> int32 {
    return x + y
}

# Function with IO effect
fn read_file(path: str) -> str throws IO {
    return read(path)
}

# Function with state mutation
fn increment_counter() -> int32 mutates State {
    global counter
    counter += 1
    return counter
}
```

### Type Aliases

```triton
# Simple alias
type Float = float32
type Int = int32

# Generic alias
type Matrix<T> = Tensor<T, [?, ?]>
type Vector<T> = Tensor<T, [?]>

# Ternary aliases
type TernaryWeights = TernaryTensor<trit, [?, ?]>
type TernaryActivations = TernaryTensor<trit, [?, ?]>
```

### Existential Types

Hide implementation details:

```triton
# Existential type
type SomeLayer = exists Shape. Layer<Tensor<float32, Shape>, Tensor<float32, Shape>>

# User can't see concrete shape
fn create_layer() -> SomeLayer {
    return LinearLayer(weights: ...)
}
```

## Type System Examples

### Complete Network Type

```triton
layer TernaryMLP(
    w1: TernaryTensor<trit, [784, 256]>,
    b1: Tensor<float32, [256]>,
    w2: TernaryTensor<trit, [256, 128]>,
    b2: Tensor<float32, [128]>,
    w3: TernaryTensor<trit, [128, 10]>,
    b3: Tensor<float32, [10]>,
    input: Tensor<float32, [?, 784]>
) -> Tensor<float32, [?, 10]> {
    # Layer 1: [?, 784] @ [784, 256] = [?, 256]
    let h1 = ternary_linear(input, w1, b1)
    
    # Layer 2: [?, 256] @ [256, 128] = [?, 128]
    let h2 = ternary_linear(h1, w2, b2)
    
    # Layer 3: [?, 128] @ [128, 10] = [?, 10]
    let output = ternary_linear(h2, w3, b3)
    
    return output
}
```

### Type-Safe Quantization

```triton
# Ensures only float tensors are quantized
fn safe_quantize<S>(x: Tensor<float32, S>) -> TernaryTensor<trit, S> 
where S: Shape {
    return ternary_quantize(x)
}

# Won't compile: int32 doesn't implement Quantizable
let error = safe_quantize(Tensor<int32, [10, 10]>())
```

This comprehensive type system ensures safety and correctness for ternary neural network implementations while maintaining flexibility and expressiveness.

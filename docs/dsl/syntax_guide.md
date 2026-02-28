# Triton DSL Syntax Guide

This guide provides a comprehensive overview of Triton DSL syntax, covering expressions, statements, patterns, and common idioms for writing efficient ternary neural networks.

## Table of Contents

- [Basic Syntax](#basic-syntax)
- [Expression Syntax](#expression-syntax)
- [Statement Syntax](#statement-syntax)
- [Pattern Examples](#pattern-examples)
- [Common Idioms](#common-idioms)

## Basic Syntax

### Program Structure

A Triton DSL program consists of layer definitions, function definitions, and optionally a main block:

```triton
# Layer definition
layer MyLayer(input: TernaryTensor<trit, [?, 784]>) -> TernaryTensor<trit, [?, 10]> {
    let weights = ternary_weights([784, 10])
    return input @ weights
}

# Function definition
fn quantize_activations(x: Tensor<float32, [?, ?]>) -> TernaryTensor<trit, [?, ?]> {
    return ternary_quantize(x)
}

# Main block
main {
    let model = MyLayer()
    print("Model initialized")
}
```

### Indentation and Whitespace

- Use 4 spaces for indentation (no tabs)
- Whitespace is not significant except for line breaks
- Multiple statements can appear on one line with semicolons

```triton
# Single line
let x = 1; let y = 2; let z = x + y

# Multi-line (preferred)
let x = 1
let y = 2
let z = x + y
```

### Comments

```triton
# Single-line comment

/* Multi-line comment
   Can span multiple lines
   Useful for documentation */

/*
 * Documentation style comment
 * Often used for layer/function documentation
 */
layer MyLayer(input: TernaryTensor<trit, [?, 784]>) -> TernaryTensor<trit, [?, 10]> {
    # Implementation comment
    return input
}
```

## Expression Syntax

### Literals

#### Numeric Literals

```triton
# Integer literals
42              # int32
0x2A            # Hexadecimal
0b101010        # Binary
0o52            # Octal

# Float literals
3.14            # float32
1.0e-5          # Scientific notation
-2.5            # Negative
.5              # Leading decimal point
5.              # Trailing decimal point
```

#### Ternary Literals

```triton
trit(-1)        # Negative ternary value
trit(0)         # Zero ternary value
trit(1)         # Positive ternary value
```

#### String Literals

```triton
"Hello, world!"         # Double-quoted string
'Single quoted'         # Single-quoted string
"Escape sequences: \n\t\r"
```

#### Tensor Literals

```triton
# Explicit tensor creation
tensor([1, 2, 3], dtype=float32)
tensor([[1, 0, -1], [1, 1, 0]], dtype=trit)

# Ternary tensor literals
ternary_tensor([[-1, 0, 1], [1, 0, -1]])
```

### Variable References

```triton
x               # Simple variable
layer.weights   # Member access
module.layer.weights  # Nested member access
```

### Arithmetic Expressions

#### Binary Operators

```triton
x + y           # Addition
x - y           # Subtraction
x * y           # Element-wise multiplication
x / y           # Division
x % y           # Modulo
x @ y           # Matrix multiplication
x ** y          # Exponentiation
```

#### Unary Operators

```triton
-x              # Negation
+x              # Positive (identity)
```

#### Operator Precedence

From highest to lowest:

1. Member access: `.`
2. Function calls: `()`
3. Exponentiation: `**`
4. Unary: `+`, `-`, `!`
5. Multiplicative: `*`, `/`, `%`, `@`
6. Additive: `+`, `-`
7. Comparison: `<`, `<=`, `>`, `>=`
8. Equality: `==`, `!=`
9. Logical AND: `&&`
10. Logical OR: `||`

```triton
# Examples demonstrating precedence
let result = 2 + 3 * 4        # 14, not 20
let matmul = a @ b + c        # (a @ b) + c
let power = 2 ** 3 ** 2       # 2 ** (3 ** 2) = 512
```

### Comparison Expressions

```triton
x == y          # Equality
x != y          # Inequality
x < y           # Less than
x <= y          # Less than or equal
x > y           # Greater than
x >= y          # Greater than or equal
```

Comparisons can be chained:

```triton
0 < x < 10      # Equivalent to: (0 < x) && (x < 10)
a <= b <= c     # Equivalent to: (a <= b) && (b <= c)
```

### Logical Expressions

```triton
x && y          # Logical AND
x || y          # Logical OR
!x              # Logical NOT

# Short-circuit evaluation
x > 0 && y / x > 2    # y/x only evaluated if x > 0
```

### Conditional Expressions

```triton
# Ternary conditional
let result = condition ? value_if_true : value_if_false

# Examples
let sign = x >= 0 ? 1 : -1
let max_val = a > b ? a : b
let quantized = abs(x) < threshold ? trit(0) : sign(x)
```

### Function Calls

```triton
# Simple function call
sin(x)

# Multiple arguments
pow(x, 2)
ternary_quantize(x, threshold=0.5)

# Named arguments
conv2d(input, kernel, stride=1, padding=0)

# Keyword arguments
reshape(x, shape=[128, 256])
```

### Indexing and Slicing

```triton
# Single index
tensor[0]           # First element
tensor[-1]          # Last element

# Multiple indices
matrix[i, j]        # Element at (i, j)
tensor3d[i, j, k]   # 3D indexing

# Slicing
tensor[start:end]       # Slice from start to end
tensor[start:end:step]  # Slice with step
tensor[:10]             # First 10 elements
tensor[10:]             # From 10 to end
tensor[:]               # All elements (copy)

# Multi-dimensional slicing
matrix[:, 0]            # First column
matrix[0, :]            # First row
matrix[1:5, 2:8]        # Sub-matrix
```

### Member Access

```triton
# Attribute access
tensor.shape        # Get shape
tensor.dtype        # Get data type
tensor.device       # Get device

# Method calls
tensor.transpose()
tensor.reshape([128, 256])
tensor.sum(axis=1)
```

### List and Tuple Expressions

```triton
# Lists
let numbers = [1, 2, 3, 4, 5]
let shapes = [[128, 256], [256, 512]]

# Tuples
let pair = (x, y)
let triple = (1, 2, 3)

# Unpacking
let (a, b) = pair
let [first, ...rest] = numbers
```

## Statement Syntax

### Variable Declarations

```triton
# Immutable binding (let)
let x = 42
let y: float32 = 3.14
let z: Tensor<float32, [128, 256]> = initialize_weights()

# Type inference
let inferred = sin(x)    # Type inferred from sin return type

# Multiple bindings
let a = 1, b = 2, c = 3

# Pattern matching in binding
let (x, y) = get_coordinates()
let [first, second, ...rest] = get_list()
```

### Assignment Statements

```triton
# Simple assignment (for mutable variables)
x = 42

# Compound assignment
x += 1          # x = x + 1
x -= 2          # x = x - 2
x *= 3          # x = x * 3
x /= 4          # x = x / 4
x @= matrix     # x = x @ matrix

# Multi-target assignment
a = b = c = 0

# Tuple unpacking
(x, y) = (y, x)     # Swap values
```

### Conditional Statements

```triton
# If statement
if condition {
    # body
}

# If-else
if condition {
    # true branch
} else {
    # false branch
}

# If-else if-else chain
if x < 0 {
    return trit(-1)
} else if x > 0 {
    return trit(1)
} else {
    return trit(0)
}

# Inline if expression
let value = if condition { expr1 } else { expr2 }
```

### Loop Statements

#### For Loops

```triton
# Range-based for loop
for i in 0..10 {
    print(i)
}

# Inclusive range
for i in 0..=10 {    # 0 to 10 inclusive
    print(i)
}

# Iterate over collection
for item in collection {
    process(item)
}

# Enumerate pattern
for (index, value) in enumerate(collection) {
    print(f"{index}: {value}")
}

# With step
for i in 0..100:2 {  # Every other number
    print(i)
}
```

#### While Loops

```triton
# Basic while loop
while condition {
    # body
}

# While with break
while true {
    if should_stop {
        break
    }
}

# While with continue
while has_more {
    if should_skip {
        continue
    }
    process()
}
```

### Return Statements

```triton
# Simple return
return value

# Early return
if error_condition {
    return default_value
}

# Multiple return values (tuple)
return (result, status, metadata)

# Void return
return  # Returns None/void
```

### Block Expressions

```triton
# Block as expression
let result = {
    let temp = compute_something()
    let processed = transform(temp)
    processed  # Last expression is returned
}

# Nested blocks
{
    let x = 1
    {
        let y = x + 1
        print(y)
    }
    # y is not accessible here
}
```

## Pattern Examples

### Layer Patterns

#### Basic Ternary Layer

```triton
layer TernaryLinear(
    weights: TernaryTensor<trit, [in_features, out_features]>,
    bias: TernaryTensor<trit, [out_features]>,
    input: Tensor<float32, [batch, in_features]>
) -> Tensor<float32, [batch, out_features]> {
    # Forward pass
    let output = input @ weights
    return output + bias
}
```

#### Quantization-Aware Layer

```triton
layer QATLinear(
    weights: Tensor<float32, [in_features, out_features]>,
    input: Tensor<float32, [batch, in_features]>
) -> Tensor<float32, [batch, out_features]> {
    # Quantize weights during forward pass
    let quantized_weights = ternary_quantize(weights)
    let output = input @ quantized_weights
    
    # Straight-through estimator for gradients
    return output
}
```

#### Composite Layer

```triton
layer TernaryConvBlock(
    conv_weights: TernaryTensor<trit, [out_ch, in_ch, kh, kw]>,
    bn_gamma: Tensor<float32, [out_ch]>,
    bn_beta: Tensor<float32, [out_ch]>,
    input: Tensor<float32, [batch, in_ch, h, w]>
) -> Tensor<float32, [batch, out_ch, h2, w2]> {
    # Convolution
    let conv_out = conv2d(input, conv_weights, stride=1, padding=1)
    
    # Batch normalization
    let bn_out = batch_norm(conv_out, bn_gamma, bn_beta)
    
    # Activation
    return relu(bn_out)
}
```

### Function Patterns

#### Gradient-Safe Quantization

```triton
fn ste_quantize(x: Tensor<float32, [?, ?]>) -> TernaryTensor<trit, [?, ?]> {
    # Forward: hard quantization
    let quantized = ternary_quantize(x)
    
    # Backward: straight-through estimator
    # Gradient flows through as if identity
    return quantized with_gradient_from x
}
```

#### Custom Initialization

```triton
fn ternary_kaiming_init(shape: [int, int]) -> TernaryTensor<trit, [?, ?]> {
    # Generate random values
    let random = randn(shape, dtype=float32)
    
    # Scale based on fan-in
    let fan_in = shape[0]
    let scaled = random * sqrt(2.0 / fan_in)
    
    # Quantize to ternary
    return ternary_quantize(scaled)
}
```

#### Type-Generic Functions

```triton
fn safe_divide<T>(a: T, b: T, default: T = 0) -> T {
    return if b != 0 { a / b } else { default }
}

fn clamp<T: Comparable>(x: T, min_val: T, max_val: T) -> T {
    if x < min_val {
        return min_val
    } else if x > max_val {
        return max_val
    } else {
        return x
    }
}
```

### Control Flow Patterns

#### Guard Clauses

```triton
fn process_tensor(x: Tensor<float32, [?, ?]>) -> Tensor<float32, [?, ?]> {
    # Guard clauses for early return
    if x.shape[0] == 0 {
        return x
    }
    
    if x.shape[1] != 784 {
        panic("Invalid input dimension")
    }
    
    # Main logic
    return transform(x)
}
```

#### Error Handling Pattern

```triton
fn safe_operation(x: Tensor<float32, [?, ?]>) -> Result<Tensor<float32, [?, ?]>, Error> {
    if !is_valid(x) {
        return Err("Invalid input tensor")
    }
    
    let result = risky_operation(x)
    
    if !is_valid(result) {
        return Err("Operation produced invalid output")
    }
    
    return Ok(result)
}
```

#### Iterator Pattern

```triton
fn apply_to_layers(model: Model, transform_fn: fn(Layer) -> Layer) -> Model {
    let transformed = []
    for layer in model.layers {
        transformed.append(transform_fn(layer))
    }
    return Model(transformed)
}
```

## Common Idioms

### Ternary Quantization Idioms

#### Threshold-Based Quantization

```triton
fn threshold_quantize(x: Tensor<float32, [?, ?]>, threshold: float32 = 0.5) -> TernaryTensor<trit, [?, ?]> {
    return if abs(x) < threshold {
        trit(0)
    } else if x > 0 {
        trit(1)
    } else {
        trit(-1)
    }
}
```

#### Gradient-Preserving Quantization

```triton
fn ste_ternary_quantize(x: Tensor<float32, [?, ?]>) -> Tensor<float32, [?, ?]> {
    # Forward: quantize to {-1, 0, 1}
    let forward = ternary_quantize(x)
    
    # Backward: pass gradient through unchanged
    return stop_gradient(forward) + (x - stop_gradient(x))
}
```

### Tensor Manipulation Idioms

#### Shape Broadcasting

```triton
# Expand dimensions for broadcasting
let expanded = x.unsqueeze(1)  # [batch, features] -> [batch, 1, features]

# Repeat tensor
let repeated = x.repeat([1, 3, 1])  # Repeat along axis 1

# Flatten
let flattened = x.reshape([-1])  # Flatten to 1D
let flat_batch = x.reshape([x.shape[0], -1])  # [batch, ...] -> [batch, total_features]
```

#### Dimension Swapping

```triton
# Transpose 2D
let transposed = x.transpose()

# Permute dimensions
let permuted = x.permute([0, 2, 1, 3])  # [b, c, h, w] -> [b, h, c, w]

# Move axis
let moved = x.moveaxis(1, -1)  # Move channel to last dimension
```

### Weight Initialization Idioms

#### Ternary Weight Initialization

```triton
fn initialize_ternary_weights(shape: [int, int], sparsity: float32 = 0.3) -> TernaryTensor<trit, [?, ?]> {
    let size = shape[0] * shape[1]
    let num_zeros = int32(size * sparsity)
    let num_nonzero = size - num_zeros
    
    # Create distribution
    let weights = [
        ...([trit(-1)] * (num_nonzero / 2)),
        ...([trit(0)] * num_zeros),
        ...([trit(1)] * (num_nonzero / 2))
    ]
    
    # Shuffle and reshape
    return shuffle(weights).reshape(shape)
}
```

#### Balanced Initialization

```triton
fn balanced_ternary_init(shape: [int, int]) -> TernaryTensor<trit, [?, ?]> {
    # Equal numbers of -1, 0, 1
    let total = shape[0] * shape[1]
    let per_value = total / 3
    
    let values = [
        ...([trit(-1)] * per_value),
        ...([trit(0)] * per_value),
        ...([trit(1)] * per_value)
    ]
    
    return shuffle(values).reshape(shape)
}
```

### Optimization Idioms

#### Fused Operations

```triton
# Fuse quantization with matmul
fn quantized_linear(
    input: Tensor<float32, [batch, in_features]>,
    weights: Tensor<float32, [in_features, out_features]>
) -> Tensor<float32, [batch, out_features]> {
    # Compiler can fuse these operations
    return input @ ternary_quantize(weights)
}
```

#### In-Place Operations

```triton
# Use in-place operations for efficiency
fn normalize_inplace(x: Tensor<float32, [?, ?]>) -> Tensor<float32, [?, ?]> {
    let mean = x.mean()
    let std = x.std()
    x -= mean      # In-place subtraction
    x /= std       # In-place division
    return x
}
```

### Debugging Idioms

#### Shape Assertions

```triton
fn verify_shapes(x: Tensor<float32, [?, ?]>, expected_shape: [int, int]) {
    assert x.shape[0] == expected_shape[0], "Batch size mismatch"
    assert x.shape[1] == expected_shape[1], "Feature size mismatch"
}
```

#### Value Range Checks

```triton
fn check_ternary_values(x: TernaryTensor<trit, [?, ?]>) {
    assert all(x >= -1), "Values below -1 detected"
    assert all(x <= 1), "Values above 1 detected"
    assert all(x in [-1, 0, 1]), "Non-ternary values detected"
}
```

#### Gradient Debugging

```triton
fn debug_gradients(x: Tensor<float32, [?, ?]>, name: str) -> Tensor<float32, [?, ?]> {
    # Register hook to print gradient statistics
    x.register_hook(fn(grad) {
        print(f"{name} gradient: mean={grad.mean()}, std={grad.std()}")
        print(f"{name} gradient: min={grad.min()}, max={grad.max()}")
    })
    return x
}
```

### Model Composition Idioms

#### Sequential Composition

```triton
fn sequential(layers: [Layer]) -> fn(Tensor) -> Tensor {
    return fn(input: Tensor) {
        let output = input
        for layer in layers {
            output = layer(output)
        }
        return output
    }
}
```

#### Residual Connections

```triton
layer ResidualBlock(
    conv1: TernaryConv2d,
    conv2: TernaryConv2d,
    input: Tensor<float32, [batch, channels, h, w]>
) -> Tensor<float32, [batch, channels, h, w]> {
    let residual = input
    let out = conv1(input)
    out = relu(out)
    out = conv2(out)
    return out + residual  # Residual connection
}
```

#### Skip Connections

```triton
layer DenseBlock(
    layers: [TernaryLayer],
    input: Tensor<float32, [batch, features]>
) -> Tensor<float32, [batch, features]> {
    let outputs = [input]
    for layer in layers {
        # Concatenate all previous outputs
        let combined = concat(outputs, axis=1)
        let output = layer(combined)
        outputs.append(output)
    }
    return concat(outputs, axis=1)
}
```

## Best Practices

### Naming Conventions

```triton
# Layers: PascalCase
layer TernaryLinear(...) { }

# Functions: snake_case
fn ternary_quantize(...) { }

# Variables: snake_case
let input_tensor = ...
let learning_rate = 0.001

# Constants: UPPER_SNAKE_CASE
let MAX_ITERATIONS = 1000
let DEFAULT_THRESHOLD = 0.5
```

### Type Annotations

```triton
# Always annotate function parameters
fn process(x: Tensor<float32, [?, ?]>) -> Tensor<float32, [?, ?]> {
    return x
}

# Annotate complex types
let weights: TernaryTensor<trit, [784, 128]> = initialize_weights()

# Allow inference for simple cases
let x = 42  # Clearly int32
let y = x + 1  # Clearly int32
```

### Error Messages

```triton
# Provide helpful error messages
fn validate_input(x: Tensor<float32, [?, ?]>) {
    assert x.shape[1] == 784, 
        f"Expected 784 features, got {x.shape[1]}"
}
```

This syntax guide covers the essential patterns and idioms for writing effective Triton DSL code. Refer to the Language Specification for formal grammar and detailed semantics.

# Triton DSL Documentation

Welcome to the Triton Domain-Specific Language documentation. Triton DSL is a statically-typed language designed specifically for defining and compiling ternary neural networks.

## Documentation Structure

### Core Language Reference

1. **[Language Specification](language_spec.md)** (12KB)
   - Complete formal specification of Triton DSL
   - Lexical structure, grammar, and semantics
   - Type system fundamentals
   - Language features and design goals

2. **[Syntax Guide](syntax_guide.md)** (18KB)
   - Comprehensive syntax overview
   - Expression and statement syntax
   - Common patterns and idioms
   - Best practices and naming conventions
   - Code examples for all language constructs

3. **[Type System](type_system.md)** (18KB)
   - Complete type hierarchy
   - Type checking rules and inference
   - Generics and type constraints
   - Shape polymorphism
   - Advanced type system features

### Built-in Functions and Primitives

4. **[Built-in Functions Reference](builtin_functions.md)** (25KB)
   - Tensor creation and manipulation functions
   - Mathematical and statistical functions
   - Linear algebra operations
   - Neural network operations (conv2d, pooling, etc.)
   - Activation and loss functions
   - Complete signatures, descriptions, and examples

5. **[Quantization Primitives](quantization_primitives.md)** (25KB)
   - Quantization methods (threshold, adaptive, stochastic, learned)
   - Dequantization techniques
   - Quantization-aware operations
   - Gradient handling (STE, clipped STE, soft quantization)
   - Custom quantizer implementation
   - Advanced techniques and best practices

## Quick Start

### Hello World in Triton DSL

```triton
# Define a simple ternary linear layer
layer TernaryLinear(
    weights: TernaryTensor<trit, [784, 10]>,
    bias: Tensor<float32, [10]>,
    input: Tensor<float32, [?, 784]>
) -> Tensor<float32, [?, 10]> {
    let output = input @ weights
    return output + bias
}

# Main execution
main {
    let weights = ternary_weights([784, 10])
    let bias = zeros([10])
    let layer = TernaryLinear(weights, bias)
    
    let input = randn([32, 784])
    let output = layer(input)
    
    print(output)
}
```

### Key Concepts

#### Ternary Types
Triton DSL's primary innovation is first-class support for ternary values:

```triton
trit                              # Single ternary value: {-1, 0, 1}
TernaryTensor<trit, [m, n]>      # Tensor of ternary values
```

#### Type Safety
All types are checked at compile time:

```triton
let x: Tensor<float32, [128, 256]> = randn([128, 256])  # OK
let y: Tensor<float32, [128, 256]> = randn([256, 128])  # Error: shape mismatch
```

#### Quantization
Built-in quantization primitives:

```triton
let weights = randn([784, 128])
let quantized = ternary_quantize(weights)              # {-1, 0, 1}
let with_ste = ste_quantize(weights)                   # Gradient-preserving
```

## Common Patterns

### Defining a Ternary Neural Network

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
    # Layer 1
    let h1 = relu(input @ w1 + b1)
    
    # Layer 2
    let h2 = relu(h1 @ w2 + b2)
    
    # Output layer
    let output = h2 @ w3 + b3
    
    return output
}
```

### Quantization-Aware Training

```triton
layer QATLinear(
    weights: Tensor<float32, [in_features, out_features]>,
    input: Tensor<float32, [batch, in_features]>
) -> Tensor<float32, [batch, out_features]> {
    # Quantize weights in forward pass
    let q_weights = ste_quantize(ternary_quantize(weights))
    
    # Compute output with quantized weights
    return input @ q_weights
}
```

### Custom Quantization

```triton
fn adaptive_quantize(
    x: Tensor<float32, [?, ?]>,
    target_sparsity: float32 = 0.3
) -> TernaryTensor<trit, [?, ?]> {
    # Compute threshold for target sparsity
    let abs_x = abs(x)
    let threshold = percentile(abs_x, target_sparsity * 100)
    
    # Apply threshold quantization
    return threshold_quantize(x, threshold)
}
```

## Learning Path

### 1. For Language Learners
Start here if you're new to Triton DSL:

1. Read the [Syntax Guide](syntax_guide.md) for basic syntax
2. Study [Language Specification](language_spec.md) for complete grammar
3. Explore [Built-in Functions](builtin_functions.md) for available operations
4. Review common patterns in [Syntax Guide](syntax_guide.md#common-idioms)

### 2. For Type System Enthusiasts
Deep dive into the type system:

1. Start with [Type System](type_system.md) overview
2. Understand [Shape Polymorphism](type_system.md#shape-polymorphism)
3. Learn about [Generics and Constraints](type_system.md#generics-and-constraints)
4. Study advanced features in [Type System](type_system.md#advanced-topics)

### 3. For Neural Network Practitioners
Focus on building ternary networks:

1. Read [Quantization Primitives](quantization_primitives.md) overview
2. Learn [Quantization Methods](quantization_primitives.md#quantization-methods)
3. Understand [Gradient Handling](quantization_primitives.md#gradient-handling)
4. Study [Neural Network Operations](builtin_functions.md#neural-network-operations)
5. Review [Best Practices](quantization_primitives.md#best-practices)

### 4. For Compiler Engineers
Understand language semantics:

1. Study [Language Specification](language_spec.md) completely
2. Review [Type Checking Rules](type_system.md#type-checking-rules)
3. Understand [Type Inference](type_system.md#type-inference)
4. Explore effect tracking and optimization opportunities

## Code Examples

All documentation files include extensive code examples:

- **[Syntax Guide](syntax_guide.md)**: 50+ complete code examples
- **[Type System](type_system.md)**: 40+ type system examples
- **[Built-in Functions](builtin_functions.md)**: 100+ function examples
- **[Quantization Primitives](quantization_primitives.md)**: 60+ quantization examples

## Design Philosophy

### 1. Safety First
Triton DSL catches errors at compile time:
- Type mismatches
- Shape incompatibilities
- Ternary constraint violations

### 2. Expressiveness
Natural syntax for neural network operations:
- Layer definitions with clear parameter types
- Matrix multiplication with `@` operator
- Built-in quantization primitives

### 3. Performance
Enable aggressive optimizations:
- Static shape checking for kernel fusion
- Specialized ternary arithmetic
- Zero-cost abstractions

### 4. Interoperability
Seamless integration with existing frameworks:
- Export to PyTorch
- ONNX compatibility
- Hardware backend support

## Contributing

When contributing to Triton DSL:

1. Follow the style conventions in [Syntax Guide](syntax_guide.md#best-practices)
2. Ensure type safety as described in [Type System](type_system.md)
3. Add comprehensive examples for new features
4. Update relevant documentation sections

## Additional Resources

- **Examples Directory**: See `examples/` for complete programs
- **Tests**: See `tests/` for unit and integration tests
- **Compiler**: See `compiler/` for implementation details

## Getting Help

- Check the [Syntax Guide](syntax_guide.md) for language questions
- Review [Built-in Functions](builtin_functions.md) for API reference
- Consult [Quantization Primitives](quantization_primitives.md) for quantization help
- See [Type System](type_system.md) for type-related questions

## Version

This documentation corresponds to Triton DSL version 0.1.0.

Last updated: 2024

---

**Next Steps:**
- Start with the [Syntax Guide](syntax_guide.md) for a comprehensive overview
- Dive into [Quantization Primitives](quantization_primitives.md) for ternary networks
- Explore the [Built-in Functions Reference](builtin_functions.md) for available operations

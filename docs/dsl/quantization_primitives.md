# Triton DSL Quantization Primitives

This document provides comprehensive documentation of quantization primitives in Triton DSL, covering quantization methods, dequantization, quantization-aware operations, gradient handling, and custom quantizers for ternary neural networks.

## Table of Contents

- [Overview](#overview)
- [Quantization Methods](#quantization-methods)
- [Dequantization](#dequantization)
- [Quantization-Aware Operations](#quantization-aware-operations)
- [Gradient Handling](#gradient-handling)
- [Custom Quantizers](#custom-quantizers)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Quantization is the process of mapping continuous values to a discrete set of values. In Triton DSL, the primary focus is on **ternary quantization**, which maps floating-point values to the set {-1, 0, 1}.

### Why Ternary Quantization?

- **Memory Efficiency**: 2 bits per weight vs. 32 bits for float32 (16× reduction)
- **Computational Efficiency**: Ternary operations can use specialized hardware
- **Energy Efficiency**: Reduced data movement and simpler arithmetic
- **Model Compression**: Enables deployment on resource-constrained devices

### Quantization Challenges

1. **Information Loss**: Reducing precision loses information
2. **Gradient Flow**: Discrete functions have zero gradients almost everywhere
3. **Training Stability**: Quantized networks can be harder to optimize
4. **Accuracy Trade-off**: Balance between compression and performance

Triton DSL addresses these challenges through:
- Multiple quantization strategies
- Gradient approximation techniques (e.g., Straight-Through Estimator)
- Quantization-aware training primitives
- Flexible custom quantizer support

## Quantization Methods

### Threshold Quantization

The most basic ternary quantization method using fixed thresholds.

**Signature**:
```triton
fn threshold_quantize(
    x: Tensor<float32, [?, ?]>,
    threshold: float32 = 0.5,
    scale: float32 = 1.0
) -> TernaryTensor<trit, [?, ?]>
```

**Algorithm**:
```
quantize(x) = {
    trit(1)   if x > threshold * scale
    trit(-1)  if x < -threshold * scale
    trit(0)   otherwise
}
```

**Parameters**:
- `x`: Input tensor
- `threshold`: Quantization threshold (default: 0.5)
- `scale`: Global scaling factor

**Examples**:
```triton
# Basic threshold quantization
let weights = randn([784, 128])
let quantized = threshold_quantize(weights)

# Custom threshold
let quantized_tight = threshold_quantize(weights, threshold=0.3)
let quantized_wide = threshold_quantize(weights, threshold=0.7)

# With scaling
let scaled_weights = threshold_quantize(weights, threshold=0.5, scale=2.0)
```

**Characteristics**:
- ✓ Simple and fast
- ✓ Deterministic
- ✗ Fixed threshold may not adapt to distribution
- ✗ May produce unbalanced distributions

---

### Adaptive Threshold Quantization

Automatically determines threshold based on input statistics.

**Signature**:
```triton
fn adaptive_threshold_quantize(
    x: Tensor<float32, [?, ?]>,
    percentile: float32 = 0.7,
    target_sparsity: float32? = None
) -> TernaryTensor<trit, [?, ?]>
```

**Algorithm**:
```
threshold = percentile-th value of |x|
quantize(x) = threshold_quantize(x, threshold)
```

**Parameters**:
- `x`: Input tensor
- `percentile`: Percentile of absolute values to use as threshold (0-1)
- `target_sparsity`: Desired fraction of zeros (optional)

**Examples**:
```triton
let weights = randn([784, 128])

# 70th percentile threshold
let q1 = adaptive_threshold_quantize(weights, percentile=0.7)

# Target 30% zeros
let q2 = adaptive_threshold_quantize(weights, target_sparsity=0.3)

# Per-layer adaptation during training
fn adaptive_quantize_layer(weights: Tensor<float32, [?, ?]>) 
    -> TernaryTensor<trit, [?, ?]> {
    # Compute statistics
    let mean_abs = mean(abs(weights))
    let std_abs = std(abs(weights))
    
    # Use mean + 0.5*std as threshold
    let threshold = mean_abs + 0.5 * std_abs
    return threshold_quantize(weights, threshold)
}
```

**Characteristics**:
- ✓ Adapts to input distribution
- ✓ Can control sparsity
- ✗ Slightly more expensive
- ✗ Non-deterministic across different inputs

---

### Stochastic Quantization

Probabilistic quantization for training stability.

**Signature**:
```triton
fn stochastic_quantize(
    x: Tensor<float32, [?, ?]>,
    temperature: float32 = 1.0,
    training: bool = true
) -> TernaryTensor<trit, [?, ?]>
```

**Algorithm**:
```
p_positive = sigmoid((x - threshold) / temperature)
p_negative = sigmoid((-x - threshold) / temperature)
p_zero = 1 - p_positive - p_negative

Sample from distribution {-1: p_negative, 0: p_zero, 1: p_positive}
```

**Parameters**:
- `x`: Input tensor
- `temperature`: Controls sharpness of distribution (lower = more deterministic)
- `training`: Use stochastic mode (false = deterministic)

**Examples**:
```triton
let weights = randn([784, 128])

# Standard stochastic quantization
let q1 = stochastic_quantize(weights, training=true)

# Lower temperature (sharper distribution)
let q2 = stochastic_quantize(weights, temperature=0.5, training=true)

# Inference mode (deterministic)
let q3 = stochastic_quantize(weights, training=false)

# Training loop with annealing
fn train_with_annealing(weights: Tensor<float32, [?, ?]>, epoch: int32) 
    -> TernaryTensor<trit, [?, ?]> {
    # Anneal temperature over epochs
    let temp = max(0.1, 1.0 - epoch * 0.1)
    return stochastic_quantize(weights, temperature=temp)
}
```

**Characteristics**:
- ✓ Better gradient estimates during training
- ✓ Reduces quantization error variance
- ✓ Helps escape local minima
- ✗ Non-deterministic
- ✗ Requires careful temperature tuning

---

### Learned Quantization

Quantization with learnable thresholds and scales.

**Signature**:
```triton
fn learned_quantize(
    x: Tensor<float32, [?, ?]>,
    pos_threshold: Tensor<float32, []>,
    neg_threshold: Tensor<float32, []>,
    pos_scale: Tensor<float32, []> = 1.0,
    neg_scale: Tensor<float32, []> = 1.0
) -> TernaryTensor<trit, [?, ?]>
```

**Algorithm**:
```
quantize(x) = {
    trit(1) * pos_scale   if x > pos_threshold
    trit(-1) * neg_scale  if x < neg_threshold
    trit(0)               otherwise
}
```

**Parameters**:
- `x`: Input tensor
- `pos_threshold`: Learnable positive threshold
- `neg_threshold`: Learnable negative threshold
- `pos_scale`: Learnable positive scale
- `neg_scale`: Learnable negative scale

**Examples**:
```triton
# Define learnable parameters
let pos_threshold = tensor([0.5], requires_grad=true)
let neg_threshold = tensor([-0.5], requires_grad=true)
let pos_scale = tensor([1.0], requires_grad=true)
let neg_scale = tensor([1.0], requires_grad=true)

# Quantization with learned parameters
let weights = randn([784, 128])
let quantized = learned_quantize(
    weights, 
    pos_threshold, 
    neg_threshold,
    pos_scale,
    neg_scale
)

# Per-channel learned quantization
fn per_channel_learned_quantize(
    x: Tensor<float32, [out_channels, in_channels]>
) -> TernaryTensor<trit, [out_channels, in_channels]> {
    let pos_thresholds = tensor([0.5] * out_channels, requires_grad=true)
    let neg_thresholds = tensor([-0.5] * out_channels, requires_grad=true)
    
    # Apply per-channel thresholds
    let quantized = []
    for i in 0..out_channels {
        let channel = x[i, :]
        let q = learned_quantize(
            channel.unsqueeze(0),
            pos_thresholds[i],
            neg_thresholds[i]
        )
        quantized.append(q)
    }
    return stack(quantized)
}
```

**Characteristics**:
- ✓ Optimal thresholds learned during training
- ✓ Can adapt to layer-specific distributions
- ✓ Better accuracy than fixed thresholds
- ✗ Additional parameters to learn
- ✗ More complex training

---

### XNOR-Net Quantization

Binary weight and activation quantization with scaling factors.

**Signature**:
```triton
fn xnor_quantize(
    x: Tensor<float32, [?, ?]>,
    binary: bool = false
) -> TernaryTensor<trit, [?, ?]> | Tuple<TernaryTensor<trit, [?, ?]>, float32>
```

**Algorithm**:
```
alpha = mean(|x|)  # Scaling factor
quantized = sign(x)  # {-1, 1} or {-1, 0, 1}
```

**Parameters**:
- `x`: Input tensor
- `binary`: Use binary {-1, 1} instead of ternary

**Examples**:
```triton
let weights = randn([784, 128])

# XNOR-style quantization with scaling
let (quantized, scale) = xnor_quantize(weights)

# Apply in forward pass
fn xnor_forward(input: Tensor<float32, [?, ?]>, 
                weights: Tensor<float32, [?, ?]>) 
    -> Tensor<float32, [?, ?]> {
    let (w_quantized, w_scale) = xnor_quantize(weights)
    let (i_quantized, i_scale) = xnor_quantize(input)
    
    # Scaled computation
    return (input @ w_quantized) * w_scale * i_scale
}
```

**Characteristics**:
- ✓ Proven method from XNOR-Net paper
- ✓ Includes proper scaling
- ✓ Works well for both weights and activations
- ✗ Binary mode less expressive than ternary

## Dequantization

Converting ternary values back to floating-point representation.

### Basic Dequantization

**Signature**:
```triton
fn dequantize(
    x: TernaryTensor<trit, [?, ?]>,
    scale: float32 = 1.0,
    dtype: Type = float32
) -> Tensor<dtype, [?, ?]>
```

**Examples**:
```triton
let ternary = ternary_quantize(randn([10, 10]))

# Basic dequantization
let float_tensor = dequantize(ternary)

# With scaling
let scaled = dequantize(ternary, scale=2.0)

# Different dtype
let fp16 = dequantize(ternary, dtype=float16)
```

---

### Affine Dequantization

Dequantization with learned scale and offset.

**Signature**:
```triton
fn affine_dequantize(
    x: TernaryTensor<trit, [?, ?]>,
    scale: Tensor<float32, []> | Tensor<float32, [?]>,
    zero_point: Tensor<float32, []> | Tensor<float32, [?]> = 0.0
) -> Tensor<float32, [?, ?]>
```

**Formula**: `output = scale * x + zero_point`

**Examples**:
```triton
let ternary = ternary_quantize(randn([128, 256]))

# Per-tensor scale
let global_scale = tensor([1.5])
let result1 = affine_dequantize(ternary, global_scale)

# Per-channel scale
let channel_scales = randn([128])
let result2 = affine_dequantize(ternary, channel_scales)

# With zero point
let zero_point = tensor([0.5])
let result3 = affine_dequantize(ternary, global_scale, zero_point)
```

## Quantization-Aware Operations

Operations that integrate quantization into the computation graph.

### Quantized Linear Layer

**Signature**:
```triton
fn quantized_linear(
    input: Tensor<float32, [batch, in_features]>,
    weight: Tensor<float32, [in_features, out_features]> | TernaryTensor<trit, [in_features, out_features]>,
    bias: Tensor<float32, [out_features]>? = None,
    quantize_weights: bool = true,
    quantize_activations: bool = false
) -> Tensor<float32, [batch, out_features]>
```

**Examples**:
```triton
let input = randn([32, 784])
let weights = randn([784, 128])
let bias = zeros([128])

# Quantize weights only
let output1 = quantized_linear(input, weights, bias, 
                               quantize_weights=true, 
                               quantize_activations=false)

# Quantize both weights and activations
let output2 = quantized_linear(input, weights, bias,
                               quantize_weights=true,
                               quantize_activations=true)

# Pre-quantized weights
let q_weights = ternary_quantize(weights)
let output3 = quantized_linear(input, q_weights, bias)
```

---

### Quantized Convolution

**Signature**:
```triton
fn quantized_conv2d(
    input: Tensor<float32, [batch, in_ch, h, w]>,
    weight: Tensor<float32, [out_ch, in_ch, kh, kw]>,
    bias: Tensor<float32, [out_ch]>? = None,
    stride: int32 | [int32, int32] = 1,
    padding: int32 | [int32, int32] = 0,
    quantize_weights: bool = true,
    quantize_activations: bool = false
) -> Tensor<float32, [batch, out_ch, h', w']>
```

**Examples**:
```triton
let input = randn([1, 3, 224, 224])
let kernel = randn([64, 3, 7, 7])
let bias = zeros([64])

# Quantized convolution
let output = quantized_conv2d(
    input, kernel, bias,
    stride=2, padding=3,
    quantize_weights=true,
    quantize_activations=false
)
```

---

### Quantized Matrix Multiplication

**Signature**:
```triton
fn quantized_matmul(
    a: Tensor<float32, [m, k]>,
    b: Tensor<float32, [k, n]>,
    quantize_a: bool = false,
    quantize_b: bool = true,
    use_ste: bool = true
) -> Tensor<float32, [m, n]>
```

**Examples**:
```triton
let a = randn([128, 256])
let b = randn([256, 512])

# Quantize second matrix (typical for weights)
let c = quantized_matmul(a, b, quantize_b=true)

# Quantize both matrices
let d = quantized_matmul(a, b, quantize_a=true, quantize_b=true)
```

## Gradient Handling

Techniques for handling gradients through discrete quantization functions.

### Straight-Through Estimator (STE)

The most common gradient approximation for quantization.

**Signature**:
```triton
fn ste_quantize(
    x: Tensor<float32, [?, ?]>,
    quantize_fn: fn(Tensor<float32, [?, ?]>) -> TernaryTensor<trit, [?, ?]> = ternary_quantize
) -> Tensor<float32, [?, ?]>
```

**Gradient Behavior**:
```
Forward:  y = quantize(x)
Backward: ∂L/∂x = ∂L/∂y  (gradient passes through unchanged)
```

**Examples**:
```triton
# Basic STE quantization
let x = randn([784, 128], requires_grad=true)
let quantized = ste_quantize(x)

# Custom quantization function with STE
fn ste_custom_quantize(x: Tensor<float32, [?, ?]>) 
    -> Tensor<float32, [?, ?]> {
    let forward = threshold_quantize(x, threshold=0.7)
    let backward = x  # Gradient uses original input
    return stop_gradient(forward) + (x - stop_gradient(x))
}

# In training loop
fn training_step(weights: Tensor<float32, [?, ?]>, 
                 input: Tensor<float32, [?, ?]>,
                 target: Tensor<float32, [?, ?]>) 
    -> float32 {
    # Forward: quantize weights
    let q_weights = ste_quantize(weights)
    
    # Compute output
    let output = input @ q_weights
    
    # Loss
    let loss = mse_loss(output, target)
    
    # Backward: gradients flow through as if quantization was identity
    return loss
}
```

**Characteristics**:
- ✓ Simple and widely used
- ✓ Empirically effective
- ✗ Biased gradient estimator
- ✗ Can lead to gradient mismatch

---

### Clipped Straight-Through Estimator

STE with gradient clipping for values outside quantization range.

**Signature**:
```triton
fn clipped_ste_quantize(
    x: Tensor<float32, [?, ?]>,
    threshold: float32 = 1.0
) -> Tensor<float32, [?, ?]>
```

**Gradient Behavior**:
```
Forward:  y = quantize(x)
Backward: ∂L/∂x = {
    ∂L/∂y  if |x| <= threshold
    0      if |x| > threshold
}
```

**Examples**:
```triton
let x = randn([784, 128])

# Clipped STE - no gradient for large values
let quantized = clipped_ste_quantize(x, threshold=1.0)

# Manual implementation
fn manual_clipped_ste(x: Tensor<float32, [?, ?]>) 
    -> Tensor<float32, [?, ?]> {
    let forward = ternary_quantize(x)
    let mask = abs(x) <= 1.0
    let backward = x * mask
    return stop_gradient(forward) + (backward - stop_gradient(backward))
}
```

**Characteristics**:
- ✓ Prevents unbounded gradients
- ✓ Better than vanilla STE for extreme values
- ✗ Zero gradient for large values (vanishing gradient)

---

### Soft Quantization

Differentiable approximation to hard quantization.

**Signature**:
```triton
fn soft_quantize(
    x: Tensor<float32, [?, ?]>,
    temperature: float32 = 1.0,
    hard: bool = false
) -> Tensor<float32, [?, ?]>
```

**Formula**:
```
soft_quant(x) = tanh(x / temperature)
# As temperature → 0, converges to sign(x)
```

**Examples**:
```triton
let x = randn([784, 128])

# Soft quantization (differentiable)
let soft = soft_quantize(x, temperature=1.0)

# Sharper approximation
let sharper = soft_quantize(x, temperature=0.1)

# Hard quantization in forward, soft in backward
let hard_soft = soft_quantize(x, temperature=0.1, hard=true)

# Temperature annealing schedule
fn anneal_quantization(x: Tensor<float32, [?, ?]>, epoch: int32) 
    -> Tensor<float32, [?, ?]> {
    let temp = max(0.01, 1.0 * 0.9 ** epoch)
    return soft_quantize(x, temperature=temp)
}
```

**Characteristics**:
- ✓ Fully differentiable
- ✓ Gradients always defined
- ✓ Can anneal to hard quantization
- ✗ Not true ternary values
- ✗ Requires careful temperature tuning

---

### Gumbel-Softmax Quantization

Stochastic differentiable quantization using Gumbel-Softmax trick.

**Signature**:
```triton
fn gumbel_softmax_quantize(
    x: Tensor<float32, [?, ?]>,
    temperature: float32 = 1.0,
    hard: bool = false
) -> Tensor<float32, [?, ?]>
```

**Algorithm**:
```
# Compute logits for {-1, 0, 1}
logits = [neg_score, zero_score, pos_score]

# Add Gumbel noise
gumbel = -log(-log(uniform(0, 1)))
noisy_logits = (logits + gumbel) / temperature

# Softmax
probs = softmax(noisy_logits)

# Output
output = -1 * probs[0] + 0 * probs[1] + 1 * probs[2]
```

**Examples**:
```triton
let x = randn([784, 128])

# Gumbel-Softmax quantization
let q1 = gumbel_softmax_quantize(x, temperature=1.0)

# Hard mode (forward hard, backward soft)
let q2 = gumbel_softmax_quantize(x, temperature=0.5, hard=true)

# Training with Gumbel-Softmax
fn train_with_gumbel(weights: Tensor<float32, [?, ?]>, epoch: int32) 
    -> Tensor<float32, [?, ?]> {
    # Anneal temperature
    let temp = max(0.5, 5.0 * 0.95 ** epoch)
    
    # Use hard mode after warmup
    let hard = epoch > 10
    
    return gumbel_softmax_quantize(weights, temperature=temp, hard=hard)
}
```

**Characteristics**:
- ✓ Unbiased gradient estimator
- ✓ Better than STE theoretically
- ✓ Handles discrete distributions naturally
- ✗ More complex
- ✗ Requires temperature tuning
- ✗ Stochastic (adds noise)

## Custom Quantizers

Building custom quantization schemes for specific use cases.

### Custom Quantizer Interface

```triton
trait Quantizer {
    fn quantize(self, x: Tensor<float32, [?, ?]>) -> TernaryTensor<trit, [?, ?]>
    fn dequantize(self, x: TernaryTensor<trit, [?, ?]>) -> Tensor<float32, [?, ?]>
    fn parameters(self) -> [Tensor<float32, ?>, ...]
}
```

### Example: Percentile Quantizer

```triton
struct PercentileQuantizer {
    percentile: Tensor<float32, []>
    
    fn quantize(self, x: Tensor<float32, [?, ?]>) -> TernaryTensor<trit, [?, ?]> {
        # Compute threshold from percentile
        let threshold = percentile_value(abs(x), self.percentile)
        return threshold_quantize(x, threshold)
    }
    
    fn dequantize(self, x: TernaryTensor<trit, [?, ?]>) -> Tensor<float32, [?, ?]> {
        return dequantize(x, scale=1.0)
    }
    
    fn parameters(self) -> [Tensor<float32, ?>, ...] {
        return [self.percentile]
    }
}

# Usage
let quantizer = PercentileQuantizer(percentile=tensor([0.7], requires_grad=true))
let weights = randn([784, 128])
let quantized = quantizer.quantize(weights)
```

### Example: Layer-Wise Adaptive Quantizer

```triton
struct LayerAdaptiveQuantizer {
    pos_threshold: Tensor<float32, [num_layers]>
    neg_threshold: Tensor<float32, [num_layers]>
    scales: Tensor<float32, [num_layers]>
    
    fn quantize(self, x: Tensor<float32, [?, ?]>, layer_idx: int32) 
        -> TernaryTensor<trit, [?, ?]> {
        let pos_thresh = self.pos_threshold[layer_idx]
        let neg_thresh = self.neg_threshold[layer_idx]
        let scale = self.scales[layer_idx]
        
        return learned_quantize(x * scale, pos_thresh, neg_thresh)
    }
    
    fn dequantize(self, x: TernaryTensor<trit, [?, ?]>, layer_idx: int32) 
        -> Tensor<float32, [?, ?]> {
        let scale = self.scales[layer_idx]
        return dequantize(x, scale)
    }
    
    fn parameters(self) -> [Tensor<float32, ?>, ...] {
        return [self.pos_threshold, self.neg_threshold, self.scales]
    }
}
```

### Example: Mixed-Precision Quantizer

```triton
struct MixedPrecisionQuantizer {
    # Some layers binary, some ternary
    layer_modes: [str, ...]  # ["binary", "ternary", "ternary", ...]
    
    fn quantize(self, x: Tensor<float32, [?, ?]>, layer_idx: int32) 
        -> TernaryTensor<trit, [?, ?]> {
        let mode = self.layer_modes[layer_idx]
        
        if mode == "binary" {
            # Binary quantization {-1, 1}
            return sign(x)
        } else if mode == "ternary" {
            # Ternary quantization {-1, 0, 1}
            return ternary_quantize(x)
        } else {
            panic(f"Unknown mode: {mode}")
        }
    }
}
```

## Advanced Techniques

### Progressive Quantization

Gradually introduce quantization during training.

```triton
fn progressive_quantize(
    x: Tensor<float32, [?, ?]>,
    epoch: int32,
    warmup_epochs: int32 = 10
) -> Tensor<float32, [?, ?]> {
    if epoch < warmup_epochs {
        # Gradually blend from full precision to quantized
        let alpha = epoch / warmup_epochs
        let quantized = ste_quantize(x)
        return (1 - alpha) * x + alpha * quantized
    } else {
        # Full quantization after warmup
        return ste_quantize(x)
    }
}

# Usage in training
fn train_epoch(model: Model, epoch: int32) {
    for layer in model.layers {
        layer.weights = progressive_quantize(layer.weights, epoch)
    }
}
```

### Differentiable Sparsity

Control sparsity (number of zeros) through differentiable soft thresholding.

```triton
fn differentiable_sparse_quantize(
    x: Tensor<float32, [?, ?]>,
    target_sparsity: float32,
    temperature: float32 = 1.0
) -> Tensor<float32, [?, ?]> {
    # Soft threshold that encourages sparsity
    let threshold = compute_threshold_for_sparsity(x, target_sparsity)
    
    # Soft quantization with sparsity bias
    let zero_logit = -abs(x - threshold) / temperature
    let pos_logit = (x - threshold) / temperature
    let neg_logit = (-x - threshold) / temperature
    
    let logits = stack([neg_logit, zero_logit, pos_logit], dim=-1)
    let probs = softmax(logits, dim=-1)
    
    # Weighted sum: -1 * p[0] + 0 * p[1] + 1 * p[2]
    return -probs[:, :, 0] + probs[:, :, 2]
}
```

### Quantization-Aware Batch Normalization

Batch normalization integrated with quantization.

```triton
fn quantized_batch_norm(
    x: Tensor<float32, [batch, channels, h, w]>,
    gamma: Tensor<float32, [channels]>,
    beta: Tensor<float32, [channels]>,
    running_mean: Tensor<float32, [channels]>,
    running_var: Tensor<float32, [channels]>,
    training: bool = true
) -> Tensor<float32, [batch, channels, h, w]> {
    # Normalize
    let normalized = batch_norm(x, gamma, beta, running_mean, running_var, 
                                 training=training)
    
    # Quantize after normalization
    return ste_quantize(normalized)
}
```

## Best Practices

### 1. Start with Simple Methods

```triton
# Begin with basic threshold quantization
let quantized = threshold_quantize(weights)

# If accuracy is insufficient, try adaptive methods
let quantized = adaptive_threshold_quantize(weights, percentile=0.7)

# Only move to complex methods if necessary
let quantized = learned_quantize(weights, learned_params...)
```

### 2. Use STE for Most Cases

```triton
# STE is simple and effective for most networks
fn quantize_layer(weights: Tensor<float32, [?, ?]>) 
    -> Tensor<float32, [?, ?]> {
    return ste_quantize(ternary_quantize(weights))
}
```

### 3. Monitor Gradient Flow

```triton
fn check_gradients(x: Tensor<float32, [?, ?]>) {
    x.register_hook(fn(grad) {
        print(f"Gradient mean: {mean(grad)}")
        print(f"Gradient std: {std(grad)}")
        print(f"Fraction zero: {sum(grad == 0) / grad.numel()}")
        
        # Warn if too many gradients are zero
        if sum(grad == 0) / grad.numel() > 0.5 {
            print("WARNING: >50% of gradients are zero")
        }
    })
}
```

### 4. Use Warmup for Stable Training

```triton
fn train_with_warmup(weights: Tensor<float32, [?, ?]>, 
                     epoch: int32,
                     warmup: int32 = 5) 
    -> Tensor<float32, [?, ?]> {
    if epoch < warmup {
        # Soft quantization during warmup
        return soft_quantize(weights, temperature=1.0)
    } else {
        # Hard quantization with STE after warmup
        return ste_quantize(ternary_quantize(weights))
    }
}
```

### 5. Quantize Layers Progressively

```triton
# Quantize earlier layers first, later layers last
fn progressive_layer_quantization(model: Model, epoch: int32) {
    let num_layers = length(model.layers)
    
    for (idx, layer) in enumerate(model.layers) {
        # Earlier layers quantized sooner
        let layer_epoch = max(0, epoch - idx * 2)
        layer.weights = progressive_quantize(layer.weights, layer_epoch)
    }
}
```

### 6. Validate Quantization Quality

```triton
fn validate_quantization(
    original: Tensor<float32, [?, ?]>,
    quantized: TernaryTensor<trit, [?, ?]>
) {
    let dequantized = dequantize(quantized)
    let mse = mean((original - dequantized) ** 2)
    let mae = mean(abs(original - dequantized))
    
    print(f"Quantization MSE: {mse}")
    print(f"Quantization MAE: {mae}")
    
    # Check distribution
    let num_neg = sum(quantized == -1)
    let num_zero = sum(quantized == 0)
    let num_pos = sum(quantized == 1)
    let total = num_neg + num_zero + num_pos
    
    print(f"Distribution: {num_neg/total:.2%} / {num_zero/total:.2%} / {num_pos/total:.2%}")
}
```

This comprehensive guide covers all major quantization primitives and techniques available in Triton DSL for building efficient ternary neural networks.

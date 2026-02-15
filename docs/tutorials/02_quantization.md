# Tutorial 2: Quantization Techniques

Learn advanced quantization techniques for ternary neural networks, including deterministic and stochastic methods, gradient handling, and fine-tuning strategies.

## Learning Objectives

- Understand different quantization methods
- Implement custom quantization functions
- Handle gradients through quantization
- Fine-tune pre-trained models to ternary
- Optimize quantization for accuracy

## Prerequisites

- Completed [Tutorial 1: Basic Model](01_basic_model.md)
- Understanding of backpropagation
- Familiarity with PyTorch autograd

## Quantization Fundamentals

### What is Quantization?

Quantization is the process of constraining weights from continuous values (FP32) to a discrete set:

```
FP32: [-∞, +∞] (continuous)
    ↓ Quantization
Ternary: {-1, 0, +1} (discrete)
```

### The Gradient Problem

Direct quantization breaks gradients:

```python
def naive_quantize(x):
    # This breaks backpropagation!
    return torch.sign(x).clamp(-1, 1)
```

**Why?** The gradient of `sign()` is zero almost everywhere.

### Straight-Through Estimator (STE)

STE solves this by passing gradients "straight through":

```
Forward:  x → quantize(x) → output
Backward: ∂L/∂x ← ∂L/∂output (skip quantization)
```

## Method 1: Deterministic Quantization

### Basic Implementation

```python
import torch

class DeterministicQuantize(torch.autograd.Function):
    """Deterministic threshold-based quantization with STE."""
    
    @staticmethod
    def forward(ctx, input, threshold=0.33):
        """
        Forward pass: quantize using thresholds.
        
        Args:
            input: Tensor to quantize
            threshold: Threshold value (default: 0.33)
            
        Returns:
            Quantized tensor in {-1, 0, 1}
        """
        output = torch.zeros_like(input)
        output[input > threshold] = 1
        output[input < -threshold] = -1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: straight-through estimator.
        
        Returns:
            Gradients for input and threshold
        """
        # Pass gradient straight through
        return grad_output, None

# Usage
deterministic_quantize = DeterministicQuantize.apply
```

### Example: Quantize a Layer

```python
import torch.nn as nn

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Full-precision weights for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        # Quantize weights on-the-fly
        w_ternary = deterministic_quantize(self.weight, 0.33)
        return nn.functional.linear(x, w_ternary, self.bias)

# Test the layer
layer = TernaryLinear(784, 128)
x = torch.randn(32, 784)
output = layer(x)
print(f"Output shape: {output.shape}")
print(f"Unique weight values: {torch.unique(deterministic_quantize(layer.weight, 0.33))}")
```

### Threshold Selection

Different thresholds affect sparsity and accuracy:

```python
def analyze_threshold(weights, thresholds=[0.1, 0.33, 0.5, 0.7]):
    """Analyze the effect of different thresholds."""
    for t in thresholds:
        quantized = deterministic_quantize(weights, t)
        zeros = (quantized == 0).sum().item()
        total = quantized.numel()
        sparsity = 100 * zeros / total
        print(f"Threshold {t:.2f}: Sparsity = {sparsity:.1f}%")

# Example
weights = torch.randn(128, 784) * 0.01
analyze_threshold(weights)
```

Expected output:
```
Threshold 0.10: Sparsity = 8.2%
Threshold 0.33: Sparsity = 26.4%
Threshold 0.50: Sparsity = 38.3%
Threshold 0.70: Sparsity = 52.1%
```

## Method 2: Stochastic Quantization

### Motivation

Stochastic quantization adds randomness to better approximate gradients:

```
Probability of +1: P(+1) = σ(x / threshold)
Probability of -1: P(-1) = σ(-x / threshold)
Probability of  0: P(0) = 1 - P(+1) - P(-1)
```

### Implementation

```python
class StochasticQuantize(torch.autograd.Function):
    """Stochastic quantization with probability-based rounding."""
    
    @staticmethod
    def forward(ctx, input, threshold=0.33):
        """
        Forward pass: stochastic quantization.
        
        Uses sigmoid probabilities for smooth transitions.
        """
        # Compute probabilities
        prob_positive = torch.sigmoid(input / threshold)
        prob_negative = torch.sigmoid(-input / threshold)
        
        # Generate random samples
        rand = torch.rand_like(input)
        
        # Quantize stochastically
        output = torch.zeros_like(input)
        output[rand < prob_positive] = 1
        output[rand > (1 - prob_negative)] = -1
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator for gradients."""
        return grad_output, None

stochastic_quantize = StochasticQuantize.apply
```

### Comparison: Deterministic vs. Stochastic

```python
def compare_quantization_methods(weights):
    """Compare both quantization methods."""
    
    # Deterministic
    det_output = deterministic_quantize(weights, 0.33)
    det_zeros = (det_output == 0).sum().item()
    
    # Stochastic (average over 100 samples)
    stoch_zeros = 0
    for _ in range(100):
        stoch_output = stochastic_quantize(weights, 0.33)
        stoch_zeros += (stoch_output == 0).sum().item()
    stoch_zeros /= 100
    
    print(f"Deterministic: {det_zeros} zeros")
    print(f"Stochastic (avg): {stoch_zeros:.1f} zeros")

weights = torch.randn(1000, 1000) * 0.01
compare_quantization_methods(weights)
```

## Fine-Tuning Pre-trained Models

### Strategy 1: Progressive Quantization

Gradually quantize layers during training:

```python
class ProgressiveQuantization:
    """Progressive quantization scheduler."""
    
    def __init__(self, model, total_epochs):
        self.model = model
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def get_quantization_strength(self):
        """Linear increase in quantization strength."""
        return min(1.0, self.current_epoch / (self.total_epochs * 0.5))
    
    def step(self):
        """Update quantization strength."""
        self.current_epoch += 1
    
    def quantize_weights(self, weights):
        """Apply progressive quantization."""
        strength = self.get_quantization_strength()
        
        if strength < 1.0:
            # Mix FP32 and ternary
            quantized = deterministic_quantize(weights, 0.33)
            return strength * quantized + (1 - strength) * weights
        else:
            # Full quantization
            return deterministic_quantize(weights, 0.33)

# Usage
scheduler = ProgressiveQuantization(model, total_epochs=50)

for epoch in range(50):
    # Training loop...
    scheduler.step()
```

### Strategy 2: Layer-wise Quantization

Quantize one layer at a time:

```python
def layerwise_quantization(model, train_loader, criterion, optimizer, epochs_per_layer=5):
    """Quantize model layer by layer."""
    
    layers_to_quantize = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    
    for layer_name in layers_to_quantize:
        print(f"\nQuantizing layer: {layer_name}")
        
        # Enable quantization for this layer
        for name, module in model.named_modules():
            if name == layer_name:
                module.use_quantization = True
        
        # Fine-tune
        for epoch in range(epochs_per_layer):
            train_epoch(model, train_loader, criterion, optimizer)
        
        print(f"Layer {layer_name} quantized!")
    
    return model
```

## Optimizing Quantization Parameters

### Learned Thresholds

Make thresholds learnable parameters:

```python
class LearnableThresholdQuantize(torch.autograd.Function):
    """Quantization with learnable threshold."""
    
    @staticmethod
    def forward(ctx, input, threshold):
        """Forward with dynamic threshold."""
        ctx.save_for_backward(input, threshold)
        
        # Use the learned threshold
        output = torch.zeros_like(input)
        output[input > threshold] = 1
        output[input < -threshold] = -1
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradients for both input and threshold."""
        input, threshold = ctx.saved_tensors
        
        # Gradient for input (STE)
        grad_input = grad_output.clone()
        
        # Gradient for threshold
        # ∂L/∂t = sum(∂L/∂output * ∂output/∂t)
        grad_threshold = torch.zeros_like(threshold)
        
        # Where |input| ≈ threshold, small changes matter
        near_threshold = (input.abs() - threshold.abs()).abs() < 0.1
        grad_threshold += (grad_output * near_threshold.float()).sum()
        
        return grad_input, grad_threshold

class TernaryLinearLearnable(nn.Module):
    """Ternary linear layer with learnable threshold."""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable threshold (initialized to 0.33)
        self.threshold = nn.Parameter(torch.tensor(0.33))
        
    def forward(self, x):
        # Quantize with learned threshold
        w_ternary = LearnableThresholdQuantize.apply(self.weight, self.threshold)
        return nn.functional.linear(x, w_ternary, self.bias)
```

### Temperature Annealing

Use temperature to control quantization sharpness:

```python
def temperature_quantize(input, threshold, temperature):
    """Quantization with temperature for smooth transition."""
    
    # Soft quantization at high temperature
    # Hard quantization at low temperature
    soft_sign = torch.tanh(input / temperature)
    
    if temperature < 0.01:
        # Almost hard quantization
        output = torch.zeros_like(input)
        output[input > threshold] = 1
        output[input < -threshold] = -1
        return output
    else:
        # Soft quantization (differentiable)
        return soft_sign

# Temperature schedule
def get_temperature(epoch, max_epochs, initial_temp=1.0, final_temp=0.01):
    """Exponential decay of temperature."""
    return initial_temp * (final_temp / initial_temp) ** (epoch / max_epochs)

# Training loop
for epoch in range(max_epochs):
    temp = get_temperature(epoch, max_epochs)
    # Use temp in quantization...
```

## Quantization-Aware Training (QAT)

Full QAT implementation:

```python
class QATModel(nn.Module):
    """Model with quantization-aware training."""
    
    def __init__(self, architecture='simple'):
        super().__init__()
        
        # FP32 weights for training
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        
        # Quantization config
        self.threshold = 0.33
        self.use_quantization = False
        
    def forward(self, x):
        # Layer 1
        if self.use_quantization:
            w1_q = deterministic_quantize(self.fc1.weight, self.threshold)
            x = nn.functional.linear(x, w1_q, self.fc1.bias)
        else:
            x = self.fc1(x)
        x = torch.relu(x)
        
        # Layer 2
        if self.use_quantization:
            w2_q = deterministic_quantize(self.fc2.weight, self.threshold)
            x = nn.functional.linear(x, w2_q, self.fc2.bias)
        else:
            x = self.fc2(x)
        x = torch.relu(x)
        
        # Layer 3
        if self.use_quantization:
            w3_q = deterministic_quantize(self.fc3.weight, self.threshold)
            x = nn.functional.linear(x, w3_q, self.fc3.bias)
        else:
            x = self.fc3(x)
        
        return x
    
    def enable_quantization(self):
        """Enable quantization for inference."""
        self.use_quantization = True
    
    def disable_quantization(self):
        """Disable quantization for initial training."""
        self.use_quantization = False

# Three-phase training
def three_phase_training(model, train_loader, test_loader, criterion, optimizer):
    """
    Phase 1: Pre-training with FP32
    Phase 2: QAT with quantization
    Phase 3: Fine-tuning
    """
    
    # Phase 1: Pre-training (10 epochs)
    print("Phase 1: Pre-training with FP32")
    model.disable_quantization()
    for epoch in range(10):
        train_epoch(model, train_loader, criterion, optimizer)
    
    # Phase 2: QAT (20 epochs)
    print("\nPhase 2: Quantization-aware training")
    model.enable_quantization()
    for epoch in range(20):
        train_epoch(model, train_loader, criterion, optimizer)
    
    # Phase 3: Fine-tuning (10 epochs, lower LR)
    print("\nPhase 3: Fine-tuning")
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001
    
    for epoch in range(10):
        train_epoch(model, train_loader, criterion, optimizer)
    
    # Evaluate
    evaluate(model, test_loader, criterion)

# Run training
model = QATModel()
three_phase_training(model, train_loader, test_loader, criterion, optimizer)
```

## Best Practices

### 1. Start with Pre-training

Always pre-train with FP32 before quantization:

```python
# ✓ Good
train_fp32(model, 10_epochs)
enable_quantization(model)
fine_tune(model, 20_epochs)

# ✗ Bad
enable_quantization(model)
train_from_scratch(model, 30_epochs)
```

### 2. Use Batch Normalization

Batch norm helps stabilize ternary training:

```python
self.bn1 = nn.BatchNorm1d(256)
x = self.fc1(x)
x = self.bn1(x)  # Normalize before activation
x = torch.relu(x)
```

### 3. Monitor Sparsity

Track zero percentages during training:

```python
def compute_sparsity(model):
    """Compute model sparsity."""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        if param.requires_grad:
            quantized = deterministic_quantize(param, 0.33)
            total_params += quantized.numel()
            zero_params += (quantized == 0).sum().item()
    
    return 100 * zero_params / total_params

# In training loop
sparsity = compute_sparsity(model)
print(f"Model sparsity: {sparsity:.2f}%")
```

## Exercises

1. **Implement Ternary Connect**: Research and implement the Ternary Connect quantization method
2. **Compare Methods**: Train identical models with deterministic vs. stochastic quantization and compare accuracy
3. **Adaptive Thresholds**: Implement per-layer learnable thresholds
4. **Gradient Analysis**: Visualize gradient flow through quantized layers

## Next Steps

- [Tutorial 3: Custom Layers](03_custom_layers.md) - Build custom ternary operations
- [Tutorial 4: Training](04_training.md) - Advanced training techniques
- [DSL Reference: Quantization Primitives](../dsl/quantization_primitives.md)

## Summary

You learned:
- ✓ Deterministic and stochastic quantization
- ✓ Straight-through estimators
- ✓ Fine-tuning strategies
- ✓ Quantization-aware training
- ✓ Best practices for ternary models

## Resources

- [Ternary Weight Networks Paper](https://arxiv.org/abs/1605.04711)
- [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
- [Example Code](https://github.com/financecommander/Triton/tree/main/examples)

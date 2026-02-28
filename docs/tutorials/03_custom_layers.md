# Tutorial 3: Custom Layers

Learn how to build custom ternary layers using Triton DSL, including convolutional layers, attention mechanisms, and specialized operations.

## Learning Objectives

- Define custom ternary operations in DSL
- Implement convolutional layers with ternary weights
- Build attention mechanisms for transformers
- Create custom activation functions
- Optimize custom layers for performance

## Prerequisites

- Completed [Tutorial 1](01_basic_model.md) and [Tutorial 2](02_quantization.md)
- Understanding of CNNs and attention mechanisms
- Basic DSL syntax knowledge

## Custom Layer Basics

### Layer Definition Structure

```triton
layer LayerName(
    param1: Type1,
    param2: Type2,
    input: InputType
) -> OutputType {
    # Layer logic here
    let result = operation(input, param1, param2)
    return result
}
```

### Type System Review

```triton
# Ternary types
TernaryTensor<trit, [batch, features]>

# Float types
Tensor<float32, [batch, features]>
Tensor<float16, [batch, features]>

# Shape wildcards
Tensor<float32, [?, features]>  # Dynamic batch size
```

## Custom Convolutional Layer

### 2D Convolution

```triton
layer TernaryConv2D(
    weights: TernaryTensor<trit, [out_channels, in_channels, kernel_h, kernel_w]>,
    bias: TernaryTensor<trit, [out_channels]>,
    x: Tensor<float32, [?, in_channels, height, width]>,
    stride: int32 = 1,
    padding: int32 = 0
) -> Tensor<float32, [?, out_channels, out_h, out_w]> {
    # Convolution operation
    let conv_out = conv2d(x, weights, stride=stride, padding=padding)
    
    # Add bias (broadcast)
    let result = conv_out + bias
    
    return result
}
```

### Depthwise Separable Convolution

```triton
layer TernaryDepthwiseSeparable(
    dw_weights: TernaryTensor<trit, [in_channels, 1, 3, 3]>,
    pw_weights: TernaryTensor<trit, [out_channels, in_channels, 1, 1]>,
    dw_bias: TernaryTensor<trit, [in_channels]>,
    pw_bias: TernaryTensor<trit, [out_channels]>,
    x: Tensor<float32, [?, in_channels, h, w]>
) -> Tensor<float32, [?, out_channels, h, w]> {
    # Depthwise convolution
    let dw_out = depthwise_conv2d(x, dw_weights, padding=1)
    let dw_result = dw_out + dw_bias
    let dw_act = relu(dw_result)
    
    # Pointwise convolution
    let pw_out = conv2d(dw_act, pw_weights)
    let pw_result = pw_out + pw_bias
    
    return pw_result
}
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
from backend.pytorch.ops.quantize import quantize

class TernaryConv2d(nn.Module):
    """Custom ternary 2D convolution layer."""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x):
        # Quantize weights
        w_ternary = quantize(self.weight, method='deterministic')
        
        # Perform convolution
        return nn.functional.conv2d(
            x, w_ternary, self.bias, 
            stride=self.stride, padding=self.padding
        )

# Example usage
conv = TernaryConv2d(3, 64, kernel_size=3, padding=1)
x = torch.randn(4, 3, 32, 32)
output = conv(x)
print(f"Output shape: {output.shape}")  # [4, 64, 32, 32]
```

## Attention Mechanisms

### Self-Attention with Ternary Weights

```triton
layer TernarySelfAttention(
    query_weights: TernaryTensor<trit, [d_model, d_model]>,
    key_weights: TernaryTensor<trit, [d_model, d_model]>,
    value_weights: TernaryTensor<trit, [d_model, d_model]>,
    output_weights: TernaryTensor<trit, [d_model, d_model]>,
    x: Tensor<float32, [?, seq_len, d_model]>
) -> Tensor<float32, [?, seq_len, d_model]> {
    # Linear projections
    let Q = x @ query_weights
    let K = x @ key_weights
    let V = x @ value_weights
    
    # Attention scores
    let scores = Q @ transpose(K) / sqrt(d_model)
    let attention = softmax(scores, dim=-1)
    
    # Apply attention to values
    let attended = attention @ V
    
    # Output projection
    let output = attended @ output_weights
    
    return output
}
```

### PyTorch Implementation

```python
class TernaryAttention(nn.Module):
    """Ternary multi-head self-attention."""
    
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Ternary projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Quantize projection weights
        q_w = quantize(self.q_proj.weight, method='deterministic')
        k_w = quantize(self.k_proj.weight, method='deterministic')
        v_w = quantize(self.v_proj.weight, method='deterministic')
        
        # Linear projections with ternary weights
        Q = nn.functional.linear(x, q_w, self.q_proj.bias)
        K = nn.functional.linear(x, k_w, self.k_proj.bias)
        V = nn.functional.linear(x, v_w, self.v_proj.bias)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        out_w = quantize(self.out_proj.weight, method='deterministic')
        output = nn.functional.linear(attn_output, out_w, self.out_proj.bias)
        
        return output

# Example
attention = TernaryAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # [batch, seq_len, d_model]
output = attention(x)
print(f"Output shape: {output.shape}")  # [2, 10, 512]
```

## Custom Activation Functions

### Ternary Activation

```triton
fn ternary_activation(x: Tensor<float32, [?]>) -> Tensor<float32, [?]> {
    # Custom ternary activation
    let mask_pos = x > 0.5
    let mask_neg = x < -0.5
    let result = mask_pos * 1.0 + mask_neg * (-1.0)
    return result
}

layer TernaryActivationLayer(
    x: Tensor<float32, [?, features]>
) -> Tensor<float32, [?, features]> {
    return ternary_activation(x)
}
```

### Learnable Activation

```python
class LearnableTernaryActivation(nn.Module):
    """Learnable ternary activation function."""
    
    def __init__(self, num_features):
        super().__init__()
        # Learnable thresholds per feature
        self.threshold = nn.Parameter(torch.ones(num_features) * 0.5)
        
    def forward(self, x):
        # Apply per-feature thresholds
        threshold = self.threshold.view(1, -1)
        output = torch.zeros_like(x)
        output[x > threshold] = 1
        output[x < -threshold] = -1
        
        # STE for gradients
        return output.detach() + x - x.detach()
```

## Residual Connections

### Ternary Residual Block

```triton
layer TernaryResidualBlock(
    conv1_w: TernaryTensor<trit, [channels, channels, 3, 3]>,
    conv1_b: TernaryTensor<trit, [channels]>,
    conv2_w: TernaryTensor<trit, [channels, channels, 3, 3]>,
    conv2_b: TernaryTensor<trit, [channels]>,
    x: Tensor<float32, [?, channels, h, w]>
) -> Tensor<float32, [?, channels, h, w]> {
    # First convolution
    let conv1 = conv2d(x, conv1_w, padding=1) + conv1_b
    let act1 = relu(conv1)
    
    # Second convolution
    let conv2 = conv2d(act1, conv2_w, padding=1) + conv2_b
    
    # Residual connection
    let output = conv2 + x
    let result = relu(output)
    
    return result
}
```

### PyTorch Implementation

```python
class TernaryResidualBlock(nn.Module):
    """Ternary residual block."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = TernaryConv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = TernaryConv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = torch.relu(out)
        
        return out
```

## Batch Normalization for Ternary Layers

### Custom Batch Norm

```python
class TernaryBatchNorm(nn.Module):
    """Batch normalization optimized for ternary weights."""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                                self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_normalized + self.bias
```

## Complete Custom Model Example

### ResNet-style Architecture

```python
class TernaryResNet(nn.Module):
    """Custom ternary ResNet-style model."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution (keep FP32 for first layer)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Ternary residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        
        # First block may change dimensions
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                TernaryConv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None
        
        layers.append(TernaryResidualBlock(in_channels, out_channels, 
                                          stride, downsample))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(TernaryResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Create and test model
model = TernaryResNet(num_classes=10)
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"Output shape: {output.shape}")  # [2, 10]

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

## Performance Optimization

### Fused Operations

```python
class FusedTernaryConvBN(nn.Module):
    """Fused convolution and batch norm for ternary layers."""
    
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = TernaryConv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))
    
    def fuse(self):
        """Fuse batch norm into convolution weights for inference."""
        if not self.training:
            # Get batch norm parameters
            gamma = self.bn.weight
            beta = self.bn.bias
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps
            
            # Fuse into convolution
            std = torch.sqrt(var + eps)
            weight = self.conv.weight * (gamma / std).view(-1, 1, 1, 1)
            bias = beta - mean * gamma / std + self.conv.bias
            
            # Update weights
            self.conv.weight.data = weight
            self.conv.bias.data = bias
            
            # Remove batch norm
            self.bn = nn.Identity()
```

## Testing Custom Layers

```python
import pytest

def test_ternary_conv2d():
    """Test ternary convolution layer."""
    layer = TernaryConv2d(3, 16, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    
    # Forward pass
    output = layer(x)
    assert output.shape == (2, 16, 32, 32)
    
    # Check weight quantization
    with torch.no_grad():
        w_quantized = quantize(layer.weight, method='deterministic')
        unique_values = torch.unique(w_quantized)
        assert len(unique_values) <= 3  # {-1, 0, 1}

def test_ternary_attention():
    """Test ternary attention mechanism."""
    layer = TernaryAttention(d_model=256, num_heads=8)
    x = torch.randn(2, 10, 256)
    
    # Forward pass
    output = layer(x)
    assert output.shape == (2, 10, 256)
    
    # Test with different sequence lengths
    x_long = torch.randn(2, 50, 256)
    output_long = layer(x_long)
    assert output_long.shape == (2, 50, 256)

# Run tests
test_ternary_conv2d()
test_ternary_attention()
print("All tests passed!")
```

## Best Practices

1. **Keep first/last layers in FP32**: Better accuracy
2. **Use batch normalization**: Stabilizes training
3. **Implement gradient clipping**: Prevents explosion
4. **Test incrementally**: Verify each custom layer
5. **Profile performance**: Measure actual speedup

## Exercises

1. Implement a ternary LSTM cell
2. Create a ternary vision transformer block
3. Build a ternary grouped convolution
4. Optimize custom layers with CUDA kernels

## Next Steps

- [Tutorial 4: Training](04_training.md) - Advanced training techniques
- [Tutorial 5: Deployment](05_deployment.md) - Deploy custom models
- [Architecture: Code Generator](../architecture/code_generator.md)

## Summary

You learned:
- ✓ Building custom ternary layers in DSL
- ✓ Implementing CNNs and attention mechanisms
- ✓ Creating custom activations
- ✓ Optimizing layer performance
- ✓ Testing custom components

## Resources

- [Example Custom Layers](https://github.com/financecommander/Triton/tree/main/examples/custom_layers)
- [API Reference: Backend](../api/backend.md)
- [DSL Language Spec](../dsl/language_spec.md)

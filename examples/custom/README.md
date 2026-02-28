# Custom Layers Examples

Production-ready custom quantized layers for building advanced neural networks.

## Available Examples

### 1. Custom Ternary Convolution (`custom_conv.triton`)
Advanced ternary convolution with learnable thresholds and scaling factors.

**Features:**
- Learnable per-channel quantization thresholds
- Adaptive scaling factors (alpha)
- Running statistics tracking
- Comprehensive quantization reporting

**Usage:**
```bash
triton compile custom_conv.triton --output custom_conv.py
python custom_conv.py --train --dataset cifar10
```

### 2. Attention Mechanism (`attention_mechanism.triton`)
Multi-head attention with ternary quantization support.

**Features:**
- Ternary multi-head attention
- Efficient fused QKV projection
- Cross-attention support
- Attention visualization tools

**Usage:**
```bash
triton compile attention_mechanism.triton --output attention.py
python attention.py --train --model vit --dataset imagenet
python attention.py --visualize-attention --image sample.jpg
```

### 3. Custom Quantization (`custom_quantization.triton`)
Advanced quantization schemes beyond standard methods.

**Features:**
- Logarithmic quantization
- Learned Step Size Quantization (LSQ)
- Soft differentiable quantization
- Block-wise mixed precision
- Outlier-aware quantization

**Usage:**
```bash
triton compile custom_quantization.triton --output custom_quant.py
python custom_quant.py --train --quantization-scheme lsq --bits 4
```

## Building Custom Layers

### Layer Template

```triton
layer MyCustomLayer {
    params {
        in_features: int
        out_features: int
        custom_param: float = 1.0
    }
    
    param weight: Tensor {
        shape: [out_features, in_features]
        init: "kaiming_normal"
    }
    
    forward(x: Tensor) -> Tensor {
        // Custom forward logic
        return output
    }
    
    method custom_method() {
        // Additional functionality
    }
}
```

### Best Practices

1. **Initialization**: Use appropriate weight initialization
2. **Gradients**: Implement Straight-Through Estimator (STE) for non-differentiable operations
3. **Statistics**: Track running statistics for inference
4. **Methods**: Add helper methods for analysis and debugging

## Contributing

Submit custom layers with:
- [ ] Clear documentation
- [ ] Usage examples
- [ ] Performance benchmarks
- [ ] Tests

# Triton DSL Examples

This directory contains example implementations demonstrating Ternary Neural Networks using the Triton DSL approach.

## 📚 Interactive Notebooks

**New!** Comprehensive Jupyter notebooks for learning Triton DSL interactively:

- **[01_introduction.ipynb](notebooks/01_introduction.ipynb)** - Get started with Triton DSL and ternary neural networks
- **[02_quantization_tutorial.ipynb](notebooks/02_quantization_tutorial.ipynb)** - Deep dive into quantization techniques
- **[03_performance_analysis.ipynb](notebooks/03_performance_analysis.ipynb)** - Performance profiling and optimization

👉 **[See Notebooks README](notebooks/README.md)** for detailed guide and setup instructions.

## Examples

### 1. MNIST Ternary Neural Network (`mnist_ternary.py`)

A complete, production-ready training script for MNIST digit classification using ternary quantization.

**Quick Start:**
```bash
# Install dependencies
pip install torch torchvision numpy matplotlib seaborn scikit-learn

# Run with default settings
python mnist_ternary.py

# View all options
python mnist_ternary.py --help
```

**Features:**
- Complete TernaryNet architecture with 3 layers
- Straight-Through Estimator (STE) for gradient flow
- Deterministic and stochastic quantization
- Training loop with Adam optimizer
- Comprehensive evaluation metrics
- Model persistence (save/load)
- Visualization (training curves, weight distributions, confusion matrix)
- Full CLI interface

**Expected Results:**
- Test accuracy: ~96-97% (vs ~98.5% float32 baseline)
- Model size: ~0.06 MB (16x compression)
- Training time: ~2-3 minutes on CPU (10 epochs)

**Key Options:**
```bash
# Use stochastic quantization
python mnist_ternary.py --quantize-method stochastic

# Train for 20 epochs with larger batch size
python mnist_ternary.py --epochs 20 --batch-size 128

# Use GPU if available
python mnist_ternary.py --device cuda

# Save model to custom path
python mnist_ternary.py --save-path ./my_model.pth
```

### Testing

Run unit tests to verify the implementation:
```bash
python test_mnist_ternary.py
```

This tests:
- Ternary quantization (deterministic & stochastic)
- LinearTernary layer forward/backward passes
- TernaryNet model architecture
- Training and evaluation loops
- Model save/load functionality
- Inference latency measurement

## Understanding the Code

### Ternary Quantization

Weights are constrained to {-1, 0, 1}:
```python
def ternarize(tensor, method='deterministic'):
    if method == 'deterministic':
        # Threshold at ±0.5
        output = torch.sign(input)
        output[torch.abs(input) < 0.5] = 0
    else:
        # Stochastic: probabilistic based on magnitude
        prob = torch.abs(input).clamp(0, 1)
        output = torch.sign(input) * (torch.rand_like(input) < prob).float()
    return output
```

### Straight-Through Estimator (STE)

Gradients flow through the quantization:
```python
class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, method):
        # Quantize to {-1, 0, 1}
        return quantized_output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Pass gradient straight through
        return grad_output, None
```

### Memory Savings

For a layer with m×n weight matrix:
- Float32: m × n × 4 bytes = 4mn bytes
- Ternary: m × n × 2 bits = mn/4 bytes
- Compression: **16x** for weights

Example: LinearTernary(784, 256)
- Float32: 784 × 256 × 4 = 802,816 bytes (783 KB)
- Ternary: 784 × 256 × 2 / 8 = 50,176 bytes (49 KB)
- Savings: **16x smaller**

## Extending the Examples

To create your own ternary networks:

1. **Use LinearTernary layers:**
   ```python
   from mnist_ternary import LinearTernary, ternary_activation
   
   class MyTernaryNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = LinearTernary(input_size, hidden_size)
           self.fc2 = LinearTernary(hidden_size, output_size)
       
       def forward(self, x):
           x = ternary_activation(self.fc1(x))
           x = self.fc2(x)
           return x
   ```

2. **Train with STE:**
   ```python
   # Standard PyTorch training loop works!
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
   for data, target in train_loader:
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()  # STE handles gradient flow
       optimizer.step()
   ```

3. **Save in packed format:**
   ```python
   from mnist_ternary import save_ternary_model
   save_ternary_model(model, Path('./my_model.pth'))
   ```

## Performance Tips

1. **Quantization Method:**
   - Deterministic: Faster, more stable
   - Stochastic: Can help with accuracy in some cases

2. **Learning Rate:**
   - Start with 0.001 for Adam
   - May need to tune for your specific task

3. **Activation Quantization:**
   - Full ternary: Quantize both weights and activations
   - Hybrid: Ternary weights, float activations (better accuracy)

4. **Hardware:**
   - CPU: Works well for small models
   - GPU: Faster for larger batches/models
   - Specialized hardware: Can leverage 2-bit operations

## References

- [Ternary Weight Networks (TWN)](https://arxiv.org/abs/1605.04711)
- [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064)
- Straight-Through Estimator: Bengio et al., 2013

## Contributing

Found a bug or have an improvement? Feel free to open an issue or PR!

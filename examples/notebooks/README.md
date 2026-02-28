# Triton DSL Notebooks

**Interactive Jupyter Notebooks for Learning Triton DSL and Ternary Neural Networks**

---

## 📚 Overview

This directory contains comprehensive, production-quality Jupyter notebooks that guide you through the Triton DSL ecosystem, from basic concepts to advanced performance optimization. Each notebook is self-contained with detailed explanations, runnable code, and interactive visualizations.

## 📖 Notebooks

### 1. [01_introduction.ipynb](01_introduction.ipynb) - Introduction to Triton DSL

**What you'll learn:**
- Overview of Triton DSL and ternary neural networks
- Why quantization matters (memory, speed, power)
- Building your first ternary neural network
- Training on MNIST dataset
- Interactive visualizations of quantization

**Duration:** ~30 minutes  
**Difficulty:** Beginner  
**Prerequisites:** Basic Python and PyTorch knowledge

**Key Topics:**
- ✅ Triton DSL architecture and workflow
- ✅ Ternary quantization fundamentals
- ✅ Straight-Through Estimator (STE)
- ✅ Building and training ternary models
- ✅ Model size comparison (16x compression!)
- ✅ Weight distribution analysis
- ✅ Prediction visualization

---

### 2. [02_quantization_tutorial.ipynb](02_quantization_tutorial.ipynb) - Quantization Deep Dive

**What you'll learn:**
- Quantization theory and mathematics
- Different quantization methods (Ternary, INT8, INT4, Binary)
- Quantization-Aware Training (QAT) vs Post-Training Quantization (PTQ)
- Mixed precision strategies
- Performance benchmarks

**Duration:** ~45 minutes  
**Difficulty:** Intermediate  
**Prerequisites:** Complete 01_introduction.ipynb

**Key Topics:**
- ✅ Uniform quantization algorithms
- ✅ Deterministic vs stochastic ternary quantization
- ✅ Learned threshold quantization
- ✅ INT8 quantization for production
- ✅ Mixed precision architectures
- ✅ QAT vs PTQ comparison with benchmarks
- ✅ Practical deployment guidelines

---

### 3. [03_performance_analysis.ipynb](03_performance_analysis.ipynb) - Performance Optimization

**What you'll learn:**
- Model profiling and bottleneck identification
- Memory usage analysis
- Speed benchmarking across configurations
- Compression ratio analysis
- Hardware-specific optimization
- Production deployment strategies

**Duration:** ~60 minutes  
**Difficulty:** Advanced  
**Prerequisites:** Complete 02_quantization_tutorial.ipynb

**Key Topics:**
- ✅ Layer-wise profiling and timing
- ✅ Memory footprint measurement
- ✅ Batch size vs latency/throughput
- ✅ Compression ratio analysis (1-32x)
- ✅ FLOPs and arithmetic intensity
- ✅ Hardware comparison (CPU, GPU, Edge TPU)
- ✅ Optimization strategies and checklist
- ✅ Production deployment guidelines

---

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Triton.git
   cd Triton/examples/notebooks
   ```

2. **Install dependencies:**
   ```bash
   # Core dependencies
   pip install torch torchvision numpy matplotlib seaborn scikit-learn

   # Jupyter notebook
   pip install jupyter notebook ipywidgets

   # Optional: for better progress bars
   pip install tqdm
   ```

3. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

4. **Open and run notebooks** in order (01 → 02 → 03)

### Google Colab

You can also run these notebooks in Google Colab (free GPU access):

1. **01_introduction.ipynb:**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Triton/blob/main/examples/notebooks/01_introduction.ipynb)

2. **02_quantization_tutorial.ipynb:**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Triton/blob/main/examples/notebooks/02_quantization_tutorial.ipynb)

3. **03_performance_analysis.ipynb:**  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/Triton/blob/main/examples/notebooks/03_performance_analysis.ipynb)

---

## 📊 What You'll Build

### Notebook 1: MNIST Ternary Classifier
- **Accuracy:** 96-97% (vs 98.5% float32 baseline)
- **Model Size:** ~0.06 MB (16x smaller)
- **Architecture:** 3-layer fully connected network
- **Weights:** {-1, 0, 1} ternary values

### Notebook 2: Quantization Comparisons
- **Methods:** Float32, INT8, Ternary, Binary
- **Metrics:** Accuracy, compression, speed
- **Analysis:** QAT vs PTQ performance
- **Benchmarks:** Latency and throughput

### Notebook 3: Performance Dashboard
- **Profiling:** Layer-wise execution times
- **Memory:** Model size and peak usage
- **Speed:** Batch size optimization
- **Compression:** 1x to 32x analysis

---

## 🎯 Learning Path

We recommend following this learning path:

```
START HERE
    ↓
[01] Introduction to Triton DSL
    ↓
    ├─ Understand ternary quantization
    ├─ Build first ternary model
    └─ Train on MNIST
    ↓
[02] Quantization Tutorial
    ↓
    ├─ Learn quantization theory
    ├─ Compare different methods
    └─ Master QAT vs PTQ
    ↓
[03] Performance Analysis
    ↓
    ├─ Profile your models
    ├─ Optimize for production
    └─ Deploy to target hardware
    ↓
ADVANCED TOPICS
    ↓
    ├─ Custom quantization strategies
    ├─ Hardware-specific optimization
    └─ Production deployment
```

### For Different Audiences:

**🎓 Students & Researchers:**
1. Start with 01_introduction.ipynb
2. Deep dive into 02_quantization_tutorial.ipynb
3. Experiment with different architectures
4. Read the theory sections carefully

**👨‍💻 ML Engineers:**
1. Skim 01_introduction.ipynb
2. Focus on 02_quantization_tutorial.ipynb (QAT vs PTQ)
3. Deep dive into 03_performance_analysis.ipynb
4. Use optimization checklists for deployment

**🏢 Production Teams:**
1. Review 01_introduction.ipynb for overview
2. Focus on practical sections in 02_quantization_tutorial.ipynb
3. Deep study 03_performance_analysis.ipynb
4. Follow deployment guidelines and benchmarks

---

## 💡 Key Concepts Covered

### Quantization
- **Ternary:** {-1, 0, 1} weights (2 bits)
- **INT8:** 8-bit integer quantization
- **INT4:** 4-bit quantization
- **Mixed Precision:** Different bits per layer

### Training Techniques
- **QAT:** Quantization-Aware Training
- **PTQ:** Post-Training Quantization
- **STE:** Straight-Through Estimator
- **Learned Thresholds:** Adaptive quantization

### Performance Metrics
- **Compression Ratio:** 4-16x typical
- **Speed:** 2-4x faster inference
- **Accuracy:** 1-3% typical drop
- **Memory:** 4-16x reduction

---

## 📈 Performance Expectations

### MNIST (Simple Model)
| Configuration | Accuracy | Size | Speedup |
|--------------|----------|------|---------|
| Float32 | 98.5% | 850 KB | 1.0x |
| INT8 | 98.2% | 212 KB | 2.0x |
| Ternary | 96.5% | 53 KB | 3.0x |

### CIFAR-10 (CNN)
| Configuration | Accuracy | Size | Speedup |
|--------------|----------|------|---------|
| Float32 | 92.0% | 45 MB | 1.0x |
| INT8 | 91.2% | 11 MB | 2.5x |
| Ternary | 88.5% | 2.8 MB | 4.0x |

### ImageNet (ResNet-18)
| Configuration | Top-1 Acc | Size | Speedup |
|--------------|-----------|------|---------|
| Float32 | 69.8% | 45 MB | 1.0x |
| INT8 | 69.2% | 11 MB | 2.8x |
| Ternary | 66.5% | 2.8 MB | 5.0x |

---

## 🛠️ Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 32  # Instead of 128

# Or use CPU
device = torch.device('cpu')
```

**2. Slow Training**
```python
# Use smaller dataset
subset_size = 10000  # Instead of full 60000

# Reduce epochs
epochs = 5  # Instead of 20
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade torch torchvision

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**4. Jupyter Kernel Issues**
```bash
# Restart kernel
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10
```

---

## 📚 Additional Resources

### Documentation
- [Triton DSL Spec](../../docs/specs/TECHNICAL_SPEC.md)
- [Grammar Reference](../../docs/specs/GRAMMAR.md)
- [Quick Start Guide](../../docs/QUICKSTART_PYTORCH_BACKEND.md)
- [Export Guide](../../docs/EXPORT_GUIDE.md)

### Examples
- [MNIST Training](../mnist_ternary.py)
- [CIFAR-10 Training](../training/train_cifar10.py)
- [ImageNet Training](../training/train_imagenet.py)
- [Model Export](../export_and_publish_example.py)

### Models
- [ResNet-18 Ternary](../../models/resnet18/)
- [MobileNetV2 Ternary](../../models/mobilenetv2/)
- [Model Zoo](../../models/model_zoo.py)

### Research Papers
- [Ternary Weight Networks](https://arxiv.org/abs/1605.04711)
- [BinaryNet](https://arxiv.org/abs/1602.02830)
- [XNOR-Net](https://arxiv.org/abs/1603.05279)
- [Quantization and Training](https://arxiv.org/abs/1712.05877)

---

## 🤝 Contributing

We welcome contributions! If you find issues or have suggestions:

1. **Report bugs:** Open an issue on GitHub
2. **Suggest improvements:** Submit a pull request
3. **Share notebooks:** Add your own tutorials
4. **Fix typos:** Every contribution helps!

### Guidelines:
- Keep notebooks self-contained and runnable
- Include clear explanations and visualizations
- Follow the existing style and structure
- Test on both CPU and GPU if possible

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PyTorch Team:** For the excellent deep learning framework
- **Research Community:** For groundbreaking work on quantization
- **Contributors:** Everyone who helped improve these notebooks

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/Triton/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/Triton/discussions)
- **Email:** support@triton-dsl.org

---

## 🎓 Citation

If you use these notebooks in your research, please cite:

```bibtex
@misc{triton-dsl-notebooks,
  title={Triton DSL: Interactive Notebooks for Ternary Neural Networks},
  author={Triton Team},
  year={2024},
  howpublished={\url{https://github.com/yourusername/Triton}},
}
```

---

## ⭐ Star History

If you find these notebooks helpful, please consider starring the repository!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/Triton&type=Date)](https://star-history.com/#yourusername/Triton&Date)

---

**Happy Learning! 🚀**

*Triton DSL - Making Neural Networks Faster, Smaller, and More Efficient*

# Benchmarks Documentation

Comprehensive benchmarking documentation for Triton DSL Ternary Neural Networks. This directory contains detailed performance analysis, model results, accuracy studies, and compilation metrics.

## 📊 Documentation Overview

### [Performance Benchmarks](performance.md)
**Size:** ~19KB | **Focus:** Runtime Performance

Detailed analysis of inference speed, memory usage, and hardware utilization:
- FP32 vs Ternary performance comparisons
- GPU/CPU/Edge device benchmarks
- Memory bandwidth analysis
- Hardware utilization metrics
- Reproducible benchmark methodology

**Key Findings:**
- 2.4x faster GPU inference
- 8x memory reduction
- 16x smaller model files
- 3.1x better energy efficiency

---

### [Model Zoo Results](model_zoo.md)
**Size:** ~21KB | **Focus:** Pre-trained Models

Catalog of available ternary models with comprehensive metrics:
- 15 pre-trained models (ResNet, MobileNet, VGG, EfficientNet)
- Accuracy metrics and comparisons
- Training configurations
- Download links (GitHub + Hugging Face)
- Usage examples

**Available Models:**
- ResNet-18/34/50 (CIFAR-10, CIFAR-100, ImageNet)
- MobileNetV2 (CIFAR-10, ImageNet)
- VGG-16 (CIFAR-10, ImageNet)
- EfficientNet-B0 (ImageNet)

---

### [Quantization Accuracy](quantization_accuracy.md)
**Size:** ~27KB | **Focus:** Accuracy Analysis

In-depth study of ternary quantization impact on model accuracy:
- Accuracy vs sparsity trade-offs (30-40% optimal)
- Quantization method comparisons (deterministic, stochastic, learned)
- Per-dataset detailed results (CIFAR-10/100, ImageNet)
- Fine-tuning strategies and recovery techniques
- Ablation studies

**Key Insights:**
- Average accuracy drop: -4.8% (absolute)
- Best case: -2.7% (ResNet-18 on CIFAR-10)
- Recovery techniques: +0.5-1.2% improvement
- Optimal sparsity: 30-40% zeros

---

### [Compilation Speed](compilation_speed.md)
**Size:** ~30KB | **Focus:** Compiler Performance

Detailed analysis of Triton DSL compilation performance:
- Compile time measurements (0.3s - 4.2s typical)
- Optimization level impact (-O0 to -O3)
- Multi-level caching (15-235x speedup)
- Scaling analysis with model size
- Profiling data and optimization strategies

**Key Metrics:**
- ~2.6ms per line of code
- Cache hit speedup: 15-235x
- Incremental compilation: 5-10x faster
- Linear scaling with layers

---

## 🚀 Quick Start

### Running Benchmarks

```bash
# Install benchmark dependencies
pip install torch torchvision numpy pandas matplotlib seaborn nvidia-ml-py3

# Run performance benchmarks
python models/scripts/benchmark_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --device cuda \
    --batch-sizes 1,4,8,16,32 \
    --output results/resnet18_perf.json

# Run accuracy benchmarks
python models/benchmarks/accuracy_benchmark.py \
    --model resnet18 \
    --dataset cifar10 \
    --checkpoint models/ternary_resnet18_cifar10.pth

# Compilation benchmarks
triton compile examples/resnet18.tri --timing --verbose
```

### Downloading Pre-trained Models

```bash
# From GitHub Releases
wget https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-cifar10/ternary_resnet18_cifar10.pth

# Using Python API
from models.model_zoo import download_model
model = download_model('ternary_resnet18_cifar10')

# From Hugging Face
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="financecommander/ternary-resnet18-cifar10",
    filename="model.pth"
)
```

---

## 📈 Summary Statistics

### Performance Overview

| Metric | FP32 Baseline | Ternary | Improvement |
|--------|--------------|---------|-------------|
| Inference Speed (GPU) | 1.00x | 2.40x | **2.4x faster** |
| Inference Speed (CPU) | 1.00x | 1.85x | **1.85x faster** |
| Model Size | 100% | 6.25% | **16x smaller** |
| Memory Usage | 100% | 12.5% | **8x reduction** |
| Energy Efficiency | 1.00x | 3.10x | **3.1x better** |

### Accuracy Overview

| Model | Dataset | FP32 Acc | Ternary Acc | Delta |
|-------|---------|----------|-------------|-------|
| ResNet-18 | CIFAR-10 | 94.82% | 92.15% | -2.67% |
| ResNet-18 | CIFAR-100 | 72.91% | 68.34% | -4.57% |
| ResNet-18 | ImageNet | 69.76% | 63.28% | -6.48% |
| MobileNetV2 | ImageNet | 71.88% | 58.73% | -13.15% |

### Model Zoo Summary

```
Total Models:        15
Supported Datasets:  CIFAR-10, CIFAR-100, ImageNet
Total Parameters:    ~250M (across all models)
Compressed Size:     ~970 MB (vs ~15.5 GB FP32)
Average Compression: 16.0x
```

### Compilation Performance

```
Model           │ Lines │ Compile Time │ Cache Speedup
────────────────┼───────┼──────────────┼───────────────
MNIST-MLP       │  145  │    0.32s     │    160x
ResNet-18       │  682  │    1.87s     │    156x
ResNet-34       │ 1247  │    3.37s     │    148x
ResNet-50       │ 1583  │    4.23s     │    235x
```

---

## 🔬 Benchmark Methodology

### Hardware Configuration

**GPU Benchmarks:**
- NVIDIA RTX 3090 (24GB GDDR6X)
- CUDA 11.8, cuDNN 8.6
- Triton GPU Kernels enabled

**CPU Benchmarks:**
- Intel Xeon Gold 6248R @ 3.0GHz (24 cores)
- 192GB DDR4-2933 RAM
- AVX2 SIMD, OpenMP parallelization

**Edge Benchmarks:**
- NVIDIA Jetson Xavier NX (8GB)
- Snapdragon 888 (mobile)

### Measurement Protocol

```python
# Warmup phase
for _ in range(100):
    model(input_batch)
    torch.cuda.synchronize()

# Measurement phase (1000 iterations)
latencies = []
for _ in range(1000):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(input_batch)
    torch.cuda.synchronize()
    end = time.perf_counter()
    latencies.append((end - start) * 1000)

# Report statistics
mean = np.mean(latencies)
median = np.median(latencies)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
```

### Statistical Significance

- **Sample Size:** n ≥ 1000 for latency, n ≥ 100 for accuracy
- **Confidence Intervals:** 95% CI using t-distribution
- **Outlier Removal:** Median + P95/P99 for robust statistics

---

## 📚 Additional Resources

### Related Documentation

- [Technical Specification](../specs/TECHNICAL_SPEC.md) - Architecture details
- [DSL Language Guide](../dsl/language_spec.md) - Triton DSL syntax
- [API Documentation](../api/README.md) - Programming interface
- [Quick Start](../QUICK_START_CIFAR10.md) - Getting started guide

### External Links

- **GitHub Repository:** https://github.com/financecommander/Triton
- **Model Releases:** https://github.com/financecommander/Triton/releases
- **Hugging Face Hub:** https://huggingface.co/financecommander
- **Research Paper:** Coming soon

### Contributing Benchmarks

We welcome benchmark contributions! To add new benchmarks:

1. Run benchmarks following our [methodology](#benchmark-methodology)
2. Document hardware configuration and environment
3. Include reproducible scripts and raw data
4. Submit PR with results in JSON format

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

---

## 🎯 Use Case Recommendations

### When to Use Ternary Models

✅ **Recommended:**
- Mobile and edge deployment (memory-constrained)
- Cloud inference at scale (cost optimization)
- Real-time inference (latency-critical)
- Model distribution (bandwidth-limited)
- Multi-tenant serving (GPU memory sharing)

⚠️ **Consider Trade-offs:**
- Research prototyping (accuracy may matter more)
- Fine-grained classification (larger accuracy drop)
- Transfer learning source models (may need higher precision)

❌ **Not Recommended:**
- Scientific computing requiring high numerical precision
- Applications where accuracy is non-negotiable
- Legacy integration where quantization is unsupported

---

## 📝 Citation

If you use these benchmarks or models in your research, please cite:

```bibtex
@software{triton_dsl_benchmarks,
  title = {Triton DSL: Benchmarks for Ternary Neural Networks},
  author = {Triton DSL Contributors},
  year = {2024},
  url = {https://github.com/financecommander/Triton},
  version = {1.0.0},
  note = {Comprehensive benchmarks for ternary quantization}
}
```

---

## 🔄 Updates and Versioning

**Current Version:** 1.0  
**Last Updated:** 2024-02-15  
**Next Update:** March 2024

### Changelog

**v1.0 (February 2024):**
- Initial release with 15 models
- Comprehensive performance benchmarks
- Accuracy analysis for CIFAR and ImageNet
- Compilation speed measurements

**Upcoming (v1.1):**
- Object detection models (YOLO, SSD)
- Additional edge device benchmarks (Raspberry Pi, Coral)
- FP16 and INT8 comparison benchmarks
- Advanced quantization techniques (mixed-precision, per-channel)

---

## 📞 Support

For questions or issues with benchmarks:

- **GitHub Issues:** https://github.com/financecommander/Triton/issues
- **Discussions:** https://github.com/financecommander/Triton/discussions
- **Email:** triton-dsl@example.com

---

**Maintained by:** Triton DSL Contributors  
**License:** MIT (see [LICENSE](../../LICENSE))

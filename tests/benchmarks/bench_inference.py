"""
Benchmark: End-to-End Inference Speed

Measures inference latency for different models and batch sizes.

Models:
- MNIST classifier
- CIFAR-10 classifier

Comparison:
- Ternary
- Float32
- Int8 Quantization

Metrics:
- Latency per sample (ms)
- Throughput (samples/sec)
- 95th percentile latency

Usage:
    pytest tests/benchmarks/bench_inference.py --benchmark-json=results/inference_results.json
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pytorch.ternary_models import TernaryMNISTNet, TernaryCIFAR10Net


BATCH_SIZES = [1, 8, 32, 64]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_WARMUP = 10
N_ITERATIONS = 100


# ============================================================================
# Float32 Model Implementations
# ============================================================================

class Float32MNISTNet(nn.Module):
    """Simple CNN for MNIST with float32 weights."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Float32CIFAR10Net(nn.Module):
    """Simple CNN for CIFAR-10 with float32 weights."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# Int8 Quantized Model Implementations
# ============================================================================

def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Quantize a model to int8.
    
    Note: This is a simplified quantization. Production code would use
    torch.quantization with proper calibration.
    """
    # Use PyTorch's dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    return quantized_model


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_inference(model: nn.Module, input_tensor: torch.Tensor, 
                        n_iterations: int = N_ITERATIONS) -> Dict:
    """
    Benchmark inference latency.
    
    Returns:
        Dictionary with latency metrics
    """
    model.eval()
    model = model.to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)
    
    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(input_tensor)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            output = model(input_tensor)
            
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # Convert to ms
    
    latencies_array = np.array(latencies)
    batch_size = input_tensor.size(0)
    
    # Calculate metrics
    mean_latency = np.mean(latencies_array)
    std_latency = np.std(latencies_array)
    median_latency = np.median(latencies_array)
    p95_latency = np.percentile(latencies_array, 95)
    p99_latency = np.percentile(latencies_array, 99)
    
    # Per-sample metrics
    mean_latency_per_sample = mean_latency / batch_size
    throughput = (batch_size * 1000) / mean_latency  # samples per second
    
    return {
        'mean_ms': mean_latency,
        'std_ms': std_latency,
        'median_ms': median_latency,
        'p95_ms': p95_latency,
        'p99_ms': p99_latency,
        'mean_per_sample_ms': mean_latency_per_sample,
        'throughput_samples_per_sec': throughput,
    }


def benchmark_mnist_model(model_type: str, batch_size: int) -> Dict:
    """Benchmark MNIST model."""
    # Create input
    input_tensor = torch.randn(batch_size, 1, 28, 28)
    
    # Create model based on type
    if model_type == 'float32':
        model = Float32MNISTNet()
    elif model_type == 'ternary':
        model = TernaryMNISTNet()
    elif model_type == 'int8':
        model = Float32MNISTNet()
        model = quantize_model_int8(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Benchmark
    metrics = benchmark_inference(model, input_tensor)
    
    return {
        'dataset': 'MNIST',
        'model_type': model_type,
        'batch_size': batch_size,
        **metrics
    }


def benchmark_cifar10_model(model_type: str, batch_size: int) -> Dict:
    """Benchmark CIFAR-10 model."""
    # Create input
    input_tensor = torch.randn(batch_size, 3, 32, 32)
    
    # Create model based on type
    if model_type == 'float32':
        model = Float32CIFAR10Net()
    elif model_type == 'ternary':
        model = TernaryCIFAR10Net()
    elif model_type == 'int8':
        model = Float32CIFAR10Net()
        model = quantize_model_int8(model)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Benchmark
    metrics = benchmark_inference(model, input_tensor)
    
    return {
        'dataset': 'CIFAR-10',
        'model_type': model_type,
        'batch_size': batch_size,
        **metrics
    }


# ============================================================================
# Pytest Benchmark Tests
# ============================================================================

class TestInferenceBenchmark:
    """Inference speed benchmark suite."""
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_mnist_float32(self, benchmark, batch_size):
        """Benchmark MNIST float32 inference."""
        model = Float32MNISTNet().to(DEVICE)
        model.eval()
        input_tensor = torch.randn(batch_size, 1, 28, 28, device=DEVICE)
        
        def inference():
            with torch.no_grad():
                return model(input_tensor)
        
        benchmark.pedantic(inference, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_mnist_ternary(self, benchmark, batch_size):
        """Benchmark MNIST ternary inference."""
        model = TernaryMNISTNet().to(DEVICE)
        model.eval()
        input_tensor = torch.randn(batch_size, 1, 28, 28, device=DEVICE)
        
        def inference():
            with torch.no_grad():
                return model(input_tensor)
        
        benchmark.pedantic(inference, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_cifar10_float32(self, benchmark, batch_size):
        """Benchmark CIFAR-10 float32 inference."""
        model = Float32CIFAR10Net().to(DEVICE)
        model.eval()
        input_tensor = torch.randn(batch_size, 3, 32, 32, device=DEVICE)
        
        def inference():
            with torch.no_grad():
                return model(input_tensor)
        
        benchmark.pedantic(inference, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)
    
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test_cifar10_ternary(self, benchmark, batch_size):
        """Benchmark CIFAR-10 ternary inference."""
        model = TernaryCIFAR10Net().to(DEVICE)
        model.eval()
        input_tensor = torch.randn(batch_size, 3, 32, 32, device=DEVICE)
        
        def inference():
            with torch.no_grad():
                return model(input_tensor)
        
        benchmark.pedantic(inference, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)


# ============================================================================
# Visualization and Reporting
# ============================================================================

def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create inference benchmark visualizations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Inference Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Latency vs Batch Size (MNIST)
    ax = axes[0, 0]
    mnist_data = df[df['dataset'] == 'MNIST']
    for model_type in mnist_data['model_type'].unique():
        data = mnist_data[mnist_data['model_type'] == model_type]
        ax.plot(data['batch_size'], data['mean_per_sample_ms'], 
               marker='o', label=model_type)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency per Sample (ms)')
    ax.set_title('MNIST: Latency per Sample vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 2: Latency vs Batch Size (CIFAR-10)
    ax = axes[0, 1]
    cifar_data = df[df['dataset'] == 'CIFAR-10']
    for model_type in cifar_data['model_type'].unique():
        data = cifar_data[cifar_data['model_type'] == model_type]
        ax.plot(data['batch_size'], data['mean_per_sample_ms'], 
               marker='o', label=model_type)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Latency per Sample (ms)')
    ax.set_title('CIFAR-10: Latency per Sample vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    
    # Plot 3: Throughput Comparison (Batch Size = 32)
    ax = axes[1, 0]
    batch_32_data = df[df['batch_size'] == 32]
    
    datasets = batch_32_data['dataset'].unique()
    model_types = batch_32_data['model_type'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model_type in enumerate(model_types):
        throughputs = []
        for dataset in datasets:
            data = batch_32_data[(batch_32_data['dataset'] == dataset) & 
                                 (batch_32_data['model_type'] == model_type)]
            if len(data) > 0:
                throughputs.append(data['throughput_samples_per_sec'].values[0])
            else:
                throughputs.append(0)
        
        ax.bar(x + i*width, throughputs, width, label=model_type, alpha=0.8)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput Comparison (Batch Size = 32)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Speedup vs Batch Size
    ax = axes[1, 1]
    
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        
        # Calculate speedup: float32 / ternary
        float32_data = dataset_data[dataset_data['model_type'] == 'float32'].sort_values('batch_size')
        ternary_data = dataset_data[dataset_data['model_type'] == 'ternary'].sort_values('batch_size')
        
        if len(float32_data) > 0 and len(ternary_data) > 0:
            speedups = []
            batch_sizes = []
            for bs in BATCH_SIZES:
                f32 = float32_data[float32_data['batch_size'] == bs]
                ter = ternary_data[ternary_data['batch_size'] == bs]
                if len(f32) > 0 and len(ter) > 0:
                    speedup = f32['mean_per_sample_ms'].values[0] / ter['mean_per_sample_ms'].values[0]
                    speedups.append(speedup)
                    batch_sizes.append(bs)
            
            ax.plot(batch_sizes, speedups, marker='o', label=dataset)
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Speedup (Ternary vs Float32)')
    ax.set_title('Speedup vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax.axhline(y=2.0, color='g', linestyle='--', alpha=0.5, label='2x target')
    ax.set_xscale('log', base=2)
    
    plt.tight_layout()
    plot_path = output_dir / 'inference_benchmark.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")
    plt.close()


def generate_report():
    """Generate comprehensive inference benchmark report."""
    print("\n" + "="*80)
    print("INFERENCE BENCHMARK REPORT")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Batch Sizes: {BATCH_SIZES}")
    
    results = []
    
    # Benchmark all configurations
    for dataset_fn, dataset_name in [
        (benchmark_mnist_model, 'MNIST'),
        (benchmark_cifar10_model, 'CIFAR-10')
    ]:
        print(f"\n{dataset_name}:")
        print("-" * 80)
        
        for model_type in ['float32', 'ternary', 'int8']:
            for batch_size in BATCH_SIZES:
                result = dataset_fn(model_type, batch_size)
                results.append(result)
                
                print(f"  {model_type:8s} | Batch {batch_size:2d} | "
                      f"Latency: {result['mean_per_sample_ms']:6.3f}ms/sample | "
                      f"Throughput: {result['throughput_samples_per_sec']:7.1f} samples/sec | "
                      f"P95: {result['p95_ms']:6.2f}ms")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'inference_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    json_path = output_dir / 'inference_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    # Create visualizations
    create_visualizations(df, output_dir)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset}:")
        dataset_data = df[df['dataset'] == dataset]
        
        # Calculate average speedup
        float32_avg = dataset_data[dataset_data['model_type'] == 'float32']['mean_per_sample_ms'].mean()
        ternary_avg = dataset_data[dataset_data['model_type'] == 'ternary']['mean_per_sample_ms'].mean()
        int8_avg = dataset_data[dataset_data['model_type'] == 'int8']['mean_per_sample_ms'].mean()
        
        speedup_ternary = float32_avg / ternary_avg if ternary_avg > 0 else 0
        speedup_int8 = float32_avg / int8_avg if int8_avg > 0 else 0
        
        print(f"  Float32 avg latency: {float32_avg:.3f} ms/sample")
        print(f"  Ternary avg latency: {ternary_avg:.3f} ms/sample")
        print(f"  Int8 avg latency: {int8_avg:.3f} ms/sample")
        print(f"  Ternary speedup: {speedup_ternary:.2f}x")
        print(f"  Int8 speedup: {speedup_int8:.2f}x")
        
        if speedup_ternary >= 2.0:
            print(f"  ✓ Target met: {speedup_ternary:.2f}x >= 2.0x")
        else:
            print(f"  ✗ Target not met: {speedup_ternary:.2f}x < 2.0x")


def main():
    """Main entry point for inference benchmarks."""
    generate_report()


if __name__ == '__main__':
    main()

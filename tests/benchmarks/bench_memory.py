"""
Benchmark: Memory Usage Comparison

Measures peak memory usage during training for float32 vs ternary models.

Models tested:
- ResNet-18
- MobileNetV2
- BERT-tiny

Metrics:
- Model size on disk (MB)
- GPU memory during forward pass (MB)
- GPU memory during backward pass (MB)
- Activation memory (MB)

Usage:
    pytest tests/benchmarks/bench_memory.py --benchmark-json=results/memory_results.json
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pytorch.ternary_models import (
    TernaryResNet18, TernaryMobileNetV2, TernaryBertTiny,
    TernaryMNISTNet, TernaryCIFAR10Net
)

# Standard model implementations for comparison
from torchvision import models as torch_models


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes."""
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = os.path.getsize(tmp.name)
    return size_bytes / (1024 * 1024)


def measure_memory_usage(model: nn.Module, input_tensor: torch.Tensor, 
                         with_backward: bool = True) -> Dict[str, float]:
    """
    Measure memory usage during forward and backward passes.
    
    Returns:
        Dictionary with memory metrics
    """
    model = model.to(DEVICE)
    input_tensor = input_tensor.to(DEVICE)
    
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Forward pass
    if DEVICE == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    
    output = model(input_tensor)
    
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        forward_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        forward_memory = 0.0
    
    # Backward pass
    if with_backward:
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # Create dummy loss
        if output.dim() == 2:
            target = torch.randint(0, output.size(1), (output.size(0),), device=DEVICE)
            loss = nn.functional.cross_entropy(output, target)
        else:
            loss = output.mean()
        
        loss.backward()
        
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
            backward_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        else:
            backward_memory = 0.0
    else:
        backward_memory = 0.0
    
    # Model size
    model_size = get_model_size_mb(model)
    
    # Calculate activation memory (approximate)
    activation_memory = forward_memory - model_size if forward_memory > model_size else 0
    
    return {
        'model_size_mb': model_size,
        'forward_memory_mb': forward_memory,
        'backward_memory_mb': backward_memory,
        'activation_memory_mb': activation_memory,
        'total_memory_mb': forward_memory + backward_memory,
    }


class Float32ResNet18(nn.Module):
    """Standard ResNet-18 wrapper."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch_models.resnet18(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class Float32MobileNetV2(nn.Module):
    """Standard MobileNetV2 wrapper."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torch_models.mobilenet_v2(num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


class Float32BertTiny(nn.Module):
    """Simple BERT-like model for comparison."""
    def __init__(self, vocab_size=30522, hidden_size=128, num_layers=2, num_classes=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=2, dim_feedforward=512, batch_first=True),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.embeddings(x)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])


def benchmark_resnet18():
    """Benchmark ResNet-18 memory usage."""
    print("\n" + "="*80)
    print("ResNet-18 Memory Benchmark")
    print("="*80)
    
    input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32)
    
    # Float32 model
    float32_model = Float32ResNet18(num_classes=10)
    float32_metrics = measure_memory_usage(float32_model, input_tensor)
    
    # Ternary model
    ternary_model = TernaryResNet18(num_classes=10)
    ternary_metrics = measure_memory_usage(ternary_model, input_tensor)
    
    # Calculate reduction
    model_size_reduction = float32_metrics['model_size_mb'] / ternary_metrics['model_size_mb']
    memory_reduction = float32_metrics['total_memory_mb'] / ternary_metrics['total_memory_mb'] if ternary_metrics['total_memory_mb'] > 0 else 0
    
    print(f"\nFloat32 Model:")
    print(f"  Model Size: {float32_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {float32_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {float32_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {float32_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nTernary Model:")
    print(f"  Model Size: {ternary_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {ternary_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {ternary_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {ternary_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nReduction:")
    print(f"  Model Size: {model_size_reduction:.2f}x")
    print(f"  Total Memory: {memory_reduction:.2f}x")
    
    return {
        'model': 'ResNet-18',
        'float32': float32_metrics,
        'ternary': ternary_metrics,
        'model_size_reduction': model_size_reduction,
        'memory_reduction': memory_reduction,
    }


def benchmark_mobilenetv2():
    """Benchmark MobileNetV2 memory usage."""
    print("\n" + "="*80)
    print("MobileNetV2 Memory Benchmark")
    print("="*80)
    
    input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32)
    
    # Float32 model
    float32_model = Float32MobileNetV2(num_classes=10)
    float32_metrics = measure_memory_usage(float32_model, input_tensor)
    
    # Ternary model
    ternary_model = TernaryMobileNetV2(num_classes=10)
    ternary_metrics = measure_memory_usage(ternary_model, input_tensor)
    
    # Calculate reduction
    model_size_reduction = float32_metrics['model_size_mb'] / ternary_metrics['model_size_mb']
    memory_reduction = float32_metrics['total_memory_mb'] / ternary_metrics['total_memory_mb'] if ternary_metrics['total_memory_mb'] > 0 else 0
    
    print(f"\nFloat32 Model:")
    print(f"  Model Size: {float32_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {float32_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {float32_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {float32_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nTernary Model:")
    print(f"  Model Size: {ternary_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {ternary_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {ternary_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {ternary_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nReduction:")
    print(f"  Model Size: {model_size_reduction:.2f}x")
    print(f"  Total Memory: {memory_reduction:.2f}x")
    
    return {
        'model': 'MobileNetV2',
        'float32': float32_metrics,
        'ternary': ternary_metrics,
        'model_size_reduction': model_size_reduction,
        'memory_reduction': memory_reduction,
    }


def benchmark_bert_tiny():
    """Benchmark BERT-tiny memory usage."""
    print("\n" + "="*80)
    print("BERT-tiny Memory Benchmark")
    print("="*80)
    
    seq_length = 128
    input_tensor = torch.randint(0, 30522, (BATCH_SIZE, seq_length))
    
    # Float32 model
    float32_model = Float32BertTiny()
    float32_metrics = measure_memory_usage(float32_model, input_tensor)
    
    # Ternary model
    ternary_model = TernaryBertTiny()
    ternary_metrics = measure_memory_usage(ternary_model, input_tensor)
    
    # Calculate reduction
    model_size_reduction = float32_metrics['model_size_mb'] / ternary_metrics['model_size_mb']
    memory_reduction = float32_metrics['total_memory_mb'] / ternary_metrics['total_memory_mb'] if ternary_metrics['total_memory_mb'] > 0 else 0
    
    print(f"\nFloat32 Model:")
    print(f"  Model Size: {float32_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {float32_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {float32_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {float32_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nTernary Model:")
    print(f"  Model Size: {ternary_metrics['model_size_mb']:.2f} MB")
    print(f"  Forward Memory: {ternary_metrics['forward_memory_mb']:.2f} MB")
    print(f"  Backward Memory: {ternary_metrics['backward_memory_mb']:.2f} MB")
    print(f"  Total Memory: {ternary_metrics['total_memory_mb']:.2f} MB")
    
    print(f"\nReduction:")
    print(f"  Model Size: {model_size_reduction:.2f}x")
    print(f"  Total Memory: {memory_reduction:.2f}x")
    
    return {
        'model': 'BERT-tiny',
        'float32': float32_metrics,
        'ternary': ternary_metrics,
        'model_size_reduction': model_size_reduction,
        'memory_reduction': memory_reduction,
    }


class TestMemoryBenchmark:
    """Memory usage benchmark suite."""
    
    def test_resnet18_float32(self, benchmark):
        """Benchmark ResNet-18 float32 memory."""
        model = Float32ResNet18(num_classes=10).to(DEVICE)
        input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)
    
    def test_resnet18_ternary(self, benchmark):
        """Benchmark ResNet-18 ternary memory."""
        model = TernaryResNet18(num_classes=10).to(DEVICE)
        input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)
    
    def test_mobilenetv2_float32(self, benchmark):
        """Benchmark MobileNetV2 float32 memory."""
        model = Float32MobileNetV2(num_classes=10).to(DEVICE)
        input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)
    
    def test_mobilenetv2_ternary(self, benchmark):
        """Benchmark MobileNetV2 ternary memory."""
        model = TernaryMobileNetV2(num_classes=10).to(DEVICE)
        input_tensor = torch.randn(BATCH_SIZE, 3, 32, 32, device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)
    
    def test_bert_tiny_float32(self, benchmark):
        """Benchmark BERT-tiny float32 memory."""
        model = Float32BertTiny().to(DEVICE)
        input_tensor = torch.randint(0, 30522, (BATCH_SIZE, 128), device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)
    
    def test_bert_tiny_ternary(self, benchmark):
        """Benchmark BERT-tiny ternary memory."""
        model = TernaryBertTiny().to(DEVICE)
        input_tensor = torch.randint(0, 30522, (BATCH_SIZE, 128), device=DEVICE)
        
        def forward_pass():
            output = model(input_tensor)
            return output
        
        benchmark.pedantic(forward_pass, rounds=10, warmup_rounds=3)


def create_visualizations(results: list, output_dir: Path):
    """Create memory usage visualizations."""
    
    # Extract data for plotting
    models = [r['model'] for r in results]
    float32_sizes = [r['float32']['model_size_mb'] for r in results]
    ternary_sizes = [r['ternary']['model_size_mb'] for r in results]
    float32_memory = [r['float32']['total_memory_mb'] for r in results]
    ternary_memory = [r['ternary']['total_memory_mb'] for r in results]
    size_reductions = [r['model_size_reduction'] for r in results]
    memory_reductions = [r['memory_reduction'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Memory Usage Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Model Size Comparison
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, float32_sizes, width, label='Float32', alpha=0.8)
    ax.bar(x + width/2, ternary_sizes, width, label='Ternary', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size on Disk')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Total Memory Usage
    ax = axes[0, 1]
    ax.bar(x - width/2, float32_memory, width, label='Float32', alpha=0.8)
    ax.bar(x + width/2, ternary_memory, width, label='Ternary', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Total Memory (MB)')
    ax.set_title('Total Memory Usage (Forward + Backward)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Model Size Reduction
    ax = axes[1, 0]
    bars = ax.bar(x, size_reductions, alpha=0.8, color='green')
    ax.set_xlabel('Model')
    ax.set_ylabel('Reduction Factor')
    ax.set_title('Model Size Reduction (Float32 / Ternary)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.axhline(y=4.0, color='r', linestyle='--', label='4x Target', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    # Plot 4: Memory Reduction
    ax = axes[1, 1]
    bars = ax.bar(x, memory_reductions, alpha=0.8, color='blue')
    ax.set_xlabel('Model')
    ax.set_ylabel('Reduction Factor')
    ax.set_title('Total Memory Reduction (Float32 / Ternary)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.axhline(y=4.0, color='r', linestyle='--', label='4x Target', alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = output_dir / 'memory_benchmark.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")
    plt.close()


def main():
    """Main entry point for memory benchmarks."""
    print("Starting Memory Usage Benchmarks...")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    
    results = []
    
    # Run benchmarks
    results.append(benchmark_resnet18())
    results.append(benchmark_mobilenetv2())
    results.append(benchmark_bert_tiny())
    
    # Create output directory
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Save results
    csv_data = []
    for result in results:
        csv_data.append({
            'model': result['model'],
            'float32_model_size_mb': result['float32']['model_size_mb'],
            'ternary_model_size_mb': result['ternary']['model_size_mb'],
            'float32_forward_memory_mb': result['float32']['forward_memory_mb'],
            'ternary_forward_memory_mb': result['ternary']['forward_memory_mb'],
            'float32_backward_memory_mb': result['float32']['backward_memory_mb'],
            'ternary_backward_memory_mb': result['ternary']['backward_memory_mb'],
            'float32_total_memory_mb': result['float32']['total_memory_mb'],
            'ternary_total_memory_mb': result['ternary']['total_memory_mb'],
            'model_size_reduction': result['model_size_reduction'],
            'memory_reduction': result['memory_reduction'],
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'memory_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Save JSON
    json_path = output_dir / 'memory_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    avg_size_reduction = np.mean([r['model_size_reduction'] for r in results])
    avg_memory_reduction = np.mean([r['memory_reduction'] for r in results if r['memory_reduction'] > 0])
    
    print(f"\nAverage Model Size Reduction: {avg_size_reduction:.2f}x")
    print(f"Average Memory Reduction: {avg_memory_reduction:.2f}x")
    
    if avg_size_reduction >= 4.0:
        print(f"✓ Model size target met: {avg_size_reduction:.2f}x >= 4.0x")
    else:
        print(f"✗ Model size target not met: {avg_size_reduction:.2f}x < 4.0x")


if __name__ == '__main__':
    main()

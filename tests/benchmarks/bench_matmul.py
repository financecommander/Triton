"""
Benchmark: Matrix Multiplication Performance

Compares ternary_matmul vs torch.matmul (float32) across different matrix sizes
and sparsity levels.

Metrics:
- GFLOPS (effective operations per second)
- Memory bandwidth (GB/s)
- Latency (ms)

Usage:
    pytest tests/benchmarks/bench_matmul.py --benchmark-json=results/matmul_results.json
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pytorch.ternary_tensor import ternary_matmul, TernaryTensor


# Test configurations
MATRIX_SIZES = [128, 256, 512, 1024, 2048]
SPARSITY_LEVELS = [0.0, 0.25, 0.5, 0.75]  # Percentage of zeros
DEVICES = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
N_WARMUP = 5
N_ITERATIONS = 20


def create_sparse_matrix(size: int, sparsity: float, device: str) -> torch.Tensor:
    """Create a matrix with specified sparsity level."""
    matrix = torch.randn(size, size, device=device)
    if sparsity > 0:
        mask = torch.rand(size, size, device=device) > sparsity
        matrix = matrix * mask.float()
    return matrix


def measure_gflops(size: int, elapsed_time: float) -> float:
    """Calculate GFLOPS (billion floating point operations per second)."""
    # Matrix multiplication: 2 * N^3 operations (multiply and add)
    flops = 2 * size ** 3
    gflops = (flops / elapsed_time) / 1e9
    return gflops


def measure_bandwidth(size: int, elapsed_time: float, dtype_bytes: int = 4) -> float:
    """Calculate memory bandwidth in GB/s."""
    # Two input matrices + one output matrix
    total_bytes = 3 * size * size * dtype_bytes
    bandwidth_gbs = (total_bytes / elapsed_time) / 1e9
    return bandwidth_gbs


def benchmark_matmul(matrix_a: torch.Tensor, matrix_b: torch.Tensor, 
                     use_ternary: bool = False) -> Tuple[float, Dict]:
    """
    Benchmark matrix multiplication.
    
    Returns:
        Tuple of (mean_time_ms, metrics_dict)
    """
    device = matrix_a.device
    size = matrix_a.shape[0]
    
    # Warmup
    for _ in range(N_WARMUP):
        if use_ternary:
            _ = ternary_matmul(matrix_a, matrix_b)
        else:
            _ = torch.matmul(matrix_a, matrix_b)
    
    # Synchronize for accurate timing (especially important for CUDA)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(N_ITERATIONS):
        start = time.perf_counter()
        
        if use_ternary:
            result = ternary_matmul(matrix_a, matrix_b)
        else:
            result = torch.matmul(matrix_a, matrix_b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    times_array = np.array(times)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array)
    
    # Calculate metrics
    gflops = measure_gflops(size, mean_time / 1000)
    bandwidth = measure_bandwidth(size, mean_time / 1000)
    
    metrics = {
        'mean_ms': mean_time,
        'std_ms': std_time,
        'median_ms': np.median(times_array),
        'min_ms': np.min(times_array),
        'max_ms': np.max(times_array),
        'gflops': gflops,
        'bandwidth_gbs': bandwidth,
    }
    
    return mean_time, metrics


class TestMatMulBenchmark:
    """Matrix multiplication benchmark suite."""
    
    @pytest.mark.parametrize("size", MATRIX_SIZES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sparsity", SPARSITY_LEVELS)
    def test_matmul_float32(self, benchmark, size, device, sparsity):
        """Benchmark standard float32 matmul."""
        matrix_a = create_sparse_matrix(size, sparsity, device)
        matrix_b = create_sparse_matrix(size, sparsity, device)
        
        def run_matmul():
            return torch.matmul(matrix_a, matrix_b)
        
        result = benchmark.pedantic(run_matmul, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)
    
    @pytest.mark.parametrize("size", MATRIX_SIZES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("sparsity", SPARSITY_LEVELS)
    def test_matmul_ternary(self, benchmark, size, device, sparsity):
        """Benchmark ternary matmul."""
        matrix_a = create_sparse_matrix(size, sparsity, device)
        matrix_b = create_sparse_matrix(size, sparsity, device)
        
        def run_matmul():
            return ternary_matmul(matrix_a, matrix_b)
        
        result = benchmark.pedantic(run_matmul, rounds=N_ITERATIONS, warmup_rounds=N_WARMUP)


def generate_comparison_report():
    """Generate detailed comparison report with visualizations."""
    results = []
    
    print("\n" + "="*80)
    print("MATRIX MULTIPLICATION BENCHMARK REPORT")
    print("="*80)
    
    for device in DEVICES:
        print(f"\nDevice: {device.upper()}")
        print("-" * 80)
        
        for size in MATRIX_SIZES:
            for sparsity in SPARSITY_LEVELS:
                # Create test matrices
                matrix_a = create_sparse_matrix(size, sparsity, device)
                matrix_b = create_sparse_matrix(size, sparsity, device)
                
                # Benchmark float32
                time_float32, metrics_float32 = benchmark_matmul(matrix_a, matrix_b, use_ternary=False)
                
                # Benchmark ternary
                time_ternary, metrics_ternary = benchmark_matmul(matrix_a, matrix_b, use_ternary=True)
                
                # Calculate speedup
                speedup = time_float32 / time_ternary if time_ternary > 0 else 0
                
                result = {
                    'device': device,
                    'size': size,
                    'sparsity': sparsity,
                    'float32_ms': time_float32,
                    'ternary_ms': time_ternary,
                    'speedup': speedup,
                    'float32_gflops': metrics_float32['gflops'],
                    'ternary_gflops': metrics_ternary['gflops'],
                    'float32_bandwidth': metrics_float32['bandwidth_gbs'],
                    'ternary_bandwidth': metrics_ternary['bandwidth_gbs'],
                }
                results.append(result)
                
                print(f"Size: {size:4d} | Sparsity: {sparsity:.2f} | "
                      f"Float32: {time_float32:7.2f}ms | Ternary: {time_ternary:7.2f}ms | "
                      f"Speedup: {speedup:4.2f}x")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / 'matmul_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    # Generate visualizations
    create_visualizations(df, output_dir)
    
    # Statistical significance tests
    perform_statistical_tests(df)
    
    # Save JSON
    json_path = output_dir / 'matmul_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ JSON results saved to: {json_path}")
    
    return df


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create benchmark visualizations."""
    
    # 1. Speedup vs Matrix Size
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Matrix Multiplication Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Speedup vs Size for different sparsity levels
    ax = axes[0, 0]
    for sparsity in SPARSITY_LEVELS:
        data = df[df['sparsity'] == sparsity]
        for device in data['device'].unique():
            device_data = data[data['device'] == device]
            ax.plot(device_data['size'], device_data['speedup'], 
                   marker='o', label=f'{device} - {int(sparsity*100)}% sparse')
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Speedup (Ternary vs Float32)')
    ax.set_title('Speedup vs Matrix Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 2: GFLOPS comparison
    ax = axes[0, 1]
    sizes_to_plot = [512, 1024, 2048]
    x_pos = np.arange(len(sizes_to_plot))
    width = 0.35
    
    for device in df['device'].unique():
        device_data = df[(df['device'] == device) & (df['sparsity'] == 0.0)]
        device_data = device_data[device_data['size'].isin(sizes_to_plot)]
        
        float32_gflops = device_data['float32_gflops'].values
        ternary_gflops = device_data['ternary_gflops'].values
        
        ax.bar(x_pos - width/2, float32_gflops, width, label=f'{device} Float32', alpha=0.8)
        ax.bar(x_pos + width/2, ternary_gflops, width, label=f'{device} Ternary', alpha=0.8)
    
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('GFLOPS')
    ax.set_title('GFLOPS Comparison (0% Sparsity)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes_to_plot)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Latency vs Sparsity
    ax = axes[1, 0]
    size_to_plot = 1024
    data = df[df['size'] == size_to_plot]
    
    for device in data['device'].unique():
        device_data = data[data['device'] == device]
        ax.plot(device_data['sparsity'] * 100, device_data['float32_ms'], 
               marker='o', label=f'{device} Float32')
        ax.plot(device_data['sparsity'] * 100, device_data['ternary_ms'], 
               marker='s', label=f'{device} Ternary')
    
    ax.set_xlabel('Sparsity (%)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Latency vs Sparsity (Size={size_to_plot})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Memory Bandwidth
    ax = axes[1, 1]
    for device in df['device'].unique():
        device_data = df[(df['device'] == device) & (df['sparsity'] == 0.0)]
        ax.plot(device_data['size'], device_data['float32_bandwidth'], 
               marker='o', label=f'{device} Float32')
        ax.plot(device_data['size'], device_data['ternary_bandwidth'], 
               marker='s', label=f'{device} Ternary')
    
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Bandwidth (GB/s)')
    ax.set_title('Memory Bandwidth (0% Sparsity)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'matmul_benchmark.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_path}")
    plt.close()


def perform_statistical_tests(df: pd.DataFrame):
    """Perform statistical significance tests."""
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    # Group by device and size, compare float32 vs ternary
    for device in df['device'].unique():
        print(f"\n{device.upper()}:")
        device_data = df[df['device'] == device]
        
        for size in MATRIX_SIZES:
            size_data = device_data[device_data['size'] == size]
            
            if len(size_data) > 0:
                float32_times = size_data['float32_ms'].values
                ternary_times = size_data['ternary_ms'].values
                
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(float32_times, ternary_times)
                
                mean_speedup = size_data['speedup'].mean()
                
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                
                print(f"  Size {size:4d}: Mean Speedup = {mean_speedup:4.2f}x, "
                      f"p-value = {p_value:.6f} {significance}")


def main():
    """Main entry point for running benchmarks."""
    print("Starting Matrix Multiplication Benchmarks...")
    print(f"Testing on devices: {DEVICES}")
    print(f"Matrix sizes: {MATRIX_SIZES}")
    print(f"Sparsity levels: {[f'{s*100:.0f}%' for s in SPARSITY_LEVELS]}")
    
    df = generate_comparison_report()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for device in df['device'].unique():
        device_data = df[df['device'] == device]
        avg_speedup = device_data['speedup'].mean()
        max_speedup = device_data['speedup'].max()
        
        print(f"\n{device.upper()}:")
        print(f"  Average Speedup: {avg_speedup:.2f}x")
        print(f"  Maximum Speedup: {max_speedup:.2f}x")
        
        # Check if target is met
        if avg_speedup >= 2.0:
            print(f"  ✓ Target met: {avg_speedup:.2f}x >= 2.0x")
        else:
            print(f"  ✗ Target not met: {avg_speedup:.2f}x < 2.0x")


if __name__ == '__main__':
    main()

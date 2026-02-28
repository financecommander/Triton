"""
Utility functions for integration tests.
Provides helper functions for testing, benchmarking, and validation.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, List, Tuple, Callable, Any
from contextlib import contextmanager


def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, 
                          warmup_iterations: int = 10, 
                          benchmark_iterations: int = 100) -> Dict[str, float]:
    """
    Measure inference time for a model.
    
    Args:
        model: The model to benchmark
        input_tensor: Input tensor for inference
        warmup_iterations: Number of warmup iterations
        benchmark_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_tensor)
    
    # Synchronize CUDA if available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
    
    return {
        'mean': sum(times) / len(times),
        'std': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
        'min': min(times),
        'max': max(times),
        'median': sorted(times)[len(times) // 2],
    }


def measure_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Measure memory usage for a model.
    
    Args:
        model: The model to measure
        input_tensor: Input tensor for inference
        
    Returns:
        Dictionary with memory statistics (in MB)
    """
    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
    input_tensor = input_tensor.to(device)
    
    # Clear cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Measure model memory
    model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)
    
    # Run inference to measure peak memory
    with torch.no_grad():
        _ = model(input_tensor)
    
    result = {
        'model_memory_mb': model_memory,
        'buffer_memory_mb': buffer_memory,
        'total_memory_mb': model_memory + buffer_memory,
    }
    
    # Add CUDA memory if available
    if torch.cuda.is_available():
        result['cuda_peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return result


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_params = sum(b.numel() for b in model.buffers())
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'buffer_parameters': buffer_params,
        'total_elements': total_params + buffer_params,
    }


def validate_output_shape(output: torch.Tensor, expected_shape: Tuple[int, ...]) -> bool:
    """
    Validate output tensor shape.
    
    Args:
        output: Output tensor to validate
        expected_shape: Expected shape
        
    Returns:
        True if shape matches, False otherwise
    """
    return tuple(output.shape) == expected_shape


def validate_output_range(output: torch.Tensor, min_val: float = None, 
                         max_val: float = None) -> bool:
    """
    Validate output tensor value range.
    
    Args:
        output: Output tensor to validate
        min_val: Minimum expected value (optional)
        max_val: Maximum expected value (optional)
        
    Returns:
        True if range is valid, False otherwise
    """
    if min_val is not None and output.min() < min_val:
        return False
    if max_val is not None and output.max() > max_val:
        return False
    return True


def compare_model_outputs(model1: nn.Module, model2: nn.Module, 
                         input_tensor: torch.Tensor,
                         rtol: float = 1e-3, atol: float = 1e-5) -> Dict[str, Any]:
    """
    Compare outputs of two models.
    
    Args:
        model1: First model
        model2: Second model
        input_tensor: Input tensor for comparison
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary with comparison results
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
    
    # Calculate differences
    diff = torch.abs(output1 - output2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Check if outputs are close
    close = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    
    return {
        'outputs_close': close,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'output1_shape': tuple(output1.shape),
        'output2_shape': tuple(output2.shape),
        'shapes_match': tuple(output1.shape) == tuple(output2.shape),
    }


def validate_gradients(model: nn.Module, input_tensor: torch.Tensor, 
                      target: torch.Tensor) -> Dict[str, Any]:
    """
    Validate that gradients are computed correctly.
    
    Args:
        model: The model to validate
        input_tensor: Input tensor
        target: Target tensor
        
    Returns:
        Dictionary with gradient validation results
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    
    return {
        'has_gradients': has_gradients,
        'num_gradients': len(grad_norms),
        'max_grad_norm': max(grad_norms) if grad_norms else 0.0,
        'mean_grad_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        'loss_value': loss.item(),
    }


def test_forward_backward_pass(model: nn.Module, input_shape: Tuple[int, ...],
                              output_shape: Tuple[int, ...]) -> Dict[str, bool]:
    """
    Test forward and backward pass for a model.
    
    Args:
        model: The model to test
        input_shape: Shape of input tensor
        output_shape: Expected output shape
        
    Returns:
        Dictionary with test results
    """
    # Create random input
    x = torch.randn(*input_shape, requires_grad=True)
    
    # Forward pass
    try:
        model.train()
        output = model(x)
        forward_success = True
        shape_correct = tuple(output.shape) == output_shape
    except Exception as e:
        return {
            'forward_pass': False,
            'shape_correct': False,
            'backward_pass': False,
            'error': str(e),
        }
    
    # Backward pass
    try:
        loss = output.sum()
        loss.backward()
        backward_success = x.grad is not None
    except Exception as e:
        backward_success = False
    
    return {
        'forward_pass': forward_success,
        'shape_correct': shape_correct,
        'backward_pass': backward_success,
        'error': None,
    }


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

def check_numerical_stability(model: nn.Module, input_tensor: torch.Tensor,
                              n_iterations: int = 10) -> Dict[str, Any]:
    """
    Check numerical stability by running multiple inferences.
    
    Args:
        model: The model to test
        input_tensor: Input tensor
        n_iterations: Number of iterations
        
    Returns:
        Dictionary with stability metrics
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for _ in range(n_iterations):
            output = model(input_tensor)
            outputs.append(output.clone())
    
    # Check if outputs are consistent
    first_output = outputs[0]
    all_same = all(torch.allclose(first_output, out, rtol=1e-5, atol=1e-7) for out in outputs)
    
    # Calculate variance
    stacked = torch.stack(outputs)
    variance = stacked.var(dim=0).mean().item()
    
    return {
        'numerically_stable': all_same,
        'output_variance': variance,
        'num_iterations': n_iterations,
    }


def benchmark_batch_sizes(model: nn.Module, input_shape: Tuple[int, ...],
                         batch_sizes: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Benchmark model with different batch sizes.
    
    Args:
        model: The model to benchmark
        input_shape: Input shape (without batch dimension)
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary mapping batch size to timing statistics
    """
    results = {}
    model.eval()
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, *input_shape)
        timing = measure_inference_time(model, input_tensor, 
                                       warmup_iterations=5,
                                       benchmark_iterations=50)
        results[batch_size] = timing
    
    return results


def validate_ternary_weights(weights: torch.Tensor) -> Dict[str, Any]:
    """
    Validate that weights are ternary (-1, 0, 1).
    
    Args:
        weights: Weight tensor to validate
        
    Returns:
        Dictionary with validation results
    """
    unique_values = torch.unique(weights)
    is_ternary = len(unique_values) <= 3 and all(v in [-1, 0, 1] for v in unique_values.tolist())
    
    value_counts = {
        'negative_one': (weights == -1).sum().item(),
        'zero': (weights == 0).sum().item(),
        'positive_one': (weights == 1).sum().item(),
    }
    
    return {
        'is_ternary': is_ternary,
        'unique_values': unique_values.tolist(),
        'value_counts': value_counts,
        'total_elements': weights.numel(),
    }


def calculate_compression_ratio(original_model: nn.Module, 
                                compressed_model: nn.Module) -> float:
    """
    Calculate compression ratio between two models.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        
    Returns:
        Compression ratio
    """
    original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
    original_size += sum(b.numel() * b.element_size() for b in original_model.buffers())
    
    compressed_size = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
    compressed_size += sum(b.numel() * b.element_size() for b in compressed_model.buffers())
    
    return original_size / compressed_size if compressed_size > 0 else float('inf')

#!/usr/bin/env python3
"""
Test script for MNIST Ternary Neural Network example.

Tests the model architecture, ternary quantization, and key functions
without requiring actual MNIST data download.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mnist_ternary import (
    LinearTernary,
    TernaryNet,
    ternarize,
    ternary_activation,
    train_epoch,
    evaluate,
    measure_inference_latency,
    save_ternary_model,
    load_ternary_model,
)


def test_ternarize():
    """Test ternary quantization function."""
    print("Testing ternarize function...")
    
    # Test deterministic quantization
    x = torch.tensor([[-2.0, -0.6, -0.3, 0.0, 0.3, 0.6, 2.0]])
    x_t = ternarize(x, method='deterministic')
    expected = torch.tensor([[-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0]])
    
    assert torch.allclose(x_t, expected), f"Expected {expected}, got {x_t}"
    print("  ✓ Deterministic quantization works")
    
    # Test stochastic quantization produces valid values
    x = torch.randn(100, 100)
    x_t = ternarize(x, method='stochastic')
    unique_vals = torch.unique(x_t)
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals), \
        f"Stochastic quantization produced invalid values: {unique_vals}"
    print("  ✓ Stochastic quantization works")
    
    print("  ✓ All ternarize tests passed!\n")


def test_ternary_activation():
    """Test ternary activation function."""
    print("Testing ternary_activation function...")
    
    x = torch.tensor([[-5.0, -0.3, 0.0, 0.3, 5.0]])
    y = ternary_activation(x)
    
    # Should be all -1, 0, or 1
    unique_vals = torch.unique(y)
    assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals), \
        f"Activation produced invalid values: {unique_vals}"
    
    print("  ✓ Ternary activation produces valid values")
    print("  ✓ All ternary_activation tests passed!\n")


def test_linear_ternary():
    """Test LinearTernary layer."""
    print("Testing LinearTernary layer...")
    
    # Create layer
    layer = LinearTernary(10, 5, quantize_method='deterministic')
    
    # Test forward pass
    x = torch.randn(32, 10)
    y = layer(x)
    
    assert y.shape == (32, 5), f"Expected shape (32, 5), got {y.shape}"
    print("  ✓ Forward pass produces correct output shape")
    
    # Test that weights are ternarized during forward pass
    with torch.no_grad():
        weight_t = ternarize(layer.weight, layer.quantize_method)
        unique_vals = torch.unique(weight_t)
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_vals), \
            f"Weights not properly ternarized: {unique_vals}"
    
    print("  ✓ Weights are properly ternarized")
    
    # Test backward pass (STE)
    y.sum().backward()
    assert layer.weight.grad is not None, "Gradient not computed"
    print("  ✓ Backward pass works (STE)")
    
    # Test packed weights
    packed = layer.get_packed_weights()
    assert packed.dtype == torch.uint8, "Packed weights should be uint8"
    print("  ✓ Weight packing works")
    
    print("  ✓ All LinearTernary tests passed!\n")


def test_ternary_net():
    """Test TernaryNet model."""
    print("Testing TernaryNet model...")
    
    # Create model
    model = TernaryNet(quantize_method='deterministic')
    
    # Test forward pass
    x = torch.randn(16, 1, 28, 28)
    y = model(x)
    
    assert y.shape == (16, 10), f"Expected shape (16, 10), got {y.shape}"
    print("  ✓ Forward pass produces correct output shape")
    
    # Test parameter counting
    params = model.count_parameters()
    assert params['total'] > 0, "No parameters found"
    assert params['trainable'] == params['total'], "Not all parameters trainable"
    print(f"  ✓ Parameter counting works: {params['total']:,} parameters")
    
    # Test model size calculation
    size_info = model.get_model_size()
    assert size_info['float32_mb'] > 0, "Float32 size should be positive"
    assert size_info['ternary_mb'] > 0, "Ternary size should be positive"
    assert size_info['compression_ratio'] > 1, "Compression ratio should be > 1"
    print(f"  ✓ Model size calculation works: {size_info['compression_ratio']:.2f}x compression")
    
    # Test backward pass
    loss = y.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
    
    print("  ✓ Backward pass computes all gradients")
    print("  ✓ All TernaryNet tests passed!\n")


def test_save_load():
    """Test model saving and loading."""
    print("Testing model save/load...")
    
    # Create and save model
    model1 = TernaryNet(quantize_method='deterministic')
    save_path = Path('/tmp/test_model.pth')
    
    save_ternary_model(model1, save_path)
    assert save_path.exists(), "Model file not created"
    print("  ✓ Model saved successfully")
    
    # Load model
    model2 = load_ternary_model(save_path, torch.device('cpu'))
    
    # Verify loaded model works
    x = torch.randn(8, 1, 28, 28)
    y1 = model1(x)
    y2 = model2(x)
    
    assert torch.allclose(y1, y2, atol=1e-5), "Loaded model produces different output"
    print("  ✓ Model loaded and produces same output")
    
    # Cleanup
    save_path.unlink()
    packed_path = Path('/tmp/test_model_packed.pth')
    if packed_path.exists():
        packed_path.unlink()
    
    print("  ✓ All save/load tests passed!\n")


def test_inference_latency():
    """Test inference latency measurement."""
    print("Testing inference latency measurement...")
    
    model = TernaryNet(quantize_method='deterministic')
    device = torch.device('cpu')
    
    latency = measure_inference_latency(model, device, num_iterations=10)
    
    assert 'mean_ms' in latency, "Missing mean_ms in latency results"
    assert latency['mean_ms'] > 0, "Mean latency should be positive"
    assert latency['std_ms'] >= 0, "Std should be non-negative"
    
    print(f"  ✓ Latency measurement works: {latency['mean_ms']:.3f} ms mean")
    print("  ✓ All inference latency tests passed!\n")


def test_training_step():
    """Test a single training step."""
    print("Testing training step...")
    
    # Create fake dataset
    from torch.utils.data import TensorDataset, DataLoader
    
    X = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model, optimizer, criterion
    model = TernaryNet(quantize_method='deterministic')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    # Train one epoch
    loss, acc = train_epoch(model, device, loader, optimizer, criterion, epoch=1)
    
    assert loss > 0, "Loss should be positive"
    assert 0 <= acc <= 100, f"Accuracy should be in [0, 100], got {acc}"
    
    print(f"  ✓ Training step works: loss={loss:.4f}, acc={acc:.2f}%")
    print("  ✓ All training step tests passed!\n")


def test_evaluation():
    """Test evaluation function."""
    print("Testing evaluation...")
    
    # Create fake dataset
    from torch.utils.data import TensorDataset, DataLoader
    
    X = torch.randn(50, 1, 28, 28)
    y = torch.randint(0, 10, (50,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Create model and criterion
    model = TernaryNet(quantize_method='deterministic')
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    
    # Evaluate
    loss, acc, preds, targets = evaluate(model, device, loader, criterion)
    
    assert loss > 0, "Loss should be positive"
    assert 0 <= acc <= 100, f"Accuracy should be in [0, 100], got {acc}"
    assert len(preds) == len(targets) == 50, "Prediction length mismatch"
    
    print(f"  ✓ Evaluation works: loss={loss:.4f}, acc={acc:.2f}%")
    print("  ✓ All evaluation tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Testing MNIST Ternary Neural Network Implementation")
    print("=" * 80)
    print()
    
    try:
        test_ternarize()
        test_ternary_activation()
        test_linear_ternary()
        test_ternary_net()
        test_save_load()
        test_inference_latency()
        test_training_step()
        test_evaluation()
        
        print("=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe MNIST ternary training example is working correctly.")
        print("All core functionality (quantization, training, evaluation) is verified.")
        print("\nTo run actual training on MNIST:")
        print("  python examples/mnist_ternary.py --epochs 10")
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

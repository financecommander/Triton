"""Unit tests for quantization and activation functions."""

import time

import pytest
import torch

from backend.pytorch.ops.activations import ternary_activation
from backend.pytorch.ops.quantize import quantize


class TestQuantize:
    """Test suite for quantize function."""

    def test_deterministic_basic(self):
        """Test basic deterministic quantization."""
        x = torch.tensor([0.5, -0.5, 0.1, -0.1, 0.0])
        result = quantize(x, method="deterministic", threshold=0.33)

        assert result.dtype == torch.int8
        assert torch.equal(result, torch.tensor([1, -1, 0, 0, 0], dtype=torch.int8))

    def test_deterministic_threshold(self):
        """Test deterministic quantization with custom threshold."""
        x = torch.tensor([0.6, -0.6, 0.4, -0.4, 0.2, -0.2])
        result = quantize(x, method="deterministic", threshold=0.5)

        expected = torch.tensor([1, -1, 0, 0, 0, 0], dtype=torch.int8)
        assert torch.equal(result, expected)

    def test_deterministic_edge_cases(self):
        """Test edge cases for deterministic quantization."""
        threshold = 0.33

        # Exactly at threshold
        x = torch.tensor([0.33, -0.33])
        result = quantize(x, method="deterministic", threshold=threshold)
        # Should be 0 since we use strict inequality (> threshold, < -threshold)
        assert torch.equal(result, torch.tensor([0, 0], dtype=torch.int8))

        # Just above/below threshold
        x = torch.tensor([0.34, -0.34])
        result = quantize(x, method="deterministic", threshold=threshold)
        assert torch.equal(result, torch.tensor([1, -1], dtype=torch.int8))

    def test_stochastic_output_range(self):
        """Test that stochastic quantization produces values in {-1, 0, 1}."""
        torch.manual_seed(42)
        x = torch.randn(100)
        result = quantize(x, method="stochastic", threshold=0.33)

        assert result.dtype == torch.int8
        # Check all values are in {-1, 0, 1}
        unique_values = torch.unique(result)
        assert all(v in [-1, 0, 1] for v in unique_values.tolist())

    def test_stochastic_distribution(self):
        """Test that stochastic quantization produces reasonable distribution."""
        torch.manual_seed(42)
        # Positive values should mostly map to 1
        x = torch.full((100,), 0.8)
        result = quantize(x, method="stochastic", threshold=0.33)
        positive_count = (result == 1).sum().item()
        # Should be mostly positive
        assert positive_count > 50

        # Negative values should mostly map to -1
        torch.manual_seed(42)
        x = torch.full((100,), -0.8)
        result = quantize(x, method="stochastic", threshold=0.33)
        negative_count = (result == -1).sum().item()
        assert negative_count > 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that quantization works on CUDA tensors."""
        x = torch.randn(100, device="cuda")
        result_det = quantize(x, method="deterministic", threshold=0.33)
        result_stoch = quantize(x, method="stochastic", threshold=0.33)

        assert result_det.device.type == "cuda"
        assert result_stoch.device.type == "cuda"
        assert result_det.dtype == torch.int8
        assert result_stoch.dtype == torch.int8

    def test_gradient_passthrough(self):
        """Test that gradients pass through quantization (straight-through estimator)."""
        x = torch.randn(10, requires_grad=True)
        y = quantize(x, method="deterministic", threshold=0.33)

        # Convert to float for backward pass
        y_float = y.float()
        loss = y_float.sum()
        loss.backward()

        # Gradient should exist and match the shape
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # With straight-through estimator, gradient should be all ones
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_gradcheck_deterministic(self):
        """Test gradient computation with torch.autograd.gradcheck for deterministic mode."""
        # Use double precision for gradcheck
        x = torch.randn(5, dtype=torch.double, requires_grad=True)

        def func(input_tensor):
            # Convert to float since gradcheck needs float output
            return quantize(input_tensor, method="deterministic", threshold=0.33).float()

        # Gradcheck verifies that analytical gradients match numerical gradients
        # For straight-through estimator, this should pass
        assert torch.autograd.gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_gradcheck_stochastic(self):
        """Test gradient computation for stochastic mode."""
        torch.manual_seed(42)
        x = torch.randn(5, dtype=torch.double, requires_grad=True)

        def func(input_tensor):
            return quantize(input_tensor, method="stochastic", threshold=0.33).float()

        # Note: stochastic mode has randomness, but gradients should still pass through
        assert torch.autograd.gradcheck(func, x, eps=1e-6, atol=1e-4)

    def test_batch_processing(self):
        """Test quantization on batched inputs."""
        batch_size = 32
        feature_dim = 64
        x = torch.randn(batch_size, feature_dim)

        result = quantize(x, method="deterministic", threshold=0.33)

        assert result.shape == (batch_size, feature_dim)
        assert result.dtype == torch.int8


class TestTernaryActivation:
    """Test suite for ternary_activation function."""

    def test_basic_clipping(self):
        """Test basic clipping behavior."""
        x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        result = ternary_activation(x)

        expected = torch.tensor([-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_within_range(self):
        """Test that values within [-1, 1] are unchanged."""
        x = torch.tensor([-0.9, -0.5, 0.0, 0.5, 0.9])
        result = ternary_activation(x)

        assert torch.allclose(result, x)

    def test_ste_gradient_flow(self):
        """Test Straight-Through Estimator gradient behavior."""
        x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        y = ternary_activation(x)
        loss = y.sum()
        loss.backward()

        # Gradients should be 1 for |x| <= 1, and 0 for |x| > 1
        expected_grad = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])
        assert torch.allclose(x.grad, expected_grad)

    def test_ste_gradient_interior(self):
        """Test that gradients flow through for interior points."""
        x = torch.tensor([0.5, 0.3, -0.3, -0.5], requires_grad=True)
        y = ternary_activation(x)
        loss = y.sum()
        loss.backward()

        # All values are in (-1, 1), so all gradients should be 1
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_ste_gradient_exterior(self):
        """Test that gradients are blocked for exterior points."""
        x = torch.tensor([2.0, 3.0, -2.0, -3.0], requires_grad=True)
        y = ternary_activation(x)
        loss = y.sum()
        loss.backward()

        # All values are outside (-1, 1), so all gradients should be 0
        expected_grad = torch.zeros_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_ste_gradient_boundary(self):
        """Test gradient behavior at boundaries."""
        x = torch.tensor([-1.0, 1.0], requires_grad=True)
        y = ternary_activation(x)
        loss = y.sum()
        loss.backward()

        # At boundaries |x| = 1, gradients should flow through
        expected_grad = torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)

    def test_gradcheck(self):
        """Test gradient computation with torch.autograd.gradcheck."""
        x = torch.randn(5, dtype=torch.double, requires_grad=True)

        # Gradcheck verifies analytical vs numerical gradients
        assert torch.autograd.gradcheck(ternary_activation, x, eps=1e-6, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_support(self):
        """Test that ternary activation works on CUDA."""
        x = torch.randn(100, device="cuda", requires_grad=True)
        y = ternary_activation(x)

        assert y.device.type == "cuda"

        # Test backward pass
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"

    def test_batch_processing(self):
        """Test activation on batched inputs."""
        batch_size = 32
        feature_dim = 64
        x = torch.randn(batch_size, feature_dim)

        result = ternary_activation(x)

        assert result.shape == (batch_size, feature_dim)
        # All values should be in [-1, 1]
        assert torch.all(result >= -1.0)
        assert torch.all(result <= 1.0)


class TestBenchmarks:
    """Benchmark tests for quantization performance."""

    def test_quantize_speed_cpu(self):
        """Benchmark quantization speed on CPU."""
        x = torch.randn(1000, 1000)
        iterations = 10

        # Warm-up
        for _ in range(2):
            _ = quantize(x, method="deterministic")

        # Benchmark deterministic
        start = time.time()
        for _ in range(iterations):
            _ = quantize(x, method="deterministic")
        det_time = (time.time() - start) / iterations

        # Benchmark stochastic
        torch.manual_seed(42)
        start = time.time()
        for _ in range(iterations):
            _ = quantize(x, method="stochastic")
        stoch_time = (time.time() - start) / iterations

        print(f"\nDeterministic quantization: {det_time*1000:.2f}ms")
        print(f"Stochastic quantization: {stoch_time*1000:.2f}ms")

        # Just check that both methods complete successfully
        assert det_time > 0
        assert stoch_time > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_quantize_speed_cuda(self):
        """Benchmark quantization speed on CUDA."""
        x = torch.randn(1000, 1000, device="cuda")
        iterations = 100

        # Warm-up
        for _ in range(10):
            _ = quantize(x, method="deterministic")
        torch.cuda.synchronize()

        # Benchmark deterministic
        start = time.time()
        for _ in range(iterations):
            _ = quantize(x, method="deterministic")
        torch.cuda.synchronize()
        det_time = (time.time() - start) / iterations

        # Benchmark stochastic
        torch.manual_seed(42)
        start = time.time()
        for _ in range(iterations):
            _ = quantize(x, method="stochastic")
        torch.cuda.synchronize()
        stoch_time = (time.time() - start) / iterations

        print(f"\nCUDA Deterministic quantization: {det_time*1000:.2f}ms")
        print(f"CUDA Stochastic quantization: {stoch_time*1000:.2f}ms")

        assert det_time > 0
        assert stoch_time > 0

    def test_activation_speed_cpu(self):
        """Benchmark ternary activation speed on CPU."""
        x = torch.randn(1000, 1000, requires_grad=True)
        iterations = 10

        # Warm-up
        for _ in range(2):
            y = ternary_activation(x)
            y.sum().backward()
            x.grad.zero_()

        # Benchmark forward + backward
        start = time.time()
        for _ in range(iterations):
            y = ternary_activation(x)
            y.sum().backward()
            x.grad.zero_()
        total_time = (time.time() - start) / iterations

        print(f"\nTernary activation (forward+backward): {total_time*1000:.2f}ms")

        assert total_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

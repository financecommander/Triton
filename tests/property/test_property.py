"""
Property-Based Tests for Triton Compiler

Tests that verify mathematical properties and system invariants:
- Associativity, commutativity, distributivity
- Precision and numerical stability
- Memory consistency
- Deterministic behavior
- Conservation laws
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from typing import Tuple, List
import math
from unittest.mock import Mock

# Try to import Triton components, use mocks if not available
try:
    from backend.pytorch import ternary_matmul
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    ternary_matmul = Mock()


class TestMathematicalProperties:
    """Test mathematical properties of ternary operations."""

    @given(
        st.integers(min_value=2, max_value=64),
        st.integers(min_value=2, max_value=64),
        st.integers(min_value=2, max_value=64)
    )
    @settings(max_examples=50)
    def test_associativity_approximation(self, m, k, n):
        """Test approximate associativity: (AB)C ≈ A(BC)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Generate test matrices
        A = torch.randn(m, k, device=device, dtype=torch.float32)
        B = torch.randn(k, n, device=device, dtype=torch.float32)
        C = torch.randn(n, m, device=device, dtype=torch.float32)  # Note: C is (n, m) for (AB)C

        # Compute both orders
        AB = ternary_matmul(A, B)  # (m, k) x (k, n) -> (m, n)
        ABC1 = ternary_matmul(AB, C)  # (m, n) x (n, m) -> (m, m)

        BC = ternary_matmul(B, C)  # (k, n) x (n, m) -> (k, m)
        ABC2 = ternary_matmul(A, BC)  # (m, k) x (k, m) -> (m, m)

        # Due to quantization, exact equality is not expected
        # But results should be reasonably close
        diff = torch.abs(ABC1 - ABC2)
        relative_error = diff / (torch.abs(ABC1) + 1e-8)

        # Allow for some error due to ternary quantization
        assert relative_error.mean() < 0.5  # Less than 50% relative error on average
        assert relative_error.max() < 2.0   # Less than 200% max error

    @given(
        st.integers(min_value=2, max_value=32),
        st.integers(min_value=2, max_value=32)
    )
    @settings(max_examples=30)
    def test_distributivity_approximation(self, m, n):
        """Test approximate distributivity: A(B+C) ≈ AB + AC."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A = torch.randn(m, n, device=device, dtype=torch.float32)
        B = torch.randn(n, m, device=device, dtype=torch.float32)
        C = torch.randn(n, m, device=device, dtype=torch.float32)

        # Compute A(B+C)
        BC_sum = B + C
        ABC1 = ternary_matmul(A, BC_sum)

        # Compute AB + AC
        AB = ternary_matmul(A, B)
        AC = ternary_matmul(A, C)
        ABC2 = AB + AC

        # Check approximate equality
        diff = torch.abs(ABC1 - ABC2)
        relative_error = diff / (torch.abs(ABC2) + 1e-8)

        assert relative_error.mean() < 0.3
        assert relative_error.max() < 1.0

    def test_scalar_multiplication_consistency(self):
        """Test that scalar multiplication is consistent."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A = torch.randn(32, 32, device=device)
        B = torch.randn(32, 32, device=device)
        scalar = 3.14

        # (scalar * A) * B
        scaled_A = scalar * A
        result1 = ternary_matmul(scaled_A, B)

        # A * (scalar * B)
        scaled_B = scalar * B
        result2 = ternary_matmul(A, scaled_B)

        # scalar * (A * B)
        product = ternary_matmul(A, B)
        result3 = scalar * product

        # Results should be approximately equal
        diff12 = torch.abs(result1 - result2).mean()
        diff13 = torch.abs(result1 - result3).mean()
        diff23 = torch.abs(result2 - result3).mean()

        # Allow for quantization error
        tolerance = 1e-2
        assert diff12 < tolerance
        assert diff13 < tolerance
        assert diff23 < tolerance

    @given(st.floats(min_value=-10, max_value=10))
    @settings(max_examples=20)
    def test_homogeneity_property(self, scalar):
        """Test homogeneity: scalar * (A * B) ≈ (scalar * A) * B."""
        assume(abs(scalar) > 1e-6)  # Avoid division by zero issues

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A = torch.randn(16, 16, device=device)
        B = torch.randn(16, 16, device=device)

        # scalar * (A * B)
        product = ternary_matmul(A, B)
        result1 = scalar * product

        # (scalar * A) * B
        scaled_A = scalar * A
        result2 = ternary_matmul(scaled_A, B)

        diff = torch.abs(result1 - result2)
        relative_error = diff / (torch.abs(result1) + 1e-6)

        assert relative_error.mean() < 0.1


class TestNumericalStability:
    """Test numerical stability properties."""

    def test_gradient_flow_preservation(self):
        """Test that gradients flow properly through ternary operations."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create tensors with gradients
        A = torch.randn(32, 32, device=device, requires_grad=True)
        B = torch.randn(32, 32, device=device, requires_grad=True)

        # Forward pass
        C = ternary_matmul(A, B)
        loss = C.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are reasonable
        assert A.grad is not None
        assert B.grad is not None
        assert not torch.isnan(A.grad).any()
        assert not torch.isnan(B.grad).any()
        assert not torch.isinf(A.grad).any()
        assert not torch.isinf(B.grad).any()

        # Gradients should be non-zero (with high probability)
        assert A.grad.abs().sum() > 1e-6
        assert B.grad.abs().sum() > 1e-6

    def test_overflow_underflow_handling(self):
        """Test handling of overflow and underflow conditions."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test with very large values
        large_val = 1e20
        A = torch.full((16, 16), large_val, device=device)
        B = torch.full((16, 16), large_val, device=device)

        result = ternary_matmul(A, B)

        # Result should be finite (not inf or nan)
        assert torch.isfinite(result).all()

        # Test with very small values
        small_val = 1e-20
        A_small = torch.full((16, 16), small_val, device=device)
        B_small = torch.full((16, 16), small_val, device=device)

        result_small = ternary_matmul(A_small, B_small)

        # Result should be finite
        assert torch.isfinite(result_small).all()

    def test_precision_consistency(self):
        """Test that precision is consistent across similar operations."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test that similar matrices produce similar results
        base_A = torch.randn(32, 32, device=device)
        base_B = torch.randn(32, 32, device=device)

        # Slightly perturb matrices
        epsilon = 1e-6
        A_perturbed = base_A + epsilon * torch.randn_like(base_A)
        B_perturbed = base_B + epsilon * torch.randn_like(base_B)

        result_base = ternary_matmul(base_A, base_B)
        result_perturbed = ternary_matmul(A_perturbed, B_perturbed)

        # Results should be close (continuous function property)
        diff = torch.abs(result_base - result_perturbed)
        assert diff.mean() < epsilon * 10  # Allow some amplification due to quantization

    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_deterministic_behavior(self, seed):
        """Test that operations are deterministic for same inputs."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        A = torch.randn(32, 32, device=device)
        B = torch.randn(32, 32, device=device)

        # Run same operation multiple times
        results = []
        for _ in range(3):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            result = ternary_matmul(A, B)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], atol=1e-6)


class TestMemoryConsistency:
    """Test memory consistency and conservation."""

    def test_memory_conservation(self):
        """Test that memory usage is conserved across operations."""
        import psutil
        import gc

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        initial_memory = psutil.Process().memory_info().rss

        # Perform many operations
        for i in range(50):
            A = torch.randn(64, 64, device=device)
            B = torch.randn(64, 64, device=device)
            C = ternary_matmul(A, B)

            # Delete intermediate results
            del A, B, C

            if i % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB

    def test_tensor_lifecycle_consistency(self):
        """Test that tensors are properly managed throughout lifecycle."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create tensors
        A = torch.randn(128, 128, device=device)
        B = torch.randn(128, 128, device=device)

        # Get initial reference counts
        initial_ref_A = sys.getrefcount(A) if hasattr(sys, 'getrefcount') else 0
        initial_ref_B = sys.getrefcount(B) if hasattr(sys, 'getrefcount') else 0

        # Perform operation
        C = ternary_matmul(A, B)

        # Check that result is valid
        assert C.shape == (128, 128)
        assert C.device == A.device

        # Delete inputs
        del A, B

        # Result should still be valid
        assert C.shape == (128, 128)
        assert not torch.isnan(C).any()

        # Clean up
        del C

    def test_no_memory_corruption(self):
        """Test that operations don't cause memory corruption."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a pattern of values to detect corruption
        test_values = torch.arange(1024, device=device, dtype=torch.float32).reshape(32, 32)

        # Perform operations
        for i in range(10):
            result = ternary_matmul(test_values, test_values)

            # Result should be finite and reasonable
            assert torch.isfinite(result).all()
            assert result.shape == (32, 32)

            # For deterministic ternary quantization, results should be consistent
            if i > 0:
                # Results may vary slightly due to quantization, but should be reasonable
                assert torch.abs(result).max() < 1e10

    def test_resource_cleanup(self):
        """Test that system resources are properly cleaned up."""
        import psutil
        import gc

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get initial resource usage
        initial_cpu = psutil.cpu_percent(interval=0.1)
        initial_memory = psutil.Process().memory_info().rss

        # Perform intensive operations
        for i in range(20):
            size = 128 + i * 8  # Increasing sizes
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)
            C = ternary_matmul(A, B)
            del A, B, C

        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check resource usage after cleanup
        final_cpu = psutil.cpu_percent(interval=0.1)
        final_memory = psutil.Process().memory_info().rss

        # CPU should return to reasonable levels
        assert final_cpu < 50  # Less than 50% CPU usage

        # Memory should not have grown excessively
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB increase


class TestInvarianceProperties:
    """Test invariance properties of the system."""

    def test_shape_invariance(self):
        """Test that tensor shapes are preserved correctly."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        test_shapes = [
            (16, 16),
            (32, 64),
            (64, 32),
            (128, 128),
        ]

        for m, k in test_shapes:
            for k2, n in test_shapes:
                if k == k2:  # Compatible dimensions
                    A = torch.randn(m, k, device=device)
                    B = torch.randn(k, n, device=device)

                    C = ternary_matmul(A, B)

                    # Shape should be correct
                    assert C.shape == (m, n)

                    # Device should be preserved
                    assert C.device == A.device

    def test_dtype_preservation(self):
        """Test that data types are preserved when possible."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test with float32 (primary supported type)
        A = torch.randn(32, 32, dtype=torch.float32, device=device)
        B = torch.randn(32, 32, dtype=torch.float32, device=device)

        C = ternary_matmul(A, B)

        # Should preserve float32
        assert C.dtype == torch.float32

    def test_commutative_approximation(self):
        """Test approximate commutativity for special cases."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # For special matrices, test approximate commutativity
        # Using symmetric matrices
        A = torch.randn(32, 32, device=device)
        A_symmetric = (A + A.t()) / 2  # Make symmetric

        B = torch.randn(32, 32, device=device)
        B_symmetric = (B + B.t()) / 2  # Make symmetric

        AB = ternary_matmul(A_symmetric, B_symmetric)
        BA = ternary_matmul(B_symmetric, A_symmetric)

        # Should be approximately equal for symmetric matrices
        diff = torch.abs(AB - BA)
        relative_error = diff / (torch.abs(AB) + 1e-8)

        assert relative_error.mean() < 0.2  # Allow 20% error due to quantization

    def test_identity_preservation(self):
        """Test that identity matrices are handled correctly."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create identity matrix
        I = torch.eye(64, device=device)

        # Test I * A ≈ A
        A = torch.randn(64, 32, device=device)
        IA = ternary_matmul(I, A)

        diff = torch.abs(IA - A)
        relative_error = diff / (torch.abs(A) + 1e-8)

        assert relative_error.mean() < 0.1

        # Test A * I ≈ A
        B = torch.randn(32, 64, device=device)
        BI = ternary_matmul(B, I)

        diff = torch.abs(BI - B)
        relative_error = diff / (torch.abs(B) + 1e-8)

        assert relative_error.mean() < 0.1


class TestStatisticalProperties:
    """Test statistical properties of ternary operations."""

    def test_output_distribution_reasonableness(self):
        """Test that output distributions are reasonable."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Generate many random operations
        results = []

        for _ in range(20):
            A = torch.randn(64, 64, device=device)
            B = torch.randn(64, 64, device=device)
            C = ternary_matmul(A, B)
            results.append(C)

        # Concatenate all results
        all_results = torch.cat([r.flatten() for r in results])

        # Basic statistical checks
        mean = all_results.mean().item()
        std = all_results.std().item()
        skewness = torch.mean(((all_results - mean) / std) ** 3).item()

        # Mean should be close to zero (random inputs)
        assert abs(mean) < 1.0

        # Std should be reasonable
        assert 0.1 < std < 10.0

        # Should not be extremely skewed
        assert abs(skewness) < 2.0

    def test_quantization_effect_consistency(self):
        """Test that quantization effects are consistent."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Test that similar inputs produce similar quantized outputs
        base_A = torch.randn(32, 32, device=device)
        base_B = torch.randn(32, 32, device=device)

        # Create slightly different versions
        variations = []
        for i in range(5):
            noise_level = 0.01 * (i + 1)
            A_var = base_A + noise_level * torch.randn_like(base_A)
            B_var = base_B + noise_level * torch.randn_like(base_B)
            result = ternary_matmul(A_var, B_var)
            variations.append(result)

        # Variations should be correlated
        base_result = ternary_matmul(base_A, base_B)

        for var_result in variations:
            correlation = torch.corrcoef(torch.stack([base_result.flatten(), var_result.flatten()]))[0, 1]
            assert correlation > 0.5  # Should be reasonably correlated
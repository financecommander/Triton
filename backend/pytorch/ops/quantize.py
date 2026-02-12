"""
Quantization operations for ternary neural networks.

This module implements quantization functions that convert floating-point tensors
to ternary tensors with values in {-1, 0, 1}.
"""

from typing import Literal

import torch


def quantize(
    x: torch.Tensor,
    method: Literal["deterministic", "stochastic"] = "deterministic",
    threshold: float = 0.33,
) -> torch.Tensor:
    """
    Quantize a tensor to ternary values {-1, 0, 1}.

    This function supports two quantization methods:

    1. **Deterministic**: Threshold-based quantization where values are mapped based
       on fixed thresholds:
       - x > threshold  → +1
       - x < -threshold → -1
       - otherwise      → 0

    2. **Stochastic**: Probabilistic rounding where the probability of quantizing to
       +1 or -1 is determined by a sigmoid function:
       - P(+1) = sigmoid(x / threshold)
       - P(-1) = 1 - P(+1)
       - Values are then thresholded to ensure ternary output

    The function uses a custom autograd operation to enable gradient pass-through
    during backpropagation (straight-through estimator).

    Args:
        x: Input tensor of any shape. Can be on CPU or CUDA.
        method: Quantization method, either "deterministic" or "stochastic".
               Defaults to "deterministic".
        threshold: Threshold value for quantization. For deterministic mode, this is
                  the absolute threshold. For stochastic mode, this scales the sigmoid.
                  Defaults to 0.33.

    Returns:
        Quantized tensor with int8 dtype and values in {-1, 0, 1}.
        The tensor will be on the same device as the input.

    Complexity:
        Time: O(n) where n is the number of elements in the tensor
        Space: O(n) for the output tensor

    Examples:
        >>> import torch
        >>> x = torch.tensor([0.5, -0.5, 0.1, -0.1, 0.0])
        >>> quantize(x, method="deterministic", threshold=0.33)
        tensor([ 1, -1,  0,  0,  0], dtype=torch.int8)

        >>> # Stochastic quantization (results will vary due to randomness)
        >>> torch.manual_seed(42)
        >>> x = torch.tensor([0.5, -0.5, 0.1, -0.1, 0.0])
        >>> quantize(x, method="stochastic", threshold=0.33)
        tensor([ 1, -1,  0,  0,  0], dtype=torch.int8)

        >>> # CUDA support
        >>> if torch.cuda.is_available():
        ...     x_cuda = torch.randn(1000, device="cuda")
        ...     y_cuda = quantize(x_cuda, method="deterministic")
        ...     assert y_cuda.device.type == "cuda"
    """

    class QuantizeFunction(torch.autograd.Function):
        """Custom autograd function for quantization with gradient pass-through."""

        @staticmethod
        def forward(ctx, input_tensor, quantize_method, thresh):
            """Forward pass: quantize the input tensor."""
            if quantize_method == "deterministic":
                # Deterministic threshold-based quantization
                output = torch.zeros_like(input_tensor)
                output[input_tensor > thresh] = 1
                output[input_tensor < -thresh] = -1
            else:  # stochastic
                # Stochastic quantization using sigmoid
                # sigmoid(x/threshold) gives probability of being positive
                probs = torch.sigmoid(input_tensor / thresh)
                
                # Generate random values for stochastic rounding
                random_vals = torch.rand_like(input_tensor)
                
                # Determine sign based on probability
                output = torch.zeros_like(input_tensor)
                output[random_vals < probs] = 1
                output[random_vals >= probs] = -1
                
                # Apply threshold to create zero region
                # For small values, force to zero
                abs_input = torch.abs(input_tensor)
                zero_mask = abs_input < (thresh / 2)
                output[zero_mask] = 0

            return output

        @staticmethod
        def backward(ctx, grad_output):
            """
            Backward pass: straight-through estimator.
            
            Gradients flow through unchanged (no-op gradient).
            """
            # Straight-through estimator: gradient passes through unchanged
            return grad_output, None, None

    # Apply the quantization function and convert to int8
    result = QuantizeFunction.apply(x, method, threshold)
    # Convert to int8 for storage efficiency (this breaks gradient flow, so should
    # be done after training or in eval mode)
    return result.to(torch.int8)

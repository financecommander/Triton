"""
Activation functions for ternary neural networks.

This module implements custom activation functions with straight-through estimators
for use in ternary neural network training.
"""

import torch


class TernaryActivationFunction(torch.autograd.Function):
    """
    Custom autograd function for ternary activation with Straight-Through Estimator.
    
    This implements a hard-tanh activation that clips values to [-1, 1] in the forward
    pass, while allowing gradients to flow through in the backward pass using the
    Straight-Through Estimator (STE) technique.
    """

    @staticmethod
    def forward(ctx, input_tensor):
        """
        Forward pass: clip input to [-1, 1].
        
        Args:
            ctx: Context object for saving information for backward pass.
            input_tensor: Input tensor of any shape.
            
        Returns:
            Clipped tensor with values in [-1, 1].
            
        Complexity:
            Time: O(n) where n is the number of elements
            Space: O(n) for storing the mask
        """
        # Save mask for backward pass: True where |x| <= 1
        mask = (input_tensor.abs() <= 1.0)
        ctx.save_for_backward(mask)
        
        # Clip to [-1, 1]
        return torch.clamp(input_tensor, -1.0, 1.0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: STE with selective gradient flow.
        
        Gradients flow through if |x| <= 1, otherwise gradients are zero.
        This implements the Straight-Through Estimator technique.
        
        Args:
            ctx: Context object with saved forward pass information.
            grad_output: Gradient from the next layer.
            
        Returns:
            Gradient for the input tensor.
            
        Complexity:
            Time: O(n) where n is the number of elements
            Space: O(n) for the output gradient
        """
        mask, = ctx.saved_tensors
        
        # Apply mask: gradient flows through only where |x| <= 1
        grad_input = grad_output * mask.float()
        
        return grad_input


def ternary_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply ternary activation function with Straight-Through Estimator.
    
    This function clips the input tensor to the range [-1, 1] in the forward pass,
    which is suitable for ternary neural networks. During backpropagation, it uses
    a Straight-Through Estimator (STE) that allows gradients to flow through when
    |x| <= 1, but blocks gradients for values outside this range.
    
    The activation function can be expressed as:
        forward:  f(x) = clip(x, -1, 1) = max(-1, min(1, x))
        backward: ∂f/∂x = { 1 if |x| <= 1
                          { 0 otherwise
    
    Args:
        x: Input tensor of any shape. Can be on CPU or CUDA.
        
    Returns:
        Activated tensor with values in [-1, 1], same shape and device as input.
        
    Complexity:
        Time: O(n) where n is the number of elements in the tensor
        Space: O(n) for the output tensor and gradient mask
        
    Examples:
        >>> import torch
        >>> x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
        >>> ternary_activation(x)
        tensor([-1.0000, -0.5000,  0.0000,  0.5000,  1.0000])
        
        >>> # Gradient behavior
        >>> x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
        >>> y = ternary_activation(x)
        >>> y.sum().backward()
        >>> x.grad  # Gradients are zero for |x| > 1
        tensor([0., 1., 1., 1., 0.])
        
        >>> # CUDA support
        >>> if torch.cuda.is_available():
        ...     x_cuda = torch.randn(1000, device="cuda", requires_grad=True)
        ...     y_cuda = ternary_activation(x_cuda)
        ...     assert y_cuda.device.type == "cuda"
        ...     y_cuda.sum().backward()
        ...     assert x_cuda.grad is not None
        
    Notes:
        - This activation is differentiable and can be used in gradient-based optimization.
        - The STE technique is crucial for training ternary networks as it provides
          useful gradients even when the forward pass involves non-differentiable operations.
        - This is similar to hard-tanh but with custom gradient behavior optimized for
          ternary neural networks.
    """
    return TernaryActivationFunction.apply(x)

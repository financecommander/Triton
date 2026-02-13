"""
ONNX Export Utilities for Ternary Neural Networks

Provides comprehensive ONNX export functionality for ternary models including
ResNet-18, MobileNetV2, and other architectures with ternary quantization.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import warnings


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...],
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verbose: bool = True,
) -> bool:
    """
    Export a ternary model to ONNX format.
    
    This function handles the export of models with custom ternary quantization
    operations. Note that ONNX may not fully support all custom autograd functions,
    so the exported model may use approximations.
    
    Args:
        model: PyTorch model to export (should be in eval mode)
        output_path: Path to save ONNX model
        input_shape: Shape of input tensor (e.g., (1, 3, 224, 224) for images)
        input_names: Names for input tensors (default: ['input'])
        output_names: Names for output tensors (default: ['output'])
        opset_version: ONNX opset version (default: 13)
        dynamic_axes: Dynamic axes for inputs/outputs for variable batch size
        verbose: Print export information
        
    Returns:
        True if export succeeded, False otherwise
        
    Examples:
        >>> from models.resnet18.ternary_resnet18 import ternary_resnet18
        >>> model = ternary_resnet18(num_classes=10)
        >>> model.eval()
        >>> export_to_onnx(
        ...     model, 
        ...     Path("resnet18.onnx"),
        ...     input_shape=(1, 3, 32, 32)
        ... )
        True
    """
    # Set default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Set default dynamic axes for batch size
    if dynamic_axes is None:
        dynamic_axes = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Move to same device as model
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        if verbose:
            print(f"Exporting model to ONNX format...")
            print(f"  Input shape: {input_shape}")
            print(f"  Opset version: {opset_version}")
            print(f"  Output: {output_path}")
        
        # Suppress warnings about custom operations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        if verbose:
            print(f"✓ ONNX model exported successfully to {output_path}")
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ ONNX export failed: {e}")
            print("Note: ONNX may not fully support custom ternary quantization operations.")
            print("The exported model may use approximations or fallback implementations.")
        return False


def validate_onnx_model(
    onnx_path: Path,
    pytorch_model: nn.Module,
    input_shape: Tuple[int, ...],
    tolerance: float = 1e-3,
    verbose: bool = True
) -> bool:
    """
    Validate ONNX model against PyTorch model.
    
    Compares outputs from ONNX and PyTorch models to ensure export correctness.
    Requires onnxruntime to be installed.
    
    Args:
        onnx_path: Path to ONNX model file
        pytorch_model: Original PyTorch model
        input_shape: Shape of test input
        tolerance: Maximum allowed difference between outputs
        verbose: Print validation information
        
    Returns:
        True if validation passes, False otherwise
        
    Examples:
        >>> validate_onnx_model(
        ...     Path("resnet18.onnx"),
        ...     pytorch_model,
        ...     input_shape=(1, 3, 32, 32)
        ... )
        True
    """
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        if verbose:
            print("✗ onnxruntime not installed. Cannot validate ONNX model.")
            print("  Install with: pip install onnxruntime")
        return False
    
    try:
        # Ensure model is in eval mode
        pytorch_model.eval()
        
        # Create test input
        test_input = torch.randn(*input_shape)
        device = next(pytorch_model.parameters()).device
        test_input_device = test_input.to(device)
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input_device).cpu().numpy()
        
        # Get ONNX output
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        if verbose:
            print(f"\nONNX Model Validation:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Tolerance: {tolerance:.6f}")
        
        if max_diff <= tolerance:
            if verbose:
                print(f"✓ Validation passed (max_diff={max_diff:.6f} <= {tolerance:.6f})")
            return True
        else:
            if verbose:
                print(f"✗ Validation failed (max_diff={max_diff:.6f} > {tolerance:.6f})")
                print("  Note: Some differences may be due to custom quantization operations")
            return False
            
    except Exception as e:
        if verbose:
            print(f"✗ Validation error: {e}")
        return False


def optimize_onnx_model(
    input_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True
) -> bool:
    """
    Optimize ONNX model for inference.
    
    Applies various optimizations like constant folding, operator fusion,
    and graph simplification. Requires onnx and onnxruntime to be installed.
    
    Note: This function uses generic ONNX optimization which may not be 
    optimal for all model types. For production use, consider model-specific
    optimization tools.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model (default: overwrite input)
        verbose: Print optimization information
        
    Returns:
        True if optimization succeeded, False otherwise
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
    except ImportError:
        if verbose:
            print("✗ onnx or onnxruntime not installed. Cannot optimize.")
            print("  Install with: pip install onnx onnxruntime")
        return False
    
    try:
        if output_path is None:
            output_path = input_path
            
        if verbose:
            print(f"Optimizing ONNX model...")
            print(f"  Input: {input_path}")
            print(f"  Output: {output_path}")
        
        # Load model
        model = onnx.load(str(input_path))
        
        # Apply generic optimizations
        # Note: Using 'bert' model_type applies general-purpose optimizations
        # that work across different architectures
        optimized_model = optimizer.optimize_model(
            str(input_path),
            model_type='bert',  # Generic optimization (not model-specific)
            num_heads=0,
            hidden_size=0,
        )
        
        # Save optimized model
        optimized_model.save_model_to_file(str(output_path))
        
        if verbose:
            print(f"✓ Model optimized and saved to {output_path}")
            
        return True
        
    except Exception as e:
        if verbose:
            print(f"✗ Optimization failed: {e}")
        return False


def export_model_with_metadata(
    model: nn.Module,
    output_dir: Path,
    model_name: str,
    input_shape: Tuple[int, ...],
    metadata: Optional[Dict[str, Any]] = None,
    validate: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Export model to ONNX with metadata and validation.
    
    Creates a complete export package including the ONNX model,
    metadata JSON, and validation report.
    
    Args:
        model: PyTorch model to export
        output_dir: Directory to save export files
        model_name: Name for the model (used in filenames)
        input_shape: Shape of input tensor
        metadata: Additional metadata to save (model info, training stats, etc.)
        validate: Whether to validate exported model
        verbose: Print export information
        
    Returns:
        Dictionary with paths and export status
        
    Examples:
        >>> export_model_with_metadata(
        ...     model,
        ...     Path("exports/"),
        ...     "ternary_resnet18_cifar10",
        ...     input_shape=(1, 3, 32, 32),
        ...     metadata={"accuracy": 0.89, "dataset": "cifar10"}
        ... )
        {'onnx_path': ..., 'success': True, ...}
    """
    import json
    from datetime import datetime
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / f"{model_name}.onnx"
    metadata_path = output_dir / f"{model_name}_metadata.json"
    
    # Export to ONNX
    export_success = export_to_onnx(
        model, onnx_path, input_shape, verbose=verbose
    )
    
    if not export_success:
        return {
            'success': False,
            'onnx_path': None,
            'metadata_path': None,
            'validation_passed': False
        }
    
    # Validate if requested
    validation_passed = False
    if validate:
        validation_passed = validate_onnx_model(
            onnx_path, model, input_shape, verbose=verbose
        )
    
    # Prepare metadata
    export_metadata = {
        'model_name': model_name,
        'input_shape': list(input_shape),
        'export_date': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'onnx_opset_version': 13,
        'validated': validation_passed,
    }
    
    if metadata:
        export_metadata.update(metadata)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(export_metadata, f, indent=2)
    
    if verbose:
        print(f"\n✓ Export complete!")
        print(f"  ONNX model: {onnx_path}")
        print(f"  Metadata: {metadata_path}")
        print(f"  Validation: {'✓ Passed' if validation_passed else '✗ Failed/Skipped'}")
    
    return {
        'success': True,
        'onnx_path': str(onnx_path),
        'metadata_path': str(metadata_path),
        'validation_passed': validation_passed,
        'export_metadata': export_metadata
    }

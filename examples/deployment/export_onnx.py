#!/usr/bin/env python3
"""
ONNX Export and Optimization Script

This script demonstrates comprehensive ONNX model export with:
- Model export to ONNX format with proper configuration
- ONNX model validation and integrity checking
- ONNX model simplification for optimization
- Performance benchmarking and comparison
- Support for both full precision and quantized models

Usage:
    python export_onnx.py --model resnet18 --output model.onnx
    python export_onnx.py --model mobilenetv2 --simplify --benchmark
    python export_onnx.py --checkpoint path/to/checkpoint.pt --output custom.onnx
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import torch
import torch.nn as nn
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import onnx
    import onnxruntime as ort
    from onnx import checker, helper, shape_inference
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX/ONNXRuntime not available. Install with: pip install onnx onnxruntime")

try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False
    logger.warning("onnx-simplifier not available. Install with: pip install onnx-simplifier")

from backend.pytorch.ternary_tensor import TernaryLinear, TernaryConv2d


# ============================================================================
# Example Ternary Models
# ============================================================================

class SimpleTernaryNet(nn.Module):
    """Simple ternary CNN for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = TernaryConv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = TernaryLinear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = TernaryLinear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TernaryResNet18(nn.Module):
    """Simplified Ternary ResNet-18 for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = TernaryConv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = TernaryLinear(128, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, 
                    blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(TernaryConv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(TernaryConv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================================
# ONNX Export Functions
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    verbose: bool = False
) -> bool:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Shape of input tensor (batch_size, channels, height, width)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes configuration
        verbose: Whether to print verbose output
        
    Returns:
        True if export successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available. Cannot export model.")
        return False
    
    try:
        logger.info(f"Exporting model to ONNX: {output_path}")
        logger.info(f"Input shape: {input_shape}, Opset version: {opset_version}")
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Set model to eval mode
        model.eval()
        
        # Default dynamic axes for batch dimension
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=verbose
            )
        
        logger.info(f"✓ Model exported successfully to {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to export model: {e}")
        return False


def validate_onnx_model(
    onnx_path: Path,
    pytorch_model: Optional[nn.Module] = None,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    tolerance: float = 1e-5
) -> bool:
    """
    Validate ONNX model for correctness.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model for comparison
        input_shape: Shape of input tensor
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if validation successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available. Cannot validate model.")
        return False
    
    try:
        logger.info(f"Validating ONNX model: {onnx_path}")
        
        # Load and check ONNX model
        onnx_model = onnx.load(str(onnx_path))
        
        # Check model structure
        logger.info("Checking model structure...")
        checker.check_model(onnx_model)
        logger.info("✓ Model structure is valid")
        
        # Infer shapes
        logger.info("Inferring shapes...")
        onnx_model = shape_inference.infer_shapes(onnx_model)
        logger.info("✓ Shape inference successful")
        
        # Print model info
        graph = onnx_model.graph
        logger.info(f"  Inputs: {[i.name for i in graph.input]}")
        logger.info(f"  Outputs: {[o.name for o in graph.output]}")
        logger.info(f"  Nodes: {len(graph.node)}")
        
        # Test inference with ONNXRuntime
        logger.info("Testing ONNX Runtime inference...")
        ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        logger.info(f"✓ ONNX Runtime inference successful")
        logger.info(f"  Output shape: {ort_output.shape}")
        
        # Compare with PyTorch if model provided
        if pytorch_model is not None:
            logger.info("Comparing ONNX output with PyTorch...")
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_output = pytorch_model(
                    torch.from_numpy(test_input)
                ).numpy()
            
            # Check numerical difference
            max_diff = np.max(np.abs(pytorch_output - ort_output))
            mean_diff = np.mean(np.abs(pytorch_output - ort_output))
            
            logger.info(f"  Max difference: {max_diff:.2e}")
            logger.info(f"  Mean difference: {mean_diff:.2e}")
            
            if max_diff > tolerance:
                logger.warning(
                    f"⚠ Large numerical difference detected (max: {max_diff:.2e})"
                )
                return False
            
            logger.info("✓ Outputs match within tolerance")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Validation failed: {e}")
        return False


def simplify_onnx_model(
    input_path: Path,
    output_path: Optional[Path] = None,
    check_n: int = 3
) -> bool:
    """
    Simplify ONNX model using onnx-simplifier.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save simplified model (default: overwrite input)
        check_n: Number of test runs for verification
        
    Returns:
        True if simplification successful, False otherwise
    """
    if not ONNXSIM_AVAILABLE:
        logger.error("onnx-simplifier not available. Cannot simplify model.")
        return False
    
    try:
        logger.info(f"Simplifying ONNX model: {input_path}")
        
        if output_path is None:
            output_path = input_path
        
        # Load model
        onnx_model = onnx.load(str(input_path))
        original_nodes = len(onnx_model.graph.node)
        
        # Simplify
        simplified_model, check = onnxsim.simplify(
            onnx_model,
            check_n=check_n
        )
        
        if not check:
            logger.error("✗ Simplified model produces different outputs")
            return False
        
        # Save simplified model
        onnx.save(simplified_model, str(output_path))
        
        simplified_nodes = len(simplified_model.graph.node)
        reduction = (1 - simplified_nodes / original_nodes) * 100
        
        logger.info(f"✓ Model simplified successfully")
        logger.info(f"  Original nodes: {original_nodes}")
        logger.info(f"  Simplified nodes: {simplified_nodes}")
        logger.info(f"  Reduction: {reduction:.1f}%")
        
        # Compare file sizes
        if output_path != input_path:
            original_size = input_path.stat().st_size / 1024 / 1024
            simplified_size = output_path.stat().st_size / 1024 / 1024
            size_reduction = (1 - simplified_size / original_size) * 100
            logger.info(f"  Original size: {original_size:.2f} MB")
            logger.info(f"  Simplified size: {simplified_size:.2f} MB")
            logger.info(f"  Size reduction: {size_reduction:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Simplification failed: {e}")
        return False


def benchmark_onnx_model(
    onnx_path: Path,
    pytorch_model: Optional[nn.Module] = None,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, Any]:
    """
    Benchmark ONNX model performance.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model for comparison
        input_shape: Shape of input tensor
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with benchmark results
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX not available. Cannot benchmark model.")
        return {}
    
    results = {}
    
    try:
        logger.info(f"Benchmarking ONNX model: {onnx_path}")
        logger.info(f"Runs: {num_runs}, Warmup: {warmup_runs}")
        
        # Setup ONNX Runtime
        ort_session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        
        # Create test input
        test_input = np.random.randn(*input_shape).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        
        # Warmup
        logger.info("Running warmup...")
        for _ in range(warmup_runs):
            _ = ort_session.run(None, ort_inputs)
        
        # Benchmark ONNX
        logger.info("Benchmarking ONNX Runtime...")
        onnx_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = ort_session.run(None, ort_inputs)
            end = time.perf_counter()
            onnx_times.append((end - start) * 1000)  # Convert to ms
        
        results['onnx'] = {
            'mean_ms': np.mean(onnx_times),
            'std_ms': np.std(onnx_times),
            'min_ms': np.min(onnx_times),
            'max_ms': np.max(onnx_times),
            'throughput_fps': 1000.0 / np.mean(onnx_times)
        }
        
        logger.info(f"✓ ONNX Runtime:")
        logger.info(f"  Mean: {results['onnx']['mean_ms']:.2f} ± {results['onnx']['std_ms']:.2f} ms")
        logger.info(f"  Min: {results['onnx']['min_ms']:.2f} ms, Max: {results['onnx']['max_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['onnx']['throughput_fps']:.2f} FPS")
        
        # Benchmark PyTorch if provided
        if pytorch_model is not None:
            logger.info("Benchmarking PyTorch...")
            pytorch_model.eval()
            torch_input = torch.from_numpy(test_input)
            
            # Warmup
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = pytorch_model(torch_input)
            
            # Benchmark
            pytorch_times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = pytorch_model(torch_input)
                    end = time.perf_counter()
                    pytorch_times.append((end - start) * 1000)
            
            results['pytorch'] = {
                'mean_ms': np.mean(pytorch_times),
                'std_ms': np.std(pytorch_times),
                'min_ms': np.min(pytorch_times),
                'max_ms': np.max(pytorch_times),
                'throughput_fps': 1000.0 / np.mean(pytorch_times)
            }
            
            logger.info(f"✓ PyTorch:")
            logger.info(f"  Mean: {results['pytorch']['mean_ms']:.2f} ± {results['pytorch']['std_ms']:.2f} ms")
            logger.info(f"  Min: {results['pytorch']['min_ms']:.2f} ms, Max: {results['pytorch']['max_ms']:.2f} ms")
            logger.info(f"  Throughput: {results['pytorch']['throughput_fps']:.2f} FPS")
            
            # Calculate speedup
            speedup = results['pytorch']['mean_ms'] / results['onnx']['mean_ms']
            results['speedup'] = speedup
            
            if speedup > 1:
                logger.info(f"✓ ONNX is {speedup:.2f}x faster than PyTorch")
            else:
                logger.info(f"⚠ PyTorch is {1/speedup:.2f}x faster than ONNX")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Benchmarking failed: {e}")
        return {}


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export PyTorch models to ONNX format with validation and benchmarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export simple ternary network
  python export_onnx.py --model simple --output simple_ternary.onnx
  
  # Export with simplification and benchmarking
  python export_onnx.py --model resnet18 --simplify --benchmark
  
  # Export from checkpoint
  python export_onnx.py --checkpoint model.pt --output model.onnx
  
  # Export with custom input shape and opset
  python export_onnx.py --model simple --input-size 1 3 224 224 --opset 14
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['simple', 'resnet18'],
        default='simple',
        help='Model architecture to export'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to model checkpoint (overrides --model)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('model.onnx'),
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        metavar=('B', 'C', 'H', 'W'),
        help='Input shape (batch, channels, height, width)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=13,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model after export'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark model performance'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not ONNX_AVAILABLE:
        logger.error("ONNX/ONNXRuntime not installed. Install with:")
        logger.error("  pip install onnx onnxruntime")
        return 1
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create model
    if args.checkpoint:
        logger.info(f"Loading model from checkpoint: {args.checkpoint}")
        model = torch.load(args.checkpoint)
    else:
        logger.info(f"Creating {args.model} model...")
        if args.model == 'simple':
            model = SimpleTernaryNet(num_classes=10)
        elif args.model == 'resnet18':
            model = TernaryResNet18(num_classes=10)
        else:
            logger.error(f"Unknown model: {args.model}")
            return 1
    
    model.eval()
    logger.info(f"✓ Model loaded successfully")
    
    # Export to ONNX
    input_shape = tuple(args.input_size)
    success = export_to_onnx(
        model=model,
        output_path=args.output,
        input_shape=input_shape,
        opset_version=args.opset,
        verbose=args.verbose
    )
    
    if not success:
        logger.error("Export failed")
        return 1
    
    # Validate ONNX model
    logger.info("\n" + "="*70)
    success = validate_onnx_model(
        onnx_path=args.output,
        pytorch_model=model,
        input_shape=input_shape
    )
    
    if not success:
        logger.error("Validation failed")
        return 1
    
    # Simplify if requested
    if args.simplify:
        logger.info("\n" + "="*70)
        simplified_path = args.output.parent / f"{args.output.stem}_simplified.onnx"
        success = simplify_onnx_model(
            input_path=args.output,
            output_path=simplified_path
        )
        
        if success:
            # Validate simplified model
            success = validate_onnx_model(
                onnx_path=simplified_path,
                pytorch_model=model,
                input_shape=input_shape
            )
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("\n" + "="*70)
        results = benchmark_onnx_model(
            onnx_path=args.output,
            pytorch_model=model,
            input_shape=input_shape,
            num_runs=args.num_runs
        )
        
        if args.simplify and success:
            logger.info("\n" + "="*70)
            logger.info("Benchmarking simplified model...")
            simplified_results = benchmark_onnx_model(
                onnx_path=simplified_path,
                pytorch_model=None,
                input_shape=input_shape,
                num_runs=args.num_runs
            )
    
    logger.info("\n" + "="*70)
    logger.info("✓ Export completed successfully!")
    logger.info(f"  Output: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

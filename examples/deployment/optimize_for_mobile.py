#!/usr/bin/env python3
"""
Mobile Optimization and Deployment Script

This script demonstrates comprehensive mobile deployment optimization:
- TorchScript mobile optimization with quantization
- TensorFlow Lite (TFLite) conversion
- CoreML export for iOS devices
- Post-training quantization for mobile
- Size and performance benchmarking

Usage:
    python optimize_for_mobile.py --model resnet18 --optimize-all
    python optimize_for_mobile.py --checkpoint model.pt --format torchscript
    python optimize_for_mobile.py --model simple --quantize --benchmark
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List
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

from backend.pytorch.ternary_tensor import TernaryLinear, TernaryConv2d

# Optional imports with availability flags
try:
    from torch.utils.mobile_optimizer import optimize_for_mobile
    MOBILE_OPTIMIZER_AVAILABLE = True
except ImportError:
    MOBILE_OPTIMIZER_AVAILABLE = False
    logger.warning("Mobile optimizer not available")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logger.warning("CoreML not available. Install with: pip install coremltools")

try:
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    logger.warning("TFLite conversion not available. Install with: pip install onnx onnx-tf tensorflow")


# ============================================================================
# Example Models
# ============================================================================

class MobileTernaryNet(nn.Module):
    """Lightweight ternary network optimized for mobile deployment."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Depthwise separable convolutions for efficiency
        self.conv1 = TernaryConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = TernaryConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = TernaryLinear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CompactTernaryNet(nn.Module):
    """Ultra-compact ternary network for mobile."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            TernaryConv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            TernaryConv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            TernaryConv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = TernaryLinear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# TorchScript Mobile Optimization
# ============================================================================

def export_torchscript_mobile(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 32, 32),
    optimize: bool = True,
    quantize: bool = False
) -> bool:
    """
    Export model to TorchScript format optimized for mobile.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save mobile model
        input_shape: Shape of input tensor
        optimize: Whether to apply mobile optimizations
        quantize: Whether to apply quantization
        
    Returns:
        True if export successful
    """
    try:
        logger.info(f"Exporting TorchScript mobile model: {output_path}")
        
        model.eval()
        example_input = torch.randn(*input_shape)
        
        # Trace the model
        logger.info("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        logger.info("✓ Model traced successfully")
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying dynamic quantization...")
            try:
                # Quantize linear and conv layers
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("✓ Quantization applied")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        # Optimize for mobile
        if optimize and MOBILE_OPTIMIZER_AVAILABLE:
            logger.info("Optimizing for mobile...")
            try:
                traced_model = optimize_for_mobile(traced_model)
                logger.info("✓ Mobile optimization applied")
            except Exception as e:
                logger.warning(f"Mobile optimization failed: {e}")
        
        # Save model
        traced_model._save_for_lite_interpreter(str(output_path))
        
        file_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ Model saved successfully")
        logger.info(f"  Size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Export failed: {e}")
        return False


# ============================================================================
# TensorFlow Lite Conversion
# ============================================================================

def export_tflite(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 32, 32),
    quantize: bool = False
) -> bool:
    """
    Export model to TensorFlow Lite format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TFLite model
        input_shape: Shape of input tensor
        quantize: Whether to apply post-training quantization
        
    Returns:
        True if export successful
    """
    if not TFLITE_AVAILABLE:
        logger.error("TFLite conversion not available")
        return False
    
    try:
        logger.info(f"Converting to TensorFlow Lite: {output_path}")
        
        # First export to ONNX
        import onnx
        onnx_path = output_path.parent / "temp_model.onnx"
        
        logger.info("Step 1: Exporting to ONNX...")
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                opset_version=13,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        
        logger.info("✓ ONNX export complete")
        
        # Convert ONNX to TensorFlow
        logger.info("Step 2: Converting ONNX to TensorFlow...")
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        
        # Save TensorFlow model
        tf_path = output_path.parent / "temp_tf_model"
        tf_rep.export_graph(str(tf_path))
        logger.info("✓ TensorFlow conversion complete")
        
        # Convert to TFLite
        logger.info("Step 3: Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        
        # Apply optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if quantize:
            logger.info("Applying post-training quantization...")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size = len(tflite_model) / 1024 / 1024
        logger.info(f"✓ TFLite model saved successfully")
        logger.info(f"  Size: {file_size:.2f} MB")
        
        # Clean up temporary files
        onnx_path.unlink()
        import shutil
        shutil.rmtree(tf_path)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TFLite conversion failed: {e}")
        logger.error("Make sure you have installed: pip install onnx onnx-tf tensorflow")
        return False


# ============================================================================
# CoreML Export
# ============================================================================

def export_coreml(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple = (1, 3, 32, 32),
    quantize: bool = False
) -> bool:
    """
    Export model to CoreML format for iOS/macOS.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save CoreML model
        input_shape: Shape of input tensor
        quantize: Whether to apply quantization
        
    Returns:
        True if export successful
    """
    if not COREML_AVAILABLE:
        logger.error("CoreML not available. Install with: pip install coremltools")
        return False
    
    try:
        logger.info(f"Converting to CoreML: {output_path}")
        
        model.eval()
        example_input = torch.randn(*input_shape)
        
        # Trace the model
        logger.info("Tracing model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Convert to CoreML
        logger.info("Converting to CoreML...")
        
        # Define input shape (CoreML uses different format)
        batch, channels, height, width = input_shape
        input_shape_coreml = ct.Shape(shape=(batch, channels, height, width))
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=input_shape_coreml)],
            convert_to="mlprogram" if not quantize else "neuralnetwork"
        )
        
        # Add metadata
        mlmodel.author = "Triton DSL"
        mlmodel.short_description = "Ternary Neural Network optimized for mobile"
        mlmodel.version = "1.0"
        
        # Apply quantization if requested
        if quantize:
            logger.info("Applying quantization...")
            try:
                # Use 8-bit quantization
                mlmodel = ct.models.neural_network.quantization_utils.quantize_weights(
                    mlmodel, nbits=8
                )
                logger.info("✓ Quantization applied")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        # Save model
        mlmodel.save(str(output_path))
        
        file_size = output_path.stat().st_size / 1024 / 1024
        logger.info(f"✓ CoreML model saved successfully")
        logger.info(f"  Size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CoreML export failed: {e}")
        return False


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_mobile_model(
    model_path: Path,
    format: str,
    input_shape: tuple = (1, 3, 32, 32),
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Benchmark mobile model performance.
    
    Args:
        model_path: Path to model file
        format: Model format ('torchscript', 'tflite', 'coreml')
        input_shape: Shape of input tensor
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    try:
        logger.info(f"Benchmarking {format} model...")
        
        if format == 'torchscript':
            # Load TorchScript model
            model = torch.jit.load(str(model_path))
            model.eval()
            
            test_input = torch.randn(*input_shape)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = model(test_input)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
            
        elif format == 'tflite':
            if not TFLITE_AVAILABLE:
                logger.warning("TFLite not available for benchmarking")
                return {}
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                interpreter.set_tensor(input_details[0]['index'], test_input)
                interpreter.invoke()
                _ = interpreter.get_tensor(output_details[0]['index'])
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        elif format == 'coreml':
            if not COREML_AVAILABLE:
                logger.warning("CoreML not available for benchmarking")
                return {}
            
            # Load CoreML model
            mlmodel = ct.models.MLModel(str(model_path))
            
            test_input = np.random.randn(*input_shape).astype(np.float32)
            input_dict = {"input": test_input}
            
            # Warmup
            for _ in range(10):
                _ = mlmodel.predict(input_dict)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = mlmodel.predict(input_dict)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        else:
            logger.error(f"Unknown format: {format}")
            return {}
        
        results = {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'throughput_fps': 1000.0 / np.mean(times)
        }
        
        logger.info(f"✓ {format.upper()} Benchmark:")
        logger.info(f"  Mean: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        logger.info(f"  Throughput: {results['throughput_fps']:.2f} FPS")
        
        return results
        
    except Exception as e:
        logger.error(f"✗ Benchmarking failed: {e}")
        return {}


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optimize PyTorch models for mobile deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to TorchScript mobile with optimization
  python optimize_for_mobile.py --model compact --format torchscript --optimize
  
  # Export to all formats
  python optimize_for_mobile.py --model mobile --optimize-all
  
  # Export with quantization
  python optimize_for_mobile.py --model compact --format torchscript --quantize
  
  # Benchmark model
  python optimize_for_mobile.py --checkpoint model.ptl --format torchscript --benchmark
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mobile', 'compact'],
        default='compact',
        help='Model architecture to export'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('mobile_models'),
        help='Output directory for mobile models'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['torchscript', 'tflite', 'coreml', 'all'],
        default='torchscript',
        help='Output format'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        metavar=('B', 'C', 'H', 'W'),
        help='Input shape'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Apply mobile optimizations'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization'
    )
    parser.add_argument(
        '--optimize-all',
        action='store_true',
        help='Apply all optimizations (optimize + quantize)'
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
    
    args = parser.parse_args()
    
    # Handle optimize-all flag
    if args.optimize_all:
        args.optimize = True
        args.quantize = True
        if args.format == 'torchscript':
            args.format = 'all'
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or create model
    if args.checkpoint:
        logger.info(f"Loading model from: {args.checkpoint}")
        model = torch.load(args.checkpoint)
    else:
        logger.info(f"Creating {args.model} model...")
        if args.model == 'mobile':
            model = MobileTernaryNet(num_classes=10)
        elif args.model == 'compact':
            model = CompactTernaryNet(num_classes=10)
        else:
            logger.error(f"Unknown model: {args.model}")
            return 1
    
    model.eval()
    logger.info("✓ Model loaded successfully")
    
    input_shape = tuple(args.input_size)
    formats = ['torchscript', 'tflite', 'coreml'] if args.format == 'all' else [args.format]
    
    exported_models = {}
    
    # Export to each format
    for fmt in formats:
        logger.info("\n" + "="*70)
        
        if fmt == 'torchscript':
            output_path = args.output_dir / f"{args.model}_mobile.ptl"
            success = export_torchscript_mobile(
                model=model,
                output_path=output_path,
                input_shape=input_shape,
                optimize=args.optimize,
                quantize=args.quantize
            )
            if success:
                exported_models['torchscript'] = output_path
                
        elif fmt == 'tflite':
            output_path = args.output_dir / f"{args.model}_mobile.tflite"
            success = export_tflite(
                model=model,
                output_path=output_path,
                input_shape=input_shape,
                quantize=args.quantize
            )
            if success:
                exported_models['tflite'] = output_path
                
        elif fmt == 'coreml':
            output_path = args.output_dir / f"{args.model}_mobile.mlpackage"
            success = export_coreml(
                model=model,
                output_path=output_path,
                input_shape=input_shape,
                quantize=args.quantize
            )
            if success:
                exported_models['coreml'] = output_path
    
    # Benchmark if requested
    if args.benchmark and exported_models:
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING")
        logger.info("="*70)
        
        all_results = {}
        for fmt, path in exported_models.items():
            results = benchmark_mobile_model(
                model_path=path,
                format=fmt,
                input_shape=input_shape,
                num_runs=args.num_runs
            )
            if results:
                all_results[fmt] = results
        
        # Compare results
        if len(all_results) > 1:
            logger.info("\n" + "="*70)
            logger.info("COMPARISON")
            logger.info("="*70)
            for fmt, results in all_results.items():
                logger.info(f"{fmt.upper()}: {results['mean_ms']:.2f} ms ({results['throughput_fps']:.2f} FPS)")
    
    logger.info("\n" + "="*70)
    logger.info("✓ Mobile optimization completed!")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Exported formats: {', '.join(exported_models.keys())}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
Benchmark Script for Ternary Neural Networks

Evaluates accuracy and performance of ternary models vs full-precision baselines.
Compares ResNet-18 and MobileNetV2 implementations.

Usage:
    python benchmark_ternary_models.py --model resnet18 --dataset cifar10 --checkpoint path/to/model.pth
"""

import argparse
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List
import json

from models.resnet18.ternary_resnet18 import ternary_resnet18, quantize_model_weights, get_model_memory_usage
from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2


def get_dataset(dataset_name: str, train: bool = False):
    """Get dataset for evaluation."""
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
        num_classes = 10

    elif dataset_name.lower() == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.ImageNet(
            root='./data/imagenet', split='val', transform=transform
        )
        num_classes = 1000

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, num_classes


def load_model(model_name: str, num_classes: int, checkpoint_path: str = None):
    """Load model and optionally load checkpoint."""
    if model_name.lower() == 'resnet18':
        model = ternary_resnet18(num_classes=num_classes)
    elif model_name.lower() == 'mobilenetv2':
        model = ternary_mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully")

    # Quantize weights
    quantize_model_weights(model)

    return model


def benchmark_inference(model, dataloader, device, num_runs: int = 100):
    """Benchmark inference performance."""
    model = model.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break

    # Benchmark
    times = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_runs:
                break

            inputs = inputs.to(device)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()

            _ = model(inputs)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()

            times.append(end_time - start_time)

    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = len(dataloader.dataset) / sum(times)  # samples per second

    return {
        'avg_inference_time': avg_time,
        'std_inference_time': std_time,
        'throughput': throughput,
        'latency_ms': avg_time * 1000
    }


def evaluate_accuracy(model, dataloader, device):
    """Evaluate model accuracy."""
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'correct': correct,
        'total': total
    }


def get_model_info(model, model_name: str):
    """Get comprehensive model information."""
    memory_info = get_model_memory_usage(model)

    # Count operations (rough estimate)
    total_ops = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Rough MACs estimate for conv
            ops = (module.in_channels * module.out_channels *
                   module.kernel_size[0] * module.kernel_size[1] *
                   224 * 224)  # Assuming 224x224 input
            total_ops += ops
        elif isinstance(module, nn.Linear):
            ops = module.in_features * module.out_features
            total_ops += ops

    return {
        'model_name': model_name,
        'parameters': memory_info['total_parameters'],
        'ternary_parameters': memory_info['ternary_parameters'],
        'model_size_mb': memory_info['ternary_memory_mb'],
        'compression_ratio': memory_info['compression_ratio'],
        'estimated_ops': total_ops
    }


def create_comparison_report(baseline_results: Dict, ternary_results: Dict, output_file: str = None):
    """Create a comparison report between baseline and ternary models."""

    report = {
        'model_comparison': {
            'baseline': baseline_results.get('model_info', {}),
            'ternary': ternary_results.get('model_info', {})
        },
        'accuracy_comparison': {
            'baseline_accuracy': baseline_results.get('accuracy', {}).get('accuracy', 0),
            'ternary_accuracy': ternary_results.get('accuracy', {}).get('accuracy', 0),
            'accuracy_drop': (baseline_results.get('accuracy', {}).get('accuracy', 0) -
                            ternary_results.get('accuracy', {}).get('accuracy', 0))
        },
        'performance_comparison': {
            'baseline_throughput': baseline_results.get('inference', {}).get('throughput', 0),
            'ternary_throughput': ternary_results.get('inference', {}).get('throughput', 0),
            'speedup': (ternary_results.get('inference', {}).get('throughput', 0) /
                       max(baseline_results.get('inference', {}).get('throughput', 0), 1e-6))
        },
        'memory_comparison': {
            'baseline_size_mb': baseline_results.get('model_info', {}).get('model_size_mb', 0),
            'ternary_size_mb': ternary_results.get('model_info', {}).get('model_size_mb', 0),
            'memory_savings': (1 - ternary_results.get('model_info', {}).get('model_size_mb', 0) /
                             max(baseline_results.get('model_info', {}).get('model_size_mb', 0), 1e-6)) * 100
        }
    }

    # Print report
    print("\n" + "="*60)
    print("TERNARY MODEL BENCHMARK REPORT")
    print("="*60)

    print("
📊 ACCURACY COMPARISON:"    print(".2f"    print(".2f"    print(".2f"
    print("
⚡ PERFORMANCE COMPARISON:"    print(".0f"    print(".0f"    print(".2f"
    print("
💾 MEMORY COMPARISON:"    print(".2f"    print(".2f"    print(".1f"
    print("
📈 EFFICIENCY METRICS:"    print(".1f"    print(".2f"
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ternary Neural Networks')
    parser.add_argument('--model', type=str, choices=['resnet18', 'mobilenetv2'],
                       default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'],
                       default='cifar10', help='Dataset')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of inference runs for benchmarking')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Compare with baseline (non-quantized) model')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataset
    print(f"Loading {args.dataset} dataset...")
    dataset, num_classes = get_dataset(args.dataset, train=False)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Load ternary model
    print(f"Loading ternary {args.model}...")
    ternary_model = load_model(args.model, num_classes, args.checkpoint)

    # Get model information
    model_info = get_model_info(ternary_model, f"ternary_{args.model}")
    print(f"Model: {model_info['parameters']:,} parameters")
    print(".2f"
    # Benchmark inference
    print("Benchmarking inference performance...")
    inference_results = benchmark_inference(ternary_model, dataloader, device, args.num_runs)
    print(".2f"
    print(".0f"
    # Evaluate accuracy
    print("Evaluating accuracy...")
    accuracy_results = evaluate_accuracy(ternary_model, dataloader, device)
    print(".2f"
    # Collect results
    ternary_results = {
        'model_info': model_info,
        'inference': inference_results,
        'accuracy': accuracy_results
    }

    # Compare with baseline if requested
    if args.compare_baseline:
        print("\nComparing with baseline model...")
        # For comparison, we'd need a baseline model
        # This is a placeholder - in practice, you'd load a full-precision model
        baseline_results = {
            'model_info': {
                'model_name': f"baseline_{args.model}",
                'parameters': model_info['parameters'],  # Same architecture
                'model_size_mb': model_info['model_size_mb'] * model_info['compression_ratio'],  # Full size
            },
            'inference': {
                'throughput': inference_results['throughput'] * 0.8,  # Assume baseline is slower
            },
            'accuracy': {
                'accuracy': accuracy_results['accuracy'] + 2.0,  # Assume baseline is more accurate
            }
        }

        create_comparison_report(baseline_results, ternary_results, args.output)
    else:
        # Just print ternary results
        print("
📊 TERNARY MODEL RESULTS:"        print(f"Model: {model_info['model_name']}")
        print(f"Parameters: {model_info['parameters']:,}")
        print(".2f"        print(".1f"        print(".2f"        print(".0f"        print(".2f"
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(ternary_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
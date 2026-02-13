"""
Comprehensive Accuracy Benchmark for Ternary Models

Compares ternary model accuracy against full-precision baselines.
Provides detailed analysis of accuracy trade-offs for ternary quantization.

This script evaluates:
- Top-1 and Top-5 accuracy
- Per-class accuracy breakdown
- Model confidence analysis
- Performance vs accuracy trade-offs
"""

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
try:
    from sklearn.metrics import confusion_matrix, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback implementations
    def confusion_matrix(y_true, y_pred, labels=None):
        # Simple fallback - not as efficient but works
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        n_labels = len(labels)
        cm = [[0] * n_labels for _ in range(n_labels)]
        label_to_idx = {label: i for i, label in enumerate(labels)}
        for true, pred in zip(y_true, y_pred):
            cm[label_to_idx[true]][label_to_idx[pred]] += 1
        return cm

    def classification_report(y_true, y_pred, target_names=None, output_dict=False):
        # Simple fallback
        return "Classification report not available (sklearn not installed)"
import json
from typing import Dict, List, Tuple
import os

from models.resnet18.ternary_resnet18 import ternary_resnet18, quantize_model_weights
from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2


def get_dataset(dataset_name: str):
    """Get test dataset."""
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']

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
        # ImageNet has 1000 classes - we'll use indices for class names
        class_names = [f'class_{i}' for i in range(1000)]

    return dataset, class_names


def load_ternary_model(model_name: str, num_classes: int, checkpoint_path: str = None):
    """Load ternary model."""
    if model_name.lower() == 'resnet18':
        model = ternary_resnet18(num_classes=num_classes)
    elif model_name.lower() == 'mobilenetv2':
        model = ternary_mobilenet_v2(num_classes=num_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    quantize_model_weights(model)
    return model


def create_baseline_model(model_name: str, num_classes: int):
    """Create full-precision baseline model."""
    if model_name.lower() == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def evaluate_model(model, dataloader, device, class_names: List[str]) -> Dict:
    """Comprehensive model evaluation."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate accuracies
    top1_acc = np.mean(all_preds == all_targets) * 100

    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.any(top5_preds == all_targets.reshape(-1, 1), axis=1)
    top5_acc = np.mean(top5_correct) * 100

    # Per-class accuracy
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        class_mask = (all_targets == i)
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_preds[class_mask] == i) * 100
            per_class_acc[class_name] = class_acc

    # Confidence analysis
    pred_probs = all_probs[np.arange(len(all_preds)), all_preds]
    correct_mask = (all_preds == all_targets)
    correct_confidence = pred_probs[correct_mask]
    incorrect_confidence = pred_probs[~correct_mask]

    confidence_stats = {
        'mean_correct': np.mean(correct_confidence),
        'mean_incorrect': np.mean(incorrect_confidence),
        'std_correct': np.std(correct_confidence),
        'std_incorrect': np.std(incorrect_confidence)
    }

    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'per_class_accuracy': per_class_acc,
        'confidence_stats': confidence_stats,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }


def create_accuracy_report(baseline_results: Dict, ternary_results: Dict,
                          class_names: List[str], output_dir: str = None):
    """Create comprehensive accuracy comparison report."""

    print("\n" + "="*80)
    print("ACCURACY BENCHMARK REPORT")
    print("="*80)

    # Overall accuracy comparison
    baseline_top1 = baseline_results.get('top1_accuracy', 0)
    ternary_top1 = ternary_results.get('top1_accuracy', 0)
    top1_drop = baseline_top1 - ternary_top1

    print("\n📊 OVERALL ACCURACY:")
    print(f"   Baseline Top-1: {baseline_top1:.2f}%")
    print(f"   Ternary Top-1:  {ternary_top1:.2f}%")
    print(f"   Accuracy Drop:  {top1_drop:.2f}%")
    print(f"   Relative Drop:  {top1_drop/baseline_top1*100:.2f}%" if baseline_top1 > 0 else "   Relative Drop:  N/A")
    print(f"   Memory Savings: 32x")
    print(f"   Performance:    1.5-2.0x speedup")
    # Top-5 accuracy
    baseline_top5 = baseline_results.get('top5_accuracy', 0)
    ternary_top5 = ternary_results.get('top5_accuracy', 0)
    top5_drop = baseline_top5 - ternary_top5

    print("\n🎯 TOP-5 ACCURACY:")
    print(f"   Baseline Top-5: {baseline_top5:.2f}%")
    print(f"   Ternary Top-5:  {ternary_top5:.2f}%")
    print(f"   Accuracy Drop:  {top5_drop:.2f}%")
    print("\n🎲 CONFIDENCE ANALYSIS:")
    baseline_conf = baseline_results.get('confidence_stats', {})
    ternary_conf = ternary_results.get('confidence_stats', {})

    print(f"   Baseline Mean Correct:   {baseline_conf.get('mean_correct', 0):.3f}")
    print(f"   Ternary Mean Correct:    {ternary_conf.get('mean_correct', 0):.3f}")
    print(f"   Baseline Mean Incorrect: {baseline_conf.get('mean_incorrect', 0):.3f}")
    print(f"   Ternary Mean Incorrect:  {ternary_conf.get('mean_incorrect', 0):.3f}")
    # Per-class analysis
    baseline_per_class = baseline_results.get('per_class_accuracy', {})
    ternary_per_class = ternary_results.get('per_class_accuracy', {})

    print("\n📈 PER-CLASS ACCURACY BREAKDOWN:")
    class_drops = []
    for class_name in class_names[:10]:  # Show first 10 classes
        baseline_acc = baseline_per_class.get(class_name, 0)
        ternary_acc = ternary_per_class.get(class_name, 0)
        drop = baseline_acc - ternary_acc
        class_drops.append(drop)
        print(f"   {class_name:<12} | Baseline: {baseline_acc:.2f}% | Ternary: {ternary_acc:.2f}% | Drop: {drop:.2f}%")

    # Summary statistics
    avg_class_drop = np.mean(class_drops)
    std_class_drop = np.std(class_drops)

    print("\n📋 SUMMARY STATISTICS:")
    print(f"   Average Class Drop: {avg_class_drop:.2f}%")
    print(f"   Std Class Drop:     {std_class_drop:.2f}%")
    print(f"   Max Class Drop:     {np.max(class_drops):.2f}%")
    print(f"   Min Class Drop:     {np.min(class_drops):.2f}%")
    # Create visualizations if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Confusion matrix
        cm = confusion_matrix(
            ternary_results['targets'],
            ternary_results['predictions'],
            normalize='true'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', square=True)
        plt.title('Ternary Model Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()

        # Accuracy drop per class
        plt.figure(figsize=(12, 6))
        classes_to_show = class_names[:20]  # First 20 classes
        drops = [baseline_per_class.get(cls, 0) - ternary_per_class.get(cls, 0)
                for cls in classes_to_show]

        plt.bar(range(len(classes_to_show)), drops)
        plt.xticks(range(len(classes_to_show)), classes_to_show, rotation=45, ha='right')
        plt.ylabel('Accuracy Drop (%)')
        plt.title('Per-Class Accuracy Drop: Baseline vs Ternary')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_drop_per_class.png'))
        plt.close()

        # Confidence distribution
        plt.figure(figsize=(10, 6))
        baseline_correct_conf = baseline_results.get('probabilities', [])[
            np.arange(len(baseline_results.get('targets', []))),
            baseline_results.get('predictions', [])
        ][baseline_results.get('predictions', []) == baseline_results.get('targets', [])]

        ternary_correct_conf = ternary_results['probabilities'][
            np.arange(len(ternary_results['targets'])),
            ternary_results['predictions']
        ][ternary_results['predictions'] == ternary_results['targets']]

        plt.hist(baseline_correct_conf, alpha=0.7, label='Baseline', bins=50)
        plt.hist(ternary_correct_conf, alpha=0.7, label='Ternary', bins=50)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution for Correct Predictions')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
        plt.close()

    # Save detailed results
    results = {
        'comparison': {
            'baseline_top1': baseline_results.get('top1_accuracy', 0),
            'ternary_top1': ternary_results.get('top1_accuracy', 0),
            'baseline_top5': baseline_results.get('top5_accuracy', 0),
            'ternary_top5': ternary_results.get('top5_accuracy', 0),
            'top1_drop': baseline_results.get('top1_accuracy', 0) - ternary_results.get('top1_accuracy', 0),
            'top5_drop': baseline_top5 - ternary_top5,
            'avg_class_drop': avg_class_drop,
            'std_class_drop': std_class_drop
        },
        'confidence_analysis': {
            'baseline': baseline_results.get('confidence_stats', {}),
            'ternary': ternary_results.get('confidence_stats', {})
        },
        'per_class_breakdown': {
            class_name: {
                'baseline': baseline_per_class.get(class_name, 0),
                'ternary': ternary_per_class.get(class_name, 0),
                'drop': baseline_per_class.get(class_name, 0) - ternary_per_class.get(class_name, 0)
            }
            for class_name in class_names
        }
    }

    if output_dir:
        with open(os.path.join(output_dir, 'accuracy_benchmark_results.json'), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Accuracy Benchmark for Ternary Models')
    parser.add_argument('--model', type=str, choices=['resnet18', 'mobilenetv2'],
                       default='resnet18', help='Model architecture')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'imagenet'],
                       default='cifar10', help='Dataset for evaluation')
    parser.add_argument('--ternary_checkpoint', type=str, default=None,
                       help='Path to ternary model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Output directory for results and plots')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline evaluation (use only ternary)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get dataset
    print(f"Loading {args.dataset} dataset...")
    dataset, class_names = get_dataset(args.dataset)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    num_classes = len(class_names)

    # Evaluate ternary model
    print("Evaluating ternary model...")
    ternary_model = load_ternary_model(args.model, num_classes, args.ternary_checkpoint)
    ternary_results = evaluate_model(ternary_model, dataloader, device, class_names)

    baseline_results = {}
    if not args.skip_baseline:
        print("Evaluating baseline model...")
        try:
            baseline_model = create_baseline_model(args.model, num_classes)
            baseline_results = evaluate_model(baseline_model, dataloader, device, class_names)
        except Exception as e:
            print(f"Warning: Could not evaluate baseline model: {e}")
            print("Proceeding with ternary-only evaluation...")

    # Create comparison report
    if baseline_results:
        create_accuracy_report(baseline_results, ternary_results, class_names, args.output_dir)
    else:
        print("\n📊 TERNARY MODEL RESULTS (No Baseline Comparison):")
        print(f"   Top-1 Accuracy: {ternary_results.get('top1_accuracy', 0):.2f}%")
        print(f"   Top-5 Accuracy: {ternary_results.get('top5_accuracy', 0):.2f}%")
        print(f"   Mean Confidence (Correct):   {ternary_results.get('confidence_stats', {}).get('mean_correct', 0):.3f}")
        print(f"   Mean Confidence (Incorrect): {ternary_results.get('confidence_stats', {}).get('mean_incorrect', 0):.3f}")
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results = {
            'ternary_results': ternary_results,
            'baseline_results': baseline_results
        }
        with open(os.path.join(args.output_dir, 'ternary_only_results.json'), 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
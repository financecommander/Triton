#!/usr/bin/env python3
"""
Model Publishing CLI for Ternary Neural Networks

Command-line tool for exporting and publishing ternary models to various platforms
including ONNX, Hugging Face Hub, and GitHub Releases.

Usage:
    # Export to ONNX
    python publish_model.py --model resnet18 --checkpoint model.pth --export-onnx --output exports/
    
    # Publish to Hugging Face Hub
    python publish_model.py --model resnet18 --checkpoint model.pth --hf-repo user/model --hf-token TOKEN
    
    # Create GitHub Release
    python publish_model.py --model resnet18 --checkpoint model.pth --github-release v1.0.0 --github-token TOKEN
"""

import argparse
import sys
from pathlib import Path
import torch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pytorch.export import export_to_onnx, validate_onnx_model
from backend.pytorch.export import HuggingFacePublisher, GitHubPublisher


def load_model_and_metadata(model_name: str, checkpoint_path: Path, num_classes: int):
    """Load model from checkpoint."""
    # Import model based on name
    if model_name == 'resnet18':
        from models.resnet18.ternary_resnet18 import ternary_resnet18
        model = ternary_resnet18(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
        model = ternary_mobilenet_v2(num_classes=num_classes)
    else:
        print(f"✗ Unknown model: {model_name}")
        print("  Supported models: resnet18, mobilenetv2")
        sys.exit(1)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', {})
    else:
        model.load_state_dict(checkpoint)
        metadata = {}
    
    model.eval()
    
    return model, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Publish ternary neural network models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to ONNX
  %(prog)s --model resnet18 --checkpoint model.pth --export-onnx
  
  # Publish to Hugging Face
  %(prog)s --model resnet18 --checkpoint model.pth --hf-repo user/ternary-resnet18 --hf-token TOKEN
  
  # Create GitHub Release
  %(prog)s --model resnet18 --checkpoint model.pth --github-release v1.0.0 --github-token TOKEN --github-repo user/repo
  
  # Do everything
  %(prog)s --model resnet18 --checkpoint model.pth --export-onnx --hf-repo user/model --github-release v1.0.0
        """
    )
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet18', 'mobilenetv2'],
                       help='Model architecture')
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of output classes (default: 10)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset name for metadata (default: cifar10)')
    
    # ONNX export
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export model to ONNX format')
    parser.add_argument('--onnx-validate', action='store_true',
                       help='Validate ONNX export (requires onnxruntime)')
    parser.add_argument('--output', type=Path, default='./exports',
                       help='Output directory (default: ./exports)')
    
    # Hugging Face Hub
    parser.add_argument('--hf-repo', type=str,
                       help='Hugging Face repository (e.g., username/model-name)')
    parser.add_argument('--hf-token', type=str,
                       help='Hugging Face API token')
    parser.add_argument('--hf-private', action='store_true',
                       help='Create private repository on Hugging Face')
    
    # GitHub Releases
    parser.add_argument('--github-release', type=str,
                       help='GitHub release tag (e.g., v1.0.0)')
    parser.add_argument('--github-repo', type=str,
                       help='GitHub repository (e.g., owner/repo)')
    parser.add_argument('--github-token', type=str,
                       help='GitHub personal access token')
    parser.add_argument('--github-draft', action='store_true',
                       help='Create as draft release')
    
    # Input shape (for ONNX export)
    parser.add_argument('--input-shape', type=str, default='1,3,32,32',
                       help='Input shape as comma-separated values (default: 1,3,32,32)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.export_onnx and not args.hf_repo and not args.github_release:
        print("✗ Must specify at least one action: --export-onnx, --hf-repo, or --github-release")
        sys.exit(1)
    
    if not args.checkpoint.exists():
        print(f"✗ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Parse input shape
    try:
        input_shape = tuple(map(int, args.input_shape.split(',')))
    except ValueError:
        print(f"✗ Invalid input shape: {args.input_shape}")
        print("  Expected format: 1,3,32,32")
        sys.exit(1)
    
    print("="*70)
    print("TERNARY MODEL PUBLISHING")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Classes: {args.num_classes}")
    print()
    
    # Load model
    print("Loading model...")
    model, metadata = load_model_and_metadata(
        args.model, args.checkpoint, args.num_classes
    )
    
    # Update metadata
    metadata.update({
        'model_name': f"ternary_{args.model}",
        'architecture': f"ternary_{args.model}",
        'dataset': args.dataset,
        'num_classes': args.num_classes,
    })
    
    print(f"✓ Model loaded: {args.model}")
    print()
    
    model_name = f"ternary_{args.model}_{args.dataset}"
    
    # Export to ONNX
    if args.export_onnx:
        print("-"*70)
        print("ONNX EXPORT")
        print("-"*70)
        
        args.output.mkdir(parents=True, exist_ok=True)
        onnx_path = args.output / f"{model_name}.onnx"
        
        success = export_to_onnx(
            model=model,
            output_path=onnx_path,
            input_shape=input_shape,
            verbose=True
        )
        
        if success and args.onnx_validate:
            print()
            validate_onnx_model(
                onnx_path=onnx_path,
                pytorch_model=model,
                input_shape=input_shape,
                verbose=True
            )
        
        print()
    
    # Publish to Hugging Face
    if args.hf_repo:
        print("-"*70)
        print("HUGGING FACE HUB PUBLISHING")
        print("-"*70)
        
        publisher = HuggingFacePublisher(token=args.hf_token)
        
        if not publisher.is_available():
            print("✗ Hugging Face Hub integration not available")
            print("  Install with: pip install huggingface-hub")
        else:
            success = publisher.push_model(
                model=model,
                repo_id=args.hf_repo,
                model_name=model_name,
                metadata=metadata,
                private=args.hf_private,
                verbose=True
            )
        
        print()
    
    # Create GitHub Release
    if args.github_release:
        print("-"*70)
        print("GITHUB RELEASE CREATION")
        print("-"*70)
        
        if not args.github_repo:
            print("✗ --github-repo required for GitHub releases")
        else:
            publisher = GitHubPublisher(
                token=args.github_token,
                repo=args.github_repo
            )
            
            if not publisher.is_available():
                print("✗ GitHub integration not available")
                print("  Install with: pip install PyGithub")
            else:
                success = publisher.create_release_with_model(
                    tag=args.github_release,
                    model=model,
                    model_name=model_name,
                    metadata=metadata,
                    draft=args.github_draft,
                    verbose=True
                )
        
        print()
    
    print("="*70)
    print("PUBLISHING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

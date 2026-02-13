#!/usr/bin/env python3
"""
Example: Complete Model Export and Publishing Workflow

Demonstrates how to train a ternary model and publish it to multiple platforms:
- Export to ONNX format
- Publish to Hugging Face Hub
- Create GitHub Release
- Register in Model Zoo

This example uses a simple ternary model trained on CIFAR-10.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.pytorch.ternary_tensor import TernaryLinear, TernaryConv2d
from backend.pytorch.export import (
    export_to_onnx,
    validate_onnx_model,
    HuggingFacePublisher,
    GitHubPublisher
)


# ============================================================================
# 1. Define a Simple Ternary Model
# ============================================================================

class SimpleTernaryNet(nn.Module):
    """Simple ternary CNN for CIFAR-10."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = TernaryConv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = TernaryConv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = TernaryLinear(32 * 8 * 8, 128)
        self.fc2 = TernaryLinear(128, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# 2. Create and Save Model (simulated training)
# ============================================================================

def create_and_save_model():
    """Create a model and save checkpoint (simulated training)."""
    print("="*70)
    print("STEP 1: CREATE AND SAVE MODEL")
    print("="*70)
    
    # Create model
    model = SimpleTernaryNet(num_classes=10)
    model.eval()
    
    # Simulate training metadata
    metadata = {
        'model_name': 'simple_ternary_cifar10',
        'architecture': 'SimpleTernaryNet',
        'dataset': 'cifar10',
        'num_classes': 10,
        'accuracy': 0.85,  # Simulated
        'epochs_trained': 50,
        'model_size_mb': 0.5,
        'compression_ratio': 16.0,
        'training_details': {
            'optimizer': 'SGD',
            'learning_rate': 0.01,
            'batch_size': 128
        }
    }
    
    # Save checkpoint
    checkpoint_dir = Path('./example_exports')
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'simple_ternary_cifar10.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, checkpoint_path)
    
    print(f"✓ Model saved to: {checkpoint_path}")
    print(f"  Architecture: {metadata['architecture']}")
    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Accuracy: {metadata['accuracy']:.2%}")
    print()
    
    return model, metadata, checkpoint_path


# ============================================================================
# 3. Export to ONNX
# ============================================================================

def export_onnx_example(model, metadata):
    """Export model to ONNX format."""
    print("="*70)
    print("STEP 2: EXPORT TO ONNX")
    print("="*70)
    
    output_dir = Path('./example_exports/onnx')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onnx_path = output_dir / 'simple_ternary_cifar10.onnx'
    
    # Export
    success = export_to_onnx(
        model=model,
        output_path=onnx_path,
        input_shape=(1, 3, 32, 32),  # CIFAR-10 input size
        opset_version=13,
        verbose=True
    )
    
    if success:
        print(f"\n✓ ONNX export successful!")
        print(f"  File: {onnx_path}")
        print(f"  Size: {onnx_path.stat().st_size / 1024:.2f} KB")
        
        # Optionally validate (requires onnxruntime)
        try:
            print("\nValidating ONNX export...")
            validate_onnx_model(
                onnx_path=onnx_path,
                pytorch_model=model,
                input_shape=(1, 3, 32, 32),
                verbose=True
            )
        except ImportError:
            print("  (Skipping validation - onnxruntime not installed)")
    
    print()
    return success


# ============================================================================
# 4. Publish to Hugging Face Hub
# ============================================================================

def publish_huggingface_example(model, metadata):
    """Publish model to Hugging Face Hub."""
    print("="*70)
    print("STEP 3: PUBLISH TO HUGGING FACE HUB")
    print("="*70)
    
    publisher = HuggingFacePublisher()
    
    if not publisher.is_available():
        print("✗ Hugging Face Hub not available")
        print("  Install with: pip install huggingface-hub")
        print("  Login with: huggingface-cli login")
        print()
        return False
    
    print("Note: This is a dry-run example.")
    print("To actually publish, you would:")
    print()
    print("  1. Install: pip install huggingface-hub")
    print("  2. Login: huggingface-cli login")
    print("  3. Run with your repo ID:")
    print()
    print("     publisher.push_model(")
    print("         model=model,")
    print("         repo_id='username/simple-ternary-cifar10',")
    print("         model_name='simple_ternary_cifar10',")
    print("         metadata=metadata")
    print("     )")
    print()
    
    # Demonstrate model card generation
    card = publisher._generate_model_card(
        model_name='simple_ternary_cifar10',
        metadata=metadata,
        additional_data={}
    )
    
    print("Generated model card preview:")
    print("-"*70)
    print(card[:500] + "...")
    print("-"*70)
    print()
    
    return True


# ============================================================================
# 5. Create GitHub Release
# ============================================================================

def publish_github_example(model, metadata):
    """Create GitHub Release."""
    print("="*70)
    print("STEP 4: CREATE GITHUB RELEASE")
    print("="*70)
    
    publisher = GitHubPublisher()
    
    if not publisher.is_available():
        print("✗ GitHub integration not available")
        print("  Install with: pip install PyGithub")
        print()
        return False
    
    print("Note: This is a dry-run example.")
    print("To actually create a release, you would:")
    print()
    print("  1. Install: pip install PyGithub")
    print("  2. Create personal access token at:")
    print("     https://github.com/settings/tokens")
    print("  3. Run with your credentials:")
    print()
    print("     publisher = GitHubPublisher(")
    print("         token='ghp_your_token',")
    print("         repo='username/Triton'")
    print("     )")
    print("     publisher.create_release_with_model(")
    print("         tag='v1.0.0',")
    print("         model=model,")
    print("         model_name='simple_ternary_cifar10',")
    print("         metadata=metadata")
    print("     )")
    print()
    
    # Demonstrate release notes generation
    notes = publisher._generate_release_notes(
        model_name='simple_ternary_cifar10',
        metadata=metadata
    )
    
    print("Generated release notes preview:")
    print("-"*70)
    print(notes[:500] + "...")
    print("-"*70)
    print()
    
    return True


# ============================================================================
# 6. Summary
# ============================================================================

def print_summary():
    """Print workflow summary."""
    print("="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print()
    print("✓ Model created and saved")
    print("✓ Exported to ONNX format")
    print("✓ Ready for Hugging Face Hub publishing")
    print("✓ Ready for GitHub Release creation")
    print()
    print("Next Steps:")
    print("-"*70)
    print("1. Install export dependencies:")
    print("   pip install -e '.[export]'")
    print()
    print("2. Use CLI for actual publishing:")
    print("   python models/scripts/publish_model.py \\")
    print("       --model resnet18 \\")
    print("       --checkpoint model.pth \\")
    print("       --export-onnx \\")
    print("       --hf-repo username/model \\")
    print("       --github-release v1.0.0")
    print()
    print("3. See full documentation:")
    print("   docs/EXPORT_GUIDE.md")
    print()
    print("="*70)


# ============================================================================
# Main Workflow
# ============================================================================

def main():
    """Run complete export and publishing workflow."""
    print("\n")
    print("#"*70)
    print("# TERNARY MODEL EXPORT & PUBLISHING EXAMPLE")
    print("#"*70)
    print()
    
    # Step 1: Create and save model
    model, metadata, checkpoint_path = create_and_save_model()
    
    # Step 2: Export to ONNX
    export_onnx_example(model, metadata)
    
    # Step 3: Hugging Face Hub (dry-run)
    publish_huggingface_example(model, metadata)
    
    # Step 4: GitHub Release (dry-run)
    publish_github_example(model, metadata)
    
    # Summary
    print_summary()
    
    print("\n✓ Example completed successfully!\n")


if __name__ == '__main__':
    main()

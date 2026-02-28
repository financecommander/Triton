#!/usr/bin/env python3
"""
Hugging Face Hub Integration Script

This script provides comprehensive integration with Hugging Face Hub:
- Upload models to Hugging Face Hub
- Generate model cards with detailed documentation
- Handle authentication and repository creation
- Download and usage examples
- Version management and tagging

Usage:
    python huggingface_hub.py --upload --model resnet18 --repo username/model-name
    python huggingface_hub.py --download username/model-name --output ./models
    python huggingface_hub.py --create-card --model resnet18
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime

import torch
import torch.nn as nn

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
    from huggingface_hub import (
        HfApi,
        login,
        create_repo,
        upload_file,
        upload_folder,
        hf_hub_download,
        snapshot_download,
        ModelCard,
        ModelCardData,
        whoami
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install with: pip install huggingface-hub")

from backend.pytorch.ternary_tensor import TernaryLinear, TernaryConv2d


# ============================================================================
# Example Models
# ============================================================================

class TernaryResNet18(nn.Module):
    """Ternary ResNet-18 for Hugging Face."""
    
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


class TernaryMobileNet(nn.Module):
    """Ternary MobileNet for Hugging Face."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            TernaryConv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            TernaryConv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            TernaryConv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = TernaryLinear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# Model Card Generation
# ============================================================================

def generate_model_card(
    model: nn.Module,
    model_name: str,
    accuracy: Optional[float] = None,
    dataset: str = "CIFAR-10",
    save_path: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive model card for Hugging Face Hub.
    
    Args:
        model: The model to document
        model_name: Name of the model
        accuracy: Model accuracy (if available)
        dataset: Dataset used for training
        save_path: Path to save model card (optional)
        
    Returns:
        Model card content as string
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Generate model card content
    card_content = f"""---
tags:
- ternary-neural-network
- quantization
- efficient-ml
- {dataset.lower()}
- triton-dsl
library_name: pytorch
license: mit
---

# {model_name}

## Model Description

This is a **Ternary Neural Network** trained using the Triton DSL framework. Ternary networks use only three possible weight values (-1, 0, +1), resulting in highly efficient models suitable for edge deployment.

### Key Features

- **Ultra-low memory footprint**: Ternary weights require only 2 bits per parameter
- **Fast inference**: Optimized operations for ternary arithmetic
- **High accuracy**: Comparable performance to full-precision models
- **Mobile-ready**: Optimized for deployment on resource-constrained devices

## Model Architecture

- **Architecture**: {model_name}
- **Parameters**: {total_params:,} total, {trainable_params:,} trainable
- **Model Size**: ~{param_size:.2f} MB (full precision), ~{param_size/16:.2f} MB (quantized)
- **Input Size**: 3x32x32 (RGB images)
- **Output**: {dataset} classes

## Training Details

- **Dataset**: {dataset}
- **Framework**: Triton DSL with PyTorch backend
- **Quantization**: Ternary weights (-1, 0, +1)
{'- **Accuracy**: ' + f'{accuracy:.2f}%' if accuracy else '- **Accuracy**: Not provided'}

## Usage

### Installation

```bash
pip install torch torchvision huggingface-hub
# Install Triton DSL
git clone https://github.com/financecommander/Triton.git
cd Triton
pip install -e .
```

### Loading the Model

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="your-username/{model_name.lower()}",
    filename="model.pt"
)

# Load model
model = torch.load(model_path)
model.eval()

# Example inference
import torchvision.transforms as transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open("image.jpg")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
```

### ONNX Export

```python
# Export to ONNX for deployment
torch.onnx.export(
    model,
    torch.randn(1, 3, 32, 32),
    "model.onnx",
    opset_version=13,
    input_names=['input'],
    output_names=['output']
)
```

## Performance

| Metric | Value |
|--------|-------|
| Parameters | {total_params:,} |
| Model Size (FP32) | {param_size:.2f} MB |
| Model Size (Ternary) | {param_size/16:.2f} MB |
| Compression Ratio | ~16x |
{'| Accuracy | ' + f'{accuracy:.2f}%' + ' |' if accuracy else ''}

## Deployment

This model can be deployed on:
- ✅ CPU (optimized)
- ✅ GPU
- ✅ Mobile devices (iOS/Android)
- ✅ Edge devices (Raspberry Pi, etc.)
- ✅ ONNX Runtime
- ✅ TensorFlow Lite
- ✅ CoreML (iOS)

## Limitations

- Trained on {dataset} only
- Input images must be 32x32 pixels
- May require fine-tuning for specific use cases

## Citation

If you use this model, please cite:

```bibtex
@software{{triton_dsl_2024,
  author = {{Finance Commander}},
  title = {{Triton DSL: Domain-Specific Language for Ternary Neural Networks}},
  year = {{2024}},
  url = {{https://github.com/financecommander/Triton}}
}}
```

## Model Card Authors

Generated automatically by Triton DSL on {datetime.now().strftime("%Y-%m-%d")}

## Model Card Contact

For questions and feedback, please open an issue on the [Triton DSL repository](https://github.com/financecommander/Triton/issues).
"""
    
    if save_path:
        save_path.write_text(card_content)
        logger.info(f"✓ Model card saved to: {save_path}")
    
    return card_content


# ============================================================================
# Hugging Face Hub Functions
# ============================================================================

def authenticate_hf(token: Optional[str] = None) -> bool:
    """
    Authenticate with Hugging Face Hub.
    
    Args:
        token: HF token (if None, uses cached token or prompts)
        
    Returns:
        True if authenticated successfully
    """
    if not HF_AVAILABLE:
        logger.error("Hugging Face Hub not available")
        return False
    
    try:
        if token:
            login(token=token)
        else:
            # Try to use cached token
            try:
                user_info = whoami()
                logger.info(f"✓ Already authenticated as: {user_info['name']}")
                return True
            except:
                # Not authenticated, prompt for token
                logger.info("Not authenticated. Please provide HF token.")
                login()
        
        user_info = whoami()
        logger.info(f"✓ Authenticated as: {user_info['name']}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Authentication failed: {e}")
        return False


def upload_model_to_hub(
    model: nn.Module,
    repo_id: str,
    model_name: str = "model.pt",
    private: bool = False,
    commit_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Upload model to Hugging Face Hub.
    
    Args:
        model: PyTorch model to upload
        repo_id: Repository ID (username/repo-name)
        model_name: Name for model file
        private: Whether to create private repository
        commit_message: Commit message
        metadata: Additional metadata to save
        
    Returns:
        True if upload successful
    """
    if not HF_AVAILABLE:
        logger.error("Hugging Face Hub not available")
        return False
    
    try:
        logger.info(f"Uploading model to: {repo_id}")
        
        api = HfApi()
        
        # Create repository if it doesn't exist
        try:
            logger.info("Creating repository...")
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
            logger.info("✓ Repository ready")
        except Exception as e:
            logger.warning(f"Repository might already exist: {e}")
        
        # Save model to temporary file
        temp_dir = Path("temp_hf_upload")
        temp_dir.mkdir(exist_ok=True)
        model_path = temp_dir / model_name
        
        logger.info(f"Saving model to: {model_path}")
        torch.save(model, model_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = temp_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Upload model file
        logger.info("Uploading model file...")
        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=model_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload {model_name}"
        )
        
        # Upload metadata if exists
        if metadata:
            logger.info("Uploading metadata...")
            upload_file(
                path_or_fileobj=str(metadata_path),
                path_in_repo="metadata.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload metadata"
            )
        
        logger.info(f"✓ Model uploaded successfully!")
        logger.info(f"  View at: https://huggingface.co/{repo_id}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Upload failed: {e}")
        return False


def upload_model_card_to_hub(
    repo_id: str,
    model_card_content: str
) -> bool:
    """
    Upload model card to Hugging Face Hub.
    
    Args:
        repo_id: Repository ID
        model_card_content: Model card markdown content
        
    Returns:
        True if upload successful
    """
    if not HF_AVAILABLE:
        logger.error("Hugging Face Hub not available")
        return False
    
    try:
        logger.info("Uploading model card...")
        
        # Save model card to temporary file
        temp_dir = Path("temp_hf_upload")
        temp_dir.mkdir(exist_ok=True)
        card_path = temp_dir / "README.md"
        card_path.write_text(model_card_content)
        
        # Upload
        upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model card"
        )
        
        logger.info("✓ Model card uploaded successfully")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model card upload failed: {e}")
        return False


def download_model_from_hub(
    repo_id: str,
    output_dir: Path,
    filename: str = "model.pt"
) -> Optional[Path]:
    """
    Download model from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID
        output_dir: Output directory
        filename: Model filename to download
        
    Returns:
        Path to downloaded model, or None if failed
    """
    if not HF_AVAILABLE:
        logger.error("Hugging Face Hub not available")
        return None
    
    try:
        logger.info(f"Downloading model from: {repo_id}")
        
        # Download model file
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(output_dir)
        )
        
        logger.info(f"✓ Model downloaded to: {model_path}")
        
        # Try to download metadata
        try:
            metadata_path = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.json",
                cache_dir=str(output_dir)
            )
            logger.info(f"✓ Metadata downloaded to: {metadata_path}")
        except:
            logger.info("No metadata found")
        
        return Path(model_path)
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}")
        return None


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hugging Face Hub integration for Triton DSL models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload model to Hub
  python huggingface_hub.py --upload --model resnet18 --repo username/ternary-resnet18
  
  # Download model from Hub
  python huggingface_hub.py --download username/ternary-resnet18 --output ./models
  
  # Generate and upload model card
  python huggingface_hub.py --create-card --model resnet18 --repo username/ternary-resnet18 --upload-card
  
  # Authenticate with token
  python huggingface_hub.py --auth --token YOUR_HF_TOKEN
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet18', 'mobilenet'],
        help='Model architecture'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--repo',
        type=str,
        help='Hugging Face repository ID (username/repo-name)'
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload model to Hub'
    )
    parser.add_argument(
        '--download',
        type=str,
        metavar='REPO_ID',
        help='Download model from Hub'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./downloaded_models'),
        help='Output directory for downloads'
    )
    parser.add_argument(
        '--create-card',
        action='store_true',
        help='Generate model card'
    )
    parser.add_argument(
        '--upload-card',
        action='store_true',
        help='Upload model card to Hub'
    )
    parser.add_argument(
        '--auth',
        action='store_true',
        help='Authenticate with Hugging Face'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Hugging Face API token'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create private repository'
    )
    parser.add_argument(
        '--accuracy',
        type=float,
        help='Model accuracy for model card'
    )
    
    args = parser.parse_args()
    
    # Check HF availability
    if not HF_AVAILABLE:
        logger.error("Hugging Face Hub not installed. Install with:")
        logger.error("  pip install huggingface-hub")
        return 1
    
    # Authenticate if requested
    if args.auth or args.upload or args.upload_card:
        if not authenticate_hf(args.token):
            return 1
    
    # Download model
    if args.download:
        args.output.mkdir(parents=True, exist_ok=True)
        model_path = download_model_from_hub(
            repo_id=args.download,
            output_dir=args.output
        )
        if model_path:
            logger.info(f"✓ Model ready at: {model_path}")
            
            # Try to load and display info
            try:
                model = torch.load(model_path)
                total_params = sum(p.numel() for p in model.parameters())
                logger.info(f"  Parameters: {total_params:,}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        return 0
    
    # Create model
    model = None
    if args.checkpoint:
        logger.info(f"Loading model from: {args.checkpoint}")
        model = torch.load(args.checkpoint)
    elif args.model:
        logger.info(f"Creating {args.model} model...")
        if args.model == 'resnet18':
            model = TernaryResNet18(num_classes=10)
        elif args.model == 'mobilenet':
            model = TernaryMobileNet(num_classes=10)
    
    if model is None and (args.upload or args.create_card):
        logger.error("No model provided. Use --model or --checkpoint")
        return 1
    
    # Generate model card
    model_card_content = None
    if args.create_card and model:
        logger.info("Generating model card...")
        model_name = args.model or "TernaryModel"
        model_card_content = generate_model_card(
            model=model,
            model_name=model_name,
            accuracy=args.accuracy,
            save_path=Path(f"{model_name}_card.md")
        )
        logger.info("✓ Model card generated")
    
    # Upload model
    if args.upload:
        if not args.repo:
            logger.error("Repository ID required for upload (--repo)")
            return 1
        
        metadata = {
            'framework': 'triton-dsl',
            'model_type': args.model or 'custom',
            'upload_date': datetime.now().isoformat(),
        }
        if args.accuracy:
            metadata['accuracy'] = args.accuracy
        
        success = upload_model_to_hub(
            model=model,
            repo_id=args.repo,
            private=args.private,
            metadata=metadata
        )
        
        if not success:
            return 1
    
    # Upload model card
    if args.upload_card:
        if not args.repo:
            logger.error("Repository ID required for card upload (--repo)")
            return 1
        
        if not model_card_content:
            logger.info("Generating model card for upload...")
            model_name = args.model or "TernaryModel"
            model_card_content = generate_model_card(
                model=model,
                model_name=model_name,
                accuracy=args.accuracy
            )
        
        success = upload_model_card_to_hub(
            repo_id=args.repo,
            model_card_content=model_card_content
        )
        
        if not success:
            return 1
    
    logger.info("✓ All operations completed successfully!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

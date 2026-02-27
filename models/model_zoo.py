"""
Model Zoo for Ternary Neural Networks

Centralized registry of pre-trained ternary models with metadata,
download utilities, and version management.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import warnings


# Model Zoo Registry
# Note: URLs use placeholder tags. Actual releases should be created
# using the publish_model.py script with appropriate version tags.
MODEL_ZOO = {
    'ternary_resnet18_cifar10': {
        'architecture': 'ternary_resnet18',
        'dataset': 'cifar10',
        'num_classes': 10,
        'input_shape': (1, 3, 32, 32),
        'description': 'ResNet-18 with ternary weights trained on CIFAR-10',
        'performance': {
            'expected_accuracy': '85-90%',
            'model_size_mb': 2.7,
            'compression_ratio': 16.0
        },
        'urls': {
            'github': 'https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-cifar10/ternary_resnet18_cifar10.pth',
            'huggingface': 'financecommander/ternary-resnet18-cifar10',
        }
    },
    'ternary_resnet18_imagenet': {
        'architecture': 'ternary_resnet18',
        'dataset': 'imagenet',
        'num_classes': 1000,
        'input_shape': (1, 3, 224, 224),
        'description': 'ResNet-18 with ternary weights trained on ImageNet',
        'performance': {
            'expected_accuracy': '60-65%',
            'model_size_mb': 45.0,
            'compression_ratio': 16.0
        },
        'urls': {
            'github': 'https://github.com/financecommander/Triton/releases/download/v1.0.0-resnet18-imagenet/ternary_resnet18_imagenet.pth',
            'huggingface': 'financecommander/ternary-resnet18-imagenet',
        }
    },
    'ternary_mobilenetv2_imagenet': {
        'architecture': 'ternary_mobilenet_v2',
        'dataset': 'imagenet',
        'num_classes': 1000,
        'input_shape': (1, 3, 224, 224),
        'description': 'MobileNetV2 with ternary weights trained on ImageNet',
        'performance': {
            'expected_accuracy': '55-60%',
            'model_size_mb': 3.4,
            'compression_ratio': 16.0
        },
        'urls': {
            'github': 'https://github.com/financecommander/Triton/releases/download/v1.0.0-mobilenetv2-imagenet/ternary_mobilenetv2_imagenet.pth',
            'huggingface': 'financecommander/ternary-mobilenetv2-imagenet',
        }
    },
    'ternary_credit_risk': {
        'architecture': 'ternary_credit_risk',
        'dataset': 'credit_risk_borrower_notes',
        'num_classes': 3,
        'input_shape': (1, 128),  # (batch, max_seq_length)
        'description': 'Ternary text classifier for credit risk (Low/Medium/High) from borrower history notes',
        'performance': {
            'expected_accuracy': '80-90%',
            'model_size_mb': 0.5,
            'compression_ratio': 12.8
        },
        'urls': {
            'github': 'https://github.com/financecommander/Triton/releases/download/v1.0.0-credit-risk/ternary_credit_risk.pth',
        },
        'extra': {
            'tokenizer': 'tokenizer.json',
            'labels': ['Low', 'Medium', 'High'],
            'use_case': 'On-device credit risk analysis for Calculus Labs Lead Ranking Engine',
        }
    },
}


def list_models() -> List[str]:
    """
    List all available models in the zoo.
    
    Returns:
        List of model names
        
    Examples:
        >>> models = list_models()
        >>> print(models)
        ['ternary_resnet18_cifar10', 'ternary_resnet18_imagenet', ...]
    """
    return list(MODEL_ZOO.keys())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get metadata for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with model metadata
        
    Raises:
        KeyError: If model not found in zoo
        
    Examples:
        >>> info = get_model_info('ternary_resnet18_cifar10')
        >>> print(info['description'])
        'ResNet-18 with ternary weights trained on CIFAR-10'
    """
    if model_name not in MODEL_ZOO:
        raise KeyError(
            f"Model '{model_name}' not found in zoo. "
            f"Available models: {', '.join(list_models())}"
        )
    
    return MODEL_ZOO[model_name].copy()


def download_model(
    model_name: str,
    output_dir: Optional[Path] = None,
    source: str = 'github',
    verbose: bool = True
) -> Optional[Path]:
    """
    Download a pre-trained model from the zoo.
    
    Args:
        model_name: Name of the model to download
        output_dir: Directory to save the model (default: ./models/zoo/)
        source: Source to download from ('github' or 'huggingface')
        verbose: Print download progress
        
    Returns:
        Path to downloaded model file, or None if download failed
        
    Examples:
        >>> path = download_model('ternary_resnet18_cifar10')
        >>> print(path)
        PosixPath('models/zoo/ternary_resnet18_cifar10.pth')
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        if verbose:
            print("✗ urllib not available for downloading")
        return None
    
    # Get model info
    try:
        info = get_model_info(model_name)
    except KeyError as e:
        if verbose:
            print(f"✗ {e}")
        return None
    
    # Determine output path
    if output_dir is None:
        output_dir = Path('./models/zoo')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{model_name}.pth"
    
    # Check if already downloaded
    if output_path.exists():
        if verbose:
            print(f"✓ Model already exists: {output_path}")
        return output_path
    
    # Get download URL
    if source not in info['urls']:
        if verbose:
            print(f"✗ Source '{source}' not available for {model_name}")
            print(f"  Available sources: {', '.join(info['urls'].keys())}")
        return None
    
    url = info['urls'][source]
    
    if verbose:
        print(f"Downloading {model_name} from {source}...")
        print(f"  URL: {url}")
        print(f"  Output: {output_path}")
    
    try:
        # For Hugging Face, construct full URL
        if source == 'huggingface':
            # Use huggingface_hub if available
            try:
                from huggingface_hub import hf_hub_download
                repo_id = url
                filename = f"{model_name}.pth"
                
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(output_dir)
                )
                
                # Copy to expected location
                import shutil
                shutil.copy(downloaded_path, output_path)
                
            except ImportError:
                if verbose:
                    print("✗ huggingface_hub not installed")
                    print("  Install with: pip install huggingface-hub")
                return None
        else:
            # Direct download for GitHub
            urllib.request.urlretrieve(url, output_path)
        
        if verbose:
            print(f"✓ Model downloaded successfully to {output_path}")
        
        return output_path
        
    except (urllib.error.URLError, Exception) as e:
        if verbose:
            print(f"✗ Download failed: {e}")
            print("  Note: Model may not be published yet")
        return None


def load_pretrained(
    model_name: str,
    download_if_missing: bool = True,
    verbose: bool = True
):
    """
    Load a pre-trained model from the zoo.
    
    Args:
        model_name: Name of the model
        download_if_missing: Download if not found locally
        verbose: Print loading progress
        
    Returns:
        Loaded PyTorch model
        
    Examples:
        >>> model = load_pretrained('ternary_resnet18_cifar10')
        >>> model.eval()
    """
    import torch
    
    # Get model info
    try:
        info = get_model_info(model_name)
    except KeyError as e:
        if verbose:
            print(f"✗ {e}")
        return None
    
    # Check local cache
    cache_path = Path('./models/zoo') / f"{model_name}.pth"
    
    if not cache_path.exists() and download_if_missing:
        if verbose:
            print(f"Model not found locally, downloading...")
        cache_path = download_model(model_name, verbose=verbose)
        if cache_path is None:
            return None
    
    if not cache_path.exists():
        if verbose:
            print(f"✗ Model file not found: {cache_path}")
        return None
    
    # Load checkpoint first (some models need metadata from it)
    if verbose:
        print(f"Loading {model_name}...")

    checkpoint = torch.load(cache_path, map_location='cpu')

    # Load model architecture
    architecture = info['architecture']
    num_classes = info['num_classes']

    if architecture == 'ternary_resnet18':
        from models.resnet18.ternary_resnet18 import ternary_resnet18
        model = ternary_resnet18(num_classes=num_classes)
    elif architecture == 'ternary_mobilenet_v2':
        from models.mobilenetv2.ternary_mobilenetv2 import ternary_mobilenet_v2
        model = ternary_mobilenet_v2(num_classes=num_classes)
    elif architecture == 'ternary_credit_risk':
        from models.credit_risk.ternary_credit_risk import TernaryCreditRiskNet
        vocab_size = checkpoint.get('vocab_size', 5000)
        model = TernaryCreditRiskNet(vocab_size=vocab_size)
    else:
        if verbose:
            print(f"✗ Unknown architecture: {architecture}")
        return None

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    if verbose:
        print(f"✓ Model loaded successfully")
    
    return model


def print_zoo_summary():
    """Print a summary of all models in the zoo."""
    print("\n" + "="*80)
    print("TERNARY MODEL ZOO")
    print("="*80)
    
    for i, (name, info) in enumerate(MODEL_ZOO.items(), 1):
        print(f"\n{i}. {name}")
        print(f"   Architecture: {info['architecture']}")
        print(f"   Dataset: {info['dataset']}")
        print(f"   Classes: {info['num_classes']}")
        print(f"   Description: {info['description']}")
        print(f"   Expected Accuracy: {info['performance']['expected_accuracy']}")
        print(f"   Model Size: {info['performance']['model_size_mb']} MB")
        print(f"   Compression: {info['performance']['compression_ratio']}x")
        print(f"   Sources: {', '.join(info['urls'].keys())}")
    
    print("\n" + "="*80)
    print(f"Total models: {len(MODEL_ZOO)}")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Print zoo summary when run as script
    print_zoo_summary()
    
    print("Usage Examples:")
    print("-"*80)
    print("# List all models")
    print(">>> from models.model_zoo import list_models")
    print(">>> models = list_models()")
    print()
    print("# Get model info")
    print(">>> from models.model_zoo import get_model_info")
    print(">>> info = get_model_info('ternary_resnet18_cifar10')")
    print()
    print("# Download model")
    print(">>> from models.model_zoo import download_model")
    print(">>> path = download_model('ternary_resnet18_cifar10')")
    print()
    print("# Load pre-trained model")
    print(">>> from models.model_zoo import load_pretrained")
    print(">>> model = load_pretrained('ternary_resnet18_cifar10')")
    print("-"*80)

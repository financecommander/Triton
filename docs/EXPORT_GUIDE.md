# Model Export and Publishing Guide

This guide covers exporting ternary neural network models and publishing them to various platforms.

## Table of Contents

1. [ONNX Export](#onnx-export)
2. [Hugging Face Hub Publishing](#hugging-face-hub-publishing)
3. [GitHub Releases](#github-releases)
4. [Model Zoo](#model-zoo)
5. [Complete Publishing Workflow](#complete-publishing-workflow)

## ONNX Export

Export your ternary models to ONNX format for cross-platform deployment.

### Basic Usage

```python
from pathlib import Path
from backend.pytorch.export import export_to_onnx

# Load your model
model = ternary_resnet18(num_classes=10)
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# Export to ONNX
export_to_onnx(
    model=model,
    output_path=Path("resnet18.onnx"),
    input_shape=(1, 3, 32, 32),  # CIFAR-10 input size
    opset_version=13,
    verbose=True
)
```

### Validate ONNX Model

```python
from backend.pytorch.export import validate_onnx_model

# Validate exported model matches PyTorch model
validate_onnx_model(
    onnx_path=Path("resnet18.onnx"),
    pytorch_model=model,
    input_shape=(1, 3, 32, 32),
    tolerance=1e-3,
    verbose=True
)
```

### Export with Metadata

```python
from backend.pytorch.export.onnx_exporter import export_model_with_metadata

result = export_model_with_metadata(
    model=model,
    output_dir=Path("exports/"),
    model_name="ternary_resnet18_cifar10",
    input_shape=(1, 3, 32, 32),
    metadata={
        'dataset': 'cifar10',
        'accuracy': 0.89,
        'model_size_mb': 2.7
    },
    validate=True,
    verbose=True
)

print(f"ONNX model: {result['onnx_path']}")
print(f"Metadata: {result['metadata_path']}")
```

## Hugging Face Hub Publishing

Publish your models to Hugging Face Hub for easy sharing and distribution.

### Setup

```bash
# Install Hugging Face Hub
pip install huggingface-hub

# Login to Hugging Face (one-time setup)
huggingface-cli login
```

### Publish a Model

```python
from backend.pytorch.export import HuggingFacePublisher

# Initialize publisher (uses cached token if logged in)
publisher = HuggingFacePublisher()

# Push model to Hub
publisher.push_model(
    model=model,
    repo_id="username/ternary-resnet18-cifar10",
    model_name="ternary_resnet18_cifar10",
    metadata={
        'dataset': 'cifar10',
        'accuracy': 0.89,
        'architecture': 'ternary_resnet18',
        'model_size_mb': 2.7,
        'compression_ratio': 16.0
    },
    private=False,  # Set to True for private repos
    verbose=True
)
```

### Upload Checkpoint Only

```python
from pathlib import Path

publisher.push_checkpoint(
    checkpoint_path=Path("model.pth"),
    repo_id="username/ternary-resnet18-cifar10",
    commit_message="Upload trained checkpoint",
    verbose=True
)
```

### Create Model Collection

```python
# Note: Collections must be created via web interface
# This will print instructions
publisher.create_model_collection(
    collection_name="ternary-neural-networks",
    model_repos=[
        "username/ternary-resnet18-cifar10",
        "username/ternary-mobilenetv2-imagenet"
    ],
    description="Memory-efficient ternary neural networks",
    verbose=True
)
```

## GitHub Releases

Publish models as GitHub releases for version-controlled distribution.

### Setup

```bash
# Install PyGithub
pip install PyGithub

# Create a personal access token with 'repo' scope at:
# https://github.com/settings/tokens
```

### Create Release with Model

```python
from backend.pytorch.export import GitHubPublisher

# Initialize publisher
publisher = GitHubPublisher(
    token="ghp_your_token_here",
    repo="username/Triton"
)

# Create release with model
publisher.create_release_with_model(
    tag="v1.0.0",
    model=model,
    model_name="ternary_resnet18_cifar10",
    metadata={
        'dataset': 'cifar10',
        'accuracy': 0.89,
        'architecture': 'ternary_resnet18'
    },
    release_name="Ternary ResNet-18 CIFAR-10 v1.0.0",
    release_notes="First stable release",
    draft=False,
    prerelease=False,
    verbose=True
)
```

### Upload Checkpoint to Existing Release

```python
from pathlib import Path

publisher.upload_checkpoint(
    release_tag="v1.0.0",
    checkpoint_path=Path("checkpoint_epoch_100.pth"),
    label="Training Checkpoint - Epoch 100",
    verbose=True
)
```

### List Releases

```python
tags = publisher.list_releases(verbose=True)
print(f"Available releases: {tags}")
```

## Model Zoo

Access pre-trained models from the centralized model zoo.

### List Available Models

```python
from models.model_zoo import list_models, get_model_info, print_zoo_summary

# List all models
models = list_models()
print(models)

# Print detailed summary
print_zoo_summary()

# Get info for specific model
info = get_model_info('ternary_resnet18_cifar10')
print(f"Architecture: {info['architecture']}")
print(f"Dataset: {info['dataset']}")
print(f"Expected accuracy: {info['performance']['expected_accuracy']}")
```

### Download Pre-trained Model

```python
from models.model_zoo import download_model
from pathlib import Path

# Download from GitHub
model_path = download_model(
    model_name='ternary_resnet18_cifar10',
    output_dir=Path('./models/zoo'),
    source='github',  # or 'huggingface'
    verbose=True
)

print(f"Downloaded to: {model_path}")
```

### Load Pre-trained Model

```python
from models.model_zoo import load_pretrained

# Load model (downloads if needed)
model = load_pretrained(
    model_name='ternary_resnet18_cifar10',
    download_if_missing=True,
    verbose=True
)

model.eval()

# Use for inference
import torch
with torch.no_grad():
    output = model(input_tensor)
```

## Complete Publishing Workflow

Use the CLI tool for a complete publishing workflow:

### Command-Line Interface

```bash
# Export to ONNX only
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --export-onnx \
    --output exports/

# Publish to Hugging Face Hub
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --hf-repo username/ternary-resnet18 \
    --hf-token $HF_TOKEN

# Create GitHub Release
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --github-release v1.0.0 \
    --github-repo username/Triton \
    --github-token $GITHUB_TOKEN

# Do everything at once
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --dataset cifar10 \
    --num-classes 10 \
    --export-onnx \
    --onnx-validate \
    --hf-repo username/ternary-resnet18 \
    --github-release v1.0.0 \
    --github-repo username/Triton
```

### Full Example Script

```python
#!/usr/bin/env python3
"""Complete model publishing workflow."""

import torch
from pathlib import Path
from backend.pytorch.export import (
    export_to_onnx,
    HuggingFacePublisher,
    GitHubPublisher
)
from models.resnet18.ternary_resnet18 import ternary_resnet18

# Load trained model
model = ternary_resnet18(num_classes=10)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

metadata = {
    'dataset': 'cifar10',
    'accuracy': checkpoint.get('accuracy', 0.0),
    'architecture': 'ternary_resnet18',
    'epochs_trained': checkpoint.get('epoch', 0) + 1,
    'model_size_mb': 2.7,
    'compression_ratio': 16.0
}

model_name = "ternary_resnet18_cifar10"

# 1. Export to ONNX
print("Exporting to ONNX...")
export_to_onnx(
    model=model,
    output_path=Path(f"exports/{model_name}.onnx"),
    input_shape=(1, 3, 32, 32),
    verbose=True
)

# 2. Publish to Hugging Face Hub
print("\nPublishing to Hugging Face Hub...")
hf_publisher = HuggingFacePublisher()
hf_publisher.push_model(
    model=model,
    repo_id=f"username/{model_name}",
    model_name=model_name,
    metadata=metadata,
    verbose=True
)

# 3. Create GitHub Release
print("\nCreating GitHub Release...")
gh_publisher = GitHubPublisher(repo="username/Triton")
gh_publisher.create_release_with_model(
    tag="v1.0.0",
    model=model,
    model_name=model_name,
    metadata=metadata,
    verbose=True
)

print("\n✓ Publishing complete!")
```

## Environment Variables

For automated workflows, use environment variables for authentication:

```bash
# Hugging Face
export HF_TOKEN="hf_your_token_here"

# GitHub
export GITHUB_TOKEN="ghp_your_token_here"

# Then use in scripts:
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --hf-repo username/model \
    --hf-token $HF_TOKEN \
    --github-release v1.0.0 \
    --github-token $GITHUB_TOKEN
```

## Best Practices

1. **Version Your Models**: Use semantic versioning for releases (v1.0.0, v1.1.0, etc.)

2. **Include Metadata**: Always include comprehensive metadata (dataset, accuracy, training details)

3. **Validate ONNX Exports**: Use `--onnx-validate` to ensure export correctness

4. **Test Before Publishing**: Validate models locally before pushing to public repositories

5. **Use Descriptive Names**: Follow naming convention: `ternary_{architecture}_{dataset}`

6. **Document Your Models**: Include clear usage instructions in model cards and READMEs

7. **Private First**: Test with private repositories before making public

8. **Track Versions**: Keep track of which checkpoints correspond to which releases

## Troubleshooting

### ONNX Export Issues

```python
# If export fails with custom operations:
# 1. Ensure model is in eval mode
model.eval()

# 2. Try with lower opset version
export_to_onnx(..., opset_version=11)

# 3. Disable constant folding
torch.onnx.export(..., do_constant_folding=False)
```

### Hugging Face Authentication

```bash
# If authentication fails:
huggingface-cli logout
huggingface-cli login

# Or use token directly:
HuggingFacePublisher(token="your_token_here")
```

### GitHub Rate Limits

```python
# Check rate limit status
from github import Github
g = Github(token)
rate_limit = g.get_rate_limit()
print(f"Remaining: {rate_limit.core.remaining}/{rate_limit.core.limit}")
```

## Additional Resources

- [ONNX Documentation](https://onnx.ai/onnx/)
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [GitHub REST API](https://docs.github.com/en/rest)
- [PyGithub Documentation](https://pygithub.readthedocs.io/)

## License

MIT License - See repository LICENSE file for details.

# Quick Reference: Model Export & Publishing

## Installation

```bash
# Core dependencies
pip install torch torchvision

# Export dependencies
pip install -e ".[export]"
# Includes: onnx>=1.17.0, onnxruntime>=1.15.0, huggingface-hub>=0.19.0, PyGithub>=2.1.0
```

## ONNX Export

### Python API
```python
from backend.pytorch.export import export_to_onnx

export_to_onnx(
    model=model,
    output_path="model.onnx",
    input_shape=(1, 3, 32, 32)
)
```

### CLI
```bash
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --export-onnx \
    --output exports/
```

## Hugging Face Hub

### Setup
```bash
pip install huggingface-hub
huggingface-cli login
```

### Python API
```python
from backend.pytorch.export import HuggingFacePublisher

publisher = HuggingFacePublisher()
publisher.push_model(
    model=model,
    repo_id="username/model-name",
    model_name="ternary_model",
    metadata={"accuracy": 0.89}
)
```

### CLI
```bash
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --hf-repo username/ternary-resnet18 \
    --hf-token $HF_TOKEN
```

## GitHub Releases

### Setup
```bash
pip install PyGithub
# Create token: https://github.com/settings/tokens
export GITHUB_TOKEN="ghp_your_token"
```

### Python API
```python
from backend.pytorch.export import GitHubPublisher

publisher = GitHubPublisher(
    token="ghp_token",
    repo="username/Triton"
)
publisher.create_release_with_model(
    tag="v1.0.0",
    model=model,
    model_name="ternary_model",
    metadata={"accuracy": 0.89}
)
```

### CLI
```bash
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint model.pth \
    --github-release v1.0.0 \
    --github-repo username/Triton \
    --github-token $GITHUB_TOKEN
```

## Model Zoo

### List Models
```python
from models.model_zoo import list_models, print_zoo_summary

models = list_models()
print_zoo_summary()
```

### Download & Load
```python
from models.model_zoo import load_pretrained

model = load_pretrained('ternary_resnet18_cifar10')
model.eval()
```

## Complete Workflow

```bash
# Export everything at once
python models/scripts/publish_model.py \
    --model resnet18 \
    --checkpoint best_model.pth \
    --dataset cifar10 \
    --num-classes 10 \
    --export-onnx \
    --onnx-validate \
    --hf-repo username/ternary-resnet18 \
    --hf-token $HF_TOKEN \
    --github-release v1.0.0 \
    --github-repo username/Triton \
    --github-token $GITHUB_TOKEN
```

## Environment Variables

```bash
export HF_TOKEN="hf_your_token"
export GITHUB_TOKEN="ghp_your_token"
```

## Common Issues

### ONNX Export Fails
```python
# Ensure model is in eval mode
model.eval()

# Try lower opset version
export_to_onnx(..., opset_version=11)
```

### Authentication Issues
```bash
# Hugging Face
huggingface-cli logout
huggingface-cli login

# GitHub
# Regenerate token with 'repo' scope
```

## Documentation

- Full Guide: [docs/EXPORT_GUIDE.md](../docs/EXPORT_GUIDE.md)
- Example: [examples/export_and_publish_example.py](../examples/export_and_publish_example.py)
- API Docs: [backend/pytorch/export/](../backend/pytorch/export/)

## Support

- Issues: https://github.com/financecommander/Triton/issues
- Docs: https://github.com/financecommander/Triton/docs

"""
Hugging Face Hub Integration for Ternary Models

Provides utilities for publishing ternary neural networks to Hugging Face Hub,
including model cards, metadata, and checkpoint management.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import warnings


class HuggingFacePublisher:
    """
    Publisher for uploading ternary models to Hugging Face Hub.
    
    Handles authentication, model card generation, and file uploads to create
    a complete model repository on Hugging Face Hub.
    
    Examples:
        >>> publisher = HuggingFacePublisher(token="hf_...")
        >>> publisher.push_model(
        ...     model=my_model,
        ...     repo_id="username/ternary-resnet18",
        ...     model_name="ternary_resnet18",
        ...     metadata={"dataset": "cifar10", "accuracy": 0.89}
        ... )
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize Hugging Face Hub publisher.
        
        Args:
            token: Hugging Face API token. If None, will try to use cached token.
        """
        try:
            from huggingface_hub import HfApi, create_repo, upload_file
            self.HfApi = HfApi
            self.create_repo = create_repo
            self.upload_file = upload_file
            self._hub_available = True
        except ImportError:
            warnings.warn(
                "huggingface_hub not installed. Install with: pip install huggingface-hub"
            )
            self._hub_available = False
            return
        
        self.token = token
        self.api = HfApi(token=token) if self._hub_available else None
    
    def is_available(self) -> bool:
        """Check if Hugging Face Hub integration is available."""
        return self._hub_available
    
    def push_model(
        self,
        model: nn.Module,
        repo_id: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        model_card_data: Optional[Dict[str, Any]] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Push a ternary model to Hugging Face Hub.
        
        Creates a repository (if needed), uploads the model checkpoint,
        and generates a model card with metadata.
        
        Args:
            model: PyTorch model to upload
            repo_id: Repository ID (e.g., "username/model-name")
            model_name: Name for the model file
            metadata: Model metadata (architecture, dataset, metrics, etc.)
            model_card_data: Additional data for model card
            commit_message: Commit message for the upload
            private: Whether to create a private repository
            verbose: Print upload progress
            
        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.is_available():
            print("✗ Hugging Face Hub not available")
            return False
        
        try:
            if verbose:
                print(f"Pushing model to Hugging Face Hub: {repo_id}")
            
            # Create repository if it doesn't exist
            try:
                self.create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True,
                    token=self.token
                )
                if verbose:
                    print(f"✓ Repository {repo_id} ready")
            except Exception as e:
                if verbose:
                    print(f"✗ Failed to create repository: {e}")
                return False
            
            # Save model to temporary file
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Save model checkpoint
                model_path = tmpdir_path / f"{model_name}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata or {}
                }, model_path)
                
                # Upload model checkpoint
                if verbose:
                    print(f"  Uploading model checkpoint...")
                self.upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo=f"{model_name}.pth",
                    repo_id=repo_id,
                    token=self.token,
                    commit_message=commit_message or f"Upload {model_name}"
                )
                
                # Generate and upload model card
                model_card = self._generate_model_card(
                    model_name=model_name,
                    metadata=metadata or {},
                    additional_data=model_card_data or {}
                )
                
                model_card_path = tmpdir_path / "README.md"
                with open(model_card_path, 'w') as f:
                    f.write(model_card)
                
                if verbose:
                    print(f"  Uploading model card...")
                self.upload_file(
                    path_or_fileobj=str(model_card_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=self.token,
                    commit_message=commit_message or "Update model card"
                )
                
                # Upload metadata JSON
                if metadata:
                    metadata_path = tmpdir_path / "metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    if verbose:
                        print(f"  Uploading metadata...")
                    self.upload_file(
                        path_or_fileobj=str(metadata_path),
                        path_in_repo="metadata.json",
                        repo_id=repo_id,
                        token=self.token,
                        commit_message=commit_message or "Upload metadata"
                    )
            
            if verbose:
                print(f"✓ Model successfully pushed to https://huggingface.co/{repo_id}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ Upload failed: {e}")
            return False
    
    def push_checkpoint(
        self,
        checkpoint_path: Path,
        repo_id: str,
        path_in_repo: Optional[str] = None,
        commit_message: Optional[str] = None,
        verbose: bool = True
    ) -> bool:
        """
        Upload a checkpoint file to Hugging Face Hub.
        
        Args:
            checkpoint_path: Path to checkpoint file
            repo_id: Repository ID
            path_in_repo: Path in repository (default: checkpoint filename)
            commit_message: Commit message
            verbose: Print progress
            
        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.is_available():
            print("✗ Hugging Face Hub not available")
            return False
        
        try:
            if path_in_repo is None:
                path_in_repo = checkpoint_path.name
            
            if verbose:
                print(f"Uploading checkpoint to {repo_id}/{path_in_repo}...")
            
            self.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.token,
                commit_message=commit_message or f"Upload {checkpoint_path.name}"
            )
            
            if verbose:
                print(f"✓ Checkpoint uploaded successfully")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ Upload failed: {e}")
            return False
    
    def create_model_collection(
        self,
        collection_name: str,
        model_repos: List[str],
        description: Optional[str] = None,
        verbose: bool = True
    ) -> bool:
        """
        Create a collection of related models on Hugging Face Hub.
        
        Args:
            collection_name: Name for the collection
            model_repos: List of model repository IDs to include
            description: Description of the collection
            verbose: Print progress
            
        Returns:
            True if collection created successfully, False otherwise
        """
        if not self.is_available():
            print("✗ Hugging Face Hub not available")
            return False
        
        if verbose:
            print(f"Creating model collection: {collection_name}")
            print(f"  Models: {', '.join(model_repos)}")
        
        # Note: Collection creation via API is limited
        # This is a placeholder for when the feature becomes available
        warnings.warn(
            "Model collections are currently managed via the web interface. "
            f"Please visit https://huggingface.co/collections to create '{collection_name}' "
            f"and add these models: {', '.join(model_repos)}"
        )
        
        return False
    
    def _generate_model_card(
        self,
        model_name: str,
        metadata: Dict[str, Any],
        additional_data: Dict[str, Any]
    ) -> str:
        """
        Generate a model card in Hugging Face format.
        
        Args:
            model_name: Name of the model
            metadata: Model metadata
            additional_data: Additional data for the card
            
        Returns:
            Model card content as string
        """
        # Extract metadata
        dataset = metadata.get('dataset', 'Unknown')
        accuracy = metadata.get('accuracy', metadata.get('final_accuracy', 'N/A'))
        architecture = metadata.get('architecture', metadata.get('model_name', model_name))
        compression = metadata.get('compression_ratio', 'N/A')
        model_size = metadata.get('model_size_mb', metadata.get('memory', {}).get('model_size_mb', 'N/A'))
        
        # Build model card
        card = f"""---
tags:
- ternary-neural-network
- quantization
- efficient-ai
- {dataset.lower() if isinstance(dataset, str) else 'vision'}
library_name: pytorch
license: mit
---

# {architecture}

A memory-efficient ternary neural network with 2-bit quantized weights (-1, 0, 1).

## Model Description

This model uses ternary quantization to compress weights to 2 bits per parameter, achieving
significant memory savings compared to standard 32-bit floating point models.

**Architecture:** {architecture}  
**Dataset:** {dataset}  
**Quantization:** Ternary (2-bit weights)

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy if isinstance(accuracy, str) else f'{accuracy:.2%}'} |
| Model Size | {model_size if isinstance(model_size, str) else f'{model_size:.2f} MB'} |
| Compression | {compression if isinstance(compression, str) else f'{compression:.1f}x'} |

## Usage

```python
import torch
from models.{architecture.split('_')[1] if '_' in architecture else architecture}.{architecture} import {architecture}

# Load model
model = {architecture}(num_classes={metadata.get('num_classes', 10)})
checkpoint = torch.load('{model_name}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

## Training Details

- **Framework:** PyTorch
- **Quantization Method:** Ternary with Straight-Through Estimator (STE)
- **Weight Values:** {{-1, 0, 1}}
- **Compression:** ~16x memory reduction vs FP32

## Technical Details

Ternary neural networks constrain weights to three values (-1, 0, 1), enabling:
- 2-bit packed storage (4 weights per byte)
- Reduced memory bandwidth requirements
- Potential for specialized hardware acceleration
- Approximate inference with minimal accuracy loss

## Citation

```bibtex
@software{{triton_ternary_{model_name},
  title = {{{architecture}: Ternary Neural Network}},
  author = {{Triton DSL Project}},
  year = {{2024}},
  url = {{https://github.com/financecommander/Triton}}
}}
```

## License

MIT License - See repository for details.

## Model Card Authors

Generated by Triton DSL Model Export System
"""
        
        return card

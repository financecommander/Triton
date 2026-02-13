"""
GitHub Releases Integration for Ternary Models

Provides utilities for publishing model checkpoints and artifacts to GitHub Releases,
enabling easy distribution and version management.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import warnings
import zipfile
import tempfile


class GitHubPublisher:
    """
    Publisher for uploading ternary models to GitHub Releases.
    
    Handles authentication, release creation, and asset uploads for distributing
    models via GitHub.
    
    Examples:
        >>> publisher = GitHubPublisher(token="ghp_...", repo="user/repo")
        >>> publisher.create_release_with_model(
        ...     tag="v1.0.0",
        ...     model=my_model,
        ...     model_name="ternary_resnet18",
        ...     metadata={"dataset": "cifar10"}
        ... )
    """
    
    def __init__(
        self,
        token: Optional[str] = None,
        repo: Optional[str] = None
    ):
        """
        Initialize GitHub publisher.
        
        Args:
            token: GitHub personal access token
            repo: Repository in format "owner/repo"
        """
        try:
            from github import Github
            self.Github = Github
            self._github_available = True
        except ImportError:
            warnings.warn(
                "PyGithub not installed. Install with: pip install PyGithub"
            )
            self._github_available = False
            return
        
        self.token = token
        self.repo_name = repo
        self.github = Github(token) if token and self._github_available else None
        self.repo = None
        
        if self.github and repo:
            try:
                self.repo = self.github.get_repo(repo)
            except Exception as e:
                warnings.warn(f"Failed to access repository {repo}: {e}")
    
    def is_available(self) -> bool:
        """Check if GitHub integration is available."""
        return self._github_available and self.github is not None
    
    def set_repository(self, repo: str) -> bool:
        """
        Set the target repository.
        
        Args:
            repo: Repository in format "owner/repo"
            
        Returns:
            True if repository was set successfully
        """
        if not self.is_available():
            return False
        
        try:
            self.repo_name = repo
            self.repo = self.github.get_repo(repo)
            return True
        except Exception as e:
            warnings.warn(f"Failed to access repository {repo}: {e}")
            return False
    
    def create_release_with_model(
        self,
        tag: str,
        model: nn.Module,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        release_name: Optional[str] = None,
        release_notes: Optional[str] = None,
        draft: bool = False,
        prerelease: bool = False,
        verbose: bool = True
    ) -> bool:
        """
        Create a GitHub release with model checkpoint.
        
        Args:
            tag: Git tag for the release (e.g., "v1.0.0")
            model: PyTorch model to upload
            model_name: Name for the model file
            metadata: Model metadata
            release_name: Name for the release (default: tag)
            release_notes: Release notes/description
            draft: Create as draft release
            prerelease: Mark as prerelease
            verbose: Print progress
            
        Returns:
            True if release created successfully
        """
        if not self.is_available() or not self.repo:
            if verbose:
                print("✗ GitHub integration not available or repository not set")
            return False
        
        try:
            if verbose:
                print(f"Creating GitHub release {tag} in {self.repo_name}...")
            
            # Generate release notes if not provided
            if release_notes is None:
                release_notes = self._generate_release_notes(
                    model_name, metadata or {}
                )
            
            # Create release
            release = self.repo.create_git_release(
                tag=tag,
                name=release_name or tag,
                message=release_notes,
                draft=draft,
                prerelease=prerelease
            )
            
            if verbose:
                print(f"✓ Release {tag} created")
            
            # Create temporary directory for files
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Save model checkpoint
                model_path = tmpdir_path / f"{model_name}.pth"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'metadata': metadata or {},
                    'tag': tag
                }, model_path)
                
                # Upload model checkpoint
                if verbose:
                    print(f"  Uploading {model_name}.pth...")
                release.upload_asset(
                    str(model_path),
                    label=f"{model_name} Model Checkpoint",
                    content_type="application/octet-stream"
                )
                
                # Save and upload metadata
                if metadata:
                    metadata_path = tmpdir_path / f"{model_name}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    if verbose:
                        print(f"  Uploading metadata...")
                    release.upload_asset(
                        str(metadata_path),
                        label=f"{model_name} Metadata",
                        content_type="application/json"
                    )
                
                # Create and upload README
                readme = self._generate_readme(model_name, metadata or {}, tag)
                readme_path = tmpdir_path / "README.md"
                with open(readme_path, 'w') as f:
                    f.write(readme)
                
                if verbose:
                    print(f"  Uploading README...")
                release.upload_asset(
                    str(readme_path),
                    label="Model Documentation",
                    content_type="text/markdown"
                )
                
                # Create ZIP package
                zip_path = tmpdir_path / f"{model_name}_package.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(model_path, f"{model_name}.pth")
                    if metadata:
                        zipf.write(metadata_path, f"{model_name}_metadata.json")
                    zipf.write(readme_path, "README.md")
                
                if verbose:
                    print(f"  Uploading package...")
                release.upload_asset(
                    str(zip_path),
                    label=f"{model_name} Complete Package",
                    content_type="application/zip"
                )
            
            if verbose:
                print(f"✓ Release complete: {release.html_url}")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ Release creation failed: {e}")
            return False
    
    def upload_checkpoint(
        self,
        release_tag: str,
        checkpoint_path: Path,
        label: Optional[str] = None,
        verbose: bool = True
    ) -> bool:
        """
        Upload a checkpoint to an existing release.
        
        Args:
            release_tag: Tag of the release
            checkpoint_path: Path to checkpoint file
            label: Label for the asset
            verbose: Print progress
            
        Returns:
            True if upload succeeded
        """
        if not self.is_available() or not self.repo:
            if verbose:
                print("✗ GitHub integration not available")
            return False
        
        try:
            # Get release
            release = self.repo.get_release(release_tag)
            
            if verbose:
                print(f"Uploading {checkpoint_path.name} to release {release_tag}...")
            
            # Upload asset
            release.upload_asset(
                str(checkpoint_path),
                label=label or checkpoint_path.name,
                content_type="application/octet-stream"
            )
            
            if verbose:
                print(f"✓ Checkpoint uploaded successfully")
            
            return True
            
        except Exception as e:
            if verbose:
                print(f"✗ Upload failed: {e}")
            return False
    
    def list_releases(self, verbose: bool = True) -> List[str]:
        """
        List all releases in the repository.
        
        Args:
            verbose: Print release information
            
        Returns:
            List of release tags
        """
        if not self.is_available() or not self.repo:
            return []
        
        try:
            releases = list(self.repo.get_releases())
            tags = [r.tag_name for r in releases]
            
            if verbose:
                print(f"Releases in {self.repo_name}:")
                for release in releases:
                    print(f"  {release.tag_name}: {release.title}")
            
            return tags
            
        except Exception as e:
            if verbose:
                print(f"✗ Failed to list releases: {e}")
            return []
    
    def _generate_release_notes(
        self,
        model_name: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate release notes for the model."""
        architecture = metadata.get('architecture', model_name)
        dataset = metadata.get('dataset', 'Unknown')
        accuracy = metadata.get('accuracy', metadata.get('final_accuracy', 'N/A'))
        model_size = metadata.get('model_size_mb', metadata.get('memory', {}).get('model_size_mb', 'N/A'))
        compression = metadata.get('compression_ratio', 'N/A')
        
        notes = f"""# {architecture} - Ternary Neural Network

A memory-efficient neural network with ternary quantized weights (-1, 0, 1).

## Model Information

- **Architecture:** {architecture}
- **Dataset:** {dataset}
- **Quantization:** Ternary (2-bit weights)
- **Accuracy:** {accuracy if isinstance(accuracy, str) else f'{accuracy:.2%}'}
- **Model Size:** {model_size if isinstance(model_size, str) else f'{model_size:.2f} MB'}
- **Compression:** {compression if isinstance(compression, str) else f'{compression:.1f}x vs FP32'}

## Download

Download the complete package or individual files:

- `{model_name}_package.zip` - Complete package with model, metadata, and documentation
- `{model_name}.pth` - PyTorch model checkpoint
- `{model_name}_metadata.json` - Model metadata and training info
- `README.md` - Usage instructions

## Usage

```python
import torch

# Load model checkpoint
checkpoint = torch.load('{model_name}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# View metadata
print(checkpoint['metadata'])
```

## Requirements

- PyTorch >= 2.0.0
- Python >= 3.10

See README.md for detailed usage instructions.
"""
        return notes
    
    def _generate_readme(
        self,
        model_name: str,
        metadata: Dict[str, Any],
        tag: str
    ) -> str:
        """Generate README for the model."""
        architecture = metadata.get('architecture', model_name)
        dataset = metadata.get('dataset', 'Unknown')
        
        readme = f"""# {architecture}

Ternary Neural Network model from the Triton DSL project.

## Installation

```bash
# Install PyTorch
pip install torch>=2.0.0

# Download model
wget https://github.com/{self.repo_name}/releases/download/{tag}/{model_name}.pth
```

## Usage

```python
import torch
from models.{architecture.split('_')[1] if '_' in architecture else architecture}.{architecture} import {architecture}

# Load model
model = {architecture}(num_classes={metadata.get('num_classes', 10)})
checkpoint = torch.load('{model_name}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1)
```

## Model Details

- **Dataset:** {dataset}
- **Quantization:** Ternary (2-bit weights: -1, 0, 1)
- **Framework:** PyTorch
- **License:** MIT

## Citation

```bibtex
@software{{triton_ternary,
  title = {{Triton: DSL for Ternary Neural Networks}},
  author = {{Triton DSL Project}},
  year = {{2024}},
  url = {{https://github.com/financecommander/Triton}}
}}
```

## License

MIT License - See repository for details.
"""
        return readme

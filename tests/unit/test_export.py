"""
Unit tests for model export functionality.

Tests ONNX export, Hugging Face Hub integration, and GitHub Releases publishing.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Import export utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.pytorch.export import export_to_onnx, validate_onnx_model
from backend.pytorch.export import HuggingFacePublisher, GitHubPublisher


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    model.eval()
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# ONNX Export Tests
# ============================================================================

class TestONNXExport:
    """Tests for ONNX export functionality."""
    
    def test_export_simple_model(self, simple_model, temp_dir):
        """Test basic ONNX export."""
        output_path = temp_dir / "model.onnx"
        
        success = export_to_onnx(
            model=simple_model,
            output_path=output_path,
            input_shape=(1, 10),
            verbose=False
        )
        
        assert success, "ONNX export should succeed"
        assert output_path.exists(), "ONNX file should be created"
    
    def test_export_with_dynamic_axes(self, simple_model, temp_dir):
        """Test ONNX export with dynamic batch size."""
        output_path = temp_dir / "model_dynamic.onnx"
        
        success = export_to_onnx(
            model=simple_model,
            output_path=output_path,
            input_shape=(1, 10),
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            verbose=False
        )
        
        assert success, "ONNX export with dynamic axes should succeed"
        assert output_path.exists(), "ONNX file should be created"
    
    def test_export_creates_directory(self, simple_model, temp_dir):
        """Test that export creates output directory if needed."""
        output_path = temp_dir / "subdir" / "model.onnx"
        
        success = export_to_onnx(
            model=simple_model,
            output_path=output_path,
            input_shape=(1, 10),
            verbose=False
        )
        
        assert success, "Export should succeed"
        assert output_path.exists(), "ONNX file should be created"
        assert output_path.parent.exists(), "Parent directory should be created"
    
    def test_export_different_opset_versions(self, simple_model, temp_dir):
        """Test ONNX export with different opset versions."""
        for opset in [11, 12, 13]:
            output_path = temp_dir / f"model_opset{opset}.onnx"
            
            success = export_to_onnx(
                model=simple_model,
                output_path=output_path,
                input_shape=(1, 10),
                opset_version=opset,
                verbose=False
            )
            
            assert success, f"Export with opset {opset} should succeed"
            assert output_path.exists(), f"ONNX file for opset {opset} should be created"
    
    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
        reason="onnxruntime required for validation"
    )
    def test_validate_onnx_model(self, simple_model, temp_dir):
        """Test ONNX model validation."""
        output_path = temp_dir / "model.onnx"
        
        # Export model
        export_success = export_to_onnx(
            model=simple_model,
            output_path=output_path,
            input_shape=(1, 10),
            verbose=False
        )
        
        assert export_success, "Export should succeed"
        
        # Validate model
        validation_success = validate_onnx_model(
            onnx_path=output_path,
            pytorch_model=simple_model,
            input_shape=(1, 10),
            verbose=False
        )
        
        assert validation_success, "Validation should succeed"


# ============================================================================
# Hugging Face Hub Tests
# ============================================================================

class TestHuggingFacePublisher:
    """Tests for Hugging Face Hub integration."""
    
    def test_publisher_initialization_without_token(self):
        """Test publisher initialization without token."""
        publisher = HuggingFacePublisher()
        # Should not raise error, but may not be available
        assert isinstance(publisher, HuggingFacePublisher)
    
    def test_publisher_availability_check(self):
        """Test availability check."""
        publisher = HuggingFacePublisher()
        is_available = publisher.is_available()
        assert isinstance(is_available, bool)
    
    @patch('backend.pytorch.export.huggingface_hub.HfApi')
    def test_push_model_creates_repo(self, mock_hf_api, simple_model, temp_dir):
        """Test that push_model creates repository."""
        publisher = HuggingFacePublisher(token="test_token")
        
        if not publisher.is_available():
            pytest.skip("Hugging Face Hub not available")
        
        # Mock the API calls
        with patch.object(publisher, 'create_repo') as mock_create_repo:
            with patch.object(publisher, 'upload_file') as mock_upload_file:
                success = publisher.push_model(
                    model=simple_model,
                    repo_id="test/model",
                    model_name="test_model",
                    metadata={"test": "data"},
                    verbose=False
                )
                
                # Should attempt to create repo
                mock_create_repo.assert_called_once()
    
    def test_model_card_generation(self, simple_model):
        """Test model card generation."""
        publisher = HuggingFacePublisher()
        
        if not publisher.is_available():
            pytest.skip("Hugging Face Hub not available")
        
        metadata = {
            'dataset': 'test_dataset',
            'accuracy': 0.95,
            'architecture': 'test_model',
            'model_size_mb': 10.5
        }
        
        card = publisher._generate_model_card(
            model_name="test_model",
            metadata=metadata,
            additional_data={}
        )
        
        assert isinstance(card, str)
        assert 'test_dataset' in card
        assert 'test_model' in card
        assert len(card) > 100, "Model card should have substantial content"


# ============================================================================
# GitHub Publisher Tests
# ============================================================================

class TestGitHubPublisher:
    """Tests for GitHub Releases integration."""
    
    def test_publisher_initialization_without_token(self):
        """Test publisher initialization without token."""
        publisher = GitHubPublisher()
        assert isinstance(publisher, GitHubPublisher)
    
    def test_publisher_availability_check(self):
        """Test availability check."""
        publisher = GitHubPublisher()
        is_available = publisher.is_available()
        assert isinstance(is_available, bool)
    
    def test_set_repository(self):
        """Test repository setting."""
        publisher = GitHubPublisher(token="test_token")
        
        if not publisher.is_available():
            pytest.skip("GitHub integration not available")
        
        # Should not raise error even with invalid repo
        # (will fail when trying to access it)
        result = publisher.set_repository("invalid/repo")
        assert isinstance(result, bool)
    
    def test_release_notes_generation(self, simple_model):
        """Test release notes generation."""
        publisher = GitHubPublisher(repo="test/repo")
        
        if not publisher.is_available():
            pytest.skip("GitHub integration not available")
        
        metadata = {
            'architecture': 'test_model',
            'dataset': 'test_dataset',
            'accuracy': 0.89,
            'model_size_mb': 5.2
        }
        
        notes = publisher._generate_release_notes(
            model_name="test_model",
            metadata=metadata
        )
        
        assert isinstance(notes, str)
        assert 'test_model' in notes
        assert 'test_dataset' in notes
        assert len(notes) > 100, "Release notes should have substantial content"
    
    def test_readme_generation(self, simple_model):
        """Test README generation."""
        publisher = GitHubPublisher(repo="test/repo")
        
        if not publisher.is_available():
            pytest.skip("GitHub integration not available")
        
        metadata = {
            'architecture': 'test_model',
            'dataset': 'test_dataset',
            'num_classes': 10
        }
        
        readme = publisher._generate_readme(
            model_name="test_model",
            metadata=metadata,
            tag="v1.0.0"
        )
        
        assert isinstance(readme, str)
        assert 'test_model' in readme
        assert 'Installation' in readme
        assert 'Usage' in readme
        assert len(readme) > 100, "README should have substantial content"


# ============================================================================
# Integration Tests
# ============================================================================

class TestExportIntegration:
    """Integration tests for export functionality."""
    
    def test_export_workflow(self, simple_model, temp_dir):
        """Test complete export workflow."""
        # Export to ONNX
        onnx_path = temp_dir / "model.onnx"
        export_success = export_to_onnx(
            model=simple_model,
            output_path=onnx_path,
            input_shape=(1, 10),
            verbose=False
        )
        
        assert export_success, "ONNX export should succeed"
        assert onnx_path.exists(), "ONNX file should exist"
        
        # Save metadata
        metadata = {
            'model_name': 'test_model',
            'architecture': 'simple',
            'dataset': 'test'
        }
        
        metadata_path = temp_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        assert metadata_path.exists(), "Metadata file should exist"
        
        # Verify metadata
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata == metadata, "Metadata should match"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

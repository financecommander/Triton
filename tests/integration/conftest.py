"""
Pytest fixtures for integration tests.
Provides common test data, models, and utilities.
"""

import pytest
import torch
import torch.nn as nn
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import LayerDef, Param, Program
from backend.pytorch.codegen import generate_pytorch_code, PyTorchCodeGenerator


@pytest.fixture(scope="session")
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_layer_def():
    """Simple ternary layer definition for testing."""
    return LayerDef(
        name="SimpleTernaryLayer",
        params=[
            Param(name="weights", param_type="TernaryTensor", shape=[64, 128]),
            Param(name="bias", param_type="TernaryTensor", shape=[128]),
        ],
        body=[]
    )


@pytest.fixture
def multi_layer_def():
    """Multi-parameter ternary layer definition."""
    return LayerDef(
        name="MultiLayerTernary",
        params=[
            Param(name="weights1", param_type="TernaryTensor", shape=[32, 64]),
            Param(name="weights2", param_type="TernaryTensor", shape=[64, 128]),
            Param(name="weights3", param_type="TernaryTensor", shape=[128, 256]),
            Param(name="bias", param_type="TernaryTensor", shape=[256]),
        ],
        body=[]
    )


@pytest.fixture
def compiled_simple_model(simple_layer_def):
    """Generate and instantiate a simple compiled model."""
    code = generate_pytorch_code(simple_layer_def)
    namespace = {}
    exec(code, namespace)
    model_class = namespace["SimpleTernaryLayer"]
    return model_class()


@pytest.fixture
def compiled_multi_model(multi_layer_def):
    """Generate and instantiate a multi-layer compiled model."""
    code = generate_pytorch_code(multi_layer_def)
    namespace = {}
    exec(code, namespace)
    model_class = namespace["MultiLayerTernary"]
    return model_class()


@pytest.fixture
def sample_input_2d():
    """Sample 2D input tensor for testing."""
    return torch.randn(4, 64)  # Batch size 4, feature dim 64


@pytest.fixture
def sample_input_4d():
    """Sample 4D input tensor for CNN testing."""
    return torch.randn(2, 3, 32, 32)  # Batch size 2, 3 channels, 32x32 image


@pytest.fixture
def cifar10_like_input():
    """CIFAR-10 like input tensor."""
    return torch.randn(8, 3, 32, 32)  # Batch size 8, RGB, 32x32


@pytest.fixture
def imagenet_like_input():
    """ImageNet-like input tensor."""
    return torch.randn(2, 3, 224, 224)  # Batch size 2, RGB, 224x224


@pytest.fixture(scope="session")
def pytorch_generator():
    """Reusable PyTorch code generator instance."""
    return PyTorchCodeGenerator()


class SimpleModel(nn.Module):
    """Simple PyTorch model for comparison testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def reference_pytorch_model():
    """Reference PyTorch model for comparison."""
    return SimpleModel()


class CNNModel(nn.Module):
    """Simple CNN model for testing."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def reference_cnn_model():
    """Reference CNN model for testing."""
    return CNNModel()


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        'warmup_iterations': 10,
        'benchmark_iterations': 100,
        'batch_sizes': [1, 4, 8, 16, 32],
        'input_sizes': [(64,), (128,), (256,), (512,)],
    }


@pytest.fixture
def quantization_config():
    """Configuration for quantization tests."""
    return {
        'fp32_to_fp16': True,
        'fp16_to_int8': True,
        'fp16_to_ternary': True,
        'calibration_samples': 100,
        'threshold': 0.7,
    }


def create_test_model_code(layer_name: str, in_features: int, out_features: int) -> str:
    """Helper to create test model code."""
    layer_def = LayerDef(
        name=layer_name,
        params=[
            Param(name="weights", param_type="TernaryTensor", shape=[in_features, out_features]),
            Param(name="bias", param_type="TernaryTensor", shape=[out_features]),
        ],
        body=[]
    )
    return generate_pytorch_code(layer_def)


@pytest.fixture
def model_factory():
    """Factory function for creating test models."""
    def _create_model(in_features: int, out_features: int, name: str = "TestModel"):
        code = create_test_model_code(name, in_features, out_features)
        namespace = {}
        exec(code, namespace)
        return namespace[name]()
    return _create_model


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def training_config():
    """Configuration for training tests."""
    return {
        'epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_samples': 1000,
        'num_classes': 10,
        'input_dim': 64,
    }


@pytest.fixture
def mock_dataset(training_config):
    """Mock dataset for training tests."""
    num_samples = training_config['num_samples']
    input_dim = training_config['input_dim']
    num_classes = training_config['num_classes']
    
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    return torch.utils.data.TensorDataset(X, y)


@pytest.fixture
def mock_dataloader(mock_dataset, training_config):
    """Mock dataloader for training tests."""
    return torch.utils.data.DataLoader(
        mock_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True
    )

"""
Triton DSL Examples
===================

This package contains production-quality examples demonstrating Triton DSL
capabilities for building memory-efficient neural networks with ternary and
mixed-precision quantization.

Quick Start
-----------
>>> import examples
>>> from examples.mnist_ternary import TernaryNet, train, evaluate
>>> 
>>> # Create and train a ternary model
>>> model = TernaryNet()
>>> train(model, epochs=10)
>>> accuracy = evaluate(model)

Example Categories
------------------
- basic/: Foundational model architectures
- quantization/: Advanced quantization techniques
- custom/: Custom quantized layer implementations
- training/: Production training scripts
- deployment/: Model export and serving
- notebooks/: Interactive tutorials

For more information, see the README.md in each subdirectory.
"""

__version__ = "0.1.0"
__author__ = "Finance Commander"

# Version info
VERSION = __version__

# Export commonly used functions
try:
    from .mnist_ternary import (
        TernaryNet,
        train as train_mnist,
        evaluate as evaluate_mnist,
        save_ternary_model,
        load_ternary_model,
    )
    __all__ = [
        "TernaryNet",
        "train_mnist",
        "evaluate_mnist",
        "save_ternary_model",
        "load_ternary_model",
    ]
except ImportError:
    # mnist_ternary might not be available if dependencies not installed
    __all__ = []

# Example categories
CATEGORIES = {
    "basic": "Foundational model architectures with ternary quantization",
    "quantization": "Advanced quantization techniques and QAT",
    "custom": "Custom quantized layer implementations",
    "training": "Production-ready training scripts",
    "deployment": "Model export and deployment examples",
    "notebooks": "Interactive Jupyter notebook tutorials",
}

def list_examples():
    """List all available examples by category."""
    from pathlib import Path
    
    examples_dir = Path(__file__).parent
    
    for category, description in CATEGORIES.items():
        category_path = examples_dir / category
        if category_path.exists():
            print(f"\n{category.upper()}: {description}")
            print("-" * 60)
            
            # List files
            for file in sorted(category_path.glob("*")):
                if file.is_file() and file.suffix in [".py", ".triton", ".ipynb", ".md"]:
                    print(f"  - {file.name}")

def get_example_path(category, name):
    """Get the path to a specific example file.
    
    Args:
        category: Example category (basic, quantization, custom, etc.)
        name: Example name (e.g., "simple_mlp.triton")
    
    Returns:
        Path to the example file
    """
    from pathlib import Path
    
    examples_dir = Path(__file__).parent
    example_path = examples_dir / category / name
    
    if not example_path.exists():
        raise FileNotFoundError(f"Example not found: {category}/{name}")
    
    return example_path

# Provide helpful info when imported
def _print_info():
    """Print helpful information about the examples package."""
    print("Triton DSL Examples")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"\nAvailable categories: {', '.join(CATEGORIES.keys())}")
    print("\nUse `examples.list_examples()` to see all examples")
    print("Use `examples.get_example_path(category, name)` to get file paths")
    print("\nFor more info: https://github.com/financecommander/Triton")

# Print info when imported interactively
if __name__ != "__main__":
    import sys
    if hasattr(sys, 'ps1'):  # Interactive mode
        _print_info()

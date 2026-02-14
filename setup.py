#!/usr/bin/env python3
"""
Triton DSL - Setup Configuration
Enhanced setup.py with extras_require, version checking, and install validation.
"""

import sys
import os
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

# Version requirements
MIN_PYTHON_VERSION = (3, 10)
RECOMMENDED_PYTHON_VERSION = (3, 11)

# Read version from pyproject.toml
def get_version():
    """Extract version from pyproject.toml"""
    pyproject_path = Path(__file__).parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path) as f:
            for line in f:
                if line.startswith("version"):
                    return line.split("=")[1].strip().strip('"')
    return "0.1.0"

# Check Python version
def check_python_version():
    """Validate Python version meets minimum requirements"""
    current = sys.version_info[:2]
    if current < MIN_PYTHON_VERSION:
        sys.stderr.write(
            f"ERROR: Triton DSL requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+\n"
            f"Current Python version: {current[0]}.{current[1]}\n"
        )
        sys.exit(1)
    elif current < RECOMMENDED_PYTHON_VERSION:
        print(
            f"WARNING: Python {RECOMMENDED_PYTHON_VERSION[0]}.{RECOMMENDED_PYTHON_VERSION[1]}+ "
            f"is recommended for best performance. Current: {current[0]}.{current[1]}"
        )

# Core dependencies
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "numpy>=1.24.0",
    "ply>=3.11",
    "jinja2>=3.0.0",
]

# Optional dependency groups (extras_require)
EXTRAS_REQUIRE = {
    # Development dependencies
    "dev": [
        # Testing
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-benchmark>=4.0.0",
        "hypothesis>=6.0.0",
        # Linting and formatting
        "black>=23.0.0",
        "flake8>=7.0.0",
        "mypy>=1.7.0",
        "pylint>=3.0.0",
        "ruff>=0.1.0",
        # Documentation
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=2.0.0",
        "sphinx-autodoc-typehints>=1.25.0",
        "myst-parser>=2.0.0",
        # Development tools
        "ipython>=8.18.0",
        "jupyterlab>=4.0.0",
        "jupyter>=1.0.0",
        # Type stubs
        "types-setuptools>=69.0.0",
    ],
    
    # GPU support
    "gpu": [
        "cuda-python>=12.0.0",
        "triton>=2.1.0",
    ],
    
    # Examples and demos
    "examples": [
        "torchvision>=0.16.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
    ],
    
    # Model export
    "export": [
        "onnx>=1.17.0",
        "onnxruntime>=1.15.0",
        "huggingface-hub>=0.19.0",
        "PyGithub>=2.1.0",
    ],
}

# Combine all extras for complete installation
EXTRAS_REQUIRE["all"] = [
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
]

# Custom install command with validation
class CustomInstallCommand(install):
    """Custom install command with version checking"""
    
    def run(self):
        check_python_version()
        print("Installing Triton DSL...")
        install.run(self)
        self.validate_installation()
    
    def validate_installation(self):
        """Validate that core dependencies are installed correctly"""
        print("\nValidating installation...")
        try:
            import torch
            import numpy
            import ply
            import jinja2
            print(f"✓ PyTorch {torch.__version__}")
            print(f"✓ NumPy {numpy.__version__}")
            print(f"✓ PLY (Python Lex-Yacc)")
            print(f"✓ Jinja2 {jinja2.__version__}")
            print("\n✅ Installation validated successfully!")
        except ImportError as e:
            print(f"\n⚠️  Warning: Could not validate installation: {e}")

# Custom develop command with validation
class CustomDevelopCommand(develop):
    """Custom develop command with version checking"""
    
    def run(self):
        check_python_version()
        print("Installing Triton DSL in development mode...")
        develop.run(self)

# Read long description from README
def read_readme():
    """Read README.md for long description"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return "Domain-Specific Language for Ternary Neural Networks"

# Main setup configuration
if __name__ == "__main__":
    check_python_version()
    
    setup(
        name="triton-dsl",
        version=get_version(),
        description="Domain-Specific Language for Ternary Neural Networks",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        author="Finance Commander",
        author_email="dev@financecommander.com",
        url="https://github.com/financecommander/Triton",
        project_urls={
            "Documentation": "https://github.com/financecommander/Triton/docs",
            "Source": "https://github.com/financecommander/Triton",
            "Issues": "https://github.com/financecommander/Triton/issues",
        },
        license="MIT",
        packages=find_packages(
            include=["compiler*", "backend*", "kernels*", "examples*"]
        ),
        python_requires=">=3.10",
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        cmdclass={
            "install": CustomInstallCommand,
            "develop": CustomDevelopCommand,
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Software Development :: Compilers",
        ],
        keywords=[
            "machine-learning",
            "neural-networks",
            "quantization",
            "dsl",
            "compiler",
            "ternary",
            "tnn",
        ],
        # Note: entry_points should be defined in pyproject.toml [project.scripts]
        # to avoid conflicts. Uncomment if needed:
        # entry_points={
        #     "console_scripts": [
        #         "triton-dsl=compiler.cli:main",
        #     ],
        # },
    )

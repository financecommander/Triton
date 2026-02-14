# Installation Guide

Complete installation guide for Triton DSL with requirements files, optional dependencies, and compatibility information.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation Options](#installation-options)
- [Requirements Files](#requirements-files)
- [Optional Dependencies](#optional-dependencies)
- [Compatibility Matrix](#compatibility-matrix)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Quick Start

### Minimal Installation (Production)

For production use with core functionality only:

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install from PyPI (when published)
pip install triton-dsl
```

### Development Installation

For development with testing, linting, and documentation tools:

```bash
# Install base + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Or using setup.py
pip install -e ".[dev]"
```

---

## Installation Options

### Option 1: Using Requirements Files (Recommended)

Requirements files provide pinned versions for reproducible installations:

```bash
# 1. Production (core dependencies only)
pip install -r requirements.txt

# 2. Development (adds testing, linting, docs)
pip install -r requirements.txt -r requirements-dev.txt

# 3. GPU Support (adds CUDA, Triton)
pip install -r requirements.txt -r requirements-gpu.txt

# 4. Examples (adds visualization, datasets)
pip install -r requirements.txt -r requirements-examples.txt

# 5. Complete Installation (all optional dependencies)
pip install -r requirements.txt -r requirements-dev.txt -r requirements-gpu.txt -r requirements-examples.txt
```

### Option 2: Using setup.py Extras

Install with optional dependency groups:

```bash
# Development
pip install -e ".[dev]"

# GPU support
pip install -e ".[gpu]"

# Examples
pip install -e ".[examples]"

# Model export
pip install -e ".[export]"

# Complete installation
pip install -e ".[all]"

# Multiple groups
pip install -e ".[dev,gpu,examples]"
```

### Option 3: Using pyproject.toml (pip 21.3+)

Modern pip versions support pyproject.toml directly:

```bash
# Install with optional dependencies
pip install -e ".[dev]"
pip install -e ".[all]"
```

---

## Requirements Files

### requirements.txt (Production)

**Core dependencies for running Triton DSL compiler and runtime**

- `torch>=2.1.0` - PyTorch deep learning framework
- `numpy>=1.24.0` - Numerical computing
- `ply>=3.11` - Lexer/parser (Python Lex-Yacc)
- `jinja2>=3.0.0` - Code generation templates

**Install:** `pip install -r requirements.txt`

### requirements-dev.txt (Development)

**Development tools, testing, linting, and documentation**

Testing:
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `hypothesis>=6.0.0` - Property-based testing

Linting & Formatting:
- `black>=23.0.0` - Code formatter
- `flake8>=7.0.0` - Style checker
- `mypy>=1.7.0` - Static type checker
- `pylint>=3.0.0` - Code analyzer
- `ruff>=0.1.0` - Fast Python linter

Documentation:
- `sphinx>=7.2.0` - Documentation generator
- `sphinx-rtd-theme>=2.0.0` - ReadTheDocs theme
- `sphinx-autodoc-typehints>=1.25.0` - Type hints in docs
- `myst-parser>=2.0.0` - Markdown support

Development Tools:
- `ipython>=8.18.0` - Enhanced Python REPL
- `jupyterlab>=4.0.0` - Jupyter notebooks
- `jupyter>=1.0.0` - Jupyter core

**Install:** `pip install -r requirements.txt -r requirements-dev.txt`

### requirements-gpu.txt (GPU Support)

**CUDA and GPU acceleration dependencies**

- `torch==2.1.0+cu121` - PyTorch with CUDA 12.1
- `triton>=2.1.0` - OpenAI Triton for GPU kernels
- `cuda-python>=12.0.0` - CUDA Python bindings

**Prerequisites:**
- NVIDIA GPU with CUDA Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- CUDA Toolkit 12.0+ installed on system
- cuDNN 8.9+ (optional, for optimizations)

**Install:** `pip install -r requirements.txt -r requirements-gpu.txt`

### requirements-examples.txt (Examples/Demos)

**Dependencies for running examples and demos**

- `torchvision>=0.16.0` - Computer vision datasets and models
- `matplotlib>=3.7.0` - Plotting and visualization
- `seaborn>=0.12.0` - Statistical visualization
- `pandas>=2.0.0` - Data analysis
- `scipy>=1.11.0` - Scientific computing
- `scikit-learn>=1.3.0` - Machine learning utilities

**Install:** `pip install -r requirements.txt -r requirements-examples.txt`

---

## Optional Dependencies

### Development (`dev`)

Install for contributing to Triton DSL:

```bash
pip install -e ".[dev]"
```

Includes: testing frameworks, linters, formatters, documentation tools, and development utilities.

### GPU Support (`gpu`)

Install for CUDA-accelerated GPU operations:

```bash
pip install -e ".[gpu]"
```

**Note:** Requires NVIDIA GPU and CUDA Toolkit 12.0+ installed on your system.

### Examples (`examples`)

Install for running examples and demos:

```bash
pip install -e ".[examples]"
```

Includes: torchvision, visualization libraries, and ML utilities.

### Export (`export`)

Install for model export capabilities:

```bash
pip install -e ".[export]"
```

Includes: ONNX export, HuggingFace Hub publishing, GitHub release creation.

### All (`all`)

Complete installation with all optional dependencies:

```bash
pip install -e ".[all]"
```

---

## Compatibility Matrix

### Python Versions

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.10 | ✅ Supported | Minimum required version |
| 3.11 | ✅ Recommended | Best performance |
| 3.12 | ✅ Supported | Latest features |
| 3.9 | ❌ Not Supported | Use Python 3.10+ |

### PyTorch Versions

| PyTorch | Status | CUDA Support |
|---------|--------|--------------|
| 2.1.0+ | ✅ Supported | CUDA 11.8, 12.1 |
| 2.0.x | ⚠️ Partial | May work but untested |
| 1.x | ❌ Not Supported | Use PyTorch 2.1+ |

### CUDA Versions (GPU Support)

| CUDA Version | Status | GPU Architecture |
|--------------|--------|------------------|
| 12.1 | ✅ Recommended | Volta, Turing, Ampere, Ada, Hopper |
| 12.0 | ✅ Supported | Volta, Turing, Ampere, Ada, Hopper |
| 11.8 | ⚠️ Limited | Volta, Turing, Ampere |
| 11.x | ❌ Not Recommended | Use CUDA 12.0+ |

### Operating Systems

| OS | Status | Notes |
|----|--------|-------|
| Linux (Ubuntu 20.04+) | ✅ Fully Supported | Recommended for production |
| Linux (Other distros) | ✅ Supported | Debian, Fedora, Arch, etc. |
| macOS (Intel) | ✅ Supported | CPU-only (no GPU support) |
| macOS (Apple Silicon) | ⚠️ Experimental | CPU-only, use MPS for acceleration |
| Windows 10/11 | ✅ Supported | WSL2 recommended for best experience |

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# 1. Update system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# 2. Create virtual environment
python3 -m venv triton-env
source triton-env/bin/activate

# 3. Install Triton DSL
pip install -r requirements.txt -r requirements-dev.txt

# 4. For GPU support (requires CUDA installed)
pip install -r requirements-gpu.txt
```

### macOS

```bash
# 1. Install Python 3.10+ (using Homebrew)
brew install python@3.11

# 2. Create virtual environment
python3.11 -m venv triton-env
source triton-env/bin/activate

# 3. Install Triton DSL
pip install -r requirements.txt -r requirements-dev.txt

# Note: GPU support not available on macOS
# Use MPS (Metal Performance Shaders) for acceleration on Apple Silicon
```

### Windows

```bash
# Option 1: Using WSL2 (Recommended)
# Follow Linux instructions in WSL2 Ubuntu

# Option 2: Native Windows
# 1. Install Python 3.10+ from python.org
# 2. Open PowerShell or Command Prompt

# 3. Create virtual environment
python -m venv triton-env
.\triton-env\Scripts\activate

# 4. Install Triton DSL
pip install -r requirements.txt -r requirements-dev.txt

# 5. For GPU support (requires CUDA installed)
pip install -r requirements-gpu.txt
```

### Docker

```dockerfile
# Example Dockerfile for Triton DSL
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Copy requirements
COPY requirements*.txt /app/

# Install dependencies
WORKDIR /app
RUN pip install -r requirements.txt -r requirements-gpu.txt

# Copy source
COPY . /app

# Install Triton DSL
RUN pip install -e .

CMD ["python3", "-m", "pytest"]
```

---

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Fails

**Problem:** `pip install torch` fails or takes too long

**Solution:**
```bash
# Use specific PyTorch index
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Or for CPU-only
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 2. CUDA Not Found

**Problem:** `RuntimeError: CUDA not available`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

#### 3. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'compiler'`

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Triton"
```

#### 4. Version Conflicts

**Problem:** Dependency version conflicts

**Solution:**
```bash
# Use virtual environment (recommended)
python -m venv fresh-env
source fresh-env/bin/activate  # Linux/macOS
# OR
.\fresh-env\Scripts\activate  # Windows

# Install with pinned versions
pip install -r requirements.txt
```

#### 5. Memory Issues

**Problem:** Out of memory during installation

**Solution:**
```bash
# Increase pip timeout and use no-cache
pip install --no-cache-dir -r requirements.txt

# Install one at a time
pip install torch==2.1.0
pip install numpy==1.24.3
# ... continue
```

### Getting Help

- **GitHub Issues:** https://github.com/financecommander/Triton/issues
- **Documentation:** https://github.com/financecommander/Triton/docs
- **Discussions:** https://github.com/financecommander/Triton/discussions

---

## Verification

After installation, verify everything is working:

```bash
# 1. Check Python version
python --version  # Should be 3.10+

# 2. Verify core imports
python -c "import torch; import numpy; import ply; import jinja2; print('✓ Core dependencies OK')"

# 3. Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 4. Check CUDA availability (if GPU installed)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Run minimal test
python -m pytest tests/unit/test_lexer.py -v

# 6. Check Triton DSL imports (after pip install -e .)
python -c "from compiler.lexer import TernaryLexer; from backend.pytorch import TernaryLinear; print('✓ Triton DSL imports OK')"
```

### Expected Output

```
Python 3.11.x
✓ Core dependencies OK
PyTorch: 2.1.0
CUDA available: True
✓ Triton DSL imports OK
```

---

## Development Workflow

For contributors and developers:

```bash
# 1. Clone repository
git clone https://github.com/financecommander/Triton.git
cd Triton

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows

# 3. Install in development mode with all dependencies
pip install -e ".[all]"

# 4. Run tests
pytest tests/

# 5. Run linters
black .
ruff check .
mypy compiler/ backend/

# 6. Build documentation
cd docs
sphinx-build -b html . _build/
```

---

## Updating Dependencies

To update to the latest versions:

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade torch

# Regenerate requirements with current versions
pip freeze > requirements-frozen.txt
```

---

## Uninstallation

To completely remove Triton DSL:

```bash
# Uninstall package
pip uninstall triton-dsl

# Remove virtual environment (if used)
deactivate
rm -rf venv/  # or triton-env/

# Clean up cache
pip cache purge
```

---

**Last Updated:** 2024-02-14
**Version:** 0.1.0

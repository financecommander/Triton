# Development Environment Setup

This guide walks you through setting up a complete development environment for contributing to Triton DSL.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Repository Setup](#repository-setup)
- [Virtual Environment](#virtual-environment)
- [Development Dependencies](#development-dependencies)
- [IDE Configuration](#ide-configuration)
- [Build and Test Setup](#build-and-test-setup)
- [Docker Development Environment](#docker-development-environment)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

**Python 3.10 or higher**
```bash
# Check Python version
python --version  # Should be 3.10+

# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev

# On macOS with Homebrew
brew install python@3.10

# On Windows, download from python.org
```

**Git**
```bash
# Ubuntu/Debian
sudo apt-get install git

# macOS
brew install git

# Verify installation
git --version
```

**Build Tools (Linux)**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# Fedora/RHEL
sudo dnf groupinstall "Development Tools"
```

### Optional but Recommended

**CUDA Toolkit (for GPU development)**
```bash
# Check if CUDA is available
nvidia-smi

# Install CUDA Toolkit 12.0+ from:
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
```

**Docker (for containerized development)**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker run hello-world
```

## Repository Setup

### 1. Fork the Repository

1. Navigate to https://github.com/financecommander/Triton
2. Click the "Fork" button in the top-right corner
3. Wait for the fork to complete

### 2. Clone Your Fork

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Triton.git
cd Triton

# Add upstream remote
git remote add upstream https://github.com/financecommander/Triton.git

# Verify remotes
git remote -v
# Output should show:
# origin    https://github.com/YOUR_USERNAME/Triton.git (fetch)
# origin    https://github.com/YOUR_USERNAME/Triton.git (push)
# upstream  https://github.com/financecommander/Triton.git (fetch)
# upstream  https://github.com/financecommander/Triton.git (push)
```

### 3. Configure Git

```bash
# Set your identity
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Enable helpful defaults
git config --global pull.rebase true
git config --global fetch.prune true
git config --global diff.colorMoved zebra
```

## Virtual Environment

### Using venv (Recommended)

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show path to venv)
which python
```

### Using conda (Alternative)

```bash
# Create conda environment
conda create -n triton-dev python=3.10
conda activate triton-dev

# Install pip in conda environment
conda install pip
```

### Environment Variables

Add to your `.bashrc`, `.zshrc`, or virtual environment activation script:

```bash
# Add to venv/bin/activate or ~/.bashrc
export TRITON_DEV=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Adjust for your GPU setup
```

## Development Dependencies

### Install Core Dependencies

```bash
# Ensure virtual environment is activated
which python  # Should point to venv/bin/python

# Install in development mode with all optional dependencies
pip install -e ".[dev,benchmark,examples,cuda,export]"

# Or install dependencies individually:
pip install -e .                    # Core dependencies
pip install -e ".[dev]"             # Development tools
pip install -e ".[benchmark]"       # Benchmarking tools
pip install -e ".[examples]"        # Example dependencies
pip install -e ".[cuda]"           # CUDA support
pip install -e ".[export]"         # Model export tools
```

### Core Dependencies Installed

- **PyTorch** (≥2.1.0) - Deep learning framework
- **torchvision** (≥0.16.0) - Vision utilities
- **NumPy** (≥1.24.0) - Numerical computing
- **PLY** (≥3.11) - Parser generator
- **Jinja2** (≥3.0.0) - Template engine

### Development Dependencies

- **pytest** (≥7.4.0) - Testing framework
- **pytest-cov** (≥4.1.0) - Coverage reporting
- **black** (≥23.0.0) - Code formatter
- **ruff** (≥0.1.0) - Fast linter
- **mypy** (≥1.7.0) - Type checker
- **hypothesis** (≥6.0.0) - Property-based testing

### Verify Installation

```bash
# Check installed packages
pip list | grep -E "torch|pytest|black|ruff|mypy"

# Test imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytest; print(f'pytest: {pytest.__version__}')"
python -c "import black; print(f'black: {black.__version__}')"
python -c "import ruff; print('ruff installed')"

# Check CUDA availability (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## IDE Configuration

### Visual Studio Code

**1. Install VS Code**
```bash
# Ubuntu/Debian
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list'
sudo apt-get update
sudo apt-get install code

# macOS
brew install --cask visual-studio-code
```

**2. Install Extensions**
```bash
# Install via CLI
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
code --install-extension njpwerner.autodocstring
code --install-extension ms-vscode.makefile-tools
```

**3. Workspace Settings**

Create `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests"
  ],
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/parser.out": true,
    "**/parsetab.py": true
  }
}
```

**4. Launch Configuration**

Create `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Python: Run Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"],
      "console": "integratedTerminal"
    },
    {
      "name": "Python: Run Single Test",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["${file}", "-v", "-s"],
      "console": "integratedTerminal"
    },
    {
      "name": "Python: MNIST Example",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/examples/mnist_ternary.py",
      "args": ["--epochs", "5"],
      "console": "integratedTerminal"
    }
  ]
}
```

### PyCharm

**1. Open Project**
- File → Open → Select Triton directory

**2. Configure Python Interpreter**
- File → Settings → Project → Python Interpreter
- Click gear icon → Add → Existing environment
- Select `venv/bin/python`

**3. Configure Code Style**
- File → Settings → Tools → Black
  - ✓ Enable Black formatter
  - Arguments: `--line-length 100`
  
**4. Configure Testing**
- File → Settings → Tools → Python Integrated Tools
  - Default test runner: pytest
  - ✓ pytest

**5. External Tools - Black**
- File → Settings → Tools → External Tools → Add
  - Name: Black
  - Program: `$PyInterpreterDirectory$/black`
  - Arguments: `--line-length 100 $FilePath$`
  - Working directory: `$ProjectFileDir$`

**6. External Tools - Ruff**
- File → Settings → Tools → External Tools → Add
  - Name: Ruff
  - Program: `$PyInterpreterDirectory$/ruff`
  - Arguments: `check $FilePath$`
  - Working directory: `$ProjectFileDir$`

**7. File Watchers (Optional)**
- Install "File Watchers" plugin
- Settings → Tools → File Watchers → Add
  - Black watcher: Run on save
  - Ruff watcher: Run on save

## Build and Test Setup

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_parser.py -v

# Run specific test function
pytest tests/unit/test_parser.py::test_parse_declaration -v

# Run with coverage
pytest tests/ --cov=compiler --cov=backend --cov-report=html

# Run only unit tests
pytest tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run with markers
pytest -m "not slow" -v  # Skip slow tests
```

### Code Quality Checks

```bash
# Format code with Black
black compiler/ backend/ kernels/ tests/ examples/

# Check formatting without changes
black --check compiler/ backend/ kernels/ tests/ examples/

# Lint with Ruff
ruff check compiler/ backend/ kernels/ tests/ examples/

# Auto-fix linting issues
ruff check --fix compiler/ backend/ kernels/ tests/ examples/

# Type check with mypy
mypy compiler/ backend/ kernels/

# Run all checks (typical pre-commit workflow)
black --check . && ruff check . && mypy compiler/ backend/ kernels/ && pytest tests/ -v
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before commits:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10
        args: [--line-length=100]
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1000]
```

## Docker Development Environment

### Dockerfile for Development

Create `Dockerfile.dev`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/triton

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev,benchmark,examples,cuda,export]"

# Create non-root user
RUN useradd -m -u 1000 developer && \
    chown -R developer:developer /workspace
USER developer

# Default command
CMD ["/bin/bash"]
```

### Docker Compose Setup

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  triton-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace/triton
      - triton-cache:/home/developer/.cache
    environment:
      - PYTHONPATH=/workspace/triton
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
    shm_size: '2gb'
    tty: true
    stdin_open: true
    command: /bin/bash

volumes:
  triton-cache:
```

### Using Docker

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Enter container
docker-compose exec triton-dev bash

# Inside container, run tests
pytest tests/ -v

# Stop container
docker-compose down

# Build without cache (fresh start)
docker-compose build --no-cache
```

### Docker Without Compose

```bash
# Build image
docker build -t triton-dev:latest -f Dockerfile.dev .

# Run container
docker run -it --rm \
  --gpus all \
  -v $(pwd):/workspace/triton \
  -e PYTHONPATH=/workspace/triton \
  triton-dev:latest

# Run tests in container
docker run --rm \
  -v $(pwd):/workspace/triton \
  triton-dev:latest \
  pytest tests/ -v
```

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'compiler'`**
```bash
# Solution: Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

**Issue: `parser.out` or `parsetab.py` causing issues**
```bash
# Solution: Remove PLY cache files
find . -name "parser.out" -delete
find . -name "parsetab.py" -delete
```

**Issue: PyTorch CUDA not available**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Issue: Black/Ruff not found in IDE**
```bash
# Ensure tools are installed in the same environment
which python
which black
which ruff

# If not found, reinstall
pip install black ruff
```

**Issue: Tests fail with `FileNotFoundError`**
```bash
# Ensure you're running from repository root
cd /path/to/Triton
pytest tests/ -v
```

**Issue: Permission denied in Docker**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER .

# Or run container as current user
docker run --user $(id -u):$(id -g) ...
```

### Getting Help

- **GitHub Issues**: https://github.com/financecommander/Triton/issues
- **Discussions**: https://github.com/financecommander/Triton/discussions
- Check existing documentation in `docs/`

### Next Steps

After setting up your development environment:

1. Read [Code Style Guide](code_style.md)
2. Review [Testing Requirements](testing.md)
3. Understand [PR Process](pr_process.md)
4. Check [Technical Specification](../specs/TECHNICAL_SPEC.md)
5. Try running the examples:
   ```bash
   python examples/mnist_ternary.py
   ```

## Quick Reference

```bash
# Essential commands
source venv/bin/activate               # Activate environment
pip install -e ".[dev]"               # Install dependencies
pytest tests/ -v                       # Run tests
black .                                # Format code
ruff check .                          # Lint code
mypy compiler/ backend/ kernels/      # Type check
git fetch upstream                     # Sync with upstream
git rebase upstream/main               # Update branch
```

Welcome to Triton DSL development! 🚀

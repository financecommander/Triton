# Requirements Quick Reference

Quick reference guide for Triton DSL dependency management.

## Available Requirements Files

| File | Purpose | Install Command |
|------|---------|-----------------|
| `requirements.txt` | Production (core only) | `pip install -r requirements.txt` |
| `requirements-dev.txt` | Development (testing, linting, docs) | `pip install -r requirements.txt -r requirements-dev.txt` |
| `requirements-gpu.txt` | GPU support (CUDA, Triton) | `pip install -r requirements.txt -r requirements-gpu.txt` |
| `requirements-examples.txt` | Examples and demos | `pip install -r requirements.txt -r requirements-examples.txt` |

## Common Installation Scenarios

### 1. Minimal Production Install
```bash
pip install -r requirements.txt
```
**Includes:** torch, numpy, ply, jinja2

### 2. Development Environment
```bash
pip install -r requirements.txt -r requirements-dev.txt
# OR using setup.py
pip install -e ".[dev]"
```
**Includes:** All testing, linting, documentation, and development tools

### 3. GPU Development
```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-gpu.txt
# OR using setup.py
pip install -e ".[dev,gpu]"
```
**Includes:** Development tools + CUDA/GPU support

### 4. Complete Installation
```bash
pip install -r requirements.txt -r requirements-dev.txt -r requirements-gpu.txt -r requirements-examples.txt
# OR using setup.py
pip install -e ".[all]"
```
**Includes:** Everything (dev, GPU, examples, export)

### 5. Examples Only
```bash
pip install -r requirements.txt -r requirements-examples.txt
# OR using setup.py
pip install -e ".[examples]"
```
**Includes:** Visualization and demo dependencies

## Optional Dependency Groups (setup.py)

Use these with `pip install -e ".[group]"`:

| Group | Description | Key Packages |
|-------|-------------|--------------|
| `dev` | Development tools | pytest, black, mypy, sphinx |
| `gpu` | GPU acceleration | triton, cuda-python |
| `examples` | Examples/demos | torchvision, matplotlib, scikit-learn |
| `export` | Model export | onnx, huggingface-hub, PyGithub |
| `all` | Complete install | All of the above |

### Combining Groups

Install multiple groups together:
```bash
pip install -e ".[dev,gpu]"
pip install -e ".[dev,examples,export]"
```

## Core Dependencies

### Production (requirements.txt)
```
torch>=2.1.0,<3.0.0          # PyTorch deep learning
numpy>=1.24.0,<3.0.0         # Numerical computing
ply>=3.11,<4.0.0             # Lexer/parser
jinja2>=3.1.0,<4.0.0         # Code generation
```

### Development (requirements-dev.txt)
```
Testing:
  pytest>=7.4.0              # Test framework
  pytest-cov>=4.1.0          # Coverage
  pytest-benchmark>=4.0.0    # Benchmarks
  hypothesis>=6.0.0          # Property testing

Code Quality:
  black>=23.12.0             # Formatter
  flake8>=7.0.0              # Linter
  mypy>=1.7.0                # Type checker
  pylint>=3.0.0              # Code analyzer
  ruff>=0.1.8                # Fast linter

Documentation:
  sphinx>=7.2.0              # Doc generator
  sphinx-rtd-theme>=2.0.0    # RTD theme

Dev Tools:
  ipython>=8.18.0            # Enhanced REPL
  jupyterlab>=4.0.0          # Notebooks
```

### GPU Support (requirements-gpu.txt)
```
torch>=2.1.0                 # PyTorch + CUDA 12.1
triton>=2.1.0                # OpenAI Triton
cuda-python>=12.0.0          # CUDA bindings
```
**Requires:** CUDA Toolkit 12.0+ on system

### Examples (requirements-examples.txt)
```
torchvision>=0.16.0          # Computer vision
matplotlib>=3.7.0            # Plotting
seaborn>=0.12.0              # Stats visualization
pandas>=2.0.0                # Data analysis
scipy>=1.11.0                # Scientific computing
scikit-learn>=1.3.0          # ML utilities
```

## Version Constraints

All requirements use **flexible version ranges** for compatibility:
- `>=X.Y.Z,<MAJOR+1.0.0` - Allow minor/patch updates
- `==X.Y.Z` - Pinned versions (only for critical dependencies)

### Why Flexible Ranges?

✅ Compatible with newer releases  
✅ Automatic security updates  
✅ Easier dependency resolution  
✅ Better long-term maintainability

If you need **reproducible builds**, use:
```bash
pip freeze > requirements-lock.txt
```

## Updating Dependencies

### Update All Packages
```bash
pip install --upgrade -r requirements.txt
```

### Update Specific Package
```bash
pip install --upgrade torch
```

### Generate Lock File
```bash
# Install current versions
pip install -r requirements.txt

# Freeze exact versions
pip freeze > requirements-lock.txt
```

## System Requirements

### Python Version
- **Minimum:** Python 3.10
- **Recommended:** Python 3.11
- **Supported:** Python 3.10, 3.11, 3.12

### PyTorch Version
- **Minimum:** PyTorch 2.1.0
- **Recommended:** Latest 2.x release
- **CUDA:** 11.8, 12.1, 12.4 (for GPU)

### GPU Requirements (Optional)
- **NVIDIA GPU:** Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- **CUDA Toolkit:** 12.0+ installed on system
- **Driver:** Latest NVIDIA drivers recommended

### Storage
- **Minimal:** ~500 MB (core dependencies)
- **Development:** ~2 GB (with dev tools)
- **Complete:** ~5 GB (all dependencies + models)

## Virtual Environment Setup

### Using venv (Built-in)
```bash
# Create environment
python3 -m venv triton-env

# Activate
source triton-env/bin/activate     # Linux/macOS
.\triton-env\Scripts\activate      # Windows

# Install
pip install -r requirements.txt

# Deactivate when done
deactivate
```

### Using conda
```bash
# Create environment
conda create -n triton python=3.11

# Activate
conda activate triton

# Install
pip install -r requirements.txt

# Deactivate when done
conda deactivate
```

## Troubleshooting

### Issue: Package Version Conflict
```bash
# Solution: Use fresh virtual environment
python3 -m venv fresh-env
source fresh-env/bin/activate
pip install -r requirements.txt
```

### Issue: PyTorch Download Slow
```bash
# Solution: Use specific index
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory
```bash
# Solution: Install without cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Import Errors
```bash
# Solution: Install in editable mode
pip install -e .
```

## Platform-Specific Notes

### Linux
✅ Fully supported (recommended)  
✅ GPU support available  
✅ All features work

### macOS
✅ CPU-only support  
⚠️ No CUDA support  
⚠️ Use MPS for acceleration (Apple Silicon)

### Windows
✅ Native Windows supported  
✅ GPU support available  
💡 WSL2 recommended for best experience

## CI/CD Integration

### GitHub Actions
```yaml
- name: Install dependencies
  run: |
    pip install -r requirements.txt -r requirements-dev.txt
```

### Docker
```dockerfile
COPY requirements*.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt
```

### Testing
```bash
# Install test dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run tests
pytest tests/
```

## More Information

- **Full Installation Guide:** [INSTALLATION.md](INSTALLATION.md)
- **Project README:** [README.md](README.md)
- **Contributing Guide:** (coming soon)

---

**Last Updated:** 2024-02-14  
**Version:** 0.1.0

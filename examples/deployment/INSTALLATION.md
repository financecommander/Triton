# Installation Guide for Deployment Examples

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Docker (for Docker deployment)
- Git (for cloning repository)

## Basic Installation

### 1. Install Core Dependencies

```bash
# Install PyTorch and basic dependencies
pip install torch>=2.1.0 torchvision>=0.16.0 numpy>=1.24.0
```

### 2. Install Triton DSL

```bash
# Clone repository
git clone https://github.com/financecommander/Triton.git
cd Triton

# Install in development mode
pip install -e .
```

## Optional Dependencies

### ONNX Export

```bash
pip install onnx>=1.15.0 onnxruntime>=1.16.0 onnx-simplifier
```

### Mobile Optimization

```bash
# TensorFlow Lite
pip install tensorflow>=2.0.0 onnx-tf

# CoreML (macOS only)
pip install coremltools
```

### Hugging Face Hub

```bash
pip install huggingface-hub>=0.19.0
```

### Docker Deployment

```bash
# Install from requirements file
pip install -r examples/deployment/docker_deployment/requirements.txt

# Or install individually
pip install flask>=3.0.0 flask-cors>=4.0.0 gunicorn>=21.2.0
pip install prometheus-client>=0.19.0 Pillow>=10.0.0
```

## Complete Installation (All Features)

```bash
# Install all optional dependencies
pip install onnx onnxruntime onnx-simplifier \
    tensorflow onnx-tf \
    coremltools \
    huggingface-hub \
    flask flask-cors gunicorn \
    prometheus-client Pillow
```

Or use the export optional dependencies:

```bash
pip install -e ".[export]"
```

## Docker Installation

### Install Docker

**Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
```bash
brew install --cask docker
```

**Windows:**
Download from https://docs.docker.com/desktop/install/windows-install/

### Install Docker Compose

```bash
# Ubuntu/Debian
sudo apt-get install docker-compose-plugin

# macOS
brew install docker-compose

# Or install as Python package
pip install docker-compose
```

### GPU Support (Optional)

For NVIDIA GPU support:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Verification

### Verify Installation

```bash
# Navigate to deployment examples
cd examples/deployment

# Run verification script
./verify_deployment.sh
```

### Test Individual Components

```bash
# Test ONNX export (without actual export)
python export_onnx.py --help

# Test mobile optimization (without actual export)
python optimize_for_mobile.py --help

# Test Hugging Face integration (without upload)
python huggingface_hub.py --help

# Test Docker setup
cd docker_deployment
docker-compose config
```

## Troubleshooting

### Issue: Module not found

```bash
# Ensure Triton is installed
pip install -e .

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Permission denied (Docker)

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

### Issue: CUDA/GPU not available

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Missing dependencies

```bash
# Install all dependencies
pip install -r examples/deployment/docker_deployment/requirements.txt
pip install onnx onnxruntime onnx-simplifier tensorflow onnx-tf coremltools huggingface-hub
```

## Platform-Specific Notes

### macOS

- CoreML is only available on macOS
- Use `brew` for Docker installation
- Some bash scripts may need minor adjustments for BSD tools

### Windows

- Use WSL2 for best compatibility
- Docker Desktop required
- Git Bash or PowerShell recommended

### Linux

- Full support for all features
- GPU support available with NVIDIA drivers
- Package managers vary by distribution

## Next Steps

After installation:

1. **Read the documentation**
   ```bash
   cat examples/deployment/README.md
   ```

2. **Try the examples**
   ```bash
   python export_onnx.py --model simple --output test.onnx
   ```

3. **Run the test suite**
   ```bash
   cd docker_deployment
   python test_api.py
   ```

4. **Deploy your model**
   ```bash
   docker-compose up -d
   ```

## Support

If you encounter issues:

1. Check the README files in each directory
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Check Python and Docker versions
5. Open an issue on GitHub with full error details

## License

MIT License - see LICENSE file for details

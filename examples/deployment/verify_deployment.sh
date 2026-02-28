#!/bin/bash
# Verification script for deployment examples

echo "=========================================="
echo "Triton DSL Deployment Examples Verification"
echo "=========================================="
echo ""

# Check if all files exist
echo "Checking files..."
files=(
    "export_onnx.py"
    "optimize_for_mobile.py"
    "huggingface_hub.py"
    "README.md"
    "DEPLOYMENT_SUMMARY.md"
    "docker_deployment/Dockerfile"
    "docker_deployment/app.py"
    "docker_deployment/requirements.txt"
    "docker_deployment/docker-compose.yml"
    "docker_deployment/README.md"
    "docker_deployment/nginx.conf"
    "docker_deployment/prometheus.yml"
    "docker_deployment/test_api.py"
)

missing=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (missing)"
        missing=$((missing + 1))
    fi
done

echo ""
if [ $missing -eq 0 ]; then
    echo "✓ All files present (${#files[@]}/${#files[@]})"
else
    echo "✗ Missing $missing files"
    exit 1
fi

echo ""
echo "Checking Python syntax..."
python_files=(
    "export_onnx.py"
    "optimize_for_mobile.py"
    "huggingface_hub.py"
    "docker_deployment/app.py"
    "docker_deployment/test_api.py"
)

for file in "${python_files[@]}"; do
    if python3 -m py_compile "$file" 2>/dev/null; then
        echo "✓ $file"
    else
        echo "✗ $file (syntax error)"
        exit 1
    fi
done

echo ""
echo "Checking file permissions..."
for file in "${python_files[@]}"; do
    if [ -x "$file" ]; then
        echo "✓ $file (executable)"
    else
        echo "  $file (not executable - this is OK)"
    fi
done

echo ""
echo "File statistics:"
total_lines=0
total_size=0

for file in "${python_files[@]}"; do
    lines=$(wc -l < "$file")
    size=$(wc -c < "$file")
    total_lines=$((total_lines + lines))
    total_size=$((total_size + size))
done

echo "  Total Python files: ${#python_files[@]}"
echo "  Total lines of code: $total_lines"
echo "  Total size: $((total_size / 1024)) KB"

echo ""
echo "=========================================="
echo "✓ Verification complete!"
echo "=========================================="
echo ""
echo "Quick start commands:"
echo "  python export_onnx.py --help"
echo "  python optimize_for_mobile.py --help"
echo "  python huggingface_hub.py --help"
echo "  cd docker_deployment && docker-compose up -d"

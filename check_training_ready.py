#!/usr/bin/env python3
"""
Training Readiness Checker

Validates that everything is ready to launch CIFAR-10 training.
"""

import sys
import os
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70)

def check_command(cmd, description):
    """Check if a command is available"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        return True
    except:
        return False

def check_python_module(module_name):
    """Check if a Python module is installed"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def main():
    print_header("CIFAR-10 Training Readiness Check")
    
    all_checks = []
    
    # Check Python
    print("\n1. Checking Python...")
    python_version = sys.version.split()[0]
    print(f"   ✓ Python {python_version}")
    all_checks.append(True)
    
    # Check required files
    print("\n2. Checking required files...")
    required_files = [
        "models/scripts/train_ternary_models.py",
        "launch_training.sh",
        "START_HERE.md",
        "docs/CIFAR10_TRAINING_GUIDE.md",
        "examples/cifar10_training_examples.sh"
    ]
    
    files_ok = True
    for filepath in required_files:
        if check_file_exists(filepath):
            print(f"   ✓ {filepath}")
        else:
            print(f"   ✗ {filepath} NOT FOUND")
            files_ok = False
    all_checks.append(files_ok)
    
    # Check directories
    print("\n3. Checking/creating directories...")
    directories = ["checkpoints", "logs", "results", "data"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"   ✓ Created {directory}/")
        else:
            print(f"   ✓ {directory}/ exists")
    all_checks.append(True)
    
    # Check Python dependencies
    print("\n4. Checking Python dependencies...")
    dependencies = {
        "torch": "PyTorch (deep learning framework)",
        "torchvision": "TorchVision (datasets and models)",
        "numpy": "NumPy (numerical computing)",
        "tensorboard": "TensorBoard (visualization, optional)"
    }
    
    deps_installed = 0
    deps_missing = []
    
    for module, desc in dependencies.items():
        if check_python_module(module):
            print(f"   ✓ {module} - {desc}")
            deps_installed += 1
        else:
            print(f"   ✗ {module} - {desc}")
            deps_missing.append(module)
    
    all_checks.append(len(deps_missing) == 0)
    
    # Check training script syntax
    print("\n5. Validating training script...")
    try:
        import ast
        with open("models/scripts/train_ternary_models.py", "r") as f:
            code = f.read()
        ast.parse(code)
        print("   ✓ Training script syntax valid")
        all_checks.append(True)
    except Exception as e:
        print(f"   ✗ Training script has syntax errors: {e}")
        all_checks.append(False)
    
    # Check for checkpoints
    print("\n6. Checking for existing checkpoints...")
    if os.path.exists("checkpoints"):
        checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
        if checkpoints:
            print(f"   ✓ Found {len(checkpoints)} checkpoint(s):")
            for cp in checkpoints[:5]:  # Show first 5
                print(f"     - {cp}")
            if len(checkpoints) > 5:
                print(f"     ... and {len(checkpoints) - 5} more")
        else:
            print("   ⚠ No checkpoints found (starting fresh)")
    
    # Summary
    print_header("Summary")
    
    if all(all_checks):
        print("\n✓ All checks passed!")
        print("\nYour training environment is ready!")
        print("\nNext steps:")
        print("  1. Read LAUNCH_GUIDE.md or START_HERE.md")
        print("  2. Run: ./launch_training.sh")
        print("  3. Or use direct command from documentation")
        return 0
    else:
        print("\n✗ Some checks failed.")
        
        if len(deps_missing) > 0:
            print("\nMissing dependencies:")
            for dep in deps_missing:
                print(f"  - {dep}")
            print("\nTo install:")
            print(f"  pip install {' '.join(deps_missing)}")
        
        print("\nPlease fix the issues above before launching training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

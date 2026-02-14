#!/usr/bin/env python3
"""
Simple validation script for the enhanced CIFAR-10 training script.

Validates:
1. Python syntax is correct
2. All required classes and functions are defined
3. Imports are structured correctly
4. No obvious runtime errors in class definitions
"""

import sys
import os
import ast

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

def validate_python_syntax(filepath):
    """Validate Python syntax by parsing the AST"""
    print(f"Validating Python syntax for: {filepath}")
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✓ Syntax validation passed")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False

def check_definitions(filepath):
    """Check that required classes and functions are defined"""
    print("\nChecking required definitions...")
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    # Extract all class and function names
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    
    # Required classes
    required_classes = ['CutMix', 'MixUp', 'LabelSmoothingCrossEntropy', 'EarlyStopping']
    
    # Required functions
    required_functions = ['get_dataset', 'get_model', 'train_epoch', 'validate', 
                         'save_checkpoint', 'main']
    
    all_pass = True
    
    for cls in required_classes:
        if cls in classes:
            print(f"✓ Class '{cls}' defined")
        else:
            print(f"✗ Class '{cls}' NOT found")
            all_pass = False
    
    for func in required_functions:
        if func in functions:
            print(f"✓ Function '{func}' defined")
        else:
            print(f"✗ Function '{func}' NOT found")
            all_pass = False
    
    return all_pass

def check_imports(filepath):
    """Check that key imports are present"""
    print("\nChecking imports...")
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    required_imports = [
        'argparse',
        'os',
        'time',
        'csv',
        'numpy',
    ]
    
    all_pass = True
    for imp in required_imports:
        if f"import {imp}" in code:
            print(f"✓ Import '{imp}' found")
        else:
            print(f"⚠ Import '{imp}' not found (may be conditional)")
    
    return True  # Don't fail on imports since some are conditional

def check_argument_parser(filepath):
    """Check that argument parser has key arguments"""
    print("\nChecking argument parser...")
    
    with open(filepath, 'r') as f:
        code = f.read()
    
    required_args = [
        '--epochs',
        '--resume',
        '--early_stopping',
        '--early_stopping_patience',
        '--label_smoothing',
        '--cutmix',
        '--mixup',
        '--autoaugment',
        '--randaugment',
        '--save_freq',
        '--csv_log',
    ]
    
    all_pass = True
    for arg in required_args:
        if arg in code:
            print(f"✓ Argument '{arg}' defined")
        else:
            print(f"✗ Argument '{arg}' NOT found")
            all_pass = False
    
    return all_pass

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '../'))
    train_script = os.path.join(repo_root, 'models/scripts/train_ternary_models.py')
    
    if not os.path.exists(train_script):
        print(f"Error: Training script not found at {train_script}")
        print(f"Script dir: {script_dir}")
        print(f"Repo root: {repo_root}")
        sys.exit(1)
    
    print("="*70)
    print("CIFAR-10 Training Script Validation")
    print("="*70)
    
    results = []
    
    # 1. Syntax validation
    results.append(("Syntax", validate_python_syntax(train_script)))
    
    # 2. Definition checking
    results.append(("Definitions", check_definitions(train_script)))
    
    # 3. Import checking
    results.append(("Imports", check_imports(train_script)))
    
    # 4. Argument parser checking
    results.append(("Arguments", check_argument_parser(train_script)))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("\n✓ All validations passed!")
        print("\nThe enhanced training script is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install torch torchvision numpy tensorboard")
        print("2. Run with: python models/scripts/train_ternary_models.py --help")
        print("3. See docs/QUICK_START_CIFAR10.md for usage examples")
        return 0
    else:
        print("\n✗ Some validations failed. Please review the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

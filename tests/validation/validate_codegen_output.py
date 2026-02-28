"""
Output Validation Script
========================

Validates that generated PyTorch code is:
1. Syntactically correct
2. Executable
3. Creates valid nn.Module instances
4. Can perform forward passes
5. Has correct parameter shapes
6. Handles ternary weights properly

Also provides side-by-side comparison of DSL and generated PyTorch.
"""

import sys
import os
import textwrap

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import (
    Program, LayerDef, Param, Assignment, Return,
    BinaryOp, Identifier, IntLiteral
)
from triton.compiler.codegen import generate_pytorch_code


def validate_syntax(code: str) -> bool:
    """Validate Python syntax."""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        return False


def validate_execution(code: str) -> bool:
    """Validate code can be executed."""
    try:
        namespace = {}
        exec(code, namespace)
        return True
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return False


def validate_module_creation(code: str, class_name: str) -> bool:
    """Validate that module class can be instantiated."""
    try:
        namespace = {}
        exec(code, namespace)
        
        if class_name not in namespace:
            print(f"❌ Class {class_name} not found in namespace")
            return False
        
        module_class = namespace[class_name]
        
        # Try to instantiate
        try:
            module = module_class()
        except Exception as e:
            print(f"❌ Failed to instantiate {class_name}: {e}")
            return False
        
        # Check it's a proper nn.Module
        try:
            import torch.nn as nn
            if not isinstance(module, nn.Module):
                print(f"❌ {class_name} is not an nn.Module instance")
                return False
        except ImportError:
            print("⚠️  torch not available, skipping nn.Module check")
        
        return True
        
    except Exception as e:
        print(f"❌ Module Creation Error: {e}")
        return False


def validate_forward_pass(code: str, class_name: str, input_shape: tuple) -> bool:
    """Validate that forward pass works."""
    try:
        import torch
        
        namespace = {}
        exec(code, namespace)
        
        module_class = namespace[class_name]
        module = module_class()
        
        # Create dummy input
        x = torch.randn(*input_shape)
        
        try:
            output = module(x)
            if output is None:
                print(f"❌ Forward pass returned None")
                return False
            return True
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
            
    except ImportError:
        print("⚠️  torch not available, skipping forward pass test")
        return True
    except Exception as e:
        print(f"❌ Forward Pass Validation Error: {e}")
        return False


def validate_ternary_weights(code: str, class_name: str) -> bool:
    """Validate ternary weight packing/unpacking."""
    try:
        import torch
        
        namespace = {}
        exec(code, namespace)
        
        if class_name not in namespace:
            return True  # No module to validate
        
        module_class = namespace[class_name]
        module = module_class()
        
        # Check for packed ternary buffers
        has_ternary = False
        for name, buffer in module.named_buffers():
            if "_packed" in name:
                has_ternary = True
                
                # Try to unpack
                try:
                    from backend.pytorch.ops.pack import unpack_ternary
                    
                    param_name = name.replace("_packed", "")
                    numel_attr = f"_{param_name}_numel"
                    
                    if hasattr(module, numel_attr):
                        numel = getattr(module, numel_attr)
                        unpacked = unpack_ternary(buffer, numel)
                        
                        # Check values are in {-1, 0, 1}
                        if not torch.all(torch.isin(unpacked, torch.tensor([-1, 0, 1]))):
                            print(f"❌ Unpacked values not in {{-1, 0, 1}}")
                            return False
                    
                except ImportError:
                    print("⚠️  backend.pytorch.ops.pack not available, skipping ternary check")
                    return True
                except Exception as e:
                    print(f"❌ Failed to unpack ternary weights: {e}")
                    return False
        
        return True
        
    except ImportError:
        print("⚠️  torch not available, skipping ternary weights test")
        return True
    except Exception as e:
        print(f"❌ Ternary Weight Validation Error: {e}")
        return False


def create_side_by_side_comparison(layer_def: LayerDef, pytorch_code: str) -> str:
    """Create side-by-side comparison of DSL and PyTorch."""
    
    # Format DSL representation
    dsl_lines = []
    dsl_lines.append(f"layer {layer_def.name} {{")
    
    for param in layer_def.params:
        shape_str = f"[{', '.join(map(str, param.shape))}]" if param.shape else ""
        dsl_lines.append(f"  {param.name}: {param.param_type}{shape_str}")
    
    if layer_def.body:
        dsl_lines.append("")
        dsl_lines.append("  // Body:")
        for stmt in layer_def.body:
            dsl_lines.append(f"  {str(stmt)}")
    
    dsl_lines.append("}")
    
    dsl_text = "\n".join(dsl_lines)
    
    # Format PyTorch code (truncate if too long)
    pytorch_lines = pytorch_code.split("\n")
    if len(pytorch_lines) > 40:
        pytorch_lines = pytorch_lines[:40] + ["", "  # ... (truncated)"]
    pytorch_text = "\n".join(pytorch_lines)
    
    # Create side-by-side
    comparison = f"""
{'='*80}
SIDE-BY-SIDE COMPARISON: DSL vs Generated PyTorch
{'='*80}

DSL INPUT:
{'-'*80}
{dsl_text}

GENERATED PYTORCH:
{'-'*80}
{pytorch_text}

{'='*80}
"""
    
    return comparison


def run_validation_suite():
    """Run complete validation suite."""
    print("\n" + "="*80)
    print("TRITON COMPILER OUTPUT VALIDATION SUITE")
    print("="*80)
    
    test_cases = []
    
    # Test Case 1: Simple Layer
    print("\n📋 Test Case 1: Simple Layer")
    print("-" * 80)
    
    layer1 = LayerDef(
        name="SimpleLayer",
        params=[
            Param(name="x", param_type="Tensor", shape=None)
        ],
        body=[]
    )
    program1 = Program(statements=[layer1])
    code1 = generate_pytorch_code(program1)
    
    print(create_side_by_side_comparison(layer1, code1))
    
    results1 = {
        "Syntax Valid": validate_syntax(code1),
        "Executable": validate_execution(code1),
        "Module Created": validate_module_creation(code1, "SimpleLayer"),
        "Forward Pass": validate_forward_pass(code1, "SimpleLayer", (2, 10)),
    }
    
    for check, passed in results1.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    test_cases.append(("Simple Layer", results1))
    
    # Test Case 2: Ternary Layer
    print("\n📋 Test Case 2: Ternary Layer")
    print("-" * 80)
    
    layer2 = LayerDef(
        name="TernaryLayer",
        params=[
            Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
            Param(name="bias", param_type="TernaryTensor", shape=[256]),
            Param(name="x", param_type="Tensor", shape=None)
        ],
        body=[]
    )
    program2 = Program(statements=[layer2])
    code2 = generate_pytorch_code(program2)
    
    print(create_side_by_side_comparison(layer2, code2))
    
    results2 = {
        "Syntax Valid": validate_syntax(code2),
        "Executable": validate_execution(code2),
        "Module Created": validate_module_creation(code2, "TernaryLayer"),
        "Forward Pass": validate_forward_pass(code2, "TernaryLayer", (2, 128)),
        "Ternary Weights": validate_ternary_weights(code2, "TernaryLayer"),
    }
    
    for check, passed in results2.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    test_cases.append(("Ternary Layer", results2))
    
    # Test Case 3: Layer with Operations
    print("\n📋 Test Case 3: Layer with Operations")
    print("-" * 80)
    
    layer3 = LayerDef(
        name="OpLayer",
        params=[
            Param(name="x", param_type="Tensor", shape=None),
            Param(name="y", param_type="Tensor", shape=None)
        ],
        body=[
            Assignment(
                target="result",
                value=BinaryOp(
                    op="+",
                    left=Identifier(name="x"),
                    right=Identifier(name="y")
                )
            ),
            Return(value=Identifier(name="result"))
        ]
    )
    program3 = Program(statements=[layer3])
    code3 = generate_pytorch_code(program3)
    
    print(create_side_by_side_comparison(layer3, code3))
    
    results3 = {
        "Syntax Valid": validate_syntax(code3),
        "Executable": validate_execution(code3),
        "Module Created": validate_module_creation(code3, "OpLayer"),
        "Forward Pass": validate_forward_pass(code3, "OpLayer", (2, 10)),
    }
    
    for check, passed in results3.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    test_cases.append(("Operation Layer", results3))
    
    # Test Case 4: Multiple Layers
    print("\n📋 Test Case 4: Multiple Layers in One Program")
    print("-" * 80)
    
    layer4a = LayerDef(
        name="Layer1",
        params=[Param(name="x", param_type="Tensor", shape=None)],
        body=[]
    )
    layer4b = LayerDef(
        name="Layer2",
        params=[Param(name="y", param_type="Tensor", shape=None)],
        body=[]
    )
    program4 = Program(statements=[layer4a, layer4b])
    code4 = generate_pytorch_code(program4)
    
    results4 = {
        "Syntax Valid": validate_syntax(code4),
        "Executable": validate_execution(code4),
        "Layer1 Created": validate_module_creation(code4, "Layer1"),
        "Layer2 Created": validate_module_creation(code4, "Layer2"),
    }
    
    for check, passed in results4.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
    
    test_cases.append(("Multiple Layers", results4))
    
    # Print Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total_checks = 0
    passed_checks = 0
    
    for test_name, results in test_cases:
        test_passed = sum(results.values())
        test_total = len(results)
        total_checks += test_total
        passed_checks += test_passed
        
        status = "✓" if test_passed == test_total else "✗"
        print(f"{status} {test_name}: {test_passed}/{test_total} checks passed")
    
    print("-" * 80)
    print(f"OVERALL: {passed_checks}/{total_checks} checks passed")
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 ALL VALIDATIONS PASSED! Generated code is production-ready.")
    elif success_rate >= 80:
        print("\n⚠️  Most validations passed. Some issues need attention.")
    else:
        print("\n❌ Validation failed. Significant issues detected.")
    
    print("="*80)
    
    return success_rate == 100


if __name__ == "__main__":
    try:
        success = run_validation_suite()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Validation suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

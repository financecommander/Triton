#!/usr/bin/env python3
"""
Simple Test Runner for Codegen Module
No external dependencies required.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import (
    Program, LayerDef, Param, Assignment, Return,
    BinaryOp, Identifier, IntLiteral
)
from triton.compiler.codegen import (
    IROpcode, IRValue, IRInstruction, IRBasicBlock, IRFunction, IRModule,
    ASTToIRConverter, ConstantFoldingPass, DeadCodeEliminationPass,
    PyTorchCodeGenerator, CodeGenerationPipeline,
    generate_pytorch_code, CodeFormatter
)


def test_ir_value_creation():
    """Test IRValue creation."""
    value = IRValue(name="%x", dtype="float32")
    assert value.name == "%x"
    assert value.dtype == "float32"
    assert not value.is_constant
    print("✓ IR value creation")


def test_ir_instruction():
    """Test IR instruction creation."""
    left = IRValue(name="%a", dtype="float32")
    right = IRValue(name="%b", dtype="float32")
    result = IRValue(name="%c", dtype="float32")
    
    inst = IRInstruction(
        opcode=IROpcode.ADD,
        result=result,
        operands=[left, right]
    )
    
    assert inst.opcode == IROpcode.ADD
    assert len(inst.operands) == 2
    print("✓ IR instruction creation")


def test_convert_empty_program():
    """Test converting empty program."""
    program = Program(statements=[])
    converter = ASTToIRConverter()
    ir_module = converter.convert_program(program)
    
    assert ir_module.name == "main"
    assert len(ir_module.functions) == 0
    print("✓ Convert empty program")


def test_convert_simple_layer():
    """Test converting simple layer."""
    layer = LayerDef(
        name="SimpleLayer",
        params=[Param(name="x", param_type="Tensor", shape=None)],
        body=[]
    )
    
    program = Program(statements=[layer])
    converter = ASTToIRConverter()
    ir_module = converter.convert_program(program)
    
    assert "SimpleLayer" in ir_module.functions
    func = ir_module.functions["SimpleLayer"]
    assert func.name == "SimpleLayer"
    print("✓ Convert simple layer")


def test_convert_binary_operation():
    """Test converting binary operations."""
    assignment = Assignment(
        target="a",
        value=BinaryOp(
            op="+",
            left=Identifier(name="x"),
            right=Identifier(name="y")
        )
    )
    
    layer = LayerDef(
        name="AddLayer",
        params=[
            Param(name="x", param_type="Tensor", shape=None),
            Param(name="y", param_type="Tensor", shape=None)
        ],
        body=[assignment]
    )
    
    program = Program(statements=[layer])
    converter = ASTToIRConverter()
    ir_module = converter.convert_program(program)
    
    func = ir_module.functions["AddLayer"]
    instructions = func.get_all_instructions()
    
    opcodes = [inst.opcode for inst in instructions]
    assert IROpcode.ADD in opcodes
    print("✓ Convert binary operation")


def test_constant_folding():
    """Test constant folding optimization."""
    module = IRModule(name="test")
    func = IRFunction(name="test", params=[], return_type="int32")
    block = IRBasicBlock(name="entry")
    
    const1 = IRValue(name="%c1", dtype="int32", is_constant=True, constant_value=2)
    const2 = IRValue(name="%c2", dtype="int32", is_constant=True, constant_value=3)
    result = IRValue(name="%result", dtype="int32")
    
    add_inst = IRInstruction(
        opcode=IROpcode.ADD,
        result=result,
        operands=[const1, const2]
    )
    block.append(add_inst)
    
    func.blocks["entry"] = block
    module.functions["test"] = func
    
    opt_pass = ConstantFoldingPass()
    modified = opt_pass.run(module)
    
    assert modified
    instructions = func.get_all_instructions()
    assert instructions[0].opcode == IROpcode.CONST
    assert instructions[0].attributes.get("value") == 5
    print("✓ Constant folding")


def test_dead_code_elimination():
    """Test dead code elimination."""
    module = IRModule(name="test")
    func = IRFunction(name="test", params=[], return_type="int32")
    block = IRBasicBlock(name="entry")
    
    unused = IRValue(name="%unused", dtype="int32")
    const1 = IRValue(name="%c1", dtype="int32", is_constant=True, constant_value=1)
    const2 = IRValue(name="%c2", dtype="int32", is_constant=True, constant_value=2)
    
    unused_inst = IRInstruction(
        opcode=IROpcode.ADD,
        result=unused,
        operands=[const1, const2]
    )
    block.append(unused_inst)
    
    return_inst = IRInstruction(
        opcode=IROpcode.RETURN,
        operands=[const1]
    )
    block.append(return_inst)
    
    func.blocks["entry"] = block
    module.functions["test"] = func
    
    opt_pass = DeadCodeEliminationPass()
    modified = opt_pass.run(module)
    
    assert modified
    instructions = func.get_all_instructions()
    assert len(instructions) == 1
    assert instructions[0].opcode == IROpcode.RETURN
    print("✓ Dead code elimination")


def test_generate_pytorch_code():
    """Test generating PyTorch code."""
    func = IRFunction(name="TestModule", params=[], return_type="tensor")
    block = IRBasicBlock(name="entry")
    block.append(IRInstruction(opcode=IROpcode.RETURN))
    func.blocks["entry"] = block
    
    module = IRModule(name="test")
    module.functions["TestModule"] = func
    
    generator = PyTorchCodeGenerator()
    code = generator.generate(module)
    
    assert "class TestModule(nn.Module)" in code
    assert "def __init__(self)" in code
    assert "def forward(self" in code
    print("✓ Generate PyTorch code")


def test_code_syntax_validation():
    """Test code syntax validation."""
    valid_code = "x = 1 + 2"
    valid, error = CodeFormatter.validate_syntax(valid_code)
    assert valid
    
    invalid_code = "x = 1 +"
    valid, error = CodeFormatter.validate_syntax(invalid_code)
    assert not valid
    print("✓ Code syntax validation")


def test_complete_pipeline():
    """Test complete pipeline."""
    layer = LayerDef(
        name="PipelineLayer",
        params=[Param(name="x", param_type="Tensor", shape=None)],
        body=[]
    )
    
    program = Program(statements=[layer])
    code = generate_pytorch_code(program, optimize=True)
    
    assert "class PipelineLayer" in code
    assert "def forward" in code
    
    # Validate syntax
    valid, error = CodeFormatter.validate_syntax(code)
    assert valid, f"Generated code has syntax errors: {error}"
    print("✓ Complete pipeline")


def test_ternary_layer():
    """Test ternary layer generation."""
    layer = LayerDef(
        name="TernaryLayer",
        params=[
            Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
            Param(name="x", param_type="Tensor", shape=None)
        ],
        body=[]
    )
    
    program = Program(statements=[layer])
    code = generate_pytorch_code(program)
    
    assert "TernaryLayer" in code
    assert "weights_packed" in code
    assert "unpack_ternary" in code
    print("✓ Ternary layer generation")


def test_with_operations():
    """Test layer with operations."""
    assignment = Assignment(
        target="result",
        value=BinaryOp(
            op="+",
            left=Identifier(name="x"),
            right=Identifier(name="y")
        )
    )
    ret = Return(value=Identifier(name="result"))
    
    layer = LayerDef(
        name="OpLayer",
        params=[
            Param(name="x", param_type="Tensor", shape=None),
            Param(name="y", param_type="Tensor", shape=None)
        ],
        body=[assignment, ret]
    )
    
    program = Program(statements=[layer])
    code = generate_pytorch_code(program)
    
    assert "OpLayer" in code
    valid, error = CodeFormatter.validate_syntax(code)
    assert valid
    print("✓ Layer with operations")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("TRITON COMPILER CODEGEN TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        test_ir_value_creation,
        test_ir_instruction,
        test_convert_empty_program,
        test_convert_simple_layer,
        test_convert_binary_operation,
        test_constant_folding,
        test_dead_code_elimination,
        test_generate_pytorch_code,
        test_code_syntax_validation,
        test_complete_pipeline,
        test_ternary_layer,
        test_with_operations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

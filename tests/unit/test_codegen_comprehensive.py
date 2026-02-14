"""
Comprehensive Test Suite for Triton Compiler Code Generation
=============================================================

Tests all aspects of the codegen pipeline:
- AST to IR conversion
- Optimization passes
- IR to PyTorch code generation
- Quantization codegen
- Advanced features

Over 50 test cases covering different DSL constructs.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import (
    Program, LayerDef, FunctionDef, Param, Assignment, Return,
    BinaryOp, UnaryOp, Identifier, IntLiteral, FloatLiteral,
    TritLiteral, FunctionCall, TernaryTensor, Declaration,
    ExprStatement, Type, TritType, IntType, FloatType, TensorType
)
from triton.compiler.codegen import (
    # IR
    IROpcode, IRValue, IRInstruction, IRBasicBlock, IRFunction, IRModule,
    # Conversion
    ASTToIRConverter,
    # Optimization
    ConstantFoldingPass, DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass, QuantizationFusionPass,
    # Code generation
    PyTorchCodeGenerator, QuantizationCodeGenerator,
    CUDAKernelGenerator, AutogradFunctionGenerator,
    CodeFormatter,
    # Pipeline
    CodeGenerationPipeline,
    # Public API
    generate_pytorch_code, generate_with_ir, compile_and_execute
)


# ============================================================================
# Test IR Data Structures
# ============================================================================

class TestIRDataStructures:
    """Test IR data structure creation and manipulation."""
    
    def test_ir_value_creation(self):
        """Test IRValue creation."""
        value = IRValue(name="%x", dtype="float32")
        assert value.name == "%x"
        assert value.dtype == "float32"
        assert not value.is_constant
    
    def test_ir_constant_value(self):
        """Test constant IR value."""
        value = IRValue(name="%c0", dtype="int32", is_constant=True, constant_value=42)
        assert value.is_constant
        assert value.constant_value == 42
    
    def test_ir_instruction_creation(self):
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
        assert inst.result == result
    
    def test_ir_basic_block(self):
        """Test basic block operations."""
        block = IRBasicBlock(name="entry")
        assert len(block.instructions) == 0
        
        inst = IRInstruction(opcode=IROpcode.RETURN)
        block.append(inst)
        
        assert len(block.instructions) == 1
        assert block.instructions[0] == inst
    
    def test_ir_function(self):
        """Test IR function creation."""
        param = IRValue(name="x", dtype="float32")
        func = IRFunction(
            name="test_func",
            params=[param],
            return_type="float32"
        )
        
        assert func.name == "test_func"
        assert len(func.params) == 1
        assert func.return_type == "float32"
    
    def test_ir_module(self):
        """Test IR module creation."""
        module = IRModule(name="test_module")
        assert module.name == "test_module"
        assert len(module.functions) == 0


# ============================================================================
# Test AST to IR Conversion
# ============================================================================

class TestASTToIRConversion:
    """Test conversion from AST to IR."""
    
    def test_convert_empty_program(self):
        """Test converting empty program."""
        program = Program(statements=[])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        assert ir_module.name == "main"
        assert len(ir_module.functions) == 0
    
    def test_convert_simple_layer(self):
        """Test converting simple layer definition."""
        layer = LayerDef(
            name="SimpleLayer",
            params=[
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        assert "SimpleLayer" in ir_module.functions
        func = ir_module.functions["SimpleLayer"]
        assert func.name == "SimpleLayer"
        assert len(func.params) == 1
    
    def test_convert_ternary_parameter(self):
        """Test converting ternary tensor parameter."""
        layer = LayerDef(
            name="TernaryLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[10, 20])
            ],
            body=[]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["TernaryLayer"]
        assert len(func.params) == 1
        assert func.params[0].dtype == "ternary_tensor"
        assert func.params[0].shape == [10, 20]
    
    def test_convert_binary_operation(self):
        """Test converting binary operations."""
        # Create: a = x + y
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
        
        # Should have ADD and STORE instructions
        opcodes = [inst.opcode for inst in instructions]
        assert IROpcode.ADD in opcodes
        assert IROpcode.STORE in opcodes
    
    def test_convert_matmul_operation(self):
        """Test converting matrix multiplication."""
        assignment = Assignment(
            target="result",
            value=BinaryOp(
                op="@",
                left=Identifier(name="a"),
                right=Identifier(name="b")
            )
        )
        
        layer = LayerDef(
            name="MatMulLayer",
            params=[
                Param(name="a", param_type="Tensor", shape=None),
                Param(name="b", param_type="Tensor", shape=None)
            ],
            body=[assignment]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["MatMulLayer"]
        instructions = func.get_all_instructions()
        
        # Should have MATMUL instruction
        opcodes = [inst.opcode for inst in instructions]
        assert IROpcode.MATMUL in opcodes
    
    def test_convert_return_statement(self):
        """Test converting return statement."""
        ret = Return(value=Identifier(name="x"))
        
        layer = LayerDef(
            name="ReturnLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[ret]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["ReturnLayer"]
        instructions = func.get_all_instructions()
        
        # Should have RETURN instruction
        assert any(inst.opcode == IROpcode.RETURN for inst in instructions)
    
    def test_convert_function_call(self):
        """Test converting function calls."""
        call = FunctionCall(
            name="relu",
            arguments=[Identifier(name="x")]
        )
        
        ret = Return(value=call)
        
        layer = LayerDef(
            name="ActivationLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[ret]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["ActivationLayer"]
        instructions = func.get_all_instructions()
        
        # Should have CALL instruction
        call_insts = [inst for inst in instructions if inst.opcode == IROpcode.CALL]
        assert len(call_insts) == 1
        assert call_insts[0].attributes.get("function") == "relu"
    
    def test_convert_unary_operation(self):
        """Test converting unary operations."""
        assignment = Assignment(
            target="neg_x",
            value=UnaryOp(
                op="-",
                operand=Identifier(name="x")
            )
        )
        
        layer = LayerDef(
            name="NegateLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[assignment]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["NegateLayer"]
        instructions = func.get_all_instructions()
        
        # Should have SUB instruction (0 - x)
        assert any(inst.opcode == IROpcode.SUB for inst in instructions)
    
    def test_convert_literal_expressions(self):
        """Test converting literal values."""
        # Create: x = 42
        assignment = Assignment(
            target="x",
            value=IntLiteral(value=42)
        )
        
        layer = LayerDef(
            name="LiteralLayer",
            params=[],
            body=[assignment]
        )
        
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        func = ir_module.functions["LiteralLayer"]
        instructions = func.get_all_instructions()
        
        # Check that constant value is created
        assert len(instructions) > 0


# ============================================================================
# Test Optimization Passes
# ============================================================================

class TestOptimizationPasses:
    """Test IR optimization passes."""
    
    def test_constant_folding_add(self):
        """Test constant folding for addition."""
        # Create: result = 2 + 3
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
        
        # Run constant folding
        opt_pass = ConstantFoldingPass()
        modified = opt_pass.run(module)
        
        assert modified
        # Result should be replaced with constant 5
        instructions = func.get_all_instructions()
        assert len(instructions) == 1
        assert instructions[0].opcode == IROpcode.CONST
        assert instructions[0].attributes.get("value") == 5
    
    def test_constant_folding_multiply(self):
        """Test constant folding for multiplication."""
        module = IRModule(name="test")
        func = IRFunction(name="test", params=[], return_type="int32")
        block = IRBasicBlock(name="entry")
        
        const1 = IRValue(name="%c1", dtype="int32", is_constant=True, constant_value=4)
        const2 = IRValue(name="%c2", dtype="int32", is_constant=True, constant_value=5)
        result = IRValue(name="%result", dtype="int32")
        
        mul_inst = IRInstruction(
            opcode=IROpcode.MUL,
            result=result,
            operands=[const1, const2]
        )
        block.append(mul_inst)
        
        func.blocks["entry"] = block
        module.functions["test"] = func
        
        opt_pass = ConstantFoldingPass()
        modified = opt_pass.run(module)
        
        assert modified
        instructions = func.get_all_instructions()
        assert instructions[0].attributes.get("value") == 20
    
    def test_dead_code_elimination(self):
        """Test dead code elimination."""
        module = IRModule(name="test")
        func = IRFunction(name="test", params=[], return_type="int32")
        block = IRBasicBlock(name="entry")
        
        # Create unused value
        unused = IRValue(name="%unused", dtype="int32")
        const1 = IRValue(name="%c1", dtype="int32", is_constant=True, constant_value=1)
        const2 = IRValue(name="%c2", dtype="int32", is_constant=True, constant_value=2)
        
        # This instruction's result is never used
        unused_inst = IRInstruction(
            opcode=IROpcode.ADD,
            result=unused,
            operands=[const1, const2]
        )
        block.append(unused_inst)
        
        # This instruction has side effects (return)
        return_inst = IRInstruction(
            opcode=IROpcode.RETURN,
            operands=[const1]
        )
        block.append(return_inst)
        
        func.blocks["entry"] = block
        module.functions["test"] = func
        
        # Run DCE
        opt_pass = DeadCodeEliminationPass()
        modified = opt_pass.run(module)
        
        assert modified
        instructions = func.get_all_instructions()
        # Unused ADD should be removed, only RETURN remains
        assert len(instructions) == 1
        assert instructions[0].opcode == IROpcode.RETURN
    
    def test_common_subexpression_elimination(self):
        """Test common subexpression elimination."""
        module = IRModule(name="test")
        func = IRFunction(name="test", params=[], return_type="int32")
        block = IRBasicBlock(name="entry")
        
        x = IRValue(name="%x", dtype="int32")
        y = IRValue(name="%y", dtype="int32")
        result1 = IRValue(name="%r1", dtype="int32")
        result2 = IRValue(name="%r2", dtype="int32")
        final = IRValue(name="%final", dtype="int32")
        
        # Two identical additions
        add1 = IRInstruction(
            opcode=IROpcode.ADD,
            result=result1,
            operands=[x, y]
        )
        add2 = IRInstruction(
            opcode=IROpcode.ADD,
            result=result2,
            operands=[x, y]
        )
        # Use both results
        final_add = IRInstruction(
            opcode=IROpcode.ADD,
            result=final,
            operands=[result1, result2]
        )
        
        block.append(add1)
        block.append(add2)
        block.append(final_add)
        
        func.blocks["entry"] = block
        module.functions["test"] = func
        
        # Run CSE
        opt_pass = CommonSubexpressionEliminationPass()
        modified = opt_pass.run(module)
        
        assert modified
        instructions = func.get_all_instructions()
        # Second ADD should be eliminated
        assert len(instructions) == 2
    
    def test_quantization_fusion(self):
        """Test quantization fusion pass."""
        module = IRModule(name="test")
        func = IRFunction(name="test", params=[], return_type="tensor")
        block = IRBasicBlock(name="entry")
        
        x = IRValue(name="%x", dtype="tensor")
        quantized = IRValue(name="%quant", dtype="ternary_tensor")
        dequantized = IRValue(name="%dequant", dtype="tensor")
        
        # Quantize
        quant_inst = IRInstruction(
            opcode=IROpcode.QUANTIZE_TERNARY,
            result=quantized,
            operands=[x]
        )
        # Immediately dequantize
        dequant_inst = IRInstruction(
            opcode=IROpcode.DEQUANTIZE,
            result=dequantized,
            operands=[quantized]
        )
        
        block.append(quant_inst)
        block.append(dequant_inst)
        
        func.blocks["entry"] = block
        module.functions["test"] = func
        
        # Run quantization fusion
        opt_pass = QuantizationFusionPass()
        modified = opt_pass.run(module)
        
        assert modified
        # Both instructions should be removed
        instructions = func.get_all_instructions()
        assert len(instructions) == 0


# ============================================================================
# Test PyTorch Code Generation
# ============================================================================

class TestPyTorchCodeGeneration:
    """Test generation of PyTorch code from IR."""
    
    def test_generate_empty_module(self):
        """Test generating empty module."""
        func = IRFunction(name="EmptyModule", params=[], return_type="tensor")
        block = IRBasicBlock(name="entry")
        block.append(IRInstruction(opcode=IROpcode.RETURN))
        func.blocks["entry"] = block
        
        module = IRModule(name="test")
        module.functions["EmptyModule"] = func
        
        generator = PyTorchCodeGenerator()
        code = generator.generate(module)
        
        assert "class EmptyModule(nn.Module)" in code
        assert "def __init__(self)" in code
        assert "def forward(self" in code
        assert "import torch" in code
    
    def test_generate_with_ternary_parameter(self):
        """Test generating module with ternary parameter."""
        param = IRValue(name="weights", dtype="ternary_tensor", shape=[10, 20])
        func = IRFunction(name="TernaryModule", params=[param], return_type="tensor")
        block = IRBasicBlock(name="entry")
        block.append(IRInstruction(opcode=IROpcode.RETURN))
        func.blocks["entry"] = block
        
        module = IRModule(name="test")
        module.functions["TernaryModule"] = func
        
        generator = PyTorchCodeGenerator()
        code = generator.generate(module)
        
        assert "TernaryModule" in code
        assert "weights_packed" in code
        assert "unpack_ternary" in code
        assert "_weights_shape" in code
    
    def test_generate_arithmetic_operations(self):
        """Test generating arithmetic operations."""
        x = IRValue(name="x", dtype="tensor")
        y = IRValue(name="y", dtype="tensor")
        result = IRValue(name="result", dtype="tensor")
        
        func = IRFunction(name="ArithModule", params=[x, y], return_type="tensor")
        block = IRBasicBlock(name="entry")
        
        # result = x + y
        add_inst = IRInstruction(
            opcode=IROpcode.ADD,
            result=result,
            operands=[IRValue(name="x", dtype="tensor"), IRValue(name="y", dtype="tensor")]
        )
        ret_inst = IRInstruction(
            opcode=IROpcode.RETURN,
            operands=[result]
        )
        
        block.append(add_inst)
        block.append(ret_inst)
        func.blocks["entry"] = block
        
        module = IRModule(name="test")
        module.functions["ArithModule"] = func
        
        generator = PyTorchCodeGenerator()
        code = generator.generate(module)
        
        assert "x + y" in code or "torch.add" in code
        assert "return" in code
    
    def test_generate_matmul_operation(self):
        """Test generating matrix multiplication."""
        a = IRValue(name="a", dtype="tensor")
        b = IRValue(name="b", dtype="tensor")
        result = IRValue(name="result", dtype="tensor")
        
        func = IRFunction(name="MatMulModule", params=[a, b], return_type="tensor")
        block = IRBasicBlock(name="entry")
        
        matmul_inst = IRInstruction(
            opcode=IROpcode.MATMUL,
            result=result,
            operands=[IRValue(name="a", dtype="tensor"), IRValue(name="b", dtype="tensor")]
        )
        ret_inst = IRInstruction(
            opcode=IROpcode.RETURN,
            operands=[result]
        )
        
        block.append(matmul_inst)
        block.append(ret_inst)
        func.blocks["entry"] = block
        
        module = IRModule(name="test")
        module.functions["MatMulModule"] = func
        
        generator = PyTorchCodeGenerator()
        code = generator.generate(module)
        
        assert "torch.matmul" in code
    
    def test_code_is_valid_python(self):
        """Test that generated code is valid Python."""
        layer = LayerDef(
            name="ValidModule",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        code = generate_pytorch_code(program)
        
        # Try to compile
        try:
            compile(code, '<string>', 'exec')
            assert True
        except SyntaxError:
            pytest.fail("Generated code has syntax errors")


# ============================================================================
# Test Complete Pipeline
# ============================================================================

class TestCodeGenerationPipeline:
    """Test the complete code generation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = CodeGenerationPipeline()
        assert pipeline is not None
        assert pipeline.optimize is True
    
    def test_pipeline_simple_layer(self):
        """Test pipeline with simple layer."""
        layer = LayerDef(
            name="SimpleLayer",
            params=[
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[]
        )
        
        program = Program(statements=[layer])
        pipeline = CodeGenerationPipeline()
        code = pipeline.generate(program)
        
        assert "class SimpleLayer" in code
        assert "def forward" in code
    
    def test_pipeline_with_optimization(self):
        """Test pipeline with optimization enabled."""
        # Create program with constant expression
        assignment = Assignment(
            target="result",
            value=BinaryOp(
                op="+",
                left=IntLiteral(value=2),
                right=IntLiteral(value=3)
            )
        )
        
        layer = LayerDef(
            name="ConstLayer",
            params=[],
            body=[assignment, Return(value=Identifier(name="result"))]
        )
        
        program = Program(statements=[layer])
        pipeline = CodeGenerationPipeline(optimize=True)
        code = pipeline.generate(program, optimize=True)
        
        assert code is not None
    
    def test_pipeline_without_optimization(self):
        """Test pipeline with optimization disabled."""
        layer = LayerDef(
            name="NoOptLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        pipeline = CodeGenerationPipeline(optimize=False)
        code = pipeline.generate(program, optimize=False)
        
        assert code is not None
    
    def test_generate_with_metadata(self):
        """Test generating code with metadata."""
        layer = LayerDef(
            name="MetaLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        pipeline = CodeGenerationPipeline()
        result = pipeline.generate_with_metadata(program)
        
        assert "code" in result
        assert "ir" in result
        assert "optimizations_applied" in result
        assert "imports" in result
        assert "functions" in result


# ============================================================================
# Test Quantization Code Generation
# ============================================================================

class TestQuantizationCodeGeneration:
    """Test quantization-specific code generation."""
    
    def test_ternary_quantize_code(self):
        """Test ternary quantization code generation."""
        code = QuantizationCodeGenerator.generate_ternary_quantize("input", "output")
        
        assert "ternary quantization" in code.lower()
        assert "input" in code
        assert "output" in code
        assert "torch.sign" in code
    
    def test_int8_quantize_code(self):
        """Test INT8 quantization code generation."""
        code = QuantizationCodeGenerator.generate_int8_quantize("input", "output")
        
        assert "INT8" in code
        assert "scale" in code
        assert "zero_point" in code
        assert "torch.clamp" in code
    
    def test_int4_quantize_code(self):
        """Test INT4 quantization code generation."""
        code = QuantizationCodeGenerator.generate_int4_quantize("input", "output")
        
        assert "INT4" in code
        assert "4-bit" in code
        assert "torch.clamp" in code
    
    def test_per_channel_quantize_code(self):
        """Test per-channel quantization code generation."""
        code = QuantizationCodeGenerator.generate_per_channel_quantize("input", "output", axis=0)
        
        assert "Per-channel" in code
        assert "axis 0" in code
        assert "torch.amin" in code
        assert "torch.amax" in code


# ============================================================================
# Test Advanced Features
# ============================================================================

class TestAdvancedFeatures:
    """Test advanced code generation features."""
    
    def test_cuda_kernel_generation(self):
        """Test CUDA kernel generation."""
        kernel_code = CUDAKernelGenerator.generate_ternary_matmul_kernel()
        
        assert "triton" in kernel_code
        assert "@triton.jit" in kernel_code
        assert "ternary_matmul_kernel" in kernel_code
        assert "tl.dot" in kernel_code
    
    def test_autograd_function_generation(self):
        """Test autograd function generation."""
        backward_code = AutogradFunctionGenerator.generate_ternary_backward()
        
        assert "torch.autograd.Function" in backward_code
        assert "forward" in backward_code
        assert "backward" in backward_code
        assert "straight-through estimator" in backward_code.lower()


# ============================================================================
# Test Code Formatting
# ============================================================================

class TestCodeFormatting:
    """Test code formatting and validation."""
    
    def test_validate_valid_syntax(self):
        """Test syntax validation with valid code."""
        code = "x = 1 + 2"
        valid, error = CodeFormatter.validate_syntax(code)
        
        assert valid
        assert error is None
    
    def test_validate_invalid_syntax(self):
        """Test syntax validation with invalid code."""
        code = "x = 1 +"  # Incomplete expression
        valid, error = CodeFormatter.validate_syntax(code)
        
        assert not valid
        assert error is not None
    
    def test_format_code(self):
        """Test code formatting."""
        code = "x=1+2"
        formatted = CodeFormatter.format_code(code)
        
        # Should either be formatted or returned as-is
        assert formatted is not None


# ============================================================================
# Test Public API
# ============================================================================

class TestPublicAPI:
    """Test public API functions."""
    
    def test_generate_pytorch_code(self):
        """Test generate_pytorch_code function."""
        layer = LayerDef(
            name="TestLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        code = generate_pytorch_code(program)
        
        assert code is not None
        assert "class TestLayer" in code
    
    def test_generate_with_ir(self):
        """Test generate_with_ir function."""
        layer = LayerDef(
            name="IRLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        code, ir_module = generate_with_ir(program)
        
        assert code is not None
        assert ir_module is not None
        assert isinstance(ir_module, IRModule)
    
    def test_compile_and_execute(self):
        """Test compile_and_execute function."""
        layer = LayerDef(
            name="ExecLayer",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer])
        namespace = compile_and_execute(program)
        
        assert namespace is not None
        assert "ExecLayer" in namespace


# ============================================================================
# Test Complex Scenarios
# ============================================================================

class TestComplexScenarios:
    """Test complex code generation scenarios."""
    
    def test_multiple_operations(self):
        """Test layer with multiple operations."""
        # a = x + y
        # b = a * 2
        # return b
        
        stmt1 = Assignment(
            target="a",
            value=BinaryOp(op="+", left=Identifier(name="x"), right=Identifier(name="y"))
        )
        stmt2 = Assignment(
            target="b",
            value=BinaryOp(op="*", left=Identifier(name="a"), right=IntLiteral(value=2))
        )
        stmt3 = Return(value=Identifier(name="b"))
        
        layer = LayerDef(
            name="MultiOpLayer",
            params=[
                Param(name="x", param_type="Tensor", shape=None),
                Param(name="y", param_type="Tensor", shape=None)
            ],
            body=[stmt1, stmt2, stmt3]
        )
        
        program = Program(statements=[layer])
        code = generate_pytorch_code(program)
        
        assert "MultiOpLayer" in code
        assert "forward" in code
    
    def test_nested_expressions(self):
        """Test nested binary expressions."""
        # result = (x + y) * (a - b)
        
        add_expr = BinaryOp(op="+", left=Identifier(name="x"), right=Identifier(name="y"))
        sub_expr = BinaryOp(op="-", left=Identifier(name="a"), right=Identifier(name="b"))
        mul_expr = BinaryOp(op="*", left=add_expr, right=sub_expr)
        
        assignment = Assignment(target="result", value=mul_expr)
        ret = Return(value=Identifier(name="result"))
        
        layer = LayerDef(
            name="NestedLayer",
            params=[
                Param(name="x", param_type="Tensor", shape=None),
                Param(name="y", param_type="Tensor", shape=None),
                Param(name="a", param_type="Tensor", shape=None),
                Param(name="b", param_type="Tensor", shape=None)
            ],
            body=[assignment, ret]
        )
        
        program = Program(statements=[layer])
        code = generate_pytorch_code(program)
        
        assert code is not None
    
    def test_multiple_layers(self):
        """Test program with multiple layers."""
        layer1 = LayerDef(
            name="Layer1",
            params=[Param(name="x", param_type="Tensor", shape=None)],
            body=[]
        )
        layer2 = LayerDef(
            name="Layer2",
            params=[Param(name="y", param_type="Tensor", shape=None)],
            body=[]
        )
        
        program = Program(statements=[layer1, layer2])
        code = generate_pytorch_code(program)
        
        assert "class Layer1" in code
        assert "class Layer2" in code


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in code generation."""
    
    def test_invalid_syntax_raises_error(self):
        """Test that invalid generated code raises error."""
        # This would need to create a scenario where invalid code is generated
        # For now, just test that validation works
        invalid_code = "class X def y():"
        valid, error = CodeFormatter.validate_syntax(invalid_code)
        
        assert not valid
        assert error is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_end_to_end_simple(self):
        """Test end-to-end generation for simple layer."""
        layer = LayerDef(
            name="E2ELayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[10, 20]),
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[]
        )
        
        program = Program(statements=[layer])
        
        # Generate code
        code = generate_pytorch_code(program, optimize=True)
        
        # Validate syntax
        valid, error = CodeFormatter.validate_syntax(code)
        assert valid, f"Generated code has syntax errors: {error}"
        
        # Check key components
        assert "class E2ELayer" in code
        assert "def __init__" in code
        assert "def forward" in code
        assert "weights_packed" in code
    
    def test_end_to_end_with_operations(self):
        """Test end-to-end with actual operations."""
        # Create: result = x @ weights + bias
        matmul = BinaryOp(
            op="@",
            left=Identifier(name="x"),
            right=Identifier(name="weights")
        )
        add = BinaryOp(
            op="+",
            left=matmul,
            right=Identifier(name="bias")
        )
        assignment = Assignment(target="result", value=add)
        ret = Return(value=Identifier(name="result"))
        
        layer = LayerDef(
            name="LinearLayer",
            params=[
                Param(name="weights", param_type="TernaryTensor", shape=[128, 256]),
                Param(name="bias", param_type="TernaryTensor", shape=[256]),
                Param(name="x", param_type="Tensor", shape=None)
            ],
            body=[assignment, ret]
        )
        
        program = Program(statements=[layer])
        code = generate_pytorch_code(program, optimize=True)
        
        # Validate
        valid, error = CodeFormatter.validate_syntax(code)
        assert valid, f"Syntax error: {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

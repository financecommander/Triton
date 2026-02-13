"""
Unit tests for the TypeChecker.
"""

import pytest
from compiler.ast.nodes import (
    Program, TritLiteral, IntLiteral, FloatLiteral, TernaryTensor,
    Identifier, BinaryOp, UnaryOp, FunctionCall,
    Assignment, Return, ExprStatement, Param, FunctionDef, LayerDef,
    TritType, IntType, FloatType, TensorType
)
from compiler.typechecker.validator import TypeChecker, TypeError as TritonTypeError


class TestTritValidation:
    """Tests for trit value validation."""
    
    def test_valid_trit_values(self):
        """Test that valid trit values (-1, 0, 1) pass validation."""
        checker = TypeChecker()
        
        # Test each valid trit value
        for value in [-1, 0, 1]:
            node = TritLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0, f"Valid trit value {value} should not produce errors"
    
    def test_invalid_trit_values(self):
        """Test that invalid trit values produce errors."""
        checker = TypeChecker()
        
        # Test invalid values
        for value in [-2, 2, 5, 100]:
            node = TritLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 1
            assert "must be -1, 0, or 1" in errors[0].message
            assert errors[0].lineno == 1


class TestTernaryTensorValidation:
    """Tests for TernaryTensor validation."""
    
    def test_valid_tensor_shape_matches_values(self):
        """Test that tensor shape matching value count passes."""
        checker = TypeChecker()
        
        # 2x3 tensor with 6 values
        node = TernaryTensor(
            shape=[2, 3],
            values=[-1, 0, 1, -1, 0, 1],
            lineno=1, col_offset=0
        )
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        assert len(errors) == 0
    
    def test_tensor_shape_mismatch(self):
        """Test that shape mismatch produces error."""
        checker = TypeChecker()
        
        # 2x3 tensor but only 5 values (should be 6)
        node = TernaryTensor(
            shape=[2, 3],
            values=[-1, 0, 1, -1, 0],
            lineno=1, col_offset=0
        )
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "expects 6 values, got 5" in errors[0].message
    
    def test_tensor_invalid_trit_values(self):
        """Test that invalid trit values in tensor produce errors."""
        checker = TypeChecker()
        
        # Tensor with invalid value (2)
        node = TernaryTensor(
            shape=[2, 2],
            values=[-1, 0, 2, 1],
            lineno=1, col_offset=0
        )
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        assert len(errors) >= 1
        assert any("must be -1, 0, or 1" in err.message for err in errors)
    
    def test_multidimensional_tensor(self):
        """Test 3D tensor validation."""
        checker = TypeChecker()
        
        # 2x2x2 tensor with 8 values
        node = TernaryTensor(
            shape=[2, 2, 2],
            values=[-1, 0, 1, -1, 0, 1, -1, 0],
            lineno=1, col_offset=0
        )
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        assert len(errors) == 0


class TestBinaryOperationValidation:
    """Tests for binary operation type validation."""
    
    def test_compatible_trit_operations(self):
        """Test operations between compatible trit types."""
        checker = TypeChecker()
        
        # Create two trit literals
        left = TritLiteral(value=1, lineno=1, col_offset=0)
        right = TritLiteral(value=-1, lineno=1, col_offset=5)
        
        for op in ['+', '-', '*']:
            binary_op = BinaryOp(left=left, op=op, right=right, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=binary_op)])
            errors = checker.validate(program)
            assert len(errors) == 0, f"Operation {op} between trits should be valid"
    
    def test_incompatible_type_operations(self):
        """Test operations between incompatible types."""
        checker = TypeChecker()
        
        # Trit and Float are incompatible
        left = TritLiteral(value=1, lineno=1, col_offset=0)
        right = FloatLiteral(value=3.14, lineno=1, col_offset=5)
        
        binary_op = BinaryOp(left=left, op='+', right=right, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=binary_op)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "compatible types" in errors[0].message
    
    def test_tensor_operations(self):
        """Test operations on tensors."""
        checker = TypeChecker()
        
        # Create two tensors and assign them
        tensor1 = TernaryTensor(shape=[2, 2], values=[1, 0, -1, 1], lineno=1, col_offset=0)
        tensor2 = TernaryTensor(shape=[2, 2], values=[-1, 1, 0, -1], lineno=2, col_offset=0)
        
        assign1 = Assignment(target="t1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="t2", value=tensor2, lineno=2, col_offset=0)
        
        # Add tensors
        id1 = Identifier(name="t1", lineno=3, col_offset=0)
        id2 = Identifier(name="t2", lineno=3, col_offset=5)
        binary_op = BinaryOp(left=id1, op='+', right=id2, lineno=3, col_offset=0)
        
        program = Program(statements=[assign1, assign2, ExprStatement(expr=binary_op)])
        errors = checker.validate(program)
        assert len(errors) == 0


class TestMatrixMultiplication:
    """Tests for matrix multiplication dimension validation."""
    
    def test_valid_matrix_multiplication(self):
        """Test valid matrix multiplication with compatible dimensions."""
        checker = TypeChecker()
        
        # (2, 3) @ (3, 4) -> (2, 4)
        tensor1 = TernaryTensor(
            shape=[2, 3],
            values=[1, 0, -1, 1, 0, -1],
            lineno=1, col_offset=0
        )
        tensor2 = TernaryTensor(
            shape=[3, 4],
            values=[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1],
            lineno=2, col_offset=0
        )
        
        assign1 = Assignment(target="m1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="m2", value=tensor2, lineno=2, col_offset=0)
        
        id1 = Identifier(name="m1", lineno=3, col_offset=0)
        id2 = Identifier(name="m2", lineno=3, col_offset=5)
        matmul = BinaryOp(left=id1, op='@', right=id2, lineno=3, col_offset=0)
        
        program = Program(statements=[assign1, assign2, ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 0
    
    def test_incompatible_matrix_dimensions(self):
        """Test matrix multiplication with incompatible dimensions."""
        checker = TypeChecker()
        
        # (2, 3) @ (2, 4) -> Error: inner dimensions don't match
        tensor1 = TernaryTensor(
            shape=[2, 3],
            values=[1, 0, -1, 1, 0, -1],
            lineno=1, col_offset=0
        )
        tensor2 = TernaryTensor(
            shape=[2, 4],
            values=[1, 0, -1, 1, 0, -1, 1, 0],
            lineno=2, col_offset=0
        )
        
        assign1 = Assignment(target="m1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="m2", value=tensor2, lineno=2, col_offset=0)
        
        id1 = Identifier(name="m1", lineno=3, col_offset=0)
        id2 = Identifier(name="m2", lineno=3, col_offset=5)
        matmul = BinaryOp(left=id1, op='@', right=id2, lineno=3, col_offset=0)
        
        program = Program(statements=[assign1, assign2, ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "dimension mismatch" in errors[0].message.lower()
        assert "Inner dimensions must match" in errors[0].message
    
    def test_non_tensor_matrix_multiplication(self):
        """Test matrix multiplication on non-tensor types."""
        checker = TypeChecker()
        
        # Cannot do @ on scalars
        left = IntLiteral(value=5, lineno=1, col_offset=0)
        right = IntLiteral(value=3, lineno=1, col_offset=5)
        matmul = BinaryOp(left=left, op='@', right=right, lineno=1, col_offset=0)
        
        program = Program(statements=[ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "requires tensor operands" in errors[0].message.lower()


class TestFunctionSignatures:
    """Tests for function signature validation."""
    
    def test_valid_function_call(self):
        """Test valid function call with matching signature."""
        checker = TypeChecker()
        
        # Define function: def foo(x: trit, y: trit) -> trit
        func_def = FunctionDef(
            name="foo",
            params=[
                Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0),
                Param(name="y", type_annotation=TritType(), lineno=1, col_offset=0)
            ],
            return_type=TritType(),
            body=[Return(value=Identifier(name="x", lineno=2, col_offset=0), lineno=2, col_offset=0)],
            lineno=1, col_offset=0
        )
        
        # Call function: foo(1, -1)
        func_call = FunctionCall(
            name="foo",
            args=[
                TritLiteral(value=1, lineno=3, col_offset=0),
                TritLiteral(value=-1, lineno=3, col_offset=5)
            ],
            lineno=3, col_offset=0
        )
        
        program = Program(statements=[func_def, ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 0
    
    def test_function_wrong_argument_count(self):
        """Test function call with wrong number of arguments."""
        checker = TypeChecker()
        
        # Define function: def foo(x: trit) -> trit
        func_def = FunctionDef(
            name="foo",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[Return(value=Identifier(name="x", lineno=2, col_offset=0), lineno=2, col_offset=0)],
            lineno=1, col_offset=0
        )
        
        # Call function with 2 arguments: foo(1, -1)
        func_call = FunctionCall(
            name="foo",
            args=[
                TritLiteral(value=1, lineno=3, col_offset=0),
                TritLiteral(value=-1, lineno=3, col_offset=5)
            ],
            lineno=3, col_offset=0
        )
        
        program = Program(statements=[func_def, ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "expects 1 arguments, got 2" in errors[0].message
    
    def test_function_wrong_argument_type(self):
        """Test function call with wrong argument type."""
        checker = TypeChecker()
        
        # Define function: def foo(x: trit) -> trit
        func_def = FunctionDef(
            name="foo",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[Return(value=Identifier(name="x", lineno=2, col_offset=0), lineno=2, col_offset=0)],
            lineno=1, col_offset=0
        )
        
        # Call function with float: foo(3.14)
        func_call = FunctionCall(
            name="foo",
            args=[FloatLiteral(value=3.14, lineno=3, col_offset=0)],
            lineno=3, col_offset=0
        )
        
        program = Program(statements=[func_def, ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "argument 0 expects" in errors[0].message
    
    def test_undefined_function(self):
        """Test calling undefined function."""
        checker = TypeChecker()
        
        # Call undefined function
        func_call = FunctionCall(
            name="undefined_func",
            args=[TritLiteral(value=1, lineno=1, col_offset=0)],
            lineno=1, col_offset=0
        )
        
        program = Program(statements=[ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Undefined function" in errors[0].message


class TestErrorCollection:
    """Tests for error collection (don't fail on first error)."""
    
    def test_multiple_errors_collected(self):
        """Test that multiple errors are collected before raising."""
        checker = TypeChecker()
        
        # Create multiple errors:
        # 1. Invalid trit value
        # 2. Tensor shape mismatch
        # 3. Undefined variable
        
        invalid_trit = TritLiteral(value=5, lineno=1, col_offset=0)
        invalid_tensor = TernaryTensor(
            shape=[2, 2],
            values=[1, 0, -1],  # Only 3 values, needs 4
            lineno=2, col_offset=0
        )
        undefined_var = Identifier(name="undefined", lineno=3, col_offset=0)
        
        program = Program(statements=[
            ExprStatement(expr=invalid_trit),
            ExprStatement(expr=invalid_tensor),
            ExprStatement(expr=undefined_var)
        ])
        
        errors = checker.validate(program)
        # Should have at least 3 errors (one from each issue)
        assert len(errors) >= 3
    
    def test_error_includes_line_info(self):
        """Test that errors include line and column information."""
        checker = TypeChecker()
        
        node = TritLiteral(value=10, lineno=42, col_offset=15)
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        
        assert len(errors) == 1
        assert errors[0].lineno == 42
        assert errors[0].col_offset == 15
        assert "Line 42, Col 15" in str(errors[0])


class TestTypeCheckerUsage:
    """Tests for TypeChecker usage pattern."""
    
    def test_example_usage_pattern(self):
        """Test the example usage pattern from requirements."""
        # Example usage:
        # checker = TypeChecker()
        # errors = checker.validate(ast)
        # if errors:
        #     raise TypeError(f"Type errors found: {errors}")
        
        checker = TypeChecker()
        
        # Valid AST
        valid_ast = Program(statements=[
            ExprStatement(expr=TritLiteral(value=1, lineno=1, col_offset=0))
        ])
        
        errors = checker.validate(valid_ast)
        assert errors == []
        
        # Invalid AST
        invalid_ast = Program(statements=[
            ExprStatement(expr=TritLiteral(value=10, lineno=1, col_offset=0))
        ])
        
        errors = checker.validate(invalid_ast)
        assert len(errors) > 0
        
        # Can raise TypeError with formatted message
        error_msg = f"Type errors found: {errors}"
        assert "Type errors found:" in error_msg


class TestVariableScoping:
    """Tests for variable scoping in functions."""
    
    def test_function_parameter_scoping(self):
        """Test that function parameters are in scope within function body."""
        checker = TypeChecker()
        
        func_def = FunctionDef(
            name="test",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Return(value=Identifier(name="x", lineno=2, col_offset=0), lineno=2, col_offset=0)
            ],
            lineno=1, col_offset=0
        )
        
        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 0
    
    def test_undefined_variable_in_function(self):
        """Test that undefined variables in function body are caught."""
        checker = TypeChecker()
        
        func_def = FunctionDef(
            name="test",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Return(value=Identifier(name="y", lineno=2, col_offset=0), lineno=2, col_offset=0)
            ],
            lineno=1, col_offset=0
        )
        
        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Undefined variable 'y'" in errors[0].message

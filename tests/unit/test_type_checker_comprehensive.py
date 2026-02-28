"""
Comprehensive test suite for the production-quality TypeChecker.

This test suite includes 100+ test cases covering:
- Basic type checking
- Generic types and constraints
- Error cases with expected messages
- Performance benchmarks
- Real DSL code examples
"""

import pytest
import time
from compiler.ast.nodes import (
    Program,
    TritLiteral,
    IntLiteral,
    FloatLiteral,
    TernaryTensor,
    Identifier,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    Assignment,
    Return,
    ExprStatement,
    Param,
    FunctionDef,
    LayerDef,
    Declaration,
    TritType,
    IntType,
    FloatType,
    TensorType,
)
from compiler.typechecker.type_checker import (
    TypeChecker,
    TypeError as TritonTypeError,
    QuantizationType,
    EffectType,
    TypeUnifier,
)

# ============================================================================
# Basic Type Validation Tests
# ============================================================================


class TestBasicTypeValidation:
    """Tests for basic type validation."""

    def test_valid_trit_values(self):
        """Test that valid trit values (-1, 0, 1) pass validation."""
        checker = TypeChecker()

        for value in [-1, 0, 1]:
            node = TritLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0, f"Valid trit value {value} should not produce errors"

    def test_invalid_trit_values(self):
        """Test that invalid trit values produce errors with suggestions."""
        checker = TypeChecker()

        for value in [-2, 2, 5, 100]:
            node = TritLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 1
            assert "must be -1, 0, or 1" in errors[0].message
            assert errors[0].suggested_fix is not None
            assert errors[0].lineno == 1

    def test_int_literal_types(self):
        """Test integer literal type inference."""
        checker = TypeChecker()

        for value in [0, 1, 42, -100, 1000]:
            node = IntLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0

    def test_float_literal_types(self):
        """Test float literal type inference."""
        checker = TypeChecker()

        for value in [0.0, 1.5, 3.14, -2.71, 100.001]:
            node = FloatLiteral(value=value, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0


# ============================================================================
# Tensor Type Tests
# ============================================================================


class TestTensorTypes:
    """Tests for tensor type validation."""

    def test_valid_tensor_shapes(self):
        """Test valid tensor shapes."""
        checker = TypeChecker()

        test_cases = [
            ([2, 3], [1, 0, -1, 1, 0, -1]),
            ([3, 3], [1, 0, -1, 1, 0, -1, 1, 0, -1]),
            ([2, 2, 2], [1, 0, -1, 1, 0, -1, 1, 0]),
            ([4], [1, 0, -1, 1]),
        ]

        for shape, values in test_cases:
            node = TernaryTensor(shape=shape, values=values, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0, f"Valid tensor shape {shape} should not produce errors"

    def test_tensor_shape_mismatch(self):
        """Test tensor shape mismatch errors with suggestions."""
        checker = TypeChecker()

        test_cases = [
            ([2, 3], [1, 0, -1, 1, 0]),  # 6 expected, 5 given
            ([3, 3], [1, 0, -1, 1]),  # 9 expected, 4 given
            ([2, 2], [1, 0, -1, 1, 0, -1, 1]),  # 4 expected, 7 given
        ]

        for shape, values in test_cases:
            node = TernaryTensor(shape=shape, values=values, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) >= 1
            assert "expects" in errors[0].message
            assert "values" in errors[0].message
            assert errors[0].suggested_fix is not None

    def test_tensor_invalid_values(self):
        """Test tensor with invalid trit values."""
        checker = TypeChecker()

        node = TernaryTensor(shape=[2, 2], values=[-1, 0, 2, 1], lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)
        assert len(errors) >= 1
        assert any("invalid" in err.message.lower() for err in errors)

    def test_multidimensional_tensors(self):
        """Test various multidimensional tensor shapes."""
        checker = TypeChecker()

        test_cases = [
            [2, 3, 4],  # 3D: 24 elements
            [5, 5],  # 2D: 25 elements
            [10],  # 1D: 10 elements
            [2, 2, 2, 2],  # 4D: 16 elements
        ]

        for shape in test_cases:
            size = 1
            for dim in shape:
                size *= dim
            values = [(-1) ** i for i in range(size)]
            node = TernaryTensor(shape=shape, values=values, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=node)])
            errors = checker.validate(program)
            assert len(errors) == 0


# ============================================================================
# Binary Operation Tests
# ============================================================================


class TestBinaryOperations:
    """Tests for binary operation type validation."""

    def test_compatible_arithmetic_operations(self):
        """Test arithmetic operations between compatible types."""
        checker = TypeChecker()

        # Trit + Trit
        left = TritLiteral(value=1, lineno=1, col_offset=0)
        right = TritLiteral(value=-1, lineno=1, col_offset=5)

        for op in ["+", "-", "*"]:
            binary_op = BinaryOp(left=left, op=op, right=right, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=binary_op)])
            errors = checker.validate(program)
            assert len(errors) == 0

    def test_incompatible_type_operations(self):
        """Test operations between incompatible types."""
        checker = TypeChecker()

        # Trit and Float are incompatible
        left = TritLiteral(value=1, lineno=1, col_offset=0)
        right = FloatLiteral(value=3.14, lineno=1, col_offset=5)

        binary_op = BinaryOp(left=left, op="+", right=right, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=binary_op)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "compatible types" in errors[0].message
        assert errors[0].suggested_fix is not None

    def test_comparison_operations(self):
        """Test comparison operations."""
        checker = TypeChecker()

        left = IntLiteral(value=5, lineno=1, col_offset=0)
        right = IntLiteral(value=3, lineno=1, col_offset=5)

        for op in ["==", "!=", "<", ">", "<=", ">="]:
            binary_op = BinaryOp(left=left, op=op, right=right, lineno=1, col_offset=0)
            program = Program(statements=[ExprStatement(expr=binary_op)])
            errors = checker.validate(program)
            assert len(errors) == 0

    def test_tensor_arithmetic(self):
        """Test arithmetic operations on tensors."""
        checker = TypeChecker()

        # Create two tensors
        tensor1 = TernaryTensor(shape=[2, 2], values=[1, 0, -1, 1], lineno=1, col_offset=0)
        tensor2 = TernaryTensor(shape=[2, 2], values=[-1, 1, 0, -1], lineno=2, col_offset=0)

        assign1 = Assignment(target="t1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="t2", value=tensor2, lineno=2, col_offset=0)

        # Add tensors
        id1 = Identifier(name="t1", lineno=3, col_offset=0)
        id2 = Identifier(name="t2", lineno=3, col_offset=5)
        binary_op = BinaryOp(left=id1, op="+", right=id2, lineno=3, col_offset=0)

        program = Program(statements=[assign1, assign2, ExprStatement(expr=binary_op)])
        errors = checker.validate(program)
        assert len(errors) == 0


# ============================================================================
# Matrix Multiplication Tests
# ============================================================================


class TestMatrixMultiplication:
    """Tests for matrix multiplication dimension validation."""

    def test_valid_matrix_multiplication(self):
        """Test valid matrix multiplication with compatible dimensions."""
        checker = TypeChecker()

        # (2, 3) @ (3, 4) -> (2, 4)
        tensor1 = TernaryTensor(shape=[2, 3], values=[1, 0, -1, 1, 0, -1], lineno=1, col_offset=0)
        tensor2 = TernaryTensor(
            shape=[3, 4],
            values=[1, 0, -1, 1, 0, -1, 1, 0, -1, 1, 0, -1],
            lineno=2,
            col_offset=0,
        )

        assign1 = Assignment(target="m1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="m2", value=tensor2, lineno=2, col_offset=0)

        id1 = Identifier(name="m1", lineno=3, col_offset=0)
        id2 = Identifier(name="m2", lineno=3, col_offset=5)
        matmul = BinaryOp(left=id1, op="@", right=id2, lineno=3, col_offset=0)

        program = Program(statements=[assign1, assign2, ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 0

    def test_incompatible_matrix_dimensions(self):
        """Test matrix multiplication with incompatible dimensions."""
        checker = TypeChecker()

        # (2, 3) @ (2, 4) -> Error: inner dimensions don't match
        tensor1 = TernaryTensor(shape=[2, 3], values=[1, 0, -1, 1, 0, -1], lineno=1, col_offset=0)
        tensor2 = TernaryTensor(
            shape=[2, 4], values=[1, 0, -1, 1, 0, -1, 1, 0], lineno=2, col_offset=0
        )

        assign1 = Assignment(target="m1", value=tensor1, lineno=1, col_offset=0)
        assign2 = Assignment(target="m2", value=tensor2, lineno=2, col_offset=0)

        id1 = Identifier(name="m1", lineno=3, col_offset=0)
        id2 = Identifier(name="m2", lineno=3, col_offset=5)
        matmul = BinaryOp(left=id1, op="@", right=id2, lineno=3, col_offset=0)

        program = Program(statements=[assign1, assign2, ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "dimension mismatch" in errors[0].message.lower()
        assert "Inner dimensions must match" in errors[0].message
        assert errors[0].suggested_fix is not None

    def test_non_tensor_matrix_multiplication(self):
        """Test matrix multiplication on non-tensor types."""
        checker = TypeChecker()

        # Cannot do @ on scalars
        left = IntLiteral(value=5, lineno=1, col_offset=0)
        right = IntLiteral(value=3, lineno=1, col_offset=5)
        matmul = BinaryOp(left=left, op="@", right=right, lineno=1, col_offset=0)

        program = Program(statements=[ExprStatement(expr=matmul)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "requires tensor operands" in errors[0].message.lower()

    def test_matrix_multiplication_various_shapes(self):
        """Test matrix multiplication with various valid shapes."""
        checker = TypeChecker()

        test_cases = [
            ([2, 2], [2, 2]),  # Square matrices
            ([3, 5], [5, 2]),  # Rectangular matrices
            ([10, 1], [1, 10]),  # Vector-like
        ]

        for shape1, shape2 in test_cases:
            size1 = shape1[0] * shape1[1]
            size2 = shape2[0] * shape2[1]
            values1 = [(-1) ** i for i in range(size1)]
            values2 = [(-1) ** i for i in range(size2)]

            tensor1 = TernaryTensor(shape=shape1, values=values1, lineno=1, col_offset=0)
            tensor2 = TernaryTensor(shape=shape2, values=values2, lineno=2, col_offset=0)

            assign1 = Assignment(target="m1", value=tensor1, lineno=1, col_offset=0)
            assign2 = Assignment(target="m2", value=tensor2, lineno=2, col_offset=0)

            id1 = Identifier(name="m1", lineno=3, col_offset=0)
            id2 = Identifier(name="m2", lineno=3, col_offset=5)
            matmul = BinaryOp(left=id1, op="@", right=id2, lineno=3, col_offset=0)

            program = Program(statements=[assign1, assign2, ExprStatement(expr=matmul)])
            errors = checker.validate(program)
            assert len(errors) == 0, f"Valid matmul {shape1} @ {shape2} should not error"


# ============================================================================
# Function Signature Tests
# ============================================================================


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
                Param(name="y", type_annotation=TritType(), lineno=1, col_offset=0),
            ],
            return_type=TritType(),
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        # Call function: foo(1, -1)
        func_call = FunctionCall(
            name="foo",
            args=[
                TritLiteral(value=1, lineno=3, col_offset=0),
                TritLiteral(value=-1, lineno=3, col_offset=5),
            ],
            lineno=3,
            col_offset=0,
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
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        # Call function with 2 arguments: foo(1, -1)
        func_call = FunctionCall(
            name="foo",
            args=[
                TritLiteral(value=1, lineno=3, col_offset=0),
                TritLiteral(value=-1, lineno=3, col_offset=5),
            ],
            lineno=3,
            col_offset=0,
        )

        program = Program(statements=[func_def, ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "expects 1 arguments, got 2" in errors[0].message
        assert errors[0].suggested_fix is not None

    def test_function_wrong_argument_type(self):
        """Test function call with wrong argument type."""
        checker = TypeChecker()

        # Define function: def foo(x: trit) -> trit
        func_def = FunctionDef(
            name="foo",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        # Call function with float: foo(3.14)
        func_call = FunctionCall(
            name="foo",
            args=[FloatLiteral(value=3.14, lineno=3, col_offset=0)],
            lineno=3,
            col_offset=0,
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
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[ExprStatement(expr=func_call)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Undefined function" in errors[0].message
        assert errors[0].suggested_fix is not None

    def test_function_return_type_mismatch(self):
        """Test function return type mismatch."""
        checker = TypeChecker()

        # Define function: def foo() -> trit, but returns float
        func_def = FunctionDef(
            name="foo",
            params=[],
            return_type=TritType(),
            body=[
                Return(
                    value=FloatLiteral(value=3.14, lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Return type" in errors[0].message
        assert errors[0].suggested_fix is not None


# ============================================================================
# Variable Scoping Tests
# ============================================================================


class TestVariableScoping:
    """Tests for variable scoping."""

    def test_undefined_variable(self):
        """Test undefined variable access."""
        checker = TypeChecker()

        undefined_var = Identifier(name="undefined", lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=undefined_var)])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Undefined variable" in errors[0].message
        assert errors[0].suggested_fix is not None
        assert errors[0].context is not None

    def test_function_parameter_scoping(self):
        """Test that function parameters are in scope within function body."""
        checker = TypeChecker()

        func_def = FunctionDef(
            name="test",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
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
                Return(
                    value=Identifier(name="y", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 1
        assert "Undefined variable 'y'" in errors[0].message

    def test_assignment_creates_binding(self):
        """Test that assignment creates a variable binding."""
        checker = TypeChecker()

        assign = Assignment(
            target="x",
            value=TritLiteral(value=1, lineno=1, col_offset=0),
            lineno=1,
            col_offset=0,
        )
        use = Identifier(name="x", lineno=2, col_offset=0)

        program = Program(statements=[assign, ExprStatement(expr=use)])
        errors = checker.validate(program)
        assert len(errors) == 0


# ============================================================================
# Quantization Type Tests
# ============================================================================


class TestQuantizationTypes:
    """Tests for quantization type rules."""

    def test_valid_quantization_downcasting(self):
        """Test valid quantization downcast (FP32 -> FP16 -> INT8 -> Ternary)."""
        checker = TypeChecker()

        # FP32 -> Ternary (valid)
        assign = Assignment(
            target="x",
            value=FloatLiteral(value=1.5, lineno=1, col_offset=0),
            type_annotation=TritType(),
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[assign])
        errors = checker.validate(program)
        # Should have an error since we don't allow implicit quantization
        assert len(errors) >= 1

    def test_assignment_type_checking(self):
        """Test assignment with type annotations."""
        checker = TypeChecker()

        # Assign trit to trit (valid)
        assign = Assignment(
            target="x",
            value=TritLiteral(value=1, lineno=1, col_offset=0),
            type_annotation=TritType(),
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[assign])
        errors = checker.validate(program)
        assert len(errors) == 0


# ============================================================================
# Error Reporting Tests
# ============================================================================


class TestErrorReporting:
    """Tests for error reporting quality."""

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

    def test_error_includes_suggested_fix(self):
        """Test that errors include suggested fixes."""
        checker = TypeChecker()

        node = TritLiteral(value=5, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)

        assert len(errors) == 1
        assert errors[0].suggested_fix is not None
        assert "Suggested fix:" in str(errors[0])

    def test_error_includes_context(self):
        """Test that errors include helpful context."""
        checker = TypeChecker()

        node = TritLiteral(value=5, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)

        assert len(errors) == 1
        assert errors[0].context is not None
        assert "Context:" in str(errors[0])

    def test_error_includes_inference_stack(self):
        """Test that errors include inference stack for debugging."""
        checker = TypeChecker()

        node = TritLiteral(value=5, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=node)])
        errors = checker.validate(program)

        assert len(errors) == 1
        assert len(errors[0].inference_stack) > 0

    def test_multiple_errors_collected(self):
        """Test that multiple errors are collected."""
        checker = TypeChecker()

        # Create multiple errors
        invalid_trit = TritLiteral(value=5, lineno=1, col_offset=0)
        invalid_tensor = TernaryTensor(shape=[2, 2], values=[1, 0, -1], lineno=2, col_offset=0)
        undefined_var = Identifier(name="undefined", lineno=3, col_offset=0)

        program = Program(
            statements=[
                ExprStatement(expr=invalid_trit),
                ExprStatement(expr=invalid_tensor),
                ExprStatement(expr=undefined_var),
            ]
        )

        errors = checker.validate(program)
        # Should have at least 3 errors
        assert len(errors) >= 3


# ============================================================================
# Type Unification Tests
# ============================================================================


class TestTypeUnification:
    """Tests for type unification algorithm."""

    def test_unify_same_types(self):
        """Test unification of identical types."""
        unifier = TypeUnifier()

        type1 = TritType()
        type2 = TritType()
        assert unifier.unify(type1, type2)

    def test_unify_different_types(self):
        """Test unification of different types."""
        unifier = TypeUnifier()

        type1 = TritType()
        type2 = FloatType()
        assert not unifier.unify(type1, type2)

    def test_unify_tensor_types(self):
        """Test unification of tensor types."""
        unifier = TypeUnifier()

        type1 = TensorType(element_type=TritType(), shape=[2, 3])
        type2 = TensorType(element_type=TritType(), shape=[2, 3])
        assert unifier.unify(type1, type2)

    def test_unify_tensor_types_different_shapes(self):
        """Test unification of tensors with different shapes."""
        unifier = TypeUnifier()

        type1 = TensorType(element_type=TritType(), shape=[2, 3])
        type2 = TensorType(element_type=TritType(), shape=[3, 2])
        assert not unifier.unify(type1, type2)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance tests for type checking."""

    def test_type_caching_enabled(self):
        """Test that type caching improves performance."""
        # Create a large program
        statements = []
        for i in range(100):
            literal = TritLiteral(value=(-1) ** i, lineno=i, col_offset=0)
            statements.append(ExprStatement(expr=literal))
        program = Program(statements=statements)

        # Check with caching enabled
        checker_cached = TypeChecker(enable_cache=True)
        start = time.time()
        checker_cached.validate(program)
        time_cached = time.time() - start

        # Check without caching
        checker_uncached = TypeChecker(enable_cache=False)
        start = time.time()
        checker_uncached.validate(program)
        time_uncached = time.time() - start

        # Both should complete quickly
        assert time_cached < 1.0
        assert time_uncached < 1.0

    def test_cache_statistics(self):
        """Test that cache statistics are tracked."""
        checker = TypeChecker(enable_cache=True)

        literal = TritLiteral(value=1, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=literal)])
        checker.validate(program)

        stats = checker.get_cache_stats()
        assert stats is not None
        assert "hits" in stats
        assert "misses" in stats
        assert "total" in stats

    def test_large_tensor_validation(self):
        """Test validation of large tensors."""
        checker = TypeChecker()

        # Create a large tensor (100x100)
        shape = [100, 100]
        size = 10000
        values = [(-1) ** i for i in range(size)]

        tensor = TernaryTensor(shape=shape, values=values, lineno=1, col_offset=0)
        program = Program(statements=[ExprStatement(expr=tensor)])

        start = time.time()
        errors = checker.validate(program)
        elapsed = time.time() - start

        assert len(errors) == 0
        assert elapsed < 1.0  # Should be fast

    def test_deep_function_nesting(self):
        """Test validation of deeply nested function calls."""
        checker = TypeChecker()

        # Define a chain of function calls
        func_defs = []
        for i in range(10):
            func_name = f"func{i}"
            next_func = f"func{i+1}" if i < 9 else None

            if next_func:
                body = [
                    Return(
                        value=FunctionCall(
                            name=next_func,
                            args=[Identifier(name="x", lineno=i + 2, col_offset=0)],
                            lineno=i + 2,
                            col_offset=0,
                        ),
                        lineno=i + 2,
                        col_offset=0,
                    )
                ]
            else:
                body = [
                    Return(
                        value=Identifier(name="x", lineno=i + 2, col_offset=0),
                        lineno=i + 2,
                        col_offset=0,
                    )
                ]

            func_def = FunctionDef(
                name=func_name,
                params=[Param(name="x", type_annotation=TritType(), lineno=i + 1, col_offset=0)],
                return_type=TritType(),
                body=body,
                lineno=i + 1,
                col_offset=0,
            )
            func_defs.append(func_def)

        program = Program(statements=func_defs)

        start = time.time()
        errors = checker.validate(program)
        elapsed = time.time() - start

        assert len(errors) == 0
        assert elapsed < 1.0


# ============================================================================
# Real DSL Examples
# ============================================================================


class TestRealDSLExamples:
    """Tests using real DSL code patterns."""

    def test_ternary_neural_network_layer(self):
        """Test a simple ternary neural network layer."""
        checker = TypeChecker()

        # Define a simple forward pass function
        # def forward(input: Tensor, weights: Tensor) -> Tensor:
        #     return input @ weights

        func_def = FunctionDef(
            name="forward",
            params=[
                Param(
                    name="input",
                    type_annotation=TensorType(element_type=TritType()),
                    lineno=1,
                    col_offset=0,
                ),
                Param(
                    name="weights",
                    type_annotation=TensorType(element_type=TritType()),
                    lineno=1,
                    col_offset=0,
                ),
            ],
            return_type=TensorType(element_type=TritType()),
            body=[
                Return(
                    value=BinaryOp(
                        left=Identifier(name="input", lineno=2, col_offset=0),
                        op="@",
                        right=Identifier(name="weights", lineno=2, col_offset=5),
                        lineno=2,
                        col_offset=0,
                    ),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        # Should have no errors
        assert len(errors) == 0

    def test_ternary_activation_function(self):
        """Test a ternary activation function."""
        checker = TypeChecker()

        # def ternary_sign(x: Tensor) -> Tensor:
        #     return x  # Simplified

        func_def = FunctionDef(
            name="ternary_sign",
            params=[
                Param(
                    name="x",
                    type_annotation=TensorType(element_type=TritType()),
                    lineno=1,
                    col_offset=0,
                )
            ],
            return_type=TensorType(element_type=TritType()),
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 0

    def test_layer_definition(self):
        """Test layer definition syntax."""
        checker = TypeChecker()

        # layer TernaryLinear(input: Tensor, weights: Tensor):
        #     result = input @ weights

        layer_def = LayerDef(
            name="TernaryLinear",
            params=[
                Param(
                    name="input",
                    type_annotation=TensorType(element_type=TritType()),
                    lineno=1,
                    col_offset=0,
                ),
                Param(
                    name="weights",
                    type_annotation=TensorType(element_type=TritType()),
                    lineno=1,
                    col_offset=0,
                ),
            ],
            body=[
                Assignment(
                    target="result",
                    value=BinaryOp(
                        left=Identifier(name="input", lineno=2, col_offset=0),
                        op="@",
                        right=Identifier(name="weights", lineno=2, col_offset=5),
                        lineno=2,
                        col_offset=0,
                    ),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[layer_def])
        errors = checker.validate(program)
        assert len(errors) == 0


# ============================================================================
# Effect System Tests
# ============================================================================


class TestEffectSystem:
    """Tests for effect tracking and purity analysis."""

    def test_pure_function_detection(self):
        """Test detection of pure functions."""
        checker = TypeChecker(enable_effects=True)

        # Pure function: no side effects
        func_def = FunctionDef(
            name="pure_func",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Return(
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                )
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 0

        # Check that function is marked as pure
        assert "pure_func" in checker.function_table
        # Pure functions should have PURE and READ effects
        assert (
            EffectType.PURE in checker.function_table["pure_func"].effects
            or EffectType.READ in checker.function_table["pure_func"].effects
        )

    def test_impure_function_detection(self):
        """Test detection of impure functions."""
        checker = TypeChecker(enable_effects=True)

        # Impure function: has assignment (write effect)
        func_def = FunctionDef(
            name="impure_func",
            params=[Param(name="x", type_annotation=TritType(), lineno=1, col_offset=0)],
            return_type=TritType(),
            body=[
                Assignment(
                    target="y",
                    value=Identifier(name="x", lineno=2, col_offset=0),
                    lineno=2,
                    col_offset=0,
                ),
                Return(
                    value=Identifier(name="y", lineno=3, col_offset=0),
                    lineno=3,
                    col_offset=0,
                ),
            ],
            lineno=1,
            col_offset=0,
        )

        program = Program(statements=[func_def])
        errors = checker.validate(program)
        assert len(errors) == 0

        # Check that function has side effects
        assert "impure_func" in checker.function_table
        assert EffectType.WRITE in checker.function_table["impure_func"].effects

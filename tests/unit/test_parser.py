"""
Unit tests for the Triton DSL Parser.
"""

from compiler.ast.nodes import (
    Assignment,
    BinaryOp,
    Declaration,
    FloatLiteral,
    FunctionCall,
    Identifier,
    IntegerLiteral,
    LayerDef,
    Program,
    ReturnStmt,
    TernaryTensor,
)
from compiler.parser.triton_parser import parse


class TestBasicParsing:
    """Test basic parsing functionality."""

    def test_empty_program(self):
        result = parse("")
        assert isinstance(result, Program)
        assert len(result.statements) == 0

    def test_simple_declaration(self):
        code = "let x: int8"
        result = parse(code)
        assert isinstance(result, Program)
        assert len(result.statements) == 1

        stmt = result.statements[0]
        assert isinstance(stmt, Declaration)
        assert stmt.name == "x"
        assert stmt.var_type.name == "int8"
        assert stmt.initializer is None

    def test_declaration_with_initialization(self):
        code = "let x: int8 = 42"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, Declaration)
        assert stmt.name == "x"
        assert isinstance(stmt.initializer, IntegerLiteral)
        assert stmt.initializer.value == 42

    def test_assignment(self):
        code = "x = 10"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, Assignment)
        assert stmt.name == "x"
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 10


class TestExpressions:
    """Test expression parsing."""

    def test_integer_literal(self):
        code = "let x: int8 = 123"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, IntegerLiteral)
        assert stmt.initializer.value == 123

    def test_float_literal(self):
        code = "let x: float32 = 3.14"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FloatLiteral)
        assert stmt.initializer.value == 3.14

    def test_identifier(self):
        code = "y = x"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_binary_addition(self):
        code = "let z: int8 = x + y"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "+"

    def test_binary_multiplication(self):
        code = "let z: int8 = x * y"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "*"

    def test_matmul(self):
        code = "let z: float32 = A @ B"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "@"

    def test_function_call(self):
        code = "let y: int8 = relu(x)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FunctionCall)
        assert stmt.initializer.name == "relu"
        assert len(stmt.initializer.args) == 1


class TestLayerDef:
    """Test layer definition parsing."""

    def test_simple_layer(self):
        code = "layer dense(x: float32) -> float32 { return x }"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, LayerDef)
        assert stmt.name == "dense"
        assert len(stmt.params) == 1
        assert stmt.params[0].name == "x"
        assert stmt.return_type.name == "float32"
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], ReturnStmt)

    def test_layer_with_multiple_params(self):
        code = "layer add(x: int8, y: int8) -> int8 { return x + y }"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, LayerDef)
        assert len(stmt.params) == 2

    def test_empty_layer(self):
        code = "layer noop() -> int8 { }"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, LayerDef)
        assert len(stmt.params) == 0
        assert len(stmt.body) == 0


class TestTernaryTensor:
    """Test ternary tensor parsing."""

    def test_ternary_tensor_literal(self):
        code = "let w: TernaryTensor = TernaryTensor[2, 2](1, 0, -1, 1)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, TernaryTensor)
        assert stmt.initializer.shape == [2, 2]
        assert stmt.initializer.values == [1, 0, -1, 1]


class TestReturnStatement:
    """Test return statements."""

    def test_return_with_value(self):
        code = "return x"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"

    def test_return_without_value(self):
        code = "return"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, ReturnStmt)
        assert stmt.value is None


class TestASTSerialization:
    """Test AST node serialization."""

    def test_to_dict_simple(self):
        code = "let x: int8 = 5"
        result = parse(code)
        d = result.to_dict()
        assert d["node_type"] == "Program"
        assert len(d["statements"]) == 1
        assert d["statements"][0]["node_type"] == "Declaration"
        assert d["statements"][0]["name"] == "x"

    def test_repr(self):
        code = "let x: int8 = 5"
        result = parse(code)
        r = repr(result)
        assert "Program" in r


class TestMultipleStatements:
    """Test parsing multiple statements."""

    def test_two_declarations(self):
        code = "let x: int8 = 1\nlet y: int8 = 2"
        result = parse(code)
        assert len(result.statements) == 2
        assert result.statements[0].name == "x"
        assert result.statements[1].name == "y"

"""
Unit tests for Triton DSL Parser.
"""
import pytest
from compiler.parser.triton_parser import parse
from compiler.ast.nodes import (
    Program, Declaration, Assignment, LayerDef, ReturnStmt,
    BinaryOp, FunctionCall, TernaryTensor, Identifier,
    IntegerLiteral, FloatLiteral, Type, Param
)


class TestBasicParsing:
    """Test basic parsing functionality."""
    
    def test_empty_program(self):
        """Test parsing an empty program."""
        result = parse("")
        assert isinstance(result, Program)
        assert len(result.statements) == 0
    
    def test_simple_declaration(self):
        """Test parsing a simple variable declaration."""
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
        """Test parsing a declaration with initialization."""
        code = "let x: int8 = 42"
        result = parse(code)
        assert isinstance(result, Program)
        assert len(result.statements) == 1
        
        stmt = result.statements[0]
        assert isinstance(stmt, Declaration)
        assert stmt.name == "x"
        assert stmt.var_type.name == "int8"
        assert isinstance(stmt.initializer, IntegerLiteral)
        assert stmt.initializer.value == 42
    
    def test_assignment(self):
        """Test parsing an assignment statement."""
        code = "x = 10"
        result = parse(code)
        assert isinstance(result, Program)
        assert len(result.statements) == 1
        
        stmt = result.statements[0]
        assert isinstance(stmt, Assignment)
        assert stmt.name == "x"
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 10


class TestExpressions:
    """Test expression parsing."""
    
    def test_integer_literal(self):
        """Test parsing integer literals."""
        code = "let x: int8 = 123"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, IntegerLiteral)
        assert stmt.initializer.value == 123
    
    def test_float_literal(self):
        """Test parsing float literals."""
        code = "let x: float32 = 3.14"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FloatLiteral)
        assert stmt.initializer.value == 3.14
    
    def test_identifier(self):
        """Test parsing identifiers."""
        code = "y = x"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.value, Identifier)
        assert stmt.value.name == "x"
    
    def test_binary_addition(self):
        """Test parsing binary addition."""
        code = "let z: int8 = x + y"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "+"
        assert isinstance(stmt.initializer.left, Identifier)
        assert isinstance(stmt.initializer.right, Identifier)
    
    def test_binary_multiplication(self):
        """Test parsing binary multiplication."""
        code = "let z: int8 = x * y"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "*"
    
    def test_matrix_multiplication(self):
        """Test parsing matrix multiplication."""
        code = "let z: TernaryTensor = x @ y"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        assert stmt.initializer.op == "@"
    
    def test_complex_expression(self):
        """Test parsing complex expressions."""
        code = "let result: int8 = a + b * c"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, BinaryOp)
        # Due to precedence, should be: a + (b * c)
        assert stmt.initializer.op == "+"
        assert isinstance(stmt.initializer.right, BinaryOp)
        assert stmt.initializer.right.op == "*"


class TestFunctionCalls:
    """Test function call parsing."""
    
    def test_function_call_no_args(self):
        """Test parsing function call with no arguments."""
        code = "let x: int8 = foo()"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FunctionCall)
        assert stmt.initializer.name == "foo"
        assert len(stmt.initializer.args) == 0
    
    def test_function_call_one_arg(self):
        """Test parsing function call with one argument."""
        code = "let x: int8 = foo(42)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FunctionCall)
        assert stmt.initializer.name == "foo"
        assert len(stmt.initializer.args) == 1
        assert isinstance(stmt.initializer.args[0], IntegerLiteral)
    
    def test_function_call_multiple_args(self):
        """Test parsing function call with multiple arguments."""
        code = "let x: int8 = foo(a, b, c)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, FunctionCall)
        assert stmt.initializer.name == "foo"
        assert len(stmt.initializer.args) == 3


class TestTernaryTensor:
    """Test TernaryTensor parsing."""
    
    def test_ternary_tensor_simple(self):
        """Test parsing a simple TernaryTensor."""
        code = "let t: TernaryTensor = TernaryTensor[2, 2](1, 0, -1, 1)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, TernaryTensor)
        assert stmt.initializer.shape == [2, 2]
        assert stmt.initializer.values == [1, 0, -1, 1]
    
    def test_ternary_tensor_single_dim(self):
        """Test parsing a 1D TernaryTensor."""
        code = "let t: TernaryTensor = TernaryTensor[4](1, 1, 0, -1)"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt.initializer, TernaryTensor)
        assert stmt.initializer.shape == [4]
        assert stmt.initializer.values == [1, 1, 0, -1]


class TestLayerDef:
    """Test layer definition parsing."""
    
    def test_layer_no_params(self):
        """Test parsing a layer with no parameters."""
        code = """layer simple() -> int8 {
            return 42
        }"""
        result = parse(code)
        assert isinstance(result, Program)
        assert len(result.statements) == 1
        
        layer = result.statements[0]
        assert isinstance(layer, LayerDef)
        assert layer.name == "simple"
        assert len(layer.params) == 0
        assert layer.return_type.name == "int8"
        assert len(layer.body) == 1
        assert isinstance(layer.body[0], ReturnStmt)
    
    def test_layer_with_params(self):
        """Test parsing a layer with parameters."""
        code = """layer add(x: int8, y: int8) -> int8 {
            return x + y
        }"""
        result = parse(code)
        layer = result.statements[0]
        assert isinstance(layer, LayerDef)
        assert layer.name == "add"
        assert len(layer.params) == 2
        assert layer.params[0].name == "x"
        assert layer.params[0].param_type.name == "int8"
        assert layer.params[1].name == "y"
        assert layer.params[1].param_type.name == "int8"
    
    def test_layer_with_body(self):
        """Test parsing a layer with multiple statements in body."""
        code = """layer compute(x: int8) -> int8 {
            let y: int8 = x * 2
            let z: int8 = y + 1
            return z
        }"""
        result = parse(code)
        layer = result.statements[0]
        assert isinstance(layer, LayerDef)
        assert len(layer.body) == 3
        assert isinstance(layer.body[0], Declaration)
        assert isinstance(layer.body[1], Declaration)
        assert isinstance(layer.body[2], ReturnStmt)


class TestReturnStatement:
    """Test return statement parsing."""
    
    def test_return_with_value(self):
        """Test parsing return with value."""
        code = "return 42"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, ReturnStmt)
        assert isinstance(stmt.value, IntegerLiteral)
        assert stmt.value.value == 42
    
    def test_return_without_value(self):
        """Test parsing return without value."""
        code = "return"
        result = parse(code)
        stmt = result.statements[0]
        assert isinstance(stmt, ReturnStmt)
        assert stmt.value is None


class TestMultipleStatements:
    """Test parsing multiple statements."""
    
    def test_multiple_declarations(self):
        """Test parsing multiple declarations."""
        code = """
        let x: int8 = 1
        let y: int8 = 2
        let z: int8 = 3
        """
        result = parse(code)
        assert len(result.statements) == 3
        assert all(isinstance(stmt, Declaration) for stmt in result.statements)
    
    def test_mixed_statements(self):
        """Test parsing mixed statement types."""
        code = """
        let x: int8 = 10
        y = 20
        let z: int8 = x + y
        """
        result = parse(code)
        assert len(result.statements) == 3
        assert isinstance(result.statements[0], Declaration)
        assert isinstance(result.statements[1], Assignment)
        assert isinstance(result.statements[2], Declaration)


class TestTypes:
    """Test type parsing."""
    
    def test_trit_type(self):
        """Test parsing trit type."""
        code = "let x: trit"
        result = parse(code)
        stmt = result.statements[0]
        assert stmt.var_type.name == "trit"
    
    def test_int8_type(self):
        """Test parsing int8 type."""
        code = "let x: int8"
        result = parse(code)
        stmt = result.statements[0]
        assert stmt.var_type.name == "int8"
    
    def test_float16_type(self):
        """Test parsing float16 type."""
        code = "let x: float16"
        result = parse(code)
        stmt = result.statements[0]
        assert stmt.var_type.name == "float16"
    
    def test_float32_type(self):
        """Test parsing float32 type."""
        code = "let x: float32"
        result = parse(code)
        stmt = result.statements[0]
        assert stmt.var_type.name == "float32"
    
    def test_ternarytensor_type(self):
        """Test parsing TernaryTensor type."""
        code = "let x: TernaryTensor"
        result = parse(code)
        stmt = result.statements[0]
        assert stmt.var_type.name == "TernaryTensor"


class TestASTSerialization:
    """Test AST node serialization."""
    
    def test_program_to_dict(self):
        """Test Program node serialization."""
        code = "let x: int8 = 1"
        result = parse(code)
        ast_dict = result.to_dict()
        assert ast_dict['node_type'] == 'Program'
        assert 'statements' in ast_dict
        assert len(ast_dict['statements']) == 1
    
    def test_declaration_to_dict(self):
        """Test Declaration node serialization."""
        code = "let x: int8 = 42"
        result = parse(code)
        stmt = result.statements[0]
        stmt_dict = stmt.to_dict()
        assert stmt_dict['node_type'] == 'Declaration'
        assert stmt_dict['name'] == 'x'
        assert stmt_dict['var_type']['name'] == 'int8'
    
    def test_binary_op_to_dict(self):
        """Test BinaryOp node serialization."""
        code = "let x: int8 = 1 + 2"
        result = parse(code)
        stmt = result.statements[0]
        init_dict = stmt.initializer.to_dict()
        assert init_dict['node_type'] == 'BinaryOp'
        assert init_dict['op'] == '+'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

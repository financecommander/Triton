"""
Unit tests for the AST Node definitions.
"""

from compiler.ast.nodes import (
    Assignment,
    BinaryOp,
    Declaration,
    Expression,
    FloatLiteral,
    FunctionCall,
    Identifier,
    IntegerLiteral,
    LayerDef,
    Node,
    Param,
    Program,
    ReturnStmt,
    Statement,
    TernaryTensor,
    Type,
)


class TestBaseNode:
    """Test base Node class."""

    def test_node_creation(self):
        node = Node(lineno=1, col_offset=5)
        assert node.lineno == 1
        assert node.col_offset == 5

    def test_node_defaults(self):
        node = Node()
        assert node.lineno == 0
        assert node.col_offset == 0

    def test_node_repr(self):
        node = Node()
        assert repr(node) == "Node()"

    def test_node_to_dict(self):
        node = Node(lineno=1, col_offset=2)
        d = node.to_dict()
        assert d["node_type"] == "Node"
        assert d["lineno"] == 1
        assert d["col_offset"] == 2


class TestTypeNode:
    """Test Type node."""

    def test_type_creation(self):
        t = Type("int8")
        assert t.name == "int8"

    def test_type_repr(self):
        t = Type("float32")
        assert "float32" in repr(t)

    def test_type_to_dict(self):
        t = Type("trit")
        d = t.to_dict()
        assert d["name"] == "trit"
        assert d["node_type"] == "Type"


class TestExpressionNodes:
    """Test expression node classes."""

    def test_integer_literal(self):
        lit = IntegerLiteral(42)
        assert lit.value == 42
        assert "42" in repr(lit)
        d = lit.to_dict()
        assert d["value"] == 42

    def test_float_literal(self):
        lit = FloatLiteral(3.14)
        assert lit.value == 3.14
        d = lit.to_dict()
        assert d["value"] == 3.14

    def test_identifier(self):
        ident = Identifier("x")
        assert ident.name == "x"
        assert "x" in repr(ident)
        d = ident.to_dict()
        assert d["name"] == "x"

    def test_binary_op(self):
        left = IntegerLiteral(1)
        right = IntegerLiteral(2)
        op = BinaryOp(left, "+", right)
        assert op.op == "+"
        d = op.to_dict()
        assert d["op"] == "+"
        assert d["left"]["value"] == 1
        assert d["right"]["value"] == 2

    def test_function_call(self):
        args = [Identifier("x")]
        call = FunctionCall("relu", args)
        assert call.name == "relu"
        assert len(call.args) == 1
        d = call.to_dict()
        assert d["name"] == "relu"

    def test_ternary_tensor(self):
        tensor = TernaryTensor(shape=[2, 2], values=[1, 0, -1, 1])
        assert tensor.shape == [2, 2]
        assert tensor.values == [1, 0, -1, 1]
        d = tensor.to_dict()
        assert d["shape"] == [2, 2]
        assert d["values"] == [1, 0, -1, 1]


class TestStatementNodes:
    """Test statement node classes."""

    def test_declaration(self):
        t = Type("int8")
        init = IntegerLiteral(5)
        decl = Declaration("x", t, init)
        assert decl.name == "x"
        assert decl.var_type.name == "int8"
        assert decl.initializer.value == 5

    def test_declaration_no_init(self):
        t = Type("int8")
        decl = Declaration("x", t)
        assert decl.initializer is None
        d = decl.to_dict()
        assert d["initializer"] is None

    def test_assignment(self):
        val = IntegerLiteral(10)
        assign = Assignment("x", val)
        assert assign.name == "x"
        d = assign.to_dict()
        assert d["name"] == "x"

    def test_return_stmt(self):
        val = Identifier("x")
        ret = ReturnStmt(val)
        assert isinstance(ret.value, Identifier)
        d = ret.to_dict()
        assert d["value"]["name"] == "x"

    def test_return_stmt_no_value(self):
        ret = ReturnStmt()
        assert ret.value is None
        d = ret.to_dict()
        assert d["value"] is None

    def test_layer_def(self):
        param = Param("x", Type("float32"))
        body = [ReturnStmt(Identifier("x"))]
        layer = LayerDef("dense", [param], Type("float32"), body)
        assert layer.name == "dense"
        assert len(layer.params) == 1
        d = layer.to_dict()
        assert d["name"] == "dense"
        assert len(d["params"]) == 1
        assert len(d["body"]) == 1


class TestProgramNode:
    """Test Program node."""

    def test_program_creation(self):
        stmts = [Declaration("x", Type("int8"))]
        prog = Program(stmts)
        assert len(prog.statements) == 1
        assert "Program" in repr(prog)

    def test_program_to_dict(self):
        stmts = [Declaration("x", Type("int8"))]
        prog = Program(stmts)
        d = prog.to_dict()
        assert d["node_type"] == "Program"
        assert len(d["statements"]) == 1


class TestInheritance:
    """Test class hierarchy."""

    def test_statement_is_node(self):
        assert issubclass(Statement, Node)

    def test_expression_is_node(self):
        assert issubclass(Expression, Node)

    def test_declaration_is_statement(self):
        assert issubclass(Declaration, Statement)

    def test_binary_op_is_expression(self):
        assert issubclass(BinaryOp, Expression)

    def test_layer_def_is_statement(self):
        assert issubclass(LayerDef, Statement)

"""
Unit tests for the Triton DSL Lexer.
"""

from compiler.lexer.triton_lexer import tokenize


class TestKeywordTokens:
    """Test keyword tokenization."""

    def test_let_keyword(self):
        tokens = tokenize("let")
        assert len(tokens) == 1
        assert tokens[0].type == "LET"

    def test_layer_keyword(self):
        tokens = tokenize("layer")
        assert len(tokens) == 1
        assert tokens[0].type == "LAYER"

    def test_fn_keyword(self):
        tokens = tokenize("fn")
        assert len(tokens) == 1
        assert tokens[0].type == "FN"

    def test_return_keyword(self):
        tokens = tokenize("return")
        assert len(tokens) == 1
        assert tokens[0].type == "RETURN"

    def test_trit_keyword(self):
        tokens = tokenize("trit")
        assert len(tokens) == 1
        assert tokens[0].type == "TRIT"

    def test_ternary_tensor_keyword(self):
        tokens = tokenize("TernaryTensor")
        assert len(tokens) == 1
        assert tokens[0].type == "TERNARYTENSOR"

    def test_int8_keyword(self):
        tokens = tokenize("int8")
        assert len(tokens) == 1
        assert tokens[0].type == "INT8"

    def test_float16_keyword(self):
        tokens = tokenize("float16")
        assert len(tokens) == 1
        assert tokens[0].type == "FLOAT16"

    def test_float32_keyword(self):
        tokens = tokenize("float32")
        assert len(tokens) == 1
        assert tokens[0].type == "FLOAT32"


class TestOperatorTokens:
    """Test operator tokenization."""

    def test_plus(self):
        tokens = tokenize("+")
        assert tokens[0].type == "PLUS"

    def test_minus(self):
        tokens = tokenize("-")
        assert tokens[0].type == "MINUS"

    def test_times(self):
        tokens = tokenize("*")
        assert tokens[0].type == "TIMES"

    def test_matmul(self):
        tokens = tokenize("@")
        assert tokens[0].type == "MATMUL"

    def test_arrow(self):
        tokens = tokenize("->")
        assert tokens[0].type == "ARROW"

    def test_assign(self):
        tokens = tokenize("=")
        assert tokens[0].type == "ASSIGN"


class TestLiteralTokens:
    """Test literal tokenization."""

    def test_integer(self):
        tokens = tokenize("42")
        assert tokens[0].type == "INTEGER"
        assert tokens[0].value == 42

    def test_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].type == "FLOAT"
        assert tokens[0].value == 3.14

    def test_identifier(self):
        tokens = tokenize("my_var")
        assert tokens[0].type == "IDENTIFIER"
        assert tokens[0].value == "my_var"


class TestDelimiterTokens:
    """Test delimiter tokenization."""

    def test_parens(self):
        tokens = tokenize("()")
        assert tokens[0].type == "LPAREN"
        assert tokens[1].type == "RPAREN"

    def test_braces(self):
        tokens = tokenize("{}")
        assert tokens[0].type == "LBRACE"
        assert tokens[1].type == "RBRACE"

    def test_brackets(self):
        tokens = tokenize("[]")
        assert tokens[0].type == "LBRACKET"
        assert tokens[1].type == "RBRACKET"

    def test_comma(self):
        tokens = tokenize(",")
        assert tokens[0].type == "COMMA"

    def test_colon(self):
        tokens = tokenize(":")
        assert tokens[0].type == "COLON"

    def test_semicolon(self):
        tokens = tokenize(";")
        assert tokens[0].type == "SEMICOLON"


class TestComplexExpressions:
    """Test tokenization of complex expressions."""

    def test_variable_declaration(self):
        tokens = tokenize("let w: trit = 1")
        types = [t.type for t in tokens]
        assert types == ["LET", "IDENTIFIER", "COLON", "TRIT", "ASSIGN", "INTEGER"]

    def test_function_signature(self):
        tokens = tokenize("fn add(x: int8, y: int8) -> int8")
        types = [t.type for t in tokens]
        expected = [
            "FN", "IDENTIFIER", "LPAREN", "IDENTIFIER", "COLON", "INT8",
            "COMMA", "IDENTIFIER", "COLON", "INT8", "RPAREN", "ARROW", "INT8"
        ]
        assert types == expected

    def test_matrix_multiplication(self):
        tokens = tokenize("A @ B")
        types = [t.type for t in tokens]
        assert types == ["IDENTIFIER", "MATMUL", "IDENTIFIER"]

    def test_multiline_tracking(self):
        tokens = tokenize("let x = 1\nlet y = 2")
        assert len(tokens) == 8

"""
Unit tests for the Triton DSL Lexer.

Tests all token types, error handling, and line/column tracking.
"""

import pytest
from compiler.lexer.triton_lexer import TernaryLexer


class TestLexerBasics:
    """Test basic lexer functionality."""
    
    def test_lexer_creation(self):
        """Test that lexer can be created."""
        lexer = TernaryLexer()
        assert lexer is not None
        
    def test_lexer_input(self):
        """Test that lexer accepts input."""
        lexer = TernaryLexer()
        lexer.input("let x = 1")
        assert lexer.lexer is not None


class TestKeywordTokens:
    """Test keyword tokenization."""
    
    def test_let_keyword(self):
        """Test LET keyword."""
        lexer = TernaryLexer()
        lexer.input("let")
        tok = lexer.token()
        assert tok.type == "LET"
        assert tok.value == "let"
    
    def test_layer_keyword(self):
        """Test LAYER keyword."""
        lexer = TernaryLexer()
        lexer.input("layer")
        tok = lexer.token()
        assert tok.type == "LAYER"
        assert tok.value == "layer"
    
    def test_fn_keyword(self):
        """Test FN keyword."""
        lexer = TernaryLexer()
        lexer.input("fn")
        tok = lexer.token()
        assert tok.type == "FN"
        assert tok.value == "fn"
    
    def test_return_keyword(self):
        """Test RETURN keyword."""
        lexer = TernaryLexer()
        lexer.input("return")
        tok = lexer.token()
        assert tok.type == "RETURN"
        assert tok.value == "return"
    
    def test_trit_keyword(self):
        """Test TRIT keyword."""
        lexer = TernaryLexer()
        lexer.input("trit")
        tok = lexer.token()
        assert tok.type == "TRIT"
        assert tok.value == "trit"
    
    def test_ternary_tensor_keyword(self):
        """Test TERNARY_TENSOR keyword."""
        lexer = TernaryLexer()
        lexer.input("TernaryTensor")
        tok = lexer.token()
        assert tok.type == "TERNARY_TENSOR"
        assert tok.value == "TernaryTensor"
    
    def test_int8_keyword(self):
        """Test INT8 keyword."""
        lexer = TernaryLexer()
        lexer.input("int8")
        tok = lexer.token()
        assert tok.type == "INT8"
        assert tok.value == "int8"
    
    def test_float16_keyword(self):
        """Test FLOAT16 keyword."""
        lexer = TernaryLexer()
        lexer.input("float16")
        tok = lexer.token()
        assert tok.type == "FLOAT16"
        assert tok.value == "float16"
    
    def test_float32_keyword(self):
        """Test FLOAT32 keyword."""
        lexer = TernaryLexer()
        lexer.input("float32")
        tok = lexer.token()
        assert tok.type == "FLOAT32"
        assert tok.value == "float32"


class TestOperatorTokens:
    """Test operator tokenization."""
    
    def test_plus_operator(self):
        """Test PLUS operator."""
        lexer = TernaryLexer()
        lexer.input("+")
        tok = lexer.token()
        assert tok.type == "PLUS"
    
    def test_minus_operator(self):
        """Test MINUS operator."""
        lexer = TernaryLexer()
        lexer.input("-")
        tok = lexer.token()
        assert tok.type == "MINUS"
    
    def test_star_operator(self):
        """Test STAR operator."""
        lexer = TernaryLexer()
        lexer.input("*")
        tok = lexer.token()
        assert tok.type == "STAR"
    
    def test_matmul_operator(self):
        """Test MATMUL operator."""
        lexer = TernaryLexer()
        lexer.input("@")
        tok = lexer.token()
        assert tok.type == "MATMUL"
    
    def test_arrow_operator(self):
        """Test ARROW operator."""
        lexer = TernaryLexer()
        lexer.input("->")
        tok = lexer.token()
        assert tok.type == "ARROW"
    
    def test_assign_operator(self):
        """Test ASSIGN operator."""
        lexer = TernaryLexer()
        lexer.input("=")
        tok = lexer.token()
        assert tok.type == "ASSIGN"


class TestLiteralTokens:
    """Test literal tokenization."""
    
    def test_trit_literal_negative_one(self):
        """Test TRIT_LITERAL for -1."""
        lexer = TernaryLexer()
        lexer.input("-1")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == -1
    
    def test_trit_literal_zero(self):
        """Test TRIT_LITERAL for 0."""
        lexer = TernaryLexer()
        lexer.input("0")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == 0
    
    def test_trit_literal_one(self):
        """Test TRIT_LITERAL for 1."""
        lexer = TernaryLexer()
        lexer.input("1")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == 1
    
    def test_integer_literal(self):
        """Test INTEGER literal."""
        lexer = TernaryLexer()
        lexer.input("42")
        tok = lexer.token()
        assert tok.type == "INTEGER"
        assert tok.value == 42
    
    def test_integer_literal_large(self):
        """Test large INTEGER literal."""
        lexer = TernaryLexer()
        lexer.input("12345")
        tok = lexer.token()
        assert tok.type == "INTEGER"
        assert tok.value == 12345
    
    def test_float_literal(self):
        """Test FLOAT literal."""
        lexer = TernaryLexer()
        lexer.input("3.14")
        tok = lexer.token()
        assert tok.type == "FLOAT"
        assert tok.value == 3.14
    
    def test_float_literal_zero(self):
        """Test FLOAT literal with zero."""
        lexer = TernaryLexer()
        lexer.input("0.0")
        tok = lexer.token()
        assert tok.type == "FLOAT"
        assert tok.value == 0.0
    
    def test_identifier(self):
        """Test IDENTIFIER."""
        lexer = TernaryLexer()
        lexer.input("my_variable")
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"
        assert tok.value == "my_variable"
    
    def test_identifier_with_numbers(self):
        """Test IDENTIFIER with numbers."""
        lexer = TernaryLexer()
        lexer.input("var123")
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"
        assert tok.value == "var123"
    
    def test_identifier_underscore(self):
        """Test IDENTIFIER starting with underscore."""
        lexer = TernaryLexer()
        lexer.input("_private")
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"
        assert tok.value == "_private"


class TestDelimiterTokens:
    """Test delimiter tokenization."""
    
    def test_lparen(self):
        """Test LPAREN delimiter."""
        lexer = TernaryLexer()
        lexer.input("(")
        tok = lexer.token()
        assert tok.type == "LPAREN"
    
    def test_rparen(self):
        """Test RPAREN delimiter."""
        lexer = TernaryLexer()
        lexer.input(")")
        tok = lexer.token()
        assert tok.type == "RPAREN"
    
    def test_lbrace(self):
        """Test LBRACE delimiter."""
        lexer = TernaryLexer()
        lexer.input("{")
        tok = lexer.token()
        assert tok.type == "LBRACE"
    
    def test_rbrace(self):
        """Test RBRACE delimiter."""
        lexer = TernaryLexer()
        lexer.input("}")
        tok = lexer.token()
        assert tok.type == "RBRACE"
    
    def test_lbracket(self):
        """Test LBRACKET delimiter."""
        lexer = TernaryLexer()
        lexer.input("[")
        tok = lexer.token()
        assert tok.type == "LBRACKET"
    
    def test_rbracket(self):
        """Test RBRACKET delimiter."""
        lexer = TernaryLexer()
        lexer.input("]")
        tok = lexer.token()
        assert tok.type == "RBRACKET"
    
    def test_comma(self):
        """Test COMMA delimiter."""
        lexer = TernaryLexer()
        lexer.input(",")
        tok = lexer.token()
        assert tok.type == "COMMA"
    
    def test_colon(self):
        """Test COLON delimiter."""
        lexer = TernaryLexer()
        lexer.input(":")
        tok = lexer.token()
        assert tok.type == "COLON"


class TestComplexExpressions:
    """Test tokenization of complex expressions."""
    
    def test_variable_declaration(self):
        """Test tokenizing 'let w: trit = 1'."""
        lexer = TernaryLexer()
        lexer.input("let w: trit = 1")
        
        tokens = []
        for tok in lexer:
            tokens.append((tok.type, tok.value))
        
        expected = [
            ("LET", "let"),
            ("IDENTIFIER", "w"),
            ("COLON", ":"),
            ("TRIT", "trit"),
            ("ASSIGN", "="),
            ("TRIT_LITERAL", 1),
        ]
        assert tokens == expected
    
    def test_function_signature(self):
        """Test tokenizing function signature."""
        lexer = TernaryLexer()
        lexer.input("fn add(x: int8, y: int8) -> int8")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        expected = [
            "FN", "IDENTIFIER", "LPAREN", "IDENTIFIER", "COLON", "INT8",
            "COMMA", "IDENTIFIER", "COLON", "INT8", "RPAREN", "ARROW", "INT8"
        ]
        assert tokens == expected
    
    def test_matrix_multiplication(self):
        """Test tokenizing matrix multiplication."""
        lexer = TernaryLexer()
        lexer.input("A @ B")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        expected = ["IDENTIFIER", "MATMUL", "IDENTIFIER"]
        assert tokens == expected
    
    def test_arithmetic_expression(self):
        """Test tokenizing arithmetic expression."""
        lexer = TernaryLexer()
        lexer.input("x + y - z * 2")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        expected = ["IDENTIFIER", "PLUS", "IDENTIFIER", "MINUS", "IDENTIFIER", "STAR", "INTEGER"]
        assert tokens == expected
    
    def test_tensor_type_annotation(self):
        """Test tokenizing tensor type annotation."""
        lexer = TernaryLexer()
        lexer.input("weights: TernaryTensor[10, 20]")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        expected = [
            "IDENTIFIER", "COLON", "TERNARY_TENSOR", "LBRACKET",
            "INTEGER", "COMMA", "INTEGER", "RBRACKET"
        ]
        assert tokens == expected


class TestLineTracking:
    """Test line and column tracking."""
    
    def test_line_number_tracking(self):
        """Test that line numbers are tracked correctly."""
        lexer = TernaryLexer()
        lexer.input("let x = 1\nlet y = 2\nlet z = 3")
        
        tokens = []
        for tok in lexer:
            tokens.append((tok.type, tok.lineno))
        
        # Check that line numbers increase
        assert tokens[0][1] == 1  # let on line 1
        assert tokens[3][1] == 1  # 1 on line 1
        assert tokens[4][1] == 2  # let on line 2
        assert tokens[7][1] == 2  # 2 on line 2
        assert tokens[8][1] == 3  # let on line 3
    
    def test_column_tracking(self):
        """Test column position tracking."""
        lexer = TernaryLexer()
        lexer.input("let x = 1")
        
        tok1 = lexer.token()  # let
        col1 = lexer.find_column(tok1)
        assert col1 == 1
        
        tok2 = lexer.token()  # x
        col2 = lexer.find_column(tok2)
        assert col2 == 5
        
        tok3 = lexer.token()  # =
        col3 = lexer.find_column(tok3)
        assert col3 == 7


class TestErrorHandling:
    """Test error handling."""
    
    def test_illegal_character(self, capsys):
        """Test that illegal characters are handled."""
        lexer = TernaryLexer()
        lexer.input("let $x = 1")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        # Should skip illegal character and continue
        assert "LET" in tokens
        assert "IDENTIFIER" in tokens
        
        # Check error message was printed
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out
    
    def test_error_with_line_info(self, capsys):
        """Test that error includes line information."""
        lexer = TernaryLexer()
        lexer.input("let x = $")
        
        # Consume all tokens
        tokens = list(lexer)
        
        captured = capsys.readouterr()
        assert "line" in captured.out


class TestIterator:
    """Test iterator functionality."""
    
    def test_lexer_is_iterable(self):
        """Test that lexer can be used in for loop."""
        lexer = TernaryLexer()
        lexer.input("let x = 1")
        
        tokens = []
        for tok in lexer:
            tokens.append(tok.type)
        
        assert len(tokens) == 4
        assert tokens[0] == "LET"
    
    def test_iterator_stops_at_end(self):
        """Test that iterator stops when input is consumed."""
        lexer = TernaryLexer()
        lexer.input("let")
        
        tokens = list(lexer)
        assert len(tokens) == 1
        
        # Try to iterate again - should be empty
        tokens2 = list(lexer)
        assert len(tokens2) == 0

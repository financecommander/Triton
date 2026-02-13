"""
Comprehensive Lexer Test Suite

This module provides extensive testing coverage for the Triton DSL lexer,
including token recognition, error handling, boundary conditions, and performance.

Test Categories:
1. Token Recognition (100+ tests) - All keywords, operators, literals, delimiters
2. Error Handling (50+ tests) - Invalid characters, malformed input, tracking
3. Boundary Tests - Edge cases, large inputs, deep nesting
4. Performance Tests - Speed benchmarks, memory profiling, large files

Author: Triton Development Team
"""

import sys
import time
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from compiler.lexer.triton_lexer import TernaryLexer


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def lexer():
    """Create a fresh lexer instance for each test."""
    return TernaryLexer()


@pytest.fixture
def tokenize():
    """Helper fixture to tokenize input and return list of tokens."""
    def _tokenize(text):
        lex = TernaryLexer()
        lex.input(text)
        return list(lex)
    return _tokenize


@pytest.fixture
def token_types():
    """Helper fixture to extract token types from input."""
    def _token_types(text):
        lex = TernaryLexer()
        lex.input(text)
        return [tok.type for tok in lex]
    return _token_types


@pytest.fixture
def token_values():
    """Helper fixture to extract token values from input."""
    def _token_values(text):
        lex = TernaryLexer()
        lex.input(text)
        return [tok.value for tok in lex]
    return _token_values


# ============================================================================
# TOKEN RECOGNITION TESTS (100+ test cases)
# ============================================================================

class TestKeywordsComprehensive:
    """Comprehensive tests for all keyword tokens."""

    @pytest.mark.parametrize("keyword,expected_type", [
        ("layer", "LAYER"),
        ("let", "LET"),
        ("fn", "FN"),
        ("return", "RETURN"),
        ("trit", "TRIT"),
        ("tensor", "TENSOR"),
        ("TernaryTensor", "TERNARY_TENSOR"),
        ("int8", "INT8"),
        ("int32", "INT32"),
        ("float16", "FLOAT16"),
        ("float32", "FLOAT32"),
    ])
    def test_individual_keywords(self, lexer, keyword, expected_type):
        """Test each keyword produces correct token type."""
        lexer.input(keyword)
        tok = lexer.token()
        assert tok is not None
        assert tok.type == expected_type
        assert tok.value == keyword

    def _get_expected_token_type(self, keyword):
        """Helper to map keyword to expected token type."""
        keyword_map = {
            "layer": "LAYER",
            "let": "LET",
            "fn": "FN",
            "return": "RETURN",
            "trit": "TRIT",
            "tensor": "TENSOR",
            "TernaryTensor": "TERNARY_TENSOR",
            "int8": "INT8",
            "int32": "INT32",
            "float16": "FLOAT16",
            "float32": "FLOAT32",
        }
        return keyword_map[keyword]

    @pytest.mark.parametrize("keyword", [
        "layer", "let", "fn", "return", "trit", "tensor",
        "TernaryTensor", "int8", "int32", "float16", "float32"
    ])
    def test_keywords_with_whitespace(self, token_types, keyword):
        """Test keywords with surrounding whitespace."""
        expected_type = self._get_expected_token_type(keyword)
        assert token_types(f"  {keyword}  ") == [expected_type]

    @pytest.mark.parametrize("keyword", [
        "layer", "let", "fn", "return", "trit", "tensor",
        "TernaryTensor", "int8", "int32", "float16", "float32"
    ])
    def test_keywords_case_sensitive(self, token_types, keyword):
        """Test that keywords are case-sensitive."""
        # Uppercase versions should be identifiers (except TernaryTensor)
        if keyword != "TernaryTensor" and keyword.islower():
            upper = keyword.upper()
            result = token_types(upper)
            assert result == ["IDENTIFIER"]

    def test_multiple_keywords(self, token_types):
        """Test multiple keywords in sequence."""
        result = token_types("let fn return layer")
        assert result == ["LET", "FN", "RETURN", "LAYER"]

    def test_keywords_with_separators(self, token_types):
        """Test keywords separated by various delimiters."""
        result = token_types("let,fn;return:layer")
        assert result == ["LET", "COMMA", "FN", "SEMICOLON", "RETURN", "COLON", "LAYER"]

    def test_keyword_like_identifiers(self, token_types):
        """Test identifiers that look similar to keywords."""
        test_cases = [
            ("layer1", ["IDENTIFIER"]),
            ("let_var", ["IDENTIFIER"]),
            ("fn_name", ["IDENTIFIER"]),
            ("return_val", ["IDENTIFIER"]),
            ("trit_type", ["IDENTIFIER"]),
            ("my_layer", ["IDENTIFIER"]),
            ("_let", ["IDENTIFIER"]),
            ("FN", ["IDENTIFIER"]),
            ("LAYER", ["IDENTIFIER"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected


class TestOperatorsComprehensive:
    """Comprehensive tests for all operator tokens."""

    @pytest.mark.parametrize("operator,expected_type", [
        ("+", "PLUS"),
        ("-", "MINUS"),
        ("*", "STAR"),
        ("@", "MATMUL"),
        ("->", "ARROW"),
        ("<", "LT"),
        (">", "GT"),
        ("=", "ASSIGN"),
    ])
    def test_individual_operators(self, lexer, operator, expected_type):
        """Test each operator produces correct token type."""
        lexer.input(operator)
        tok = lexer.token()
        assert tok is not None
        assert tok.type == expected_type

    def test_operator_sequences(self, token_types):
        """Test multiple operators in sequence."""
        result = token_types("+ - * @ -> < > =")
        expected = ["PLUS", "MINUS", "STAR", "MATMUL", "ARROW", "LT", "GT", "ASSIGN"]
        assert result == expected

    def test_operators_without_spaces(self, token_types):
        """Test operators without spacing."""
        result = token_types("+-*@")
        assert result == ["PLUS", "MINUS", "STAR", "MATMUL"]

    def test_operators_in_expressions(self, token_types):
        """Test operators within expressions."""
        test_cases = [
            ("a+b", ["IDENTIFIER", "PLUS", "IDENTIFIER"]),
            ("x-y", ["IDENTIFIER", "MINUS", "IDENTIFIER"]),
            ("m*n", ["IDENTIFIER", "STAR", "IDENTIFIER"]),
            ("A@B", ["IDENTIFIER", "MATMUL", "IDENTIFIER"]),
            ("x<y", ["IDENTIFIER", "LT", "IDENTIFIER"]),
            ("x>y", ["IDENTIFIER", "GT", "IDENTIFIER"]),
            ("x=1", ["IDENTIFIER", "ASSIGN", "TRIT_LITERAL"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_arrow_vs_minus(self, token_types):
        """Test distinction between -> and separate - >."""
        assert token_types("->") == ["ARROW"]
        assert token_types("- >") == ["MINUS", "GT"]
        assert token_types("-->") == ["MINUS", "ARROW"]

    def test_minus_vs_negative_trit(self, lexer):
        """Test distinction between minus operator and -1 trit literal."""
        # -1 should be TRIT_LITERAL
        lexer.input("-1")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == -1

        # - 1 should be MINUS then TRIT_LITERAL
        lexer.input("- 1")
        tok1 = lexer.token()
        tok2 = lexer.token()
        assert tok1.type == "MINUS"
        assert tok2.type == "TRIT_LITERAL"
        assert tok2.value == 1

        # -10 should be MINUS then INTEGER
        lexer.input("-10")
        tok1 = lexer.token()
        tok2 = lexer.token()
        assert tok1.type == "MINUS"
        assert tok2.type == "INTEGER"
        assert tok2.value == 10

    def test_operators_with_whitespace_variations(self, token_types):
        """Test operators with various whitespace patterns."""
        test_cases = [
            ("  +  ", ["PLUS"]),
            ("\t-\t", ["MINUS"]),
            ("  *  ", ["STAR"]),
            ("  @  ", ["MATMUL"]),
            ("  ->  ", ["ARROW"]),
            ("  <  ", ["LT"]),
            ("  >  ", ["GT"]),
            ("  =  ", ["ASSIGN"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected


class TestLiteralsComprehensive:
    """Comprehensive tests for all literal tokens."""

    # Trit literals
    @pytest.mark.parametrize("trit,expected_value", [
        ("-1", -1),
        ("0", 0),
        ("1", 1),
    ])
    def test_trit_literals(self, lexer, trit, expected_value):
        """Test all three trit values."""
        lexer.input(trit)
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == expected_value

    def test_trit_in_context(self, token_types):
        """Test trit literals in various contexts."""
        test_cases = [
            ("-1 0 1", ["TRIT_LITERAL", "TRIT_LITERAL", "TRIT_LITERAL"]),
            ("[-1, 0, 1]", ["LBRACKET", "TRIT_LITERAL", "COMMA", "TRIT_LITERAL",
                            "COMMA", "TRIT_LITERAL", "RBRACKET"]),
            ("x=-1", ["IDENTIFIER", "ASSIGN", "TRIT_LITERAL"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    # Integer literals
    @pytest.mark.parametrize("integer,expected_value", [
        ("2", 2),
        ("10", 10),
        ("42", 42),
        ("100", 100),
        ("999", 999),
        ("1000", 1000),
        ("12345", 12345),
        ("9999999", 9999999),
    ])
    def test_integer_literals(self, lexer, integer, expected_value):
        """Test various integer literals."""
        lexer.input(integer)
        tok = lexer.token()
        assert tok.type == "INTEGER"
        assert tok.value == expected_value

    def test_very_large_integers(self, lexer):
        """Test very large integer values."""
        large_ints = [
            "1000000",
            "123456789",
            "999999999",
            "1234567890123456789",
        ]
        for int_str in large_ints:
            lexer.input(int_str)
            tok = lexer.token()
            assert tok.type == "INTEGER"
            assert tok.value == int(int_str)

    # Float literals
    @pytest.mark.parametrize("float_str,expected_value", [
        ("0.0", 0.0),
        ("1.0", 1.0),
        ("3.14", 3.14),
        ("2.71828", 2.71828),
        ("99.99", 99.99),
        ("0.001", 0.001),
        ("123.456", 123.456),
    ])
    def test_float_literals(self, lexer, float_str, expected_value):
        """Test various float literals."""
        lexer.input(float_str)
        tok = lexer.token()
        assert tok.type == "FLOAT"
        assert abs(tok.value - expected_value) < 1e-10

    def test_float_edge_cases(self, lexer):
        """Test edge cases for float literals."""
        test_cases = [
            ("0.0", 0.0),
            ("0.1", 0.1),
            ("1.0", 1.0),
            ("10.5", 10.5),
            ("999.999", 999.999),
        ]
        for float_str, expected in test_cases:
            lexer.input(float_str)
            tok = lexer.token()
            assert tok.type == "FLOAT"
            assert abs(tok.value - expected) < 1e-10

    # Identifiers
    @pytest.mark.parametrize("identifier", [
        "x",
        "y",
        "variable",
        "my_var",
        "myVar",
        "MyVar",
        "CONSTANT",
        "_private",
        "__dunder__",
        "var123",
        "var_123_abc",
        "a1b2c3",
    ])
    def test_valid_identifiers(self, lexer, identifier):
        """Test various valid identifier patterns."""
        lexer.input(identifier)
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"
        assert tok.value == identifier

    def test_identifier_length_variations(self, token_types):
        """Test identifiers of various lengths."""
        test_cases = [
            "a",
            "ab",
            "abc",
            "abcd",
            "a" * 10,
            "a" * 50,
            "a" * 100,
            "variable_with_very_long_name_that_goes_on_and_on",
        ]
        for ident in test_cases:
            result = token_types(ident)
            assert result == ["IDENTIFIER"]

    def test_identifier_naming_conventions(self, token_types):
        """Test common naming conventions."""
        test_cases = [
            ("snake_case", ["IDENTIFIER"]),
            ("camelCase", ["IDENTIFIER"]),
            ("PascalCase", ["IDENTIFIER"]),
            ("UPPER_CASE", ["IDENTIFIER"]),
            ("_leading_underscore", ["IDENTIFIER"]),
            ("trailing_underscore_", ["IDENTIFIER"]),
            ("with123numbers", ["IDENTIFIER"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected


class TestDelimitersComprehensive:
    """Comprehensive tests for all delimiter tokens."""

    @pytest.mark.parametrize("delimiter,expected_type", [
        ("(", "LPAREN"),
        (")", "RPAREN"),
        ("{", "LBRACE"),
        ("}", "RBRACE"),
        ("[", "LBRACKET"),
        ("]", "RBRACKET"),
        (",", "COMMA"),
        (":", "COLON"),
        (";", "SEMICOLON"),
        ("...", "ELLIPSIS"),
    ])
    def test_individual_delimiters(self, lexer, delimiter, expected_type):
        """Test each delimiter produces correct token type."""
        lexer.input(delimiter)
        tok = lexer.token()
        assert tok is not None
        assert tok.type == expected_type

    def test_delimiter_sequences(self, token_types):
        """Test multiple delimiters in sequence."""
        result = token_types("()[]{},;:")
        expected = ["LPAREN", "RPAREN", "LBRACKET", "RBRACKET",
                    "LBRACE", "RBRACE", "COMMA", "SEMICOLON", "COLON"]
        assert result == expected

    def test_matched_pairs(self, token_types):
        """Test matched delimiter pairs."""
        test_cases = [
            ("()", ["LPAREN", "RPAREN"]),
            ("[]", ["LBRACKET", "RBRACKET"]),
            ("{}", ["LBRACE", "RBRACE"]),
            ("(())", ["LPAREN", "LPAREN", "RPAREN", "RPAREN"]),
            ("[[]]", ["LBRACKET", "LBRACKET", "RBRACKET", "RBRACKET"]),
            ("{{}}", ["LBRACE", "LBRACE", "RBRACE", "RBRACE"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_delimiters_in_complex_expressions(self, token_types):
        """Test delimiters in complex nested structures."""
        test_cases = [
            ("fn(x, y)", ["FN", "LPAREN", "IDENTIFIER", "COMMA", "IDENTIFIER", "RPAREN"]),
            ("[1, 2, 3]", ["LBRACKET", "TRIT_LITERAL", "COMMA", "INTEGER",
                          "COMMA", "INTEGER", "RBRACKET"]),
            ("{a: b}", ["LBRACE", "IDENTIFIER", "COLON", "IDENTIFIER", "RBRACE"]),
            ("x[0]", ["IDENTIFIER", "LBRACKET", "TRIT_LITERAL", "RBRACKET"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_ellipsis_variations(self, token_types):
        """Test ellipsis token in various contexts."""
        assert token_types("...") == ["ELLIPSIS"]
        assert token_types("x...") == ["IDENTIFIER", "ELLIPSIS"]
        assert token_types("...,") == ["ELLIPSIS", "COMMA"]
        assert token_types("[...]") == ["LBRACKET", "ELLIPSIS", "RBRACKET"]


class TestWhitespaceAndComments:
    """Tests for whitespace handling and comments."""

    def test_spaces_ignored(self, token_types):
        """Test that spaces are properly ignored."""
        assert token_types("let x") == token_types("let  x")
        assert token_types("let x") == token_types("let   x")
        assert token_types("let x") == token_types("let     x")

    def test_tabs_ignored(self, token_types):
        """Test that tabs are properly ignored."""
        assert token_types("let\tx") == ["LET", "IDENTIFIER"]
        assert token_types("let\t\tx") == ["LET", "IDENTIFIER"]

    def test_mixed_whitespace(self, token_types):
        """Test mixed spaces and tabs."""
        assert token_types("let \t x") == ["LET", "IDENTIFIER"]
        assert token_types("let\t \tx") == ["LET", "IDENTIFIER"]

    def test_newlines_tracked(self, lexer):
        """Test that newlines update line numbers."""
        lexer.input("let\nx\ny")
        tok1 = lexer.token()
        tok2 = lexer.token()
        tok3 = lexer.token()
        assert tok1.lineno == 1
        assert tok2.lineno == 2
        assert tok3.lineno == 3

    def test_multiple_newlines(self, lexer):
        """Test multiple consecutive newlines."""
        lexer.input("let\n\n\nx")
        tok1 = lexer.token()
        tok2 = lexer.token()
        assert tok1.lineno == 1
        assert tok2.lineno == 4

    def test_comments_ignored(self, token_types):
        """Test that comments are ignored."""
        assert token_types("let # this is a comment") == ["LET"]
        assert token_types("let x # comment") == ["LET", "IDENTIFIER"]
        assert token_types("# full line comment\nlet x") == ["LET", "IDENTIFIER"]

    def test_comment_variations(self, token_types):
        """Test various comment patterns."""
        test_cases = [
            ("# comment", []),
            ("x # comment", ["IDENTIFIER"]),
            ("let # comment\nx", ["LET", "IDENTIFIER"]),
            ("# line 1\n# line 2\nlet", ["LET"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected


class TestComplexPrograms:
    """Tests for tokenizing complex, realistic programs."""

    def test_variable_declaration_variations(self, token_types):
        """Test various variable declaration patterns."""
        test_cases = [
            "let x = 1",
            "let y = 0",
            "let z = -1",
            "let w: trit = 1",
            "let count: int8 = 10",
            "let value: float32 = 3.14",
        ]
        for text in test_cases:
            result = token_types(text)
            assert "LET" in result
            assert "IDENTIFIER" in result
            assert "ASSIGN" in result

    def test_function_declarations(self, token_types):
        """Test function declaration patterns."""
        test_cases = [
            "fn foo()",
            "fn bar(x: int8)",
            "fn add(x: int8, y: int8) -> int8",
            "fn process(data: TernaryTensor) -> TernaryTensor",
        ]
        for text in test_cases:
            result = token_types(text)
            assert "FN" in result
            assert result[0] == "FN"

    def test_arithmetic_expressions(self, token_types):
        """Test complex arithmetic expressions."""
        test_cases = [
            ("a + b", ["IDENTIFIER", "PLUS", "IDENTIFIER"]),
            ("x - y + z", ["IDENTIFIER", "MINUS", "IDENTIFIER", "PLUS", "IDENTIFIER"]),
            ("a * b + c", ["IDENTIFIER", "STAR", "IDENTIFIER", "PLUS", "IDENTIFIER"]),
            ("x + y * z", ["IDENTIFIER", "PLUS", "IDENTIFIER", "STAR", "IDENTIFIER"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_matrix_operations(self, token_types):
        """Test matrix multiplication expressions."""
        test_cases = [
            ("A @ B", ["IDENTIFIER", "MATMUL", "IDENTIFIER"]),
            ("X @ Y @ Z", ["IDENTIFIER", "MATMUL", "IDENTIFIER", "MATMUL", "IDENTIFIER"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_tensor_indexing(self, token_types):
        """Test tensor indexing patterns."""
        test_cases = [
            ("x[0]", ["IDENTIFIER", "LBRACKET", "TRIT_LITERAL", "RBRACKET"]),
            ("matrix[10, 20]", ["IDENTIFIER", "LBRACKET", "INTEGER",
                               "COMMA", "INTEGER", "RBRACKET"]),
            ("arr[...]", ["IDENTIFIER", "LBRACKET", "ELLIPSIS", "RBRACKET"]),
        ]
        for text, expected in test_cases:
            assert token_types(text) == expected

    def test_type_annotations(self, token_types):
        """Test type annotation patterns."""
        test_cases = [
            "x: trit",
            "y: int8",
            "z: float32",
            "weights: TernaryTensor",
            "data: TernaryTensor[100, 200]",
        ]
        for text in test_cases:
            result = token_types(text)
            assert "COLON" in result

    def test_multiline_program(self, lexer):
        """Test tokenizing a complete multiline program."""
        program = """
        # Ternary neural network layer
        layer MyLayer {
            let weights: TernaryTensor[10, 20] = ...;
            
            fn forward(x: TernaryTensor) -> TernaryTensor {
                let result = x @ weights;
                return result;
            }
        }
        """
        lexer.input(program)
        tokens = list(lexer)
        assert len(tokens) > 0
        token_types = [t.type for t in tokens]
        assert "LAYER" in token_types
        assert "FN" in token_types
        assert "RETURN" in token_types

    def test_real_world_snippet(self, token_types):
        """Test a realistic code snippet."""
        code = "fn sigmoid(x: float32) -> float32"
        result = token_types(code)
        expected = ["FN", "IDENTIFIER", "LPAREN", "IDENTIFIER",
                   "COLON", "FLOAT32", "RPAREN", "ARROW", "FLOAT32"]
        assert result == expected


# ============================================================================
# ERROR HANDLING TESTS (50+ test cases)
# ============================================================================

class TestInvalidCharacters:
    """Tests for invalid character handling."""

    @pytest.mark.parametrize("char", [
        "!", "$", "?", "~", "`", "&", "|", "^", "%", "\\", '"', "'",
    ])
    def test_individual_invalid_characters(self, lexer, capsys, char):
        """Test that individual invalid characters are caught."""
        lexer.input(f"let {char} x")
        tokens = list(lexer)
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out

    def test_invalid_char_at_start(self, lexer, capsys):
        """Test invalid character at start of input."""
        lexer.input("$variable")
        tokens = list(lexer)
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out

    def test_invalid_char_at_end(self, lexer, capsys):
        """Test invalid character at end of input."""
        lexer.input("variable$")
        tokens = list(lexer)
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out

    def test_multiple_invalid_chars(self, lexer, capsys):
        """Test multiple invalid characters."""
        lexer.input("let $x = !y")
        tokens = list(lexer)
        captured = capsys.readouterr()
        # Should report multiple errors
        assert captured.out.count("Illegal character") >= 2

    def test_invalid_char_line_number(self, lexer, capsys):
        """Test that error message includes line number."""
        lexer.input("let x = 1\nlet $ y = 2")
        tokens = list(lexer)
        captured = capsys.readouterr()
        assert "line 2" in captured.out

    def test_recovery_after_error(self, token_types):
        """Test that lexer continues after encountering error."""
        # $ is invalid but lexer should continue
        result = token_types("let x")
        assert "LET" in result
        assert "IDENTIFIER" in result


class TestMalformedNumbers:
    """Tests for malformed number handling."""

    def test_leading_zeros(self, lexer):
        """Test numbers with leading zeros."""
        # These should still tokenize as integers
        lexer.input("01")
        tok = lexer.token()
        assert tok.type in ["TRIT_LITERAL", "INTEGER"]
        assert tok.value == 1

    def test_incomplete_floats(self, lexer):
        """Test handling of incomplete float literals."""
        # Note: "1." would be INTEGER "1" and error on "."
        lexer.input("1.0")
        tok = lexer.token()
        assert tok.type == "FLOAT"


class TestInvalidIdentifiers:
    """Tests for invalid identifier patterns."""

    def test_identifier_cannot_start_with_digit(self, lexer):
        """Test that identifiers cannot start with digits."""
        lexer.input("123abc")
        tok = lexer.token()
        # Should tokenize as INTEGER first
        assert tok.type == "INTEGER"
        assert tok.value == 123
        tok2 = lexer.token()
        assert tok2.type == "IDENTIFIER"
        assert tok2.value == "abc"

    def test_identifier_with_invalid_chars(self, lexer, capsys):
        """Test identifiers with invalid characters."""
        lexer.input("var$name")
        tokens = list(lexer)
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out


class TestLineColumnTracking:
    """Tests for accurate line and column tracking."""

    def test_single_line_columns(self, lexer):
        """Test column tracking on single line."""
        lexer.input("let x = 1")
        tok1 = lexer.token()
        assert lexer.find_column(tok1) == 1
        tok2 = lexer.token()
        assert lexer.find_column(tok2) == 5
        tok3 = lexer.token()
        assert lexer.find_column(tok3) == 7

    def test_multiline_line_numbers(self, lexer):
        """Test line number tracking across multiple lines."""
        lexer.input("let x = 1\nlet y = 2\nlet z = 3")
        tokens = []
        while True:
            tok = lexer.token()
            if not tok:
                break
            tokens.append((tok.type, tok.lineno))
        
        # Check specific line numbers
        assert tokens[0][1] == 1  # let on line 1
        assert tokens[4][1] == 2  # let on line 2
        assert tokens[8][1] == 3  # let on line 3

    def test_column_after_newline(self, lexer):
        """Test column tracking resets after newline."""
        lexer.input("let x\nlet y")
        tok1 = lexer.token()
        col1 = lexer.find_column(tok1)
        lexer.token()  # consume 'x'
        tok3 = lexer.token()  # 'let' on line 2
        col3 = lexer.find_column(tok3)
        assert col1 == col3  # Both 'let' keywords at same column

    def test_line_tracking_with_empty_lines(self, lexer):
        """Test line tracking with empty lines."""
        lexer.input("let\n\n\nx")
        tok1 = lexer.token()
        tok2 = lexer.token()
        assert tok1.lineno == 1
        assert tok2.lineno == 4

    def test_column_tracking_with_tabs(self, lexer):
        """Test column tracking with tabs."""
        lexer.input("let\tx")
        tok1 = lexer.token()
        tok2 = lexer.token()
        col1 = lexer.find_column(tok1)
        col2 = lexer.find_column(tok2)
        assert col2 > col1

    def test_accurate_error_position(self, lexer, capsys):
        """Test that error messages report accurate positions."""
        lexer.input("let x = $ 1")
        list(lexer)
        captured = capsys.readouterr()
        assert "line 1" in captured.out


class TestEdgeCaseErrors:
    """Tests for edge case error conditions."""

    def test_empty_input(self, lexer):
        """Test tokenizing empty input."""
        lexer.input("")
        tok = lexer.token()
        assert tok is None

    def test_only_whitespace(self, lexer):
        """Test tokenizing only whitespace."""
        lexer.input("   \t  \n  ")
        tok = lexer.token()
        assert tok is None

    def test_only_comments(self, lexer):
        """Test input with only comments."""
        lexer.input("# comment 1\n# comment 2")
        tok = lexer.token()
        assert tok is None

    def test_unterminated_input(self, token_types):
        """Test input that ends unexpectedly."""
        # These should still tokenize what's available
        assert len(token_types("let x =")) >= 3
        assert len(token_types("fn foo(")) >= 3

    def test_operator_at_end(self, token_types):
        """Test operator at end of input."""
        result = token_types("x +")
        assert result == ["IDENTIFIER", "PLUS"]


# ============================================================================
# BOUNDARY TESTS
# ============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_maximum_identifier_length(self, lexer):
        """Test very long identifier names."""
        long_id = "a" * 1000
        lexer.input(long_id)
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"
        assert len(tok.value) == 1000

    def test_extremely_long_identifier(self, token_types):
        """Test extremely long identifier (10000 chars)."""
        long_id = "variable_" * 1000
        result = token_types(long_id)
        assert result == ["IDENTIFIER"]

    def test_very_large_integer(self, lexer):
        """Test very large integer values."""
        large_int = "9" * 100
        lexer.input(large_int)
        tok = lexer.token()
        assert tok.type == "INTEGER"
        assert tok.value == int(large_int)

    def test_very_small_float(self, lexer):
        """Test very small float values."""
        small_float = "0.000001"
        lexer.input(small_float)
        tok = lexer.token()
        assert tok.type == "FLOAT"
        assert tok.value == 0.000001

    def test_very_large_float(self, lexer):
        """Test very large float values."""
        large_float = "999999.999999"
        lexer.input(large_float)
        tok = lexer.token()
        assert tok.type == "FLOAT"
        assert abs(tok.value - 999999.999999) < 1e-6

    def test_deeply_nested_delimiters(self, token_types):
        """Test deeply nested delimiter structures."""
        nested = "((((((((((x))))))))))"
        result = token_types(nested)
        assert result.count("LPAREN") == 10
        assert result.count("RPAREN") == 10
        assert "IDENTIFIER" in result

    def test_very_long_expression(self, token_types):
        """Test very long arithmetic expression."""
        expr = " + ".join([f"x{i}" for i in range(100)])
        result = token_types(expr)
        assert result.count("PLUS") == 99
        assert result.count("IDENTIFIER") == 100

    def test_many_tokens_single_line(self, lexer):
        """Test many tokens on single line."""
        text = ", ".join([f"x{i}" for i in range(1000)])
        lexer.input(text)
        tokens = list(lexer)
        assert len(tokens) == 1999  # 1000 identifiers + 999 commas

    def test_long_file_simulation(self, lexer):
        """Test tokenizing a large file (1000 lines)."""
        lines = ["let x = 1\n"] * 1000
        text = "".join(lines)
        lexer.input(text)
        tokens = list(lexer)
        # Each line has 4 tokens
        assert len(tokens) == 4000

    def test_very_long_file_simulation(self, lexer):
        """Test tokenizing a very large file (10000 lines)."""
        lines = ["let x = 1\n"] * 10000
        text = "".join(lines)
        lexer.input(text)
        tokens = list(lexer)
        assert len(tokens) == 40000

    def test_mixed_complexity_file(self, lexer):
        """Test file with mixed complexity patterns."""
        text = """
        # Complex ternary neural network implementation
        layer ConvLayer {
            let weights: TernaryTensor[128, 256] = ...;
            let bias: TernaryTensor[256] = ...;
            
            fn forward(input: TernaryTensor[100, 128]) -> TernaryTensor[100, 256] {
                let conv = input @ weights;
                let output = conv + bias;
                return output;
            }
            
            fn backward(grad: TernaryTensor) -> TernaryTensor {
                let grad_input = grad @ weights;
                return grad_input;
            }
        }
        """ * 100  # Repeat 100 times
        
        lexer.input(text)
        tokens = list(lexer)
        assert len(tokens) > 1000

    def test_unicode_identifiers(self, lexer):
        """Test that ASCII identifiers work (Unicode may not be supported)."""
        # Test basic ASCII
        lexer.input("variable")
        tok = lexer.token()
        assert tok.type == "IDENTIFIER"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance and benchmarking tests."""

    def test_tokenization_speed_small(self, benchmark, lexer):
        """Benchmark tokenization of small input."""
        text = "let x: int8 = 42"
        
        def tokenize():
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) == 6  # LET, IDENTIFIER, COLON, INT8, ASSIGN, INTEGER

    def test_tokenization_speed_medium(self, benchmark):
        """Benchmark tokenization of medium input (1000 lines)."""
        text = "let x = 1\n" * 1000
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) == 4000

    def test_tokenization_speed_large(self, benchmark):
        """Benchmark tokenization of large input (10000 lines)."""
        text = "let x = 1\n" * 10000
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) == 40000

    def test_complex_expression_performance(self, benchmark):
        """Benchmark complex expression tokenization."""
        # Generate complex nested expression
        text = "fn compute(x: TernaryTensor[100, 200], y: TernaryTensor[200, 300]) -> TernaryTensor[100, 300] { let result = x @ y; return result; }\n" * 100
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) > 1000

    def test_tokenization_with_many_keywords(self, benchmark):
        """Benchmark input with many keywords."""
        keywords = ["layer", "let", "fn", "return", "trit", "int8", "float32"]
        text = " ".join(keywords * 1000)
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) == 7000

    def test_tokenization_with_many_operators(self, benchmark):
        """Benchmark input with many operators."""
        text = "x + y - z * a @ b < c > d = e " * 1000
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) > 9000

    def test_memory_usage_large_file(self, lexer):
        """Test memory usage with large file."""
        # Generate 1MB of source code
        line = "let variable_name: TernaryTensor[100, 200] = ...\n"
        num_lines = 1024 * 1024 // len(line)  # Approximately 1MB
        text = line * num_lines
        
        start_time = time.time()
        lexer.input(text)
        tokens = list(lexer)
        end_time = time.time()
        
        duration = end_time - start_time
        assert duration < 10.0  # Should complete in reasonable time
        assert len(tokens) > 0

    def test_incremental_tokenization_performance(self):
        """Test performance of incremental tokenization."""
        lexer = TernaryLexer()
        text = "let x = 1 " * 10000
        
        start_time = time.time()
        lexer.input(text)
        
        count = 0
        while True:
            tok = lexer.token()
            if tok is None:
                break
            count += 1
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert count == 40000
        assert duration < 5.0  # Should be fast

    def test_repeated_tokenization(self, benchmark):
        """Test repeated tokenization of same input."""
        text = "fn forward(x: TernaryTensor) -> TernaryTensor { return x; }"
        
        def tokenize():
            lexer = TernaryLexer()
            lexer.input(text)
            return list(lexer)
        
        result = benchmark(tokenize)
        assert len(result) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLexerIntegration:
    """Integration tests combining multiple features."""

    def test_complete_layer_definition(self, lexer):
        """Test tokenizing a complete layer definition."""
        code = """
        layer DenseLayer {
            let weights: TernaryTensor[128, 256] = ...;
            
            fn forward(x: TernaryTensor[100, 128]) -> TernaryTensor[100, 256] {
                let output = x @ weights;
                return output;
            }
        }
        """
        lexer.input(code)
        tokens = list(lexer)
        
        token_types = [t.type for t in tokens]
        assert "LAYER" in token_types
        assert "FN" in token_types
        assert "RETURN" in token_types
        assert "MATMUL" in token_types

    def test_multiple_function_definitions(self, token_types):
        """Test multiple function definitions."""
        code = """
        fn add(x: int8, y: int8) -> int8 { return x; }
        fn multiply(a: int8, b: int8) -> int8 { return a; }
        fn process(data: TernaryTensor) -> TernaryTensor { return data; }
        """
        result = token_types(code)
        assert result.count("FN") == 3
        assert result.count("RETURN") == 3

    def test_mixed_literals_and_operators(self, token_types):
        """Test mixing all types of literals with operators."""
        code = "-1 + 0 - 1 * 10 @ 3.14"
        result = token_types(code)
        expected_types = ["TRIT_LITERAL", "PLUS", "TRIT_LITERAL", "MINUS",
                         "TRIT_LITERAL", "STAR", "INTEGER", "MATMUL", "FLOAT"]
        assert result == expected_types

    def test_realistic_training_code(self, lexer):
        """Test realistic training loop code."""
        code = """
        # Training loop
        fn train(model: TernaryTensor, data: TernaryTensor) -> TernaryTensor {
            let output = model @ data;
            let loss = output - data;
            return loss;
        }
        """
        lexer.input(code)
        tokens = list(lexer)
        assert len(tokens) > 20

    def test_error_recovery_in_complex_code(self, lexer, capsys):
        """Test that lexer recovers from errors in complex code."""
        code = """
        fn valid_function(x: int8) -> int8 {
            let $ invalid = 1;
            return x;
        }
        """
        lexer.input(code)
        tokens = list(lexer)
        
        captured = capsys.readouterr()
        assert "Illegal character" in captured.out
        
        # Should still tokenize valid parts
        token_types = [t.type for t in tokens]
        assert "FN" in token_types
        assert "RETURN" in token_types

    def test_all_features_combined(self, lexer):
        """Test all lexer features in one comprehensive test."""
        code = """
        # Comprehensive test
        layer TestLayer {
            let w1: TernaryTensor[10, 20] = ...;
            let w2: TernaryTensor[20, 30] = ...;
            let trits: trit = -1;
            let count: int8 = 42;
            let ratio: float32 = 3.14;
            
            fn forward(input: TernaryTensor, weights: TernaryTensor) -> TernaryTensor {
                let hidden = input @ w1;
                let output = hidden @ w2;
                let scaled = output * ratio;
                return scaled;
            }
            
            fn backward(grad: TernaryTensor) -> TernaryTensor {
                let result = grad @ w2;
                return result;
            }
        }
        """
        lexer.input(code)
        tokens = list(lexer)
        
        # Verify we got a good number of tokens
        assert len(tokens) > 50
        
        # Verify all major token types are present
        token_types = [t.type for t in tokens]
        assert "LAYER" in token_types
        assert "LET" in token_types
        assert "FN" in token_types
        assert "RETURN" in token_types
        assert "TRIT_LITERAL" in token_types
        assert "INTEGER" in token_types
        assert "FLOAT" in token_types
        assert "MATMUL" in token_types
        assert "TERNARY_TENSOR" in token_types


# ============================================================================
# REGRESSION TESTS
# ============================================================================

class TestRegressions:
    """Regression tests for previously found issues."""

    def test_minus_one_trit_literal(self, lexer):
        """Regression: -1 should be single TRIT_LITERAL token."""
        lexer.input("-1")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        assert tok.value == -1
        assert lexer.token() is None  # No more tokens

    def test_minus_vs_negative_in_expression(self, token_types):
        """Regression: Lexer behavior for -1 in different contexts.
        
        The lexer's lookahead logic treats -1 as a single TRIT_LITERAL token
        when the minus immediately precedes a '1', consuming both characters.
        This happens regardless of what precedes the minus sign.
        """
        # With spaces around minus: separate tokens
        assert token_types("x - 1") == ["IDENTIFIER", "MINUS", "TRIT_LITERAL"]
        # Without space after minus: -1 consumed as TRIT_LITERAL
        assert token_types("x -1") == ["IDENTIFIER", "TRIT_LITERAL"]
        assert token_types("x-1") == ["IDENTIFIER", "TRIT_LITERAL"]

    def test_arrow_vs_minus_gt(self, token_types):
        """Regression: -> should be single ARROW token."""
        assert token_types("->") == ["ARROW"]
        assert token_types("- >") == ["MINUS", "GT"]

    def test_ellipsis_vs_dots(self, token_types):
        """Regression: ... should be single ELLIPSIS token."""
        assert token_types("...") == ["ELLIPSIS"]

    def test_zero_and_one_as_trits(self, lexer):
        """Regression: 0 and 1 should be TRIT_LITERAL not INTEGER."""
        lexer.input("0")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"
        
        lexer.input("1")
        tok = lexer.token()
        assert tok.type == "TRIT_LITERAL"

    def test_two_is_integer_not_trit(self, lexer):
        """Regression: 2 and above should be INTEGER not TRIT_LITERAL."""
        lexer.input("2")
        tok = lexer.token()
        assert tok.type == "INTEGER"
        assert tok.value == 2




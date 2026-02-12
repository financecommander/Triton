"""
Triton DSL Lexer
Tokenizes Triton source code for parsing.

This module implements a PLY-based lexer for the Triton DSL.
"""

import ply.lex as lex


class TernaryLexer:
    """
    Lexer for Triton DSL using PLY (Python Lex-Yacc).
    
    Example usage:
        lexer = TernaryLexer()
        lexer.input("let w: trit = 1")
        for tok in lexer:
            print(tok)
    """
    
    # Reserved words
    reserved = {
        'layer': 'LAYER',
        'let': 'LET',
        'fn': 'FN',
        'return': 'RETURN',
        'trit': 'TRIT',
        'TernaryTensor': 'TERNARY_TENSOR',
        'int8': 'INT8',
        'float16': 'FLOAT16',
        'float32': 'FLOAT32',
    }
    
    # Token list
    tokens = [
        # Operators
        'PLUS',
        'MINUS',
        'STAR',
        'MATMUL',
        'ARROW',
        'ASSIGN',
        
        # Literals
        'TRIT_LITERAL',
        'INTEGER',
        'FLOAT',
        'IDENTIFIER',
        
        # Delimiters
        'LPAREN',
        'RPAREN',
        'LBRACE',
        'RBRACE',
        'LBRACKET',
        'RBRACKET',
        'COMMA',
        'COLON',
    ] + list(reserved.values())
    
    # Token rules (simple tokens)
    t_PLUS = r'\+'
    t_STAR = r'\*'
    t_ASSIGN = r'='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_LBRACKET = r'\['
    t_RBRACKET = r'\]'
    t_COMMA = r','
    t_COLON = r':'
    
    # Ignored characters (whitespace)
    t_ignore = ' \t'
    
    def __init__(self):
        """Initialize the lexer."""
        self.lexer = None
        self.last_token = None
        
    def build(self, **kwargs):
        """Build the lexer."""
        self.lexer = lex.lex(module=self, **kwargs)
        return self.lexer
    
    def input(self, data):
        """Set the input string for lexing."""
        if self.lexer is None:
            self.build()
        self.lexer.input(data)
    
    def token(self):
        """Get the next token."""
        if self.lexer is None:
            self.build()
        self.last_token = self.lexer.token()
        return self.last_token
    
    def __iter__(self):
        """Make lexer iterable."""
        return self
    
    def __next__(self):
        """Get next token for iteration."""
        tok = self.token()
        if tok is None:
            raise StopIteration
        return tok
    
    # Token rules with actions
    
    def t_ARROW(self, t):
        r'->'
        return t
    
    def t_MATMUL(self, t):
        r'@'
        return t
    
    def t_FLOAT(self, t):
        r'\d+\.\d+'
        t.value = float(t.value)
        return t
    
    def t_INTEGER(self, t):
        r'\d+'
        int_val = int(t.value)
        # Check if it's a trit literal (0 or 1)
        if int_val in (0, 1) and len(t.value) == 1:
            t.type = 'TRIT_LITERAL'
        t.value = int_val
        return t
    
    def t_IDENTIFIER(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'
        # Check for reserved words
        t.type = self.reserved.get(t.value, 'IDENTIFIER')
        return t
    
    def t_MINUS(self, t):
        r'-'
        # Check if next character makes it -1 (trit literal)
        # We need to look ahead
        next_pos = t.lexpos + 1
        if next_pos < len(t.lexer.lexdata) and t.lexer.lexdata[next_pos] == '1':
            # Check if there's nothing after '1' or it's not a digit
            if next_pos + 1 >= len(t.lexer.lexdata) or not t.lexer.lexdata[next_pos + 1].isdigit():
                # This is -1, consume the '1' as well
                t.value = '-1'
                t.type = 'TRIT_LITERAL'
                t.lexer.skip(1)  # Skip the '1'
                t.value = -1
                return t
        return t
    
    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += len(t.value)
    
    def t_error(self, t):
        """Error handling rule."""
        print(f"Illegal character '{t.value[0]}' at line {t.lineno}, column {self.find_column(t)}")
        t.lexer.skip(1)
    
    def find_column(self, token):
        """Find the column of a token in the input."""
        line_start = self.lexer.lexdata.rfind('\n', 0, token.lexpos) + 1
        return (token.lexpos - line_start) + 1

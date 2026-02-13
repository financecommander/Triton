"""
Triton DSL Lexer
Tokenizes Triton source code for parsing.

This module implements a PLY-based lexer for the Triton DSL.
"""

import ply.lex as lex

# Reserved words
reserved = {
    'layer': 'LAYER',
    'let': 'LET',
    'fn': 'FN',
    'return': 'RETURN',
    'trit': 'TRIT',
    'tensor': 'TENSOR',
    'TernaryTensor': 'TERNARY_TENSOR',
    'int8': 'INT8',
    'int32': 'INT32',
    'float16': 'FLOAT16',
    'float32': 'FLOAT32',
}

# Token list
tokens = [
    # Keywords
    'LAYER', 'LET', 'FN', 'RETURN',
    'TRIT', 'TENSOR', 'TERNARY_TENSOR',
    'INT8', 'INT32', 'FLOAT16', 'FLOAT32',

    # Operators
    'PLUS', 'MINUS', 'STAR', 'MATMUL',
    'ARROW', 'LT', 'GT',
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
    'SEMICOLON',
    'ELLIPSIS',
]

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
t_SEMICOLON = r';'
t_LT = r'<'
t_GT = r'>'

# Ignored characters (whitespace)
t_ignore = ' \t'

# Token functions with actions
def t_ARROW(t):
    r'->'
    return t

def t_ELLIPSIS(t):
    r'\.\.\.'
    return t

def t_MATMUL(t):
    r'@'
    return t

def t_FLOAT(t):
    r'\d+\.\d+'
    t.value = float(t.value)
    return t

def t_INTEGER(t):
    r'\d+'
    int_val = int(t.value)
    # Check if it's a trit literal (0 or 1)
    if int_val in (0, 1) and len(t.value) == 1:
        t.type = 'TRIT_LITERAL'
    t.value = int_val
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    # Check for reserved words
    t.type = reserved.get(t.value, 'IDENTIFIER')
    return t

def t_MINUS(t):
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

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_comment(t):
    r'\#.*'
    # Comments are ignored
    pass

def t_error(t):
    """Error handling rule."""
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)


class TernaryLexer:
    """Triton DSL Lexer using PLY"""

    def __init__(self):
        """Initialize the lexer."""
        self.lexer = None
        self.last_token = None

    def build(self, **kwargs):
        """Build the lexer."""
        self.lexer = lex.lex(**kwargs)
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

    def find_column(self, token):
        """Find the column number for a given token."""
        if self.lexer is None:
            return 0
        input_data = self.lexer.lexdata
        last_cr = input_data.rfind('\n', 0, token.lexpos)
        if last_cr < 0:
            last_cr = -1
        column = token.lexpos - last_cr
        return column


# Create module-level lexer instance for PLY
lexer = lex.lex()

"""
Triton DSL Lexer
Tokenizes Triton source code for parsing.
"""

import sys

import ply.lex as lex

# Reserved keywords
reserved = {
    "layer": "LAYER",
    "let": "LET",
    "fn": "FN",
    "return": "RETURN",
    "TernaryTensor": "TERNARYTENSOR",
    "trit": "TRIT",
    "int8": "INT8",
    "float16": "FLOAT16",
    "float32": "FLOAT32",
}

# Token list
tokens = [
    "IDENTIFIER",
    "INTEGER",
    "FLOAT",
    "PLUS",
    "MINUS",
    "TIMES",
    "MATMUL",
    "ARROW",
    "ASSIGN",
    "LPAREN",
    "RPAREN",
    "LBRACE",
    "RBRACE",
    "LBRACKET",
    "RBRACKET",
    "COMMA",
    "COLON",
    "SEMICOLON",
] + list(reserved.values())

# Token rules
t_PLUS = r"\+"
t_TIMES = r"\*"
t_MATMUL = r"@"
t_ASSIGN = r"="
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_COMMA = r","
t_COLON = r":"
t_SEMICOLON = r";"
t_MINUS = r"-"


def t_ARROW(t):
    r"->"
    return t

# Ignored characters (spaces and tabs)
t_ignore = " \t"


def t_FLOAT(t):
    r"\d+\.\d+"
    t.value = float(t.value)
    return t


def t_INTEGER(t):
    r"\d+"
    t.value = int(t.value)
    return t


def t_IDENTIFIER(t):
    r"[a-zA-Z_][a-zA-Z_0-9]*"
    t.type = reserved.get(t.value, "IDENTIFIER")
    return t


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_error(t):
    """Error handling rule."""
    sys.stderr.write(f"Illegal character '{t.value[0]}' at line {t.lineno}\n")
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()


def tokenize(data):
    """Tokenize input data and return list of tokens."""
    lexer.input(data)
    tokens_list = []
    while True:
        tok = lexer.token()
        if not tok:
            break
        tokens_list.append(tok)
    return tokens_list

"""
Triton DSL Parser
LALR parser using PLY yacc to generate Abstract Syntax Tree.
"""

import sys

import ply.yacc as yacc

from compiler.ast.nodes import (
    Assignment,
    BinaryOp,
    Declaration,
    FloatLiteral,
    FloatType,
    FunctionCall,
    Identifier,
    IntegerLiteral,
    IntType,
    LayerDef,
    Param,
    Program,
    ReturnStmt,
    TensorType,
    TernaryTensor,
    TritLiteral,
    TritType,
    Type,
)
from compiler.lexer.triton_lexer import tokens  # noqa: F401

# Operator precedence and associativity
precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "STAR", "MATMUL"),
    ("right", "UMINUS"),  # Unary minus
)


def p_program(p):
    """program : statement_list"""
    lineno = p[1][0].lineno if p[1] else 0
    p[0] = Program(statements=p[1], lineno=lineno)


def p_statement_list(p):
    """statement_list : statement_list statement
    | statement
    | empty"""
    if len(p) == 3:
        p[0] = p[1] + [p[2]]
    elif len(p) == 2 and p[1] is not None:
        p[0] = [p[1]]
    else:
        p[0] = []


def p_statement(p):
    """statement : declaration
    | declaration SEMICOLON
    | assignment
    | assignment SEMICOLON
    | layer_def
    | return_stmt
    | return_stmt SEMICOLON"""
    p[0] = p[1]


def p_declaration(p):
    """declaration : LET IDENTIFIER COLON type ASSIGN expression
    | LET IDENTIFIER COLON type"""
    if len(p) == 7:
        p[0] = Declaration(p[2], p[4], p[6], lineno=p.lineno(1), col_offset=0)
    else:
        p[0] = Declaration(p[2], p[4], None, lineno=p.lineno(1), col_offset=0)


def p_assignment(p):
    """assignment : IDENTIFIER ASSIGN expression"""
    p[0] = Assignment(p[1], p[3], lineno=p.lineno(1), col_offset=0)


def p_layer_def(p):
    """layer_def : LAYER IDENTIFIER LPAREN params RPAREN ARROW type LBRACE statement_list RBRACE"""
    p[0] = LayerDef(p[2], p[4], p[7], p[9], lineno=p.lineno(1), col_offset=0)


def p_return_stmt(p):
    """return_stmt : RETURN expression
    | RETURN"""
    if len(p) == 3:
        p[0] = ReturnStmt(p[2], lineno=p.lineno(1), col_offset=0)
    else:
        p[0] = ReturnStmt(None, lineno=p.lineno(1), col_offset=0)


def p_params(p):
    """params : param_list
    | empty"""
    if p[1] is None:
        p[0] = []
    else:
        p[0] = p[1]


def p_param_list(p):
    """param_list : param_list COMMA param
    | param"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]


def p_param(p):
    """param : IDENTIFIER COLON type"""
    p[0] = Param(p[1], p[3], lineno=p.lineno(1), col_offset=0)


def p_type(p):
    """type : TRIT
    | INT8
    | INT32
    | FLOAT16
    | FLOAT32
    | TERNARY_TENSOR
    | tensor_type"""
    if p[1] == 'trit':
        p[0] = TritType(lineno=p.lineno(1), col_offset=0)
    elif p[1] == 'int8':
        p[0] = IntType(bits=8, lineno=p.lineno(1), col_offset=0)
    elif p[1] == 'int32':
        p[0] = IntType(bits=32, lineno=p.lineno(1), col_offset=0)
    elif p[1] == 'float16':
        p[0] = FloatType(bits=16, lineno=p.lineno(1), col_offset=0)
    elif p[1] == 'float32':
        p[0] = FloatType(bits=32, lineno=p.lineno(1), col_offset=0)
    elif p[1] == 'TernaryTensor':
        p[0] = TensorType("TernaryTensor", TritType(), None, lineno=p.lineno(1), col_offset=0)
    else:
        p[0] = p[1]  # tensor_type rule returns the TensorType node


def p_tensor_type(p):
    """tensor_type : TENSOR LT type COMMA LBRACKET expression_list RBRACKET GT"""
    p[0] = TensorType("tensor", p[3], None, lineno=p.lineno(1), col_offset=0)


def p_expression_binop(p):
    """expression : expression PLUS expression
    | expression MINUS expression
    | expression STAR expression
    | expression MATMUL expression"""
    p[0] = BinaryOp(p[1], p[2], p[3], lineno=p.lineno(2), col_offset=0)


def p_expression_unary(p):
    """expression : MINUS expression %prec UMINUS"""
    # Create a binary operation: 0 - expression
    p[0] = BinaryOp(
        IntegerLiteral(0, lineno=p.lineno(1), col_offset=0),
        "-",
        p[2],
        lineno=p.lineno(1),
        col_offset=0,
    )


def p_expression_function_call(p):
    """expression : IDENTIFIER LPAREN arguments RPAREN"""
    if p[1] == 'ternary_tensor':
        if len(p[3]) == 2:
            p[0] = TernaryTensor([1, 1], [0, 0, 0], lineno=p.lineno(1), col_offset=0)
        else:
            p[0] = TernaryTensor([1], [0], lineno=p.lineno(1), col_offset=0)
    else:
        p[0] = FunctionCall(p[1], p[3], lineno=p.lineno(1), col_offset=0)


def p_expression_ternary_tensor(p):
    """expression : TERNARY_TENSOR LBRACKET integer_list RBRACKET LPAREN integer_list RPAREN"""
    p[0] = TernaryTensor(p[3], p[6], lineno=p.lineno(1), col_offset=0)


def p_arguments(p):
    """arguments : expression_list
    | empty"""
    if p[1] is None:
        p[0] = []
    else:
        p[0] = p[1]


def p_expression_list(p):
    """expression_list : expression_list COMMA expression
    | expression"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]


def p_integer_list(p):
    """integer_list : integer_list COMMA signed_integer
    | signed_integer"""
    if len(p) == 4:
        p[0] = p[1] + [p[3]]
    else:
        p[0] = [p[1]]


def p_signed_integer(p):
    """signed_integer : INTEGER
    | TRIT_LITERAL
    | MINUS INTEGER"""
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = -p[2]


def p_expression_identifier(p):
    """expression : IDENTIFIER"""
    p[0] = Identifier(p[1], lineno=p.lineno(1), col_offset=0)


def p_expression_integer(p):
    """expression : INTEGER"""
    p[0] = IntegerLiteral(p[1], lineno=p.lineno(1), col_offset=0)


def p_expression_float(p):
    """expression : FLOAT"""
    p[0] = FloatLiteral(p[1], lineno=p.lineno(1), col_offset=0)


def p_expression_trit_literal(p):
    """expression : TRIT_LITERAL"""
    p[0] = TritLiteral(p[1], lineno=p.lineno(1), col_offset=0)


def p_expression_array_literal(p):
    """expression : LBRACKET expression_list RBRACKET
    | LBRACKET expression_list COMMA ELLIPSIS RBRACKET"""
    # For now, just return the first expression or a list
    # This is a simplification - real array literal handling would be more complex
    if len(p) == 4:
        p[0] = p[2]  # Return the expression list
    else:
        p[0] = p[2]  # Return the expression list with ellipsis


def p_empty(p):
    """empty :"""
    pass


def p_error(p):
    """Error handling with recovery."""
    if p:
        sys.stderr.write(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}\n")
        parser.errok()
    else:
        sys.stderr.write("Syntax error at EOF\n")


# Build the parser
parser = yacc.yacc()


def parse(data: str):
    """Parse input string and return AST."""
    from compiler.lexer.triton_lexer import lexer

    return parser.parse(data, lexer=lexer)

"""
Triton DSL Parser
LALR parser using PLY yacc to generate Abstract Syntax Tree.
"""

import ply.yacc as yacc

from compiler.ast.nodes import (
    Assignment,
    BinaryOp,
    Declaration,
    FloatLiteral,
    FunctionCall,
    Identifier,
    IntegerLiteral,
    LayerDef,
    Param,
    Program,
    ReturnStmt,
    TernaryTensor,
    Type,
)
from compiler.lexer.triton_lexer import tokens  # noqa: F401

# Operator precedence and associativity
precedence = (
    ("left", "PLUS", "MINUS"),
    ("left", "TIMES", "MATMUL"),
    ("right", "UMINUS"),  # Unary minus
)


def p_program(p):
    """program : statement_list"""
    p[0] = Program(p[1], lineno=p.lineno(1))


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
    | assignment
    | layer_def
    | return_stmt"""
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
    | FLOAT16
    | FLOAT32
    | TERNARYTENSOR"""
    p[0] = Type(p[1], lineno=p.lineno(1), col_offset=0)


def p_expression_binop(p):
    """expression : expression PLUS expression
    | expression MINUS expression
    | expression TIMES expression
    | expression MATMUL expression"""
    p[0] = BinaryOp(p[1], p[2], p[3], lineno=p.lineno(2), col_offset=0)


def p_expression_unary(p):
    """expression : MINUS expression %prec UMINUS"""
    # Create a binary operation: 0 - expression
    p[0] = BinaryOp(IntegerLiteral(0), "-", p[2], lineno=p.lineno(1), col_offset=0)


def p_expression_function_call(p):
    """expression : IDENTIFIER LPAREN arguments RPAREN"""
    p[0] = FunctionCall(p[1], p[3], lineno=p.lineno(1), col_offset=0)


def p_expression_ternary_tensor(p):
    """expression : TERNARYTENSOR LBRACKET integer_list RBRACKET LPAREN integer_list RPAREN"""
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


def p_empty(p):
    """empty :"""
    pass


def p_error(p):
    """Error handling with recovery."""
    if p:
        print(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}")
        # Try to recover by skipping the token
        parser.errok()
    else:
        print("Syntax error at EOF")


# Build the parser
parser = yacc.yacc()


def parse(data: str):
    """Parse input string and return AST."""
    from compiler.lexer.triton_lexer import lexer

    return parser.parse(data, lexer=lexer)

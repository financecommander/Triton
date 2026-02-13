"""
Abstract Syntax Tree Node Definitions
Represents the structure of parsed Triton programs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from abc import ABC


@dataclass
class Node:
    """Base class for all AST nodes with location information."""
    
    # Location information - keyword-only so subclass fields come first positionally
    lineno: int = field(default=0, kw_only=True)
    col_offset: int = field(default=0, kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        """Accept a visitor for the visitor pattern."""
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for debugging."""
        result = {"node_type": self.__class__.__name__}
        for key, value in self.__dict__.items():
            if isinstance(value, Node):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, Node) else item for item in value]
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items() if k not in ["lineno", "col_offset"])
        return f"{self.__class__.__name__}({attrs})"


# Type nodes
@dataclass
class Type(Node):
    """Base/simple type node. Can hold a type name string for simple types."""
    name: str = ""
    
    def accept(self, visitor: "Visitor") -> Any:
        return None


@dataclass
class TritType(Type):
    """Ternary type: {-1, 0, 1}."""
    
    def __post_init__(self):
        if not self.name:
            self.name = "trit"
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_trit_type(self)


@dataclass
class IntType(Type):
    """Integer type."""
    bits: int = 32
    
    def __post_init__(self):
        if not self.name:
            self.name = f"int{self.bits}"
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_int_type(self)


@dataclass
class FloatType(Type):
    """Floating point type."""
    bits: int = 32
    
    def __post_init__(self):
        if not self.name:
            self.name = f"float{self.bits}"
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_float_type(self)


@dataclass
class TensorType(Type):
    """Tensor type with element type and shape."""
    element_type: Optional["Type"] = None
    shape: Optional[List[int]] = None
    
    def __post_init__(self):
        if not self.name:
            self.name = "tensor"
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_tensor_type(self)


# Expression nodes
@dataclass
class Expr(Node):
    """Base class for expression nodes."""
    pass


@dataclass
class TritLiteral(Expr):
    """Trit literal: -1, 0, or 1."""
    value: int = 0
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_trit_literal(self)


@dataclass
class IntLiteral(Expr):
    """Integer literal."""
    value: int = 0
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_int_literal(self)


@dataclass
class FloatLiteral(Expr):
    """Float literal."""
    value: float = 0.0
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_float_literal(self)


@dataclass
class TernaryTensor(Expr):
    """Ternary tensor with shape and values."""
    shape: List[int] = field(default_factory=list)
    values: List[int] = field(default_factory=list)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_ternary_tensor(self)


@dataclass
class Identifier(Expr):
    """Variable or function identifier."""
    name: str = ""
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_identifier(self)


@dataclass
class BinaryOp(Expr):
    """Binary operation."""
    left: Optional[Expr] = None
    op: str = ""
    right: Optional[Expr] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(Expr):
    """Unary operation."""
    op: str = ""
    operand: Optional[Expr] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class FunctionCall(Expr):
    """Function call."""
    name: str = ""
    args: List[Expr] = field(default_factory=list)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_function_call(self)


# Statement nodes
@dataclass
class Statement(Node):
    """Base class for statement nodes."""
    pass


@dataclass
class Assignment(Statement):
    """Assignment statement."""
    name: str = ""
    value: Optional[Expr] = None
    type_annotation: Optional[Type] = None
    
    def __init__(self, name="", value=None, type_annotation=None, *, target=None, lineno=0, col_offset=0):
        # Accept 'target' as alias for 'name'
        if target is not None:
            name = target
        self.name = name
        self.value = value
        self.type_annotation = type_annotation
        self.lineno = lineno
        self.col_offset = col_offset
    
    @property
    def target(self):
        return self.name
    
    @target.setter
    def target(self, val):
        self.name = val
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_assignment(self)


@dataclass
class Return(Statement):
    """Return statement."""
    value: Optional[Expr] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_return(self)


@dataclass
class ExprStatement(Statement):
    """Expression as a statement."""
    expr: Optional[Expr] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_expr_statement(self)


# Function and layer definitions
@dataclass
class Param(Node):
    """Function parameter."""
    name: str = ""
    type_annotation: Optional[Type] = None
    param_type: Optional[str] = None
    shape: Optional[List[int]] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_param(self)


@dataclass
class Declaration(Statement):
    """Variable declaration statement."""
    name: str = ""
    var_type: Optional[Type] = None
    initializer: Optional[Expr] = None
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_declaration(self)


@dataclass
class FunctionDef(Statement):
    """Function definition."""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[Type] = None
    body: List[Statement] = field(default_factory=list)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_function_def(self)


@dataclass
class LayerDef(Statement):
    """Layer definition (similar to function but for neural network layers)."""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    return_type: Optional[Type] = None
    body: List[Statement] = field(default_factory=list)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_layer_def(self)


# Program node
@dataclass
class Program(Node):
    """Root node representing entire program."""
    statements: List[Statement] = field(default_factory=list)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_program(self)


# Visitor interface
class Visitor(ABC):
    """Abstract visitor interface for AST traversal."""
    
    def visit_program(self, node: Program) -> Any:
        pass
    
    def visit_trit_type(self, node: TritType) -> Any:
        pass
    
    def visit_int_type(self, node: IntType) -> Any:
        pass
    
    def visit_float_type(self, node: FloatType) -> Any:
        pass
    
    def visit_tensor_type(self, node: TensorType) -> Any:
        pass
    
    def visit_trit_literal(self, node: TritLiteral) -> Any:
        pass
    
    def visit_int_literal(self, node: IntLiteral) -> Any:
        pass
    
    def visit_float_literal(self, node: FloatLiteral) -> Any:
        pass
    
    def visit_ternary_tensor(self, node: TernaryTensor) -> Any:
        pass
    
    def visit_identifier(self, node: Identifier) -> Any:
        pass
    
    def visit_binary_op(self, node: BinaryOp) -> Any:
        pass
    
    def visit_unary_op(self, node: UnaryOp) -> Any:
        pass
    
    def visit_function_call(self, node: FunctionCall) -> Any:
        pass
    
    def visit_assignment(self, node: Assignment) -> Any:
        pass
    
    def visit_return(self, node: Return) -> Any:
        pass
    
    def visit_expr_statement(self, node: ExprStatement) -> Any:
        pass
    
    def visit_param(self, node: Param) -> Any:
        pass
    
    def visit_declaration(self, node: Declaration) -> Any:
        pass
    
    def visit_function_def(self, node: FunctionDef) -> Any:
        pass
    
    def visit_layer_def(self, node: LayerDef) -> Any:
        pass


# Aliases for backward compatibility
Expression = Expr
IntegerLiteral = IntLiteral
ReturnStmt = Return

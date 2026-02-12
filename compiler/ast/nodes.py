"""
Abstract Syntax Tree Node Definitions
Represents the structure of parsed Triton programs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from abc import ABC, abstractmethod


@dataclass
class Node(ABC):
    """Base class for all AST nodes with location information."""
    
    @abstractmethod
    def accept(self, visitor: "Visitor") -> Any:
        """Accept a visitor for the visitor pattern."""
        pass
    
    # Location information for error reporting.
    # Defaults to 0 to allow easier test node creation and avoid None checks.
    # Parser should always set these to actual values from source.
    lineno: int = 0
    col_offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for debugging."""
        result = {"type": self.__class__.__name__}
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
    """Base class for type nodes."""
    pass


@dataclass
class TritType(Type):
    """Ternary type: {-1, 0, 1}."""
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_trit_type(self)


@dataclass
class IntType(Type):
    """Integer type."""
    bits: int = 32  # int8, int32, etc.
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_int_type(self)


@dataclass
class FloatType(Type):
    """Floating point type."""
    bits: int = 32  # float16, float32
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_float_type(self)


@dataclass
class TensorType(Type):
    """Tensor type with element type and shape."""
    element_type: Type = field(kw_only=True)
    shape: Optional[List[int]] = field(default=None, kw_only=True)
    
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
    value: int = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_trit_literal(self)


@dataclass
class IntLiteral(Expr):
    """Integer literal."""
    value: int = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_int_literal(self)


@dataclass
class FloatLiteral(Expr):
    """Float literal."""
    value: float = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_float_literal(self)


@dataclass
class TernaryTensor(Expr):
    """Ternary tensor with shape and values."""
    shape: List[int] = field(kw_only=True)
    values: List[int] = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_ternary_tensor(self)


@dataclass
class Identifier(Expr):
    """Variable or function identifier."""
    name: str = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_identifier(self)


@dataclass
class BinaryOp(Expr):
    """Binary operation."""
    left: Expr = field(kw_only=True)
    op: str = field(kw_only=True)  # +, -, *, /, @, etc.
    right: Expr = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(Expr):
    """Unary operation."""
    op: str = field(kw_only=True)  # -, +, not
    operand: Expr = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class FunctionCall(Expr):
    """Function call."""
    name: str = field(kw_only=True)
    args: List[Expr] = field(kw_only=True)
    
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
    target: str = field(kw_only=True)
    value: Expr = field(kw_only=True)
    type_annotation: Optional[Type] = field(default=None, kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_assignment(self)


@dataclass
class Return(Statement):
    """Return statement."""
    value: Optional[Expr] = field(default=None, kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_return(self)


@dataclass
class ExprStatement(Statement):
    """Expression as a statement."""
    expr: Expr = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_expr_statement(self)


# Function and layer definitions
@dataclass
class Param(Node):
    """Function parameter."""
    name: str = field(kw_only=True)
    type_annotation: Type = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_param(self)


@dataclass
class FunctionDef(Statement):
    """Function definition."""
    name: str = field(kw_only=True)
    params: List[Param] = field(kw_only=True)
    return_type: Optional[Type] = field(kw_only=True)
    body: List[Statement] = field(kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        return visitor.visit_function_def(self)


@dataclass
class LayerDef(Statement):
    """Layer definition (similar to function but for neural network layers)."""
    name: str = field(kw_only=True)
    params: List[Param] = field(kw_only=True)
    body: List[Statement] = field(kw_only=True)
    
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
    
    def visit_function_def(self, node: FunctionDef) -> Any:
        pass
    
    def visit_layer_def(self, node: LayerDef) -> Any:
        pass

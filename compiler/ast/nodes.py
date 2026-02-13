"""
Abstract Syntax Tree Node Definitions
Represents the structure of parsed Triton programs.
"""

from typing import Any, Dict, List, Optional


class Node:
    """Base class for all AST nodes."""

    def __init__(self, lineno: int = 0, col_offset: int = 0):
        self.lineno = lineno
        self.col_offset = col_offset

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_type": self.__class__.__name__,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
        }


class Program(Node):
    """Root node representing a complete program."""

    def __init__(self, statements: List["Statement"], lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.statements = statements

    def __repr__(self) -> str:
        return f"Program(statements={self.statements})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["statements"] = [stmt.to_dict() for stmt in self.statements]
        return result


class Statement(Node):
    """Base class for all statements."""

    pass


class Expression(Node):
    """Base class for all expressions."""

    pass


class Type(Node):
    """Represents a type annotation."""

    def __init__(self, name: str, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.name = name

    def __repr__(self) -> str:
        return f"Type(name='{self.name}')"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        return result


class Param(Node):
    """Represents a function/layer parameter."""

    def __init__(self, name: str, param_type: Type, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.name = name
        self.param_type = param_type

    def __repr__(self) -> str:
        return f"Param(name='{self.name}', type={self.param_type})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["param_type"] = self.param_type.to_dict()
        return result


class Declaration(Statement):
    """Represents a variable declaration: let x: type = expr"""

    def __init__(
        self,
        name: str,
        var_type: Type,
        initializer: Optional[Expression] = None,
        lineno: int = 0,
        col_offset: int = 0,
    ):
        super().__init__(lineno, col_offset)
        self.name = name
        self.var_type = var_type
        self.initializer = initializer

    def __repr__(self) -> str:
        return f"Declaration(name='{self.name}', type={self.var_type}, init={self.initializer})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["var_type"] = self.var_type.to_dict()
        result["initializer"] = self.initializer.to_dict() if self.initializer else None
        return result


class Assignment(Statement):
    """Represents an assignment statement: x = expr"""

    def __init__(self, name: str, value: Expression, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f"Assignment(name='{self.name}', value={self.value})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["value"] = self.value.to_dict()
        return result


class LayerDef(Statement):
    """Represents a layer definition: layer name(params) -> type { body }"""

    def __init__(
        self,
        name: str,
        params: List[Param],
        return_type: Type,
        body: List[Statement],
        lineno: int = 0,
        col_offset: int = 0,
    ):
        super().__init__(lineno, col_offset)
        self.name = name
        self.params = params
        self.return_type = return_type
        self.body = body

    def __repr__(self) -> str:
        return f"LayerDef(name='{self.name}', params={self.params}, return_type={self.return_type})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["params"] = [p.to_dict() for p in self.params]
        result["return_type"] = self.return_type.to_dict()
        result["body"] = [stmt.to_dict() for stmt in self.body]
        return result


class ReturnStmt(Statement):
    """Represents a return statement: return expr"""

    def __init__(self, value: Optional[Expression] = None, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.value = value

    def __repr__(self) -> str:
        return f"ReturnStmt(value={self.value})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["value"] = self.value.to_dict() if self.value else None
        return result


class BinaryOp(Expression):
    """Represents a binary operation: left op right"""

    def __init__(
        self, left: Expression, op: str, right: Expression, lineno: int = 0, col_offset: int = 0
    ):
        super().__init__(lineno, col_offset)
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryOp(left={self.left}, op='{self.op}', right={self.right})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["left"] = self.left.to_dict()
        result["op"] = self.op
        result["right"] = self.right.to_dict()
        return result


class FunctionCall(Expression):
    """Represents a function/layer call: name(args)"""

    def __init__(self, name: str, args: List[Expression], lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.name = name
        self.args = args

    def __repr__(self) -> str:
        return f"FunctionCall(name='{self.name}', args={self.args})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        result["args"] = [arg.to_dict() for arg in self.args]
        return result


class TernaryTensor(Expression):
    """Represents a ternary tensor literal: TernaryTensor[shape](values)"""

    def __init__(self, shape: List[int], values: List[int], lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.shape = shape
        self.values = values

    def __repr__(self) -> str:
        return f"TernaryTensor(shape={self.shape}, values={self.values})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["shape"] = self.shape
        result["values"] = self.values
        return result


class Identifier(Expression):
    """Represents an identifier (variable reference)."""

    def __init__(self, name: str, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.name = name

    def __repr__(self) -> str:
        return f"Identifier(name='{self.name}')"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["name"] = self.name
        return result


class IntegerLiteral(Expression):
    """Represents an integer literal."""

    def __init__(self, value: int, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.value = value

    def __repr__(self) -> str:
        return f"IntegerLiteral(value={self.value})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["value"] = self.value
        return result


class FloatLiteral(Expression):
    """Represents a float literal."""

    def __init__(self, value: float, lineno: int = 0, col_offset: int = 0):
        super().__init__(lineno, col_offset)
        self.value = value

    def __repr__(self) -> str:
        return f"FloatLiteral(value={self.value})"

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["value"] = self.value
        return result

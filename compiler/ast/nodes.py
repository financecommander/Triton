"""
Abstract Syntax Tree Node Definitions
Represents the structure of parsed Triton programs.
"""

from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field


@dataclass
class Node:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        result = {"type": self.__class__.__name__}
        for key, value in self.__dict__.items():
            if isinstance(value, Node):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, Node) else item for item in value]
            else:
                result[key] = value
        return result


@dataclass
class Param(Node):
    """Parameter definition."""
    name: str = ""
    param_type: str = ""
    shape: Optional[List[int]] = None


@dataclass
class TernaryTensor(Node):
    """Ternary tensor literal with shape and values."""
    shape: List[int] = field(default_factory=list)
    values: List[int] = field(default_factory=list)


@dataclass
class BinaryOp(Node):
    """Binary operation expression."""
    left: Optional['Node'] = None
    op: str = ""
    right: Optional['Node'] = None


@dataclass
class FunctionCall(Node):
    """Function call expression."""
    name: str = ""
    args: List['Node'] = field(default_factory=list)


@dataclass
class Statement(Node):
    """Base class for statements."""
    pass


@dataclass
class LayerDef(Statement):
    """Layer definition with parameters and body."""
    name: str = ""
    params: List[Param] = field(default_factory=list)
    body: List[Statement] = field(default_factory=list)

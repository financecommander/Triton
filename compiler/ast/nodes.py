"""
Abstract Syntax Tree Node Definitions
Represents the structure of parsed Triton programs.

TODO: Define AST node classes
- Base Node class with common attributes
- TernaryTensor, LayerDef, BinaryOp, FunctionCall
- Visitor pattern for traversal
- JSON serialization for debugging
"""

# GitHub Copilot: Define AST node classes with these requirements:
# - Base Node class with line/column info
# - TernaryTensor(shape: List[int], values: List[int])
# - LayerDef(name: str, params: List[Param], body: List[Statement])
# - BinaryOp(left: Expr, op: str, right: Expr)
# - FunctionCall(name: str, args: List[Expr])
# Include __repr__ and to_dict() methods for all nodes

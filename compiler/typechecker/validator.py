"""
Type Checker for Triton DSL
Validates AST nodes for type correctness.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from compiler.ast.nodes import (
    Node, Visitor, Program, Type, TritType, IntType, FloatType, TensorType,
    Expr, TritLiteral, IntLiteral, FloatLiteral, TernaryTensor, Identifier,
    BinaryOp, UnaryOp, FunctionCall,
    Statement, Assignment, Return, ExprStatement, Param, FunctionDef, LayerDef
)


@dataclass
class TypeError:
    """Represents a type error with location information."""
    message: str
    lineno: int = 0
    col_offset: int = 0
    
    def __str__(self) -> str:
        if self.lineno > 0:
            return f"Line {self.lineno}, Col {self.col_offset}: {self.message}"
        return self.message


class TypeChecker(Visitor):
    """
    Type checker that validates AST nodes using the visitor pattern.
    
    Type validation rules:
    - trit variables must hold values {-1, 0, 1}
    - TernaryTensor shape must match value count
    - Binary operations require compatible types
    - Function signatures match declarations
    - Matrix multiplication dimensions are compatible
    """
    
    def __init__(self) -> None:
        self.errors: List[TypeError] = []
        self.symbol_table: Dict[str, Type] = {}
        self.function_table: Dict[str, Tuple[List[Type], Optional[Type]]] = {}
        self.current_function_return_type: Optional[Type] = None
    
    def validate(self, ast: Node) -> List[TypeError]:
        """
        Validate an AST node and return list of errors.
        
        Args:
            ast: The AST node to validate
            
        Returns:
            List of type errors found (empty if validation succeeds)
        """
        self.errors = []
        self.symbol_table = {}
        self.function_table = {}
        self.current_function_return_type = None
        
        ast.accept(self)
        return self.errors
    
    def add_error(self, message: str, node: Node) -> None:
        """Add a type error to the error list."""
        self.errors.append(TypeError(message, node.lineno, node.col_offset))
    
    def types_compatible(self, type1: Type, type2: Type) -> bool:
        """Check if two types are compatible for operations."""
        # Same type is always compatible
        if type(type1) == type(type2):
            if isinstance(type1, IntType) and isinstance(type2, IntType):
                return type1.bits == type2.bits
            if isinstance(type1, FloatType) and isinstance(type2, FloatType):
                return type1.bits == type2.bits
            if isinstance(type1, TensorType) and isinstance(type2, TensorType):
                return (self.types_compatible(type1.element_type, type2.element_type) and
                        type1.shape == type2.shape)
            return True
        
        # Trit and Int are compatible for some operations
        if isinstance(type1, (TritType, IntType)) and isinstance(type2, (TritType, IntType)):
            return True
        
        return False
    
    def infer_type(self, node: Expr) -> Optional[Type]:
        """Infer the type of an expression."""
        # Use the visitor pattern to infer type and validate
        return node.accept(self)
    
    def visit_program(self, node: Program) -> Any:
        """Visit program node."""
        for statement in node.statements:
            statement.accept(self)
        return None
    
    def visit_trit_type(self, node: TritType) -> Any:
        """Visit trit type node."""
        return node
    
    def visit_int_type(self, node: IntType) -> Any:
        """Visit int type node."""
        return node
    
    def visit_float_type(self, node: FloatType) -> Any:
        """Visit float type node."""
        return node
    
    def visit_tensor_type(self, node: TensorType) -> Any:
        """Visit tensor type node."""
        return node
    
    def visit_trit_literal(self, node: TritLiteral) -> Any:
        """Visit trit literal and validate value is in {-1, 0, 1}."""
        if node.value not in {-1, 0, 1}:
            self.add_error(
                f"Trit literal must be -1, 0, or 1, got {node.value}",
                node
            )
        return TritType()
    
    def visit_int_literal(self, node: IntLiteral) -> Any:
        """Visit integer literal."""
        return IntType()
    
    def visit_float_literal(self, node: FloatLiteral) -> Any:
        """Visit float literal."""
        return FloatType()
    
    def visit_ternary_tensor(self, node: TernaryTensor) -> Any:
        """Visit ternary tensor and validate shape matches value count."""
        # Validate all values are in {-1, 0, 1}
        for i, value in enumerate(node.values):
            if value not in {-1, 0, 1}:
                self.add_error(
                    f"TernaryTensor value at index {i} must be -1, 0, or 1, got {value}",
                    node
                )
        
        # Validate shape matches value count
        if node.shape:
            expected_count = 1
            for dim in node.shape:
                expected_count *= dim
            
            if len(node.values) != expected_count:
                self.add_error(
                    f"TernaryTensor shape {node.shape} expects {expected_count} values, "
                    f"got {len(node.values)}",
                    node
                )
        
        return TensorType(element_type=TritType(), shape=node.shape)
    
    def visit_identifier(self, node: Identifier) -> Any:
        """Visit identifier and check it's defined."""
        if node.name not in self.symbol_table:
            self.add_error(f"Undefined variable '{node.name}'", node)
            return None
        return self.symbol_table[node.name]
    
    def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
        """Visit binary operation and validate type compatibility."""
        left_type = self.infer_type(node.left)
        right_type = self.infer_type(node.right)
        
        if left_type is None or right_type is None:
            return None
        
        # Matrix multiplication (@) requires special dimension checking
        if node.op == '@':
            if not isinstance(left_type, TensorType) or not isinstance(right_type, TensorType):
                self.add_error(
                    f"Matrix multiplication requires tensor operands, got {type(left_type).__name__} "
                    f"and {type(right_type).__name__}",
                    node
                )
                return None
            
            # Check dimension compatibility for matrix multiplication
            if left_type.shape and right_type.shape:
                if len(left_type.shape) < 2 or len(right_type.shape) < 2:
                    self.add_error(
                        "Matrix multiplication requires at least 2D tensors",
                        node
                    )
                    return None
                
                # For 2D: (m, n) @ (n, p) -> (m, p)
                if left_type.shape[-1] != right_type.shape[-2]:
                    self.add_error(
                        f"Matrix multiplication dimension mismatch: "
                        f"shape {left_type.shape} cannot multiply with shape {right_type.shape}. "
                        f"Inner dimensions must match ({left_type.shape[-1]} != {right_type.shape[-2]})",
                        node
                    )
                    return None
                
                # Result shape: combine all but last dim of left with all but first dim of right
                result_shape = left_type.shape[:-1] + [right_type.shape[-1]]
                return TensorType(element_type=left_type.element_type, shape=result_shape)
            
            return TensorType(element_type=left_type.element_type)
        
        # Other operations require compatible types
        if not self.types_compatible(left_type, right_type):
            self.add_error(
                f"Binary operation '{node.op}' requires compatible types, "
                f"got {type(left_type).__name__} and {type(right_type).__name__}",
                node
            )
            return None
        
        # Arithmetic operations preserve the type
        if node.op in {'+', '-', '*', '/'}:
            # If both are tensors, return tensor type
            if isinstance(left_type, TensorType):
                return left_type
            # Otherwise return the left type
            return left_type
        
        # Comparison operations return bool (represented as IntType for now)
        if node.op in {'==', '!=', '<', '>', '<=', '>='}:
            return IntType(bits=8)
        
        return left_type
    
    def visit_unary_op(self, node: UnaryOp) -> Optional[Type]:
        """Visit unary operation."""
        operand_type = self.infer_type(node.operand)
        
        if operand_type is None:
            return None
        
        # Unary operations preserve the type
        return operand_type
    
    def visit_function_call(self, node: FunctionCall) -> Optional[Type]:
        """Visit function call and validate signature."""
        if node.name not in self.function_table:
            self.add_error(f"Undefined function '{node.name}'", node)
            return None
        
        param_types, return_type = self.function_table[node.name]
        
        # Check argument count
        if len(node.args) != len(param_types):
            self.add_error(
                f"Function '{node.name}' expects {len(param_types)} arguments, "
                f"got {len(node.args)}",
                node
            )
            return return_type
        
        # Check argument types
        for i, (arg, expected_type) in enumerate(zip(node.args, param_types)):
            arg_type = self.infer_type(arg)
            if arg_type and not self.types_compatible(arg_type, expected_type):
                self.add_error(
                    f"Function '{node.name}' argument {i} expects {type(expected_type).__name__}, "
                    f"got {type(arg_type).__name__}",
                    node
                )
        
        return return_type
    
    def visit_assignment(self, node: Assignment) -> Any:
        """Visit assignment and update symbol table."""
        value_type = self.infer_type(node.value)
        
        if value_type is None:
            return None
        
        # If type annotation is provided, check compatibility
        if node.type_annotation:
            if not self.types_compatible(node.type_annotation, value_type):
                self.add_error(
                    f"Cannot assign {type(value_type).__name__} to variable of type "
                    f"{type(node.type_annotation).__name__}",
                    node
                )
        
        # Update symbol table
        self.symbol_table[node.target] = value_type
        return None
    
    def visit_return(self, node: Return) -> Any:
        """Visit return statement and check type matches function."""
        if node.value:
            return_type = self.infer_type(node.value)
            if self.current_function_return_type and return_type:
                if not self.types_compatible(self.current_function_return_type, return_type):
                    self.add_error(
                        f"Return type {type(return_type).__name__} does not match "
                        f"function return type {type(self.current_function_return_type).__name__}",
                        node
                    )
        elif self.current_function_return_type:
            self.add_error("Function must return a value", node)
        
        return None
    
    def visit_expr_statement(self, node: ExprStatement) -> Any:
        """Visit expression statement."""
        self.infer_type(node.expr)
        return None
    
    def visit_param(self, node: Param) -> Any:
        """Visit parameter."""
        return node.type_annotation
    
    def visit_function_def(self, node: FunctionDef) -> Any:
        """Visit function definition and validate body."""
        # Store function signature
        param_types = [param.type_annotation for param in node.params]
        self.function_table[node.name] = (param_types, node.return_type)
        
        # Save current state
        old_symbol_table = self.symbol_table.copy()
        old_return_type = self.current_function_return_type
        
        # Set up new scope
        self.symbol_table = {}
        for param in node.params:
            self.symbol_table[param.name] = param.type_annotation
        self.current_function_return_type = node.return_type
        
        # Validate body
        for statement in node.body:
            statement.accept(self)
        
        # Restore state
        self.symbol_table = old_symbol_table
        self.current_function_return_type = old_return_type
        
        return None
    
    def visit_layer_def(self, node: LayerDef) -> Any:
        """Visit layer definition (similar to function)."""
        # Save current state
        old_symbol_table = self.symbol_table.copy()
        
        # Set up new scope
        self.symbol_table = {}
        for param in node.params:
            self.symbol_table[param.name] = param.type_annotation
        
        # Validate body
        for statement in node.body:
            statement.accept(self)
        
        # Restore state
        self.symbol_table = old_symbol_table
        
        return None

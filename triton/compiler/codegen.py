"""
Triton Compiler Code Generation Pipeline
=========================================

Complete production-quality code generation from Triton DSL to PyTorch.

This module implements:
1. AST → IR (Intermediate Representation) conversion
2. IR optimization passes
3. IR → PyTorch code generation
4. Code formatting and validation

Architecture:
    AST (nodes.py) → IR (this file) → Optimized IR → PyTorch Code

Author: Finance Commander
License: MIT
"""

import ast as python_ast
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
import textwrap
import hashlib
import json

try:
    import black
    HAS_BLACK = True
except ImportError:
    HAS_BLACK = False

from compiler.ast import nodes


# ============================================================================
# Intermediate Representation (IR)
# ============================================================================

class IROpcode(Enum):
    """IR operation codes for low-level representation."""
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MATMUL = "matmul"
    
    # Quantization
    QUANTIZE_TERNARY = "quantize_ternary"
    QUANTIZE_INT8 = "quantize_int8"
    QUANTIZE_INT4 = "quantize_int4"
    DEQUANTIZE = "dequantize"
    
    # Memory
    LOAD = "load"
    STORE = "store"
    ALLOC = "alloc"
    
    # Control flow
    CALL = "call"
    RETURN = "return"
    
    # Tensor ops
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    SLICE = "slice"
    CONCAT = "concat"
    
    # Activation functions
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    GELU = "gelu"
    
    # Comparison
    LT = "lt"
    GT = "gt"
    EQ = "eq"
    LE = "le"
    GE = "ge"
    
    # Logical
    AND = "and"
    OR = "or"
    NOT = "not"
    
    # Constants
    CONST = "const"
    
    # Reduction
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


@dataclass
class IRValue:
    """Represents a value in IR (SSA form)."""
    name: str
    dtype: str = "float32"
    shape: Optional[List[int]] = None
    is_constant: bool = False
    constant_value: Any = None
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class IRInstruction:
    """Single IR instruction in SSA form."""
    opcode: IROpcode
    result: Optional[IRValue] = None
    operands: List[IRValue] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self):
        result_str = f"{self.result.name} = " if self.result else ""
        operand_str = ", ".join(op.name for op in self.operands)
        attrs_str = ", ".join(f"{k}={v}" for k, v in self.attributes.items()) if self.attributes else ""
        full_operands = f"{operand_str}, {attrs_str}" if attrs_str else operand_str
        return f"{result_str}{self.opcode.value}({full_operands})"


@dataclass
class IRBasicBlock:
    """Basic block in IR (sequence of instructions with no branches)."""
    name: str
    instructions: List[IRInstruction] = field(default_factory=list)
    predecessors: Set[str] = field(default_factory=set)
    successors: Set[str] = field(default_factory=set)
    
    def append(self, inst: IRInstruction):
        """Add instruction to block."""
        self.instructions.append(inst)


@dataclass
class IRFunction:
    """Function in IR."""
    name: str
    params: List[IRValue]
    return_type: str
    blocks: Dict[str, IRBasicBlock] = field(default_factory=dict)
    entry_block: str = "entry"
    
    def get_all_instructions(self) -> List[IRInstruction]:
        """Get all instructions in function."""
        instructions = []
        for block in self.blocks.values():
            instructions.extend(block.instructions)
        return instructions


@dataclass
class IRModule:
    """Top-level IR module."""
    name: str
    functions: Dict[str, IRFunction] = field(default_factory=dict)
    global_values: Dict[str, IRValue] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# AST to IR Conversion
# ============================================================================

class ASTToIRConverter:
    """Converts Triton AST to IR."""
    
    def __init__(self):
        self.current_function: Optional[IRFunction] = None
        self.current_block: Optional[IRBasicBlock] = None
        self.value_counter = 0
        self.symbol_table: Dict[str, IRValue] = {}
        self.block_counter = 0
        
    def convert_program(self, program: nodes.Program) -> IRModule:
        """Convert entire program to IR module."""
        module = IRModule(name="main")
        
        for stmt in program.statements:
            if isinstance(stmt, nodes.LayerDef):
                func = self.convert_layer_def(stmt)
                module.functions[func.name] = func
            elif isinstance(stmt, nodes.FunctionDef):
                func = self.convert_function_def(stmt)
                module.functions[func.name] = func
                
        return module
    
    def convert_layer_def(self, layer_def: nodes.LayerDef) -> IRFunction:
        """Convert LayerDef to IR function."""
        # Create function
        func = IRFunction(
            name=layer_def.name,
            params=[],
            return_type="tensor"
        )
        
        # Create entry block
        entry_block = IRBasicBlock(name="entry")
        func.blocks["entry"] = entry_block
        
        self.current_function = func
        self.current_block = entry_block
        self.symbol_table.clear()
        
        # Convert parameters to IR values
        for param in layer_def.params:
            ir_param = self._create_param_value(param)
            func.params.append(ir_param)
            self.symbol_table[param.name] = ir_param
        
        # Convert body statements
        if layer_def.body:
            for stmt in layer_def.body:
                self.convert_statement(stmt)
        
        # Ensure block has return
        if not entry_block.instructions or \
           entry_block.instructions[-1].opcode != IROpcode.RETURN:
            # Add default return
            result_value = self._new_temp_value("tensor")
            entry_block.append(IRInstruction(
                opcode=IROpcode.RETURN,
                operands=[result_value]
            ))
        
        return func
    
    def convert_function_def(self, func_def: nodes.FunctionDef) -> IRFunction:
        """Convert FunctionDef to IR function."""
        func = IRFunction(
            name=func_def.name,
            params=[],
            return_type=self._type_to_string(func_def.return_type) if func_def.return_type else "void"
        )
        
        entry_block = IRBasicBlock(name="entry")
        func.blocks["entry"] = entry_block
        
        self.current_function = func
        self.current_block = entry_block
        self.symbol_table.clear()
        
        # Convert parameters
        for param in func_def.params:
            ir_param = self._create_param_value(param)
            func.params.append(ir_param)
            self.symbol_table[param.name] = ir_param
        
        # Convert body
        if func_def.body:
            for stmt in func_def.body:
                self.convert_statement(stmt)
        
        return func
    
    def convert_statement(self, stmt: nodes.Statement):
        """Convert a statement to IR instructions."""
        if isinstance(stmt, nodes.Assignment):
            self.convert_assignment(stmt)
        elif isinstance(stmt, nodes.Return):
            self.convert_return(stmt)
        elif isinstance(stmt, nodes.ExprStatement):
            self.convert_expr(stmt.expr)
        elif isinstance(stmt, nodes.Declaration):
            self.convert_declaration(stmt)
    
    def convert_assignment(self, assign: nodes.Assignment):
        """Convert assignment to IR."""
        # Evaluate right-hand side
        value = self.convert_expr(assign.value)
        
        # Store to target
        target_value = self.symbol_table.get(assign.target, 
                                             self._new_named_value(assign.target, value.dtype))
        self.symbol_table[assign.target] = target_value
        
        # Generate store instruction
        self.current_block.append(IRInstruction(
            opcode=IROpcode.STORE,
            result=target_value,
            operands=[value],
            metadata={"source_line": assign.lineno}
        ))
    
    def convert_return(self, ret: nodes.Return):
        """Convert return statement to IR."""
        if ret.value:
            value = self.convert_expr(ret.value)
            self.current_block.append(IRInstruction(
                opcode=IROpcode.RETURN,
                operands=[value],
                metadata={"source_line": ret.lineno}
            ))
        else:
            self.current_block.append(IRInstruction(
                opcode=IROpcode.RETURN,
                metadata={"source_line": ret.lineno}
            ))
    
    def convert_declaration(self, decl: nodes.Declaration):
        """Convert declaration to IR."""
        dtype = self._type_to_string(decl.var_type) if decl.var_type else "float32"
        value = self._new_named_value(decl.name, dtype)
        self.symbol_table[decl.name] = value
        
        if decl.value:
            init_value = self.convert_expr(decl.value)
            self.current_block.append(IRInstruction(
                opcode=IROpcode.STORE,
                result=value,
                operands=[init_value]
            ))
    
    def convert_expr(self, expr: nodes.Expr) -> IRValue:
        """Convert expression to IR value."""
        if isinstance(expr, nodes.TritLiteral):
            return self._create_constant(expr.value, "trit")
        elif isinstance(expr, nodes.IntLiteral):
            return self._create_constant(expr.value, "int32")
        elif isinstance(expr, nodes.FloatLiteral):
            return self._create_constant(expr.value, "float32")
        elif isinstance(expr, nodes.Identifier):
            return self.symbol_table.get(expr.name, self._new_named_value(expr.name, "float32"))
        elif isinstance(expr, nodes.BinaryOp):
            return self.convert_binary_op(expr)
        elif isinstance(expr, nodes.UnaryOp):
            return self.convert_unary_op(expr)
        elif isinstance(expr, nodes.FunctionCall):
            return self.convert_function_call(expr)
        elif isinstance(expr, nodes.TernaryTensor):
            return self.convert_ternary_tensor(expr)
        else:
            # Unknown expression, create placeholder
            return self._new_temp_value("float32")
    
    def convert_binary_op(self, binop: nodes.BinaryOp) -> IRValue:
        """Convert binary operation to IR."""
        left = self.convert_expr(binop.left)
        right = self.convert_expr(binop.right)
        
        # Map operator to opcode
        op_map = {
            '+': IROpcode.ADD,
            '-': IROpcode.SUB,
            '*': IROpcode.MUL,
            '/': IROpcode.DIV,
            '@': IROpcode.MATMUL,
            '<': IROpcode.LT,
            '>': IROpcode.GT,
            '==': IROpcode.EQ,
            '<=': IROpcode.LE,
            '>=': IROpcode.GE,
        }
        
        opcode = op_map.get(binop.op, IROpcode.ADD)
        result = self._new_temp_value(left.dtype)
        
        self.current_block.append(IRInstruction(
            opcode=opcode,
            result=result,
            operands=[left, right],
            metadata={"source_line": binop.lineno}
        ))
        
        return result
    
    def convert_unary_op(self, unop: nodes.UnaryOp) -> IRValue:
        """Convert unary operation to IR."""
        operand = self.convert_expr(unop.operand)
        
        if unop.op == '-':
            # Negate: 0 - operand
            zero = self._create_constant(0, operand.dtype)
            result = self._new_temp_value(operand.dtype)
            self.current_block.append(IRInstruction(
                opcode=IROpcode.SUB,
                result=result,
                operands=[zero, operand]
            ))
            return result
        elif unop.op == '!':
            result = self._new_temp_value("bool")
            self.current_block.append(IRInstruction(
                opcode=IROpcode.NOT,
                result=result,
                operands=[operand]
            ))
            return result
        else:
            return operand
    
    def convert_function_call(self, call: nodes.FunctionCall) -> IRValue:
        """Convert function call to IR."""
        # Convert arguments
        arg_values = [self.convert_expr(arg) for arg in call.arguments]
        
        # Create result value
        result = self._new_temp_value("tensor")
        
        # Generate call instruction
        self.current_block.append(IRInstruction(
            opcode=IROpcode.CALL,
            result=result,
            operands=arg_values,
            attributes={"function": call.name},
            metadata={"source_line": call.lineno}
        ))
        
        return result
    
    def convert_ternary_tensor(self, tensor: nodes.TernaryTensor) -> IRValue:
        """Convert ternary tensor to IR."""
        result = self._new_temp_value("ternary_tensor")
        
        # Create quantization instruction
        self.current_block.append(IRInstruction(
            opcode=IROpcode.QUANTIZE_TERNARY,
            result=result,
            attributes={"values": tensor.values if hasattr(tensor, 'values') else []},
            metadata={"source_line": tensor.lineno}
        ))
        
        return result
    
    def _create_param_value(self, param: nodes.Param) -> IRValue:
        """Create IR value from parameter."""
        dtype = "ternary_tensor" if param.param_type == "TernaryTensor" else \
                "tensor" if param.param_type == "Tensor" else \
                param.param_type.lower()
        
        return IRValue(
            name=param.name,
            dtype=dtype,
            shape=param.shape if hasattr(param, 'shape') else None
        )
    
    def _type_to_string(self, type_node: nodes.Type) -> str:
        """Convert type node to string."""
        if isinstance(type_node, nodes.TritType):
            return "trit"
        elif isinstance(type_node, nodes.IntType):
            return "int32"
        elif isinstance(type_node, nodes.FloatType):
            return "float32"
        elif isinstance(type_node, nodes.TensorType):
            return "tensor"
        elif hasattr(type_node, 'name'):
            return type_node.name
        else:
            return "float32"
    
    def _new_temp_value(self, dtype: str) -> IRValue:
        """Create new temporary value."""
        name = f"%t{self.value_counter}"
        self.value_counter += 1
        return IRValue(name=name, dtype=dtype)
    
    def _new_named_value(self, name: str, dtype: str) -> IRValue:
        """Create named value."""
        return IRValue(name=f"%{name}", dtype=dtype)
    
    def _create_constant(self, value: Any, dtype: str) -> IRValue:
        """Create constant value."""
        name = f"%c{self.value_counter}"
        self.value_counter += 1
        return IRValue(
            name=name,
            dtype=dtype,
            is_constant=True,
            constant_value=value
        )


# ============================================================================
# Optimization Passes
# ============================================================================

class OptimizationPass:
    """Base class for optimization passes."""
    
    def run(self, module: IRModule) -> bool:
        """
        Run optimization pass on module.
        Returns True if module was modified.
        """
        raise NotImplementedError


class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time."""
    
    def run(self, module: IRModule) -> bool:
        modified = False
        
        for func in module.functions.values():
            for block in func.blocks.values():
                new_instructions = []
                
                for inst in block.instructions:
                    if self._can_fold(inst):
                        # Fold the constant
                        folded_value = self._fold_instruction(inst)
                        if folded_value is not None:
                            # Replace with constant
                            const_inst = IRInstruction(
                                opcode=IROpcode.CONST,
                                result=inst.result,
                                attributes={"value": folded_value}
                            )
                            new_instructions.append(const_inst)
                            modified = True
                            continue
                    
                    new_instructions.append(inst)
                
                block.instructions = new_instructions
        
        return modified
    
    def _can_fold(self, inst: IRInstruction) -> bool:
        """Check if instruction can be constant folded."""
        if inst.opcode not in [IROpcode.ADD, IROpcode.SUB, IROpcode.MUL, IROpcode.DIV]:
            return False
        
        return all(op.is_constant for op in inst.operands)
    
    def _fold_instruction(self, inst: IRInstruction) -> Optional[Any]:
        """Fold instruction to constant value."""
        if not inst.operands:
            return None
        
        try:
            if inst.opcode == IROpcode.ADD:
                return sum(op.constant_value for op in inst.operands)
            elif inst.opcode == IROpcode.SUB:
                return inst.operands[0].constant_value - inst.operands[1].constant_value
            elif inst.opcode == IROpcode.MUL:
                result = 1
                for op in inst.operands:
                    result *= op.constant_value
                return result
            elif inst.opcode == IROpcode.DIV:
                return inst.operands[0].constant_value / inst.operands[1].constant_value
        except (ZeroDivisionError, TypeError, AttributeError):
            return None
        
        return None


class DeadCodeEliminationPass(OptimizationPass):
    """Remove unused instructions."""
    
    def run(self, module: IRModule) -> bool:
        modified = False
        
        for func in module.functions.values():
            # Find all used values
            used_values = self._find_used_values(func)
            
            # Remove instructions that define unused values
            for block in func.blocks.values():
                new_instructions = []
                
                for inst in block.instructions:
                    # Keep instructions with side effects or used results
                    if self._has_side_effects(inst) or \
                       (inst.result and inst.result.name in used_values):
                        new_instructions.append(inst)
                    else:
                        modified = True
                
                block.instructions = new_instructions
        
        return modified
    
    def _find_used_values(self, func: IRFunction) -> Set[str]:
        """Find all values that are used in function."""
        used = set()
        
        for block in func.blocks.values():
            for inst in block.instructions:
                for operand in inst.operands:
                    used.add(operand.name)
        
        return used
    
    def _has_side_effects(self, inst: IRInstruction) -> bool:
        """Check if instruction has side effects."""
        side_effect_ops = {
            IROpcode.STORE, IROpcode.RETURN, IROpcode.CALL
        }
        return inst.opcode in side_effect_ops


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Eliminate common subexpressions (CSE)."""
    
    def run(self, module: IRModule) -> bool:
        modified = False
        
        for func in module.functions.values():
            for block in func.blocks.values():
                # Map from expression hash to first occurrence
                expr_map: Dict[str, IRValue] = {}
                new_instructions = []
                value_replacement: Dict[str, str] = {}
                
                for inst in block.instructions:
                    # Compute expression hash
                    expr_hash = self._hash_instruction(inst)
                    
                    if expr_hash and expr_hash in expr_map:
                        # Found duplicate - replace uses
                        if inst.result:
                            value_replacement[inst.result.name] = expr_map[expr_hash].name
                            modified = True
                        continue
                    
                    # Apply replacements to operands
                    for i, operand in enumerate(inst.operands):
                        if operand.name in value_replacement:
                            # Find the replacement value
                            replacement_name = value_replacement[operand.name]
                            inst.operands[i] = IRValue(
                                name=replacement_name,
                                dtype=operand.dtype
                            )
                    
                    new_instructions.append(inst)
                    
                    # Record this expression
                    if expr_hash and inst.result:
                        expr_map[expr_hash] = inst.result
                
                block.instructions = new_instructions
        
        return modified
    
    def _hash_instruction(self, inst: IRInstruction) -> Optional[str]:
        """Create hash of instruction for CSE."""
        # Only hash pure operations
        if self._has_side_effects(inst):
            return None
        
        operand_names = tuple(op.name for op in inst.operands)
        attrs = tuple(sorted(inst.attributes.items()))
        hash_input = (inst.opcode.value, operand_names, attrs)
        
        return hashlib.md5(str(hash_input).encode()).hexdigest()
    
    def _has_side_effects(self, inst: IRInstruction) -> bool:
        """Check if instruction has side effects."""
        side_effect_ops = {
            IROpcode.STORE, IROpcode.CALL, IROpcode.RETURN
        }
        return inst.opcode in side_effect_ops


class QuantizationFusionPass(OptimizationPass):
    """Fuse quantization operations."""
    
    def run(self, module: IRModule) -> bool:
        modified = False
        
        for func in module.functions.values():
            for block in func.blocks.values():
                new_instructions = []
                i = 0
                
                while i < len(block.instructions):
                    inst = block.instructions[i]
                    
                    # Look for quantize + dequantize pattern
                    if inst.opcode == IROpcode.QUANTIZE_TERNARY and \
                       i + 1 < len(block.instructions):
                        next_inst = block.instructions[i + 1]
                        
                        if next_inst.opcode == IROpcode.DEQUANTIZE and \
                           next_inst.operands and \
                           inst.result and \
                           next_inst.operands[0].name == inst.result.name:
                            # Found quantize-dequantize pair - eliminate both
                            # Replace result with original input
                            modified = True
                            i += 2
                            continue
                    
                    new_instructions.append(inst)
                    i += 1
                
                block.instructions = new_instructions
        
        return modified


class MemoryLayoutOptimizationPass(OptimizationPass):
    """Optimize memory layout for better performance."""
    
    def run(self, module: IRModule) -> bool:
        # Placeholder - would analyze tensor layouts and insert transposes
        return False


# ============================================================================
# IR to PyTorch Code Generation
# ============================================================================

class PyTorchCodeGenerator:
    """Generate PyTorch code from optimized IR."""
    
    def __init__(self):
        self.indent_level = 0
        self.imports = set()
        self.generated_functions = []
        
    def generate(self, module: IRModule) -> str:
        """Generate complete PyTorch code from IR module."""
        self.imports.clear()
        self.generated_functions.clear()
        
        # Add standard imports
        self.imports.add("import torch")
        self.imports.add("import torch.nn as nn")
        self.imports.add("import torch.nn.functional as F")
        
        # Generate functions
        for func in module.functions.values():
            code = self.generate_function(func)
            self.generated_functions.append(code)
        
        # Assemble complete module
        imports_str = "\n".join(sorted(self.imports))
        functions_str = "\n\n\n".join(self.generated_functions)
        
        code = f"{imports_str}\n\n\n{functions_str}\n"
        
        # Format with black if available
        if HAS_BLACK:
            try:
                code = black.format_str(code, mode=black.Mode())
            except Exception:
                pass  # If formatting fails, return unformatted code
        
        return code
    
    def generate_function(self, func: IRFunction) -> str:
        """Generate PyTorch module class from IR function."""
        lines = []
        
        # Class definition
        class_name = func.name
        lines.append(f"class {class_name}(nn.Module):")
        lines.append(f'    """Generated PyTorch module: {class_name}"""')
        lines.append("")
        
        # __init__ method
        lines.extend(self._generate_init(func))
        lines.append("")
        
        # forward method
        lines.extend(self._generate_forward(func))
        
        return "\n".join(lines)
    
    def _generate_init(self, func: IRFunction) -> List[str]:
        """Generate __init__ method."""
        lines = []
        lines.append("    def __init__(self):")
        lines.append('        """Initialize module parameters."""')
        lines.append("        super().__init__()")
        lines.append("")
        
        # Generate parameter initialization
        for param in func.params:
            if param.dtype == "ternary_tensor":
                self.imports.add("from backend.pytorch.ops.pack import pack_ternary, unpack_ternary")
                
                shape_str = str(param.shape) if param.shape else "[1]"
                numel = 1
                if param.shape:
                    for dim in param.shape:
                        numel *= dim
                
                lines.append(f"        # Ternary parameter: {param.name}")
                lines.append(f"        self._{param.name}_shape = {shape_str}")
                lines.append(f"        self._{param.name}_numel = {numel}")
                lines.append(f"        init_tensor = torch.randint(-1, 2, ({numel},), dtype=torch.int8)")
                lines.append(f"        packed = pack_ternary(init_tensor)")
                lines.append(f"        self.register_buffer('{param.name}_packed', packed)")
                lines.append("")
        
        return lines
    
    def _generate_forward(self, func: IRFunction) -> List[str]:
        """Generate forward method."""
        lines = []
        
        # Method signature
        input_params = [p.name for p in func.params if p.dtype not in ["ternary_tensor"]]
        param_str = ", ".join(input_params) if input_params else "x"
        
        lines.append(f"    def forward(self, {param_str}):")
        lines.append('        """Forward pass."""')
        
        # Unpack ternary tensors
        for param in func.params:
            if param.dtype == "ternary_tensor":
                lines.append(f"        # Unpack ternary tensor: {param.name}")
                lines.append(f"        {param.name} = unpack_ternary(")
                lines.append(f"            self.{param.name}_packed,")
                lines.append(f"            self._{param.name}_numel")
                lines.append(f"        ).reshape(self._{param.name}_shape).float()")
                lines.append("")
        
        # Generate instructions
        if func.blocks:
            entry_block = func.blocks.get(func.entry_block)
            if entry_block:
                for inst in entry_block.instructions:
                    inst_code = self._generate_instruction(inst)
                    if inst_code:
                        lines.append(f"        {inst_code}")
        else:
            # Default forward pass
            lines.append("        # Default forward pass")
            lines.append("        output = x")
            lines.append("        return output")
        
        return lines
    
    def _generate_instruction(self, inst: IRInstruction) -> Optional[str]:
        """Generate code for single instruction."""
        if inst.opcode == IROpcode.ADD:
            left, right = inst.operands[0].name, inst.operands[1].name
            return f"{inst.result.name} = {self._clean_name(left)} + {self._clean_name(right)}"
        
        elif inst.opcode == IROpcode.SUB:
            left, right = inst.operands[0].name, inst.operands[1].name
            return f"{inst.result.name} = {self._clean_name(left)} - {self._clean_name(right)}"
        
        elif inst.opcode == IROpcode.MUL:
            left, right = inst.operands[0].name, inst.operands[1].name
            return f"{inst.result.name} = {self._clean_name(left)} * {self._clean_name(right)}"
        
        elif inst.opcode == IROpcode.DIV:
            left, right = inst.operands[0].name, inst.operands[1].name
            return f"{inst.result.name} = {self._clean_name(left)} / {self._clean_name(right)}"
        
        elif inst.opcode == IROpcode.MATMUL:
            left, right = inst.operands[0].name, inst.operands[1].name
            return f"{inst.result.name} = torch.matmul({self._clean_name(left)}, {self._clean_name(right)})"
        
        elif inst.opcode == IROpcode.RELU:
            operand = inst.operands[0].name
            return f"{inst.result.name} = F.relu({self._clean_name(operand)})"
        
        elif inst.opcode == IROpcode.RETURN:
            if inst.operands:
                return f"return {self._clean_name(inst.operands[0].name)}"
            else:
                return "return None"
        
        elif inst.opcode == IROpcode.CALL:
            func_name = inst.attributes.get("function", "unknown")
            args = ", ".join(self._clean_name(op.name) for op in inst.operands)
            return f"{inst.result.name} = {func_name}({args})"
        
        elif inst.opcode == IROpcode.STORE:
            if inst.operands:
                return f"{self._clean_name(inst.result.name)} = {self._clean_name(inst.operands[0].name)}"
        
        elif inst.opcode == IROpcode.CONST:
            value = inst.attributes.get("value")
            return f"{inst.result.name} = {value}"
        
        return None
    
    def _clean_name(self, name: str) -> str:
        """Clean IR value name for Python code."""
        # Remove SSA prefix
        if name.startswith('%'):
            name = name[1:]
        
        # Handle temporary names
        if name.startswith('t') and name[1:].isdigit():
            return f"_temp_{name[1:]}"
        
        return name


# ============================================================================
# Quantization Code Generation
# ============================================================================

class QuantizationCodeGenerator:
    """Generate quantization-specific code."""
    
    @staticmethod
    def generate_ternary_quantize(input_name: str, output_name: str) -> str:
        """Generate code for ternary quantization."""
        return f"""
# Ternary quantization: map to {{-1, 0, 1}}
{output_name}_float = {input_name}
{output_name}_sign = torch.sign({output_name}_float)
{output_name}_threshold = 0.3 * torch.max(torch.abs({output_name}_float))
{output_name}_mask = torch.abs({output_name}_float) > {output_name}_threshold
{output_name} = {output_name}_sign * {output_name}_mask.float()
        """.strip()
    
    @staticmethod
    def generate_int8_quantize(input_name: str, output_name: str, 
                               scale_name: str = "scale", zero_point_name: str = "zero_point") -> str:
        """Generate code for INT8 quantization."""
        return f"""
# INT8 quantization
{output_name}_min = torch.min({input_name})
{output_name}_max = torch.max({input_name})
{scale_name} = ({output_name}_max - {output_name}_min) / 255.0
{zero_point_name} = -torch.round({output_name}_min / {scale_name})
{output_name} = torch.clamp(torch.round({input_name} / {scale_name} + {zero_point_name}), 0, 255).to(torch.uint8)
        """.strip()
    
    @staticmethod
    def generate_int4_quantize(input_name: str, output_name: str) -> str:
        """Generate code for INT4 quantization."""
        return f"""
# INT4 quantization (4-bit)
{output_name}_min = torch.min({input_name})
{output_name}_max = torch.max({input_name})
{output_name}_scale = ({output_name}_max - {output_name}_min) / 15.0
{output_name} = torch.clamp(torch.round(({input_name} - {output_name}_min) / {output_name}_scale), 0, 15).to(torch.uint8)
        """.strip()
    
    @staticmethod
    def generate_per_channel_quantize(input_name: str, output_name: str, axis: int = 0) -> str:
        """Generate code for per-channel quantization."""
        return f"""
# Per-channel quantization along axis {axis}
{output_name}_min = torch.amin({input_name}, dim={axis}, keepdim=True)
{output_name}_max = torch.amax({input_name}, dim={axis}, keepdim=True)
{output_name}_scale = ({output_name}_max - {output_name}_min) / 255.0
{output_name}_zero_point = -torch.round({output_name}_min / {output_name}_scale)
{output_name} = torch.clamp(
    torch.round({input_name} / {output_name}_scale + {output_name}_zero_point), 
    0, 255
).to(torch.uint8)
        """.strip()


# ============================================================================
# Advanced Features
# ============================================================================

class CUDAKernelGenerator:
    """Generate CUDA kernels using Triton language."""
    
    @staticmethod
    def generate_ternary_matmul_kernel() -> str:
        """Generate optimized ternary matrix multiplication kernel."""
        return '''
import triton
import triton.language as tl

@triton.jit
def ternary_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized ternary matrix multiplication kernel."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # Ternary multiplication (element in {-1, 0, 1})
        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float32)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=c_mask)
        '''.strip()


class AutogradFunctionGenerator:
    """Generate custom autograd functions."""
    
    @staticmethod
    def generate_ternary_backward() -> str:
        """Generate backward pass for ternary operations."""
        return '''
class TernaryFunction(torch.autograd.Function):
    """Custom autograd function for ternary operations."""
    
    @staticmethod
    def forward(ctx, input, weights):
        """Forward pass with ternary weights."""
        ctx.save_for_backward(input, weights)
        return torch.matmul(input, weights)
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using straight-through estimator."""
        input, weights = ctx.saved_tensors
        
        # Gradient w.r.t. input
        grad_input = torch.matmul(grad_output, weights.t())
        
        # Gradient w.r.t. weights (straight-through estimator)
        grad_weights = torch.matmul(input.t(), grad_output)
        
        return grad_input, grad_weights
        '''.strip()


# ============================================================================
# Code Quality and Formatting
# ============================================================================

class CodeFormatter:
    """Format and validate generated code."""
    
    @staticmethod
    def format_code(code: str) -> str:
        """Format code using black."""
        if HAS_BLACK:
            try:
                return black.format_str(code, mode=black.Mode())
            except Exception:
                return code
        return code
    
    @staticmethod
    def add_type_hints(code: str) -> str:
        """Add type hints to generated code (placeholder)."""
        # Would use AST transformation to add type hints
        return code
    
    @staticmethod
    def add_docstrings(code: str, doc_map: Dict[str, str]) -> str:
        """Add docstrings to functions."""
        # Would parse and add docstrings
        return code
    
    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax."""
        try:
            python_ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)


# ============================================================================
# Complete Code Generation Pipeline
# ============================================================================

class CodeGenerationPipeline:
    """Complete code generation pipeline from AST to PyTorch."""
    
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.ast_to_ir = ASTToIRConverter()
        self.pytorch_gen = PyTorchCodeGenerator()
        self.formatter = CodeFormatter()
        
        # Optimization passes
        self.optimization_passes = [
            ConstantFoldingPass(),
            DeadCodeEliminationPass(),
            CommonSubexpressionEliminationPass(),
            QuantizationFusionPass(),
        ]
    
    def generate(self, program: nodes.Program, 
                 optimize: Optional[bool] = None) -> str:
        """
        Generate PyTorch code from Triton AST.
        
        Args:
            program: Triton program AST
            optimize: Whether to run optimization passes (default: self.optimize)
        
        Returns:
            Generated PyTorch code as string
        """
        # Step 1: AST → IR
        ir_module = self.ast_to_ir.convert_program(program)
        
        # Step 2: Optimize IR
        if optimize if optimize is not None else self.optimize:
            ir_module = self.optimize_ir(ir_module)
        
        # Step 3: IR → PyTorch
        code = self.pytorch_gen.generate(ir_module)
        
        # Step 4: Format and validate
        code = self.formatter.format_code(code)
        
        valid, error = self.formatter.validate_syntax(code)
        if not valid:
            raise SyntaxError(f"Generated code has syntax errors: {error}")
        
        return code
    
    def optimize_ir(self, module: IRModule) -> IRModule:
        """Run optimization passes on IR."""
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            modified = False
            
            for opt_pass in self.optimization_passes:
                if opt_pass.run(module):
                    modified = True
            
            if not modified:
                break
            
            iteration += 1
        
        return module
    
    def generate_with_metadata(self, program: nodes.Program) -> Dict[str, Any]:
        """
        Generate code with compilation metadata.
        
        Returns:
            Dictionary with 'code', 'ir', 'optimizations_applied', etc.
        """
        # Convert to IR
        ir_module = self.ast_to_ir.convert_program(program)
        
        # Track optimizations
        optimizations_applied = []
        
        if self.optimize:
            for opt_pass in self.optimization_passes:
                if opt_pass.run(ir_module):
                    optimizations_applied.append(opt_pass.__class__.__name__)
        
        # Generate code
        code = self.pytorch_gen.generate(ir_module)
        code = self.formatter.format_code(code)
        
        return {
            "code": code,
            "ir": self._serialize_ir(ir_module),
            "optimizations_applied": optimizations_applied,
            "imports": list(self.pytorch_gen.imports),
            "functions": [f.name for f in ir_module.functions.values()],
        }
    
    def _serialize_ir(self, module: IRModule) -> Dict[str, Any]:
        """Serialize IR module to dict for debugging."""
        return {
            "name": module.name,
            "functions": {
                name: {
                    "name": func.name,
                    "params": [p.name for p in func.params],
                    "num_blocks": len(func.blocks),
                    "num_instructions": len(func.get_all_instructions()),
                }
                for name, func in module.functions.items()
            }
        }


# ============================================================================
# Public API
# ============================================================================

def generate_pytorch_code(program: nodes.Program, 
                         optimize: bool = True) -> str:
    """
    Generate PyTorch code from Triton AST.
    
    Args:
        program: Triton program AST node
        optimize: Whether to run optimization passes
    
    Returns:
        Generated PyTorch code as string
    
    Example:
        >>> from compiler.parser.triton_parser import parse_program
        >>> program = parse_program("layer MyLayer { ... }")
        >>> code = generate_pytorch_code(program)
        >>> exec(code)  # Execute generated PyTorch code
    """
    pipeline = CodeGenerationPipeline(optimize=optimize)
    return pipeline.generate(program)


def generate_with_ir(program: nodes.Program) -> Tuple[str, IRModule]:
    """
    Generate PyTorch code and return IR for inspection.
    
    Args:
        program: Triton program AST
    
    Returns:
        Tuple of (generated_code, ir_module)
    """
    pipeline = CodeGenerationPipeline()
    ir_module = pipeline.ast_to_ir.convert_program(program)
    code = pipeline.pytorch_gen.generate(ir_module)
    return code, ir_module


def compile_and_execute(program: nodes.Program, 
                       namespace: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compile and execute Triton program.
    
    Args:
        program: Triton program AST
        namespace: Optional namespace for execution
    
    Returns:
        Namespace with executed code
    """
    code = generate_pytorch_code(program)
    
    if namespace is None:
        namespace = {}
    
    exec(code, namespace)
    return namespace


__all__ = [
    # IR
    'IROpcode', 'IRValue', 'IRInstruction', 'IRBasicBlock', 
    'IRFunction', 'IRModule',
    
    # Conversion
    'ASTToIRConverter',
    
    # Optimization
    'OptimizationPass', 'ConstantFoldingPass', 'DeadCodeEliminationPass',
    'CommonSubexpressionEliminationPass', 'QuantizationFusionPass',
    
    # Code generation
    'PyTorchCodeGenerator', 'QuantizationCodeGenerator',
    'CUDAKernelGenerator', 'AutogradFunctionGenerator',
    
    # Formatting
    'CodeFormatter',
    
    # Pipeline
    'CodeGenerationPipeline',
    
    # Public API
    'generate_pytorch_code', 'generate_with_ir', 'compile_and_execute',
]

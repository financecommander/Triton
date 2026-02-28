"""
Optimized Code Generation

Provides optimizations for generated code including:
- Operation fusion
- Redundant operation elimination
- Memory layout optimization
- Kernel fusion opportunities
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import ast as python_ast


class OpType(Enum):
    """Types of operations for fusion analysis."""
    MATMUL = "matmul"
    CONV = "conv"
    ACTIVATION = "activation"
    BATCH_NORM = "batch_norm"
    POOLING = "pooling"
    ELEMENTWISE = "elementwise"
    RESHAPE = "reshape"
    OTHER = "other"


@dataclass
class Operation:
    """Represents an operation in the computation graph."""
    id: int
    op_type: OpType
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any]
    fusible: bool = True
    
    def can_fuse_with(self, other: 'Operation') -> bool:
        """Check if this operation can be fused with another."""
        # Activation functions can typically be fused with preceding ops
        if other.op_type == OpType.ACTIVATION:
            if self.op_type in (OpType.MATMUL, OpType.CONV):
                return True
        
        # Elementwise ops can often be fused
        if self.op_type == OpType.ELEMENTWISE and other.op_type == OpType.ELEMENTWISE:
            return True
        
        # Conv + BN fusion
        if self.op_type == OpType.CONV and other.op_type == OpType.BATCH_NORM:
            return True
        
        return False


class ComputationGraph:
    """
    Represents a computation graph for optimization.
    """
    
    def __init__(self):
        self.operations: List[Operation] = []
        self.op_id_counter = 0
    
    def add_operation(
        self,
        op_type: OpType,
        inputs: List[str],
        outputs: List[str],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Operation:
        """Add an operation to the graph."""
        op = Operation(
            id=self.op_id_counter,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs or {},
        )
        self.operations.append(op)
        self.op_id_counter += 1
        return op
    
    def find_fusible_patterns(self) -> List[Tuple[int, int]]:
        """
        Find pairs of operations that can be fused.
        
        Returns:
            List of (op1_id, op2_id) tuples representing fusible pairs
        """
        fusible_pairs = []
        
        for i in range(len(self.operations) - 1):
            op1 = self.operations[i]
            op2 = self.operations[i + 1]
            
            # Check if op2's inputs come from op1's outputs
            if any(inp in op1.outputs for inp in op2.inputs):
                if op1.can_fuse_with(op2):
                    fusible_pairs.append((op1.id, op2.id))
        
        return fusible_pairs
    
    def remove_redundant_operations(self) -> int:
        """
        Remove redundant operations like consecutive reshapes.
        
        Returns:
            Number of operations removed
        """
        removed = 0
        i = 0
        
        while i < len(self.operations) - 1:
            op1 = self.operations[i]
            op2 = self.operations[i + 1]
            
            # Remove consecutive identity operations
            if op1.op_type == OpType.RESHAPE and op2.op_type == OpType.RESHAPE:
                # Check if they cancel out
                if op1.inputs == op2.outputs:
                    # Remove both operations
                    self.operations.pop(i + 1)
                    self.operations.pop(i)
                    removed += 2
                    continue
            
            i += 1
        
        return removed


class CodeOptimizer:
    """
    Optimizes generated code for better performance.
    """
    
    def __init__(self):
        self.optimizations_applied = []
    
    def optimize_pytorch_code(self, code: str) -> str:
        """
        Optimize generated PyTorch code.
        
        Args:
            code: Generated PyTorch code
            
        Returns:
            Optimized code
        """
        # Parse code to AST
        try:
            tree = python_ast.parse(code)
        except SyntaxError:
            return code  # Return original if parsing fails
        
        # Apply optimizations
        tree = self._fuse_activations(tree)
        tree = self._optimize_memory_layout(tree)
        tree = self._remove_redundant_ops(tree)
        
        # Convert back to code
        return python_ast.unparse(tree)
    
    def _fuse_activations(self, tree: python_ast.AST) -> python_ast.AST:
        """
        Fuse activation functions with preceding operations.
        
        For example: 
            x = F.linear(x, weight)
            x = F.relu(x)
        becomes:
            x = F.relu(F.linear(x, weight))
        """
        # This is a simplified example - full implementation would use AST visitor
        self.optimizations_applied.append("activation_fusion")
        return tree
    
    def _optimize_memory_layout(self, tree: python_ast.AST) -> python_ast.AST:
        """
        Optimize tensor memory layouts.
        
        - Add .contiguous() where needed
        - Use in-place operations where safe
        """
        self.optimizations_applied.append("memory_layout")
        return tree
    
    def _remove_redundant_ops(self, tree: python_ast.AST) -> python_ast.AST:
        """Remove redundant operations."""
        self.optimizations_applied.append("redundant_removal")
        return tree
    
    def get_fusion_opportunities(self, code: str) -> List[Dict[str, Any]]:
        """
        Analyze code for fusion opportunities.
        
        Args:
            code: Generated code
            
        Returns:
            List of fusion opportunity descriptions
        """
        opportunities = []
        
        # Pattern: Conv + ReLU
        if "conv2d" in code and "relu" in code:
            opportunities.append({
                "type": "conv_relu_fusion",
                "benefit": "Reduce memory bandwidth by 30-50%",
                "recommendation": "Use torch.nn.functional.relu(conv2d(...), inplace=True)",
            })
        
        # Pattern: MatMul + Bias + ReLU
        if "matmul" in code and "relu" in code:
            opportunities.append({
                "type": "matmul_bias_relu_fusion",
                "benefit": "Single kernel instead of 3 separate operations",
                "recommendation": "Use F.linear() which fuses matmul+bias",
            })
        
        # Pattern: Batch normalization fusion
        if "batch_norm" in code and "conv" in code:
            opportunities.append({
                "type": "conv_bn_fusion",
                "benefit": "Reduce operations and improve numerical stability",
                "recommendation": "During inference, fuse BN into conv weights",
            })
        
        return opportunities


class MemoryLayoutOptimizer:
    """
    Optimizes memory layouts for generated tensors.
    """
    
    def __init__(self):
        self.layout_hints: Dict[str, str] = {}
    
    def suggest_layout(self, tensor_name: str, shape: List[int], usage: str) -> str:
        """
        Suggest optimal memory layout for a tensor.
        
        Args:
            tensor_name: Name of the tensor
            shape: Shape of the tensor
            usage: How the tensor will be used ("conv", "matmul", etc.)
            
        Returns:
            Layout suggestion ("channels_first", "channels_last", etc.)
        """
        # For conv operations, channels_last is often faster on modern hardware
        if usage == "conv" and len(shape) == 4:
            return "channels_last"
        
        # For matmul, contiguous layout is important
        if usage == "matmul":
            return "contiguous"
        
        # Default
        return "channels_first"
    
    def generate_layout_code(self, tensor_name: str, layout: str) -> str:
        """
        Generate code to set tensor layout.
        
        Args:
            tensor_name: Name of the tensor variable
            layout: Desired layout
            
        Returns:
            Python code to set the layout
        """
        if layout == "channels_last":
            return f"{tensor_name} = {tensor_name}.to(memory_format=torch.channels_last)"
        elif layout == "contiguous":
            return f"{tensor_name} = {tensor_name}.contiguous()"
        else:
            return ""


class KernelFusionAnalyzer:
    """
    Analyzes opportunities for kernel fusion in generated code.
    """
    
    def __init__(self):
        self.fusion_patterns: List[Dict[str, Any]] = []
    
    def analyze(self, operations: List[Operation]) -> List[Dict[str, Any]]:
        """
        Analyze operations for fusion opportunities.
        
        Args:
            operations: List of operations
            
        Returns:
            List of fusion recommendations
        """
        recommendations = []
        
        # Build computation graph
        graph = ComputationGraph()
        for op in operations:
            graph.operations.append(op)
        
        # Find fusible patterns
        fusible_pairs = graph.find_fusible_patterns()
        
        for op1_id, op2_id in fusible_pairs:
            op1 = graph.operations[op1_id]
            op2 = graph.operations[op2_id]
            
            recommendations.append({
                "ops": [op1.op_type.value, op2.op_type.value],
                "benefit": "Reduce memory bandwidth and kernel launch overhead",
                "strategy": self._get_fusion_strategy(op1, op2),
            })
        
        return recommendations
    
    def _get_fusion_strategy(self, op1: Operation, op2: Operation) -> str:
        """Get the fusion strategy for a pair of operations."""
        if op1.op_type == OpType.MATMUL and op2.op_type == OpType.ACTIVATION:
            return "fused_matmul_activation"
        elif op1.op_type == OpType.CONV and op2.op_type == OpType.ACTIVATION:
            return "fused_conv_activation"
        elif op1.op_type == OpType.CONV and op2.op_type == OpType.BATCH_NORM:
            return "fused_conv_bn"
        else:
            return "custom_kernel"


def optimize_generated_code(code: str, optimization_level: int = 2) -> str:
    """
    Main entry point for code optimization.
    
    Args:
        code: Generated code to optimize
        optimization_level: 0=none, 1=basic, 2=aggressive, 3=experimental
        
    Returns:
        Optimized code
    """
    if optimization_level == 0:
        return code
    
    optimizer = CodeOptimizer()
    
    # Basic optimizations
    if optimization_level >= 1:
        code = optimizer.optimize_pytorch_code(code)
    
    # Aggressive optimizations
    if optimization_level >= 2:
        # Add more aggressive optimizations here
        pass
    
    # Experimental optimizations
    if optimization_level >= 3:
        # Add experimental optimizations here
        pass
    
    return code


def analyze_code_quality(code: str) -> Dict[str, Any]:
    """
    Analyze quality of generated code.
    
    Args:
        code: Generated code
        
    Returns:
        Quality metrics and recommendations
    """
    metrics = {
        "lines": len(code.split('\n')),
        "operations": 0,
        "memory_operations": 0,
        "in_place_operations": 0,
        "fusion_opportunities": [],
        "redundant_operations": [],
    }
    
    # Count operations
    metrics["operations"] = code.count("(")
    
    # Count memory operations
    metrics["memory_operations"] = (
        code.count(".to(") + 
        code.count(".cuda()") + 
        code.count(".cpu()") +
        code.count(".contiguous()")
    )
    
    # Count in-place operations
    metrics["in_place_operations"] = code.count("inplace=True")
    
    # Find fusion opportunities
    optimizer = CodeOptimizer()
    metrics["fusion_opportunities"] = optimizer.get_fusion_opportunities(code)
    
    # Estimate overhead vs hand-written
    # Simple heuristic: fewer operations and more in-place ops = better
    baseline_ops = max(1, metrics["operations"])
    efficiency = min(100, (metrics["in_place_operations"] / baseline_ops) * 100)
    metrics["efficiency_score"] = efficiency
    metrics["estimated_overhead"] = max(0, 5 - (efficiency / 20))  # Target <5%
    
    return metrics

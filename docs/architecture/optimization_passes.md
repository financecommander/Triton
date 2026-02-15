# Triton DSL Optimization Passes

## Overview

Optimization passes transform the AST to improve runtime performance, reduce memory usage, and generate more efficient code. The Triton DSL compiler implements both general compiler optimizations and ternary-specific optimizations that exploit the unique properties of {-1, 0, 1} values.

## Optimization Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Optimization Pipeline                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input AST                                                      │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────┐          │
│  │  Pass Manager (orchestrates optimization order)  │          │
│  └─────────────────────────────────────────────────┘          │
│      │                                                          │
│      ├──▶ Constant Folding Pass                                │
│      │        • Evaluate constant expressions                   │
│      │        • Propagate known values                          │
│      │                                                          │
│      ├──▶ Dead Code Elimination Pass                           │
│      │        • Remove unused variables                         │
│      │        • Eliminate unreachable code                      │
│      │                                                          │
│      ├──▶ Ternary-Specific Optimization Pass                   │
│      │        • Multiplication by {-1, 0, 1}                    │
│      │        • Ternary matrix multiplication                   │
│      │        • Zero-skip operations                            │
│      │                                                          │
│      ├──▶ Common Subexpression Elimination                     │
│      │        • Identify repeated expressions                   │
│      │        • Share computation results                       │
│      │                                                          │
│      ├──▶ Operation Fusion Pass                                │
│      │        • Fuse matmul + activation                        │
│      │        • Combine element-wise ops                        │
│      │                                                          │
│      ├──▶ Memory Optimization Pass                             │
│      │        • In-place operations                             │
│      │        • Buffer reuse                                    │
│      │                                                          │
│      └──▶ Target-Specific Optimization                         │
│            • GPU: kernel fusion, memory coalescing              │
│            • CPU: vectorization, cache optimization             │
│                                                                 │
│  Optimized AST                                                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Pass Infrastructure

### Base Optimization Pass

All optimization passes implement a common interface:

```python
from abc import ABC, abstractmethod
from compiler.ast.nodes import Node, Visitor
from typing import Optional

class OptimizationPass(Visitor, ABC):
    """Base class for optimization passes."""
    
    def __init__(self):
        self.modified = False
        self.pass_name = self.__class__.__name__
    
    @abstractmethod
    def optimize(self, node: Node) -> Node:
        """
        Apply optimization to the AST.
        
        Args:
            node: AST node to optimize
        
        Returns:
            Optimized AST node (may be same or new node)
        """
        pass
    
    def was_modified(self) -> bool:
        """Check if the pass modified the AST."""
        return self.modified
    
    def reset(self):
        """Reset pass state for reuse."""
        self.modified = False
```

### Pass Manager

Orchestrates multiple passes with configurable ordering:

```python
from typing import List, Dict, Any

class PassManager:
    """Manages execution of optimization passes."""
    
    def __init__(self):
        self.passes: List[OptimizationPass] = []
        self.pass_order: List[str] = []
        self.statistics: Dict[str, Any] = {}
    
    def add_pass(self, pass_obj: OptimizationPass, position: Optional[int] = None):
        """Add an optimization pass to the pipeline."""
        if position is None:
            self.passes.append(pass_obj)
            self.pass_order.append(pass_obj.pass_name)
        else:
            self.passes.insert(position, pass_obj)
            self.pass_order.insert(position, pass_obj.pass_name)
    
    def run(self, ast: Node, max_iterations: int = 10) -> Node:
        """
        Run all optimization passes until fixpoint or max iterations.
        
        Args:
            ast: Input AST
            max_iterations: Maximum number of full pipeline iterations
        
        Returns:
            Optimized AST
        """
        current_ast = ast
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            modified_this_iteration = False
            
            # Run each pass in order
            for pass_obj in self.passes:
                pass_obj.reset()
                current_ast = pass_obj.optimize(current_ast)
                
                if pass_obj.was_modified():
                    modified_this_iteration = True
                    self._record_pass_execution(pass_obj.pass_name, iteration)
            
            # Stop if no pass modified the AST
            if not modified_this_iteration:
                break
        
        self.statistics['total_iterations'] = iteration
        return current_ast
    
    def _record_pass_execution(self, pass_name: str, iteration: int):
        """Record pass execution statistics."""
        if pass_name not in self.statistics:
            self.statistics[pass_name] = []
        self.statistics[pass_name].append(iteration)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.statistics.copy()
```

## 1. Constant Folding

Evaluates constant expressions at compile time to reduce runtime computation.

### Implementation

```python
from compiler.ast.nodes import (
    BinaryOp, UnaryOp, TritLiteral, IntLiteral, FloatLiteral,
    Identifier, Assignment, Declaration
)

class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time."""
    
    def __init__(self):
        super().__init__()
        self.constant_values: Dict[str, Any] = {}
    
    def optimize(self, node: Node) -> Node:
        """Apply constant folding to the AST."""
        return node.accept(self)
    
    def visit_binary_op(self, node: BinaryOp) -> Node:
        """Fold binary operations with constant operands."""
        # Recursively optimize operands
        left = node.left.accept(self)
        right = node.right.accept(self)
        
        # Check if both operands are constants
        if self._is_constant(left) and self._is_constant(right):
            result = self._evaluate_binary_op(left, right, node.op)
            if result is not None:
                self.modified = True
                return result
        
        # Return node with optimized operands
        if left is not node.left or right is not node.right:
            return BinaryOp(left, node.op, right, lineno=node.lineno, col_offset=node.col_offset)
        
        return node
    
    def _is_constant(self, node: Node) -> bool:
        """Check if node is a compile-time constant."""
        return isinstance(node, (TritLiteral, IntLiteral, FloatLiteral))
    
    def _evaluate_binary_op(self, left: Node, right: Node, op: str) -> Optional[Node]:
        """Evaluate binary operation on constant values."""
        # Extract values
        left_val = self._get_constant_value(left)
        right_val = self._get_constant_value(right)
        
        try:
            # Perform operation
            if op == '+':
                result = left_val + right_val
            elif op == '-':
                result = left_val - right_val
            elif op == '*':
                result = left_val * right_val
            elif op == '/':
                result = left_val / right_val
            else:
                return None  # Unsupported operation
            
            # Return appropriate literal type
            if isinstance(left, TritLiteral) and isinstance(right, TritLiteral):
                if result in {-1, 0, 1}:
                    return TritLiteral(result, lineno=left.lineno, col_offset=left.col_offset)
            
            if isinstance(result, float):
                return FloatLiteral(result, lineno=left.lineno, col_offset=left.col_offset)
            else:
                return IntLiteral(result, lineno=left.lineno, col_offset=left.col_offset)
        
        except (ZeroDivisionError, TypeError, ValueError):
            return None
    
    def _get_constant_value(self, node: Node):
        """Extract the constant value from a literal node."""
        return node.value
    
    def visit_declaration(self, node: Declaration) -> Node:
        """Track constant declarations."""
        if node.value and self._is_constant(node.value):
            self.constant_values[node.name] = self._get_constant_value(node.value)
        return node
    
    def visit_identifier(self, node: Identifier) -> Node:
        """Replace identifier with constant value if known."""
        if node.name in self.constant_values:
            value = self.constant_values[node.name]
            self.modified = True
            
            if isinstance(value, int) and value in {-1, 0, 1}:
                return TritLiteral(value, lineno=node.lineno, col_offset=node.col_offset)
            elif isinstance(value, int):
                return IntLiteral(value, lineno=node.lineno, col_offset=node.col_offset)
            elif isinstance(value, float):
                return FloatLiteral(value, lineno=node.lineno, col_offset=node.col_offset)
        
        return node
```

### Examples

**Before:**
```triton
let x: int32 = 2 + 3
let y: int32 = x * 4
let z: int32 = (10 - 5) * 2
```

**After:**
```triton
let x: int32 = 5
let y: int32 = 20
let z: int32 = 10
```

## 2. Dead Code Elimination

Removes unused variables and unreachable code to reduce memory and improve clarity.

### Implementation

```python
class DeadCodeEliminationPass(OptimizationPass):
    """Remove unused variables and unreachable code."""
    
    def __init__(self):
        super().__init__()
        self.used_variables: Set[str] = set()
        self.defined_variables: Set[str] = set()
    
    def optimize(self, node: Node) -> Node:
        """Apply dead code elimination."""
        # First pass: collect used variables
        self._collect_used_variables(node)
        
        # Second pass: remove unused definitions
        return self._eliminate_dead_code(node)
    
    def _collect_used_variables(self, node: Node):
        """Collect all variable references."""
        if isinstance(node, Identifier):
            self.used_variables.add(node.name)
        elif isinstance(node, BinaryOp):
            self._collect_used_variables(node.left)
            self._collect_used_variables(node.right)
        elif isinstance(node, Assignment):
            self._collect_used_variables(node.value)
        elif isinstance(node, Declaration):
            if node.value:
                self._collect_used_variables(node.value)
            self.defined_variables.add(node.name)
        elif isinstance(node, Program):
            for stmt in node.statements:
                self._collect_used_variables(stmt)
        elif isinstance(node, LayerDef):
            for stmt in node.body:
                self._collect_used_variables(stmt)
    
    def _eliminate_dead_code(self, node: Node) -> Node:
        """Remove unused definitions."""
        if isinstance(node, Program):
            new_statements = []
            for stmt in node.statements:
                if self._is_used(stmt):
                    new_statements.append(stmt)
                else:
                    self.modified = True
            
            return Program(statements=new_statements, lineno=node.lineno)
        
        elif isinstance(node, LayerDef):
            new_body = []
            for stmt in node.body:
                if self._is_used(stmt):
                    new_body.append(stmt)
                else:
                    self.modified = True
            
            return LayerDef(
                node.name, node.params, node.return_type, new_body,
                lineno=node.lineno, col_offset=node.col_offset
            )
        
        return node
    
    def _is_used(self, stmt: Statement) -> bool:
        """Check if statement is used."""
        if isinstance(stmt, Declaration):
            return stmt.name in self.used_variables
        elif isinstance(stmt, Assignment):
            return stmt.target in self.used_variables
        else:
            return True  # Keep other statements
```

### Examples

**Before:**
```triton
let x: trit = 1
let y: trit = 0      # unused
let z: trit = -1     # unused
return x
```

**After:**
```triton
let x: trit = 1
return x
```

## 3. Ternary-Specific Optimizations

Exploits the unique properties of ternary values {-1, 0, 1}.

### Implementation

```python
class TernaryOptimizationPass(OptimizationPass):
    """Ternary-specific optimizations."""
    
    def optimize(self, node: Node) -> Node:
        """Apply ternary optimizations."""
        return node.accept(self)
    
    def visit_binary_op(self, node: BinaryOp) -> Node:
        """Optimize ternary operations."""
        # Recursively optimize operands
        left = node.left.accept(self)
        right = node.right.accept(self)
        
        # Multiplication optimizations
        if node.op == '*':
            optimized = self._optimize_multiplication(left, right, node)
            if optimized is not None:
                self.modified = True
                return optimized
        
        # Matrix multiplication optimizations
        elif node.op == '@':
            optimized = self._optimize_matmul(left, right, node)
            if optimized is not None:
                self.modified = True
                return optimized
        
        # Return with optimized operands
        if left is not node.left or right is not node.right:
            return BinaryOp(left, node.op, right, lineno=node.lineno, col_offset=node.col_offset)
        
        return node
    
    def _optimize_multiplication(self, left: Node, right: Node, original: BinaryOp) -> Optional[Node]:
        """
        Optimize multiplication with ternary constants.
        
        x * 0  -> 0
        x * 1  -> x
        x * -1 -> -x
        0 * x  -> 0
        1 * x  -> x
        -1 * x -> -x
        """
        # Check right operand
        if isinstance(right, TritLiteral):
            if right.value == 0:
                # x * 0 = 0
                return TritLiteral(0, lineno=original.lineno, col_offset=original.col_offset)
            elif right.value == 1:
                # x * 1 = x
                return left
            elif right.value == -1:
                # x * -1 = -x
                return UnaryOp('-', left, lineno=original.lineno, col_offset=original.col_offset)
        
        # Check left operand
        if isinstance(left, TritLiteral):
            if left.value == 0:
                # 0 * x = 0
                return TritLiteral(0, lineno=original.lineno, col_offset=original.col_offset)
            elif left.value == 1:
                # 1 * x = x
                return right
            elif left.value == -1:
                # -1 * x = -x
                return UnaryOp('-', right, lineno=original.lineno, col_offset=original.col_offset)
        
        return None
    
    def _optimize_matmul(self, left: Node, right: Node, original: BinaryOp) -> Optional[Node]:
        """
        Optimize matrix multiplication for ternary tensors.
        
        Replace standard matmul with optimized ternary matmul that:
        - Skips zero elements
        - Uses add/subtract for ±1
        """
        # Check if operands are ternary tensors
        if self._is_ternary_tensor(left) and self._is_ternary_tensor(right):
            # Replace with optimized function call
            return FunctionCall(
                'ternary_matmul',
                [left, right],
                lineno=original.lineno,
                col_offset=original.col_offset
            )
        
        return None
    
    def _is_ternary_tensor(self, node: Node) -> bool:
        """Check if node represents a ternary tensor."""
        # In practice, would check type information
        # For now, simple heuristic
        return isinstance(node, (TernaryTensor, Identifier))
```

### Examples

**Multiplication by Constants:**
```triton
# Before
let a: trit = x * 0
let b: trit = x * 1
let c: trit = x * -1

# After
let a: trit = 0
let b: trit = x
let c: trit = -x
```

**Ternary Matrix Multiplication:**
```triton
# Before
let result = A @ B  # where A, B are TernaryTensor

# After
let result = ternary_matmul(A, B)  # optimized kernel
```

## 4. Common Subexpression Elimination (CSE)

Identifies and eliminates redundant computations by sharing results.

### Implementation

```python
class CSEPass(OptimizationPass):
    """Common Subexpression Elimination."""
    
    def __init__(self):
        super().__init__()
        self.expr_cache: Dict[str, str] = {}  # expr_string -> variable_name
        self.temp_counter = 0
    
    def optimize(self, node: Node) -> Node:
        """Apply CSE to the AST."""
        if isinstance(node, Program):
            return self._optimize_program(node)
        elif isinstance(node, LayerDef):
            return self._optimize_layer(node)
        return node
    
    def _optimize_program(self, program: Program) -> Program:
        """Optimize program statements."""
        new_statements = []
        
        for stmt in program.statements:
            if isinstance(stmt, Assignment):
                expr_str = self._expression_to_string(stmt.value)
                
                if expr_str in self.expr_cache:
                    # Reuse previous computation
                    cached_var = self.expr_cache[expr_str]
                    new_stmt = Assignment(
                        stmt.target,
                        Identifier(cached_var),
                        lineno=stmt.lineno,
                        col_offset=stmt.col_offset
                    )
                    new_statements.append(new_stmt)
                    self.modified = True
                else:
                    # First occurrence
                    self.expr_cache[expr_str] = stmt.target
                    new_statements.append(stmt)
            else:
                new_statements.append(stmt)
        
        return Program(statements=new_statements, lineno=program.lineno)
    
    def _expression_to_string(self, expr: Expr) -> str:
        """Convert expression to canonical string representation."""
        if isinstance(expr, BinaryOp):
            left_str = self._expression_to_string(expr.left)
            right_str = self._expression_to_string(expr.right)
            return f"({left_str} {expr.op} {right_str})"
        elif isinstance(expr, Identifier):
            return expr.name
        elif isinstance(expr, (TritLiteral, IntLiteral, FloatLiteral)):
            return str(expr.value)
        elif isinstance(expr, FunctionCall):
            args_str = ", ".join(self._expression_to_string(arg) for arg in expr.args)
            return f"{expr.name}({args_str})"
        else:
            return str(id(expr))  # Fallback to object ID
```

### Examples

**Before:**
```triton
let a = x + y
let b = x + y      # duplicate expression
let c = a * 2
let d = x + y      # duplicate expression
```

**After:**
```triton
let a = x + y
let b = a          # reuse 'a'
let c = a * 2
let d = a          # reuse 'a'
```

## 5. Operation Fusion

Combines multiple operations into single kernels to reduce memory traffic and improve performance.

### Implementation

```python
class OperationFusionPass(OptimizationPass):
    """Fuse consecutive operations into combined kernels."""
    
    def optimize(self, node: Node) -> Node:
        """Apply operation fusion."""
        if isinstance(node, LayerDef):
            return self._fuse_layer_operations(node)
        return node
    
    def _fuse_layer_operations(self, layer: LayerDef) -> LayerDef:
        """Fuse operations in layer body."""
        new_body = []
        i = 0
        
        while i < len(layer.body):
            # Pattern 1: Matmul + Activation
            if (i + 1 < len(layer.body) and
                self._is_matmul_assignment(layer.body[i]) and
                self._is_activation_assignment(layer.body[i + 1], layer.body[i].target)):
                
                fused = self._fuse_matmul_activation(layer.body[i], layer.body[i + 1])
                new_body.append(fused)
                i += 2
                self.modified = True
            
            # Pattern 2: Element-wise operations
            elif (i + 1 < len(layer.body) and
                  self._is_elementwise(layer.body[i]) and
                  self._is_elementwise(layer.body[i + 1])):
                
                fused = self._fuse_elementwise(layer.body[i], layer.body[i + 1])
                new_body.append(fused)
                i += 2
                self.modified = True
            
            else:
                new_body.append(layer.body[i])
                i += 1
        
        return LayerDef(
            layer.name, layer.params, layer.return_type, new_body,
            lineno=layer.lineno, col_offset=layer.col_offset
        )
    
    def _is_matmul_assignment(self, stmt: Statement) -> bool:
        """Check if statement is matmul assignment."""
        return (isinstance(stmt, Assignment) and
                isinstance(stmt.value, BinaryOp) and
                stmt.value.op == '@')
    
    def _is_activation_assignment(self, stmt: Statement, input_var: str) -> bool:
        """Check if statement is activation on input_var."""
        return (isinstance(stmt, Assignment) and
                isinstance(stmt.value, FunctionCall) and
                stmt.value.name in {'relu', 'sigmoid', 'tanh'} and
                len(stmt.value.args) == 1 and
                isinstance(stmt.value.args[0], Identifier) and
                stmt.value.args[0].name == input_var)
    
    def _fuse_matmul_activation(self, matmul_stmt: Assignment, 
                                activation_stmt: Assignment) -> Assignment:
        """Fuse matmul and activation into single operation."""
        matmul_op = matmul_stmt.value
        activation_func = activation_stmt.value.name
        
        # Create fused function call
        fused_call = FunctionCall(
            f'fused_matmul_{activation_func}',
            [matmul_op.left, matmul_op.right],
            lineno=matmul_stmt.lineno,
            col_offset=matmul_stmt.col_offset
        )
        
        return Assignment(
            activation_stmt.target,
            fused_call,
            lineno=activation_stmt.lineno,
            col_offset=activation_stmt.col_offset
        )
```

### Examples

**Before:**
```triton
let y = x @ weight
let z = relu(y)
```

**After:**
```triton
let z = fused_matmul_relu(x, weight)
```

## 6. Memory Optimization

Reduces memory usage through in-place operations and buffer reuse.

### Implementation

```python
class MemoryOptimizationPass(OptimizationPass):
    """Optimize memory usage."""
    
    def __init__(self):
        super().__init__()
        self.live_ranges: Dict[str, Tuple[int, int]] = {}
        self.buffer_reuse: Dict[str, str] = {}
    
    def optimize(self, node: Node) -> Node:
        """Apply memory optimizations."""
        # Analyze variable lifetimes
        self._analyze_live_ranges(node)
        
        # Find opportunities for buffer reuse
        self._find_buffer_reuse_opportunities()
        
        # Apply optimizations
        return self._apply_memory_opts(node)
    
    def _analyze_live_ranges(self, node: Node):
        """Determine when each variable is live."""
        # Implementation would track first definition and last use
        pass
    
    def _find_buffer_reuse_opportunities(self):
        """Find variables that can share buffers."""
        # If variable A's live range ends before B starts, B can reuse A's buffer
        variables = list(self.live_ranges.keys())
        
        for i, var_a in enumerate(variables):
            for var_b in variables[i+1:]:
                range_a = self.live_ranges[var_a]
                range_b = self.live_ranges[var_b]
                
                # Non-overlapping ranges - can reuse buffer
                if range_a[1] < range_b[0]:
                    self.buffer_reuse[var_b] = var_a
                    break
```

## Custom Pass Development

### Creating a Custom Pass

```python
class MyCustomPass(OptimizationPass):
    """Custom optimization pass."""
    
    def __init__(self, custom_config=None):
        super().__init__()
        self.config = custom_config or {}
    
    def optimize(self, node: Node) -> Node:
        """Apply custom optimization logic."""
        # Implement your optimization here
        return node.accept(self)
    
    def visit_binary_op(self, node: BinaryOp) -> Node:
        """Custom binary operation handling."""
        # Your optimization logic
        pass
    
    def visit_layer_def(self, node: LayerDef) -> Node:
        """Custom layer optimization."""
        # Your optimization logic
        pass
```

### Registering a Custom Pass

```python
# Create pass instance
custom_pass = MyCustomPass(custom_config={'threshold': 0.5})

# Add to pass manager
pass_manager = PassManager()
pass_manager.add_pass(ConstantFoldingPass())
pass_manager.add_pass(custom_pass)  # Add custom pass
pass_manager.add_pass(DeadCodeEliminationPass())

# Run optimization pipeline
optimized_ast = pass_manager.run(ast)
```

## Pass Ordering

The order of optimization passes affects the final result:

### Standard Pass Order

```python
def create_standard_pipeline() -> PassManager:
    """Create standard optimization pipeline."""
    manager = PassManager()
    
    # Phase 1: Simplification
    manager.add_pass(ConstantFoldingPass())
    manager.add_pass(TernaryOptimizationPass())
    
    # Phase 2: Redundancy Elimination
    manager.add_pass(CSEPass())
    manager.add_pass(DeadCodeEliminationPass())
    
    # Phase 3: High-level Fusion
    manager.add_pass(OperationFusionPass())
    
    # Phase 4: Low-level Optimization
    manager.add_pass(MemoryOptimizationPass())
    
    return manager
```

### Aggressive Optimization

```python
def create_aggressive_pipeline() -> PassManager:
    """Create aggressive optimization pipeline."""
    manager = PassManager()
    
    # Multiple iterations of key passes
    for _ in range(3):
        manager.add_pass(ConstantFoldingPass())
        manager.add_pass(TernaryOptimizationPass())
        manager.add_pass(CSEPass())
        manager.add_pass(DeadCodeEliminationPass())
    
    manager.add_pass(OperationFusionPass())
    manager.add_pass(MemoryOptimizationPass())
    
    return manager
```

### Debug-Friendly Pipeline

```python
def create_debug_pipeline() -> PassManager:
    """Create pipeline optimized for debugging."""
    manager = PassManager()
    
    # Minimal optimizations that preserve code structure
    manager.add_pass(ConstantFoldingPass())
    manager.add_pass(DeadCodeEliminationPass())
    
    # Skip aggressive optimizations that make debugging harder
    # Skip: CSE, Operation Fusion, Memory Optimization
    
    return manager
```

## Performance Analysis

### Measuring Pass Impact

```python
def measure_pass_impact(ast: Node, pass_obj: OptimizationPass) -> Dict[str, Any]:
    """Measure the impact of an optimization pass."""
    import time
    
    # Count nodes before
    nodes_before = count_ast_nodes(ast)
    
    # Run optimization
    start_time = time.time()
    optimized_ast = pass_obj.optimize(ast)
    elapsed_time = time.time() - start_time
    
    # Count nodes after
    nodes_after = count_ast_nodes(optimized_ast)
    
    return {
        'pass_name': pass_obj.pass_name,
        'nodes_before': nodes_before,
        'nodes_after': nodes_after,
        'nodes_removed': nodes_before - nodes_after,
        'reduction_pct': 100 * (nodes_before - nodes_after) / nodes_before,
        'time_seconds': elapsed_time,
        'modified': pass_obj.was_modified()
    }

def count_ast_nodes(node: Node) -> int:
    """Count total nodes in AST."""
    count = 1
    for attr in node.__dict__.values():
        if isinstance(attr, Node):
            count += count_ast_nodes(attr)
        elif isinstance(attr, list):
            for item in attr:
                if isinstance(item, Node):
                    count += count_ast_nodes(item)
    return count
```

## Best Practices

### For Pass Developers

1. **Preserve Semantics:** Ensure optimizations don't change program behavior
2. **Mark Modifications:** Set `self.modified = True` when AST changes
3. **Handle All Node Types:** Implement visitor methods for all relevant nodes
4. **Test Thoroughly:** Test on edge cases and complex programs
5. **Document Transformations:** Clearly document what each pass does
6. **Make Passes Idempotent:** Running twice should be same as running once
7. **Profile Performance:** Measure pass execution time

### For Pipeline Designers

1. **Order Matters:** Place enabling passes before dependent passes
2. **Iterate Until Fixpoint:** Some transformations enable others
3. **Balance Compile Time:** Don't over-optimize compilation speed
4. **Provide Presets:** Offer standard, aggressive, and debug modes
5. **Allow Customization:** Let users add custom passes
6. **Collect Statistics:** Track which passes are most effective

## Summary

The Triton DSL optimization infrastructure provides a flexible, extensible framework for implementing compiler optimizations. Through the pass manager architecture, standard optimizations (constant folding, DCE, CSE), and ternary-specific optimizations, the compiler generates highly efficient code while maintaining program semantics. Custom passes can be easily integrated to support domain-specific optimizations.

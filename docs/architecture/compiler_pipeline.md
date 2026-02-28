# Triton DSL Compiler Pipeline

## Overview

The Triton DSL compiler transforms high-level ternary neural network descriptions into optimized executable code for various backends (PyTorch, Triton GPU, etc.). This document describes the complete compilation pipeline from source text to target code.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Triton DSL Source Code                        │
│  layer MyTernaryLayer(x: tensor<trit, [128]>) -> tensor<trit, [64]> │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LEXER (Tokenization)                            │
│  • PLY-based lexical analyzer                                        │
│  • Converts source text to token stream                              │
│  • Recognizes keywords, operators, literals                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ Token Stream
┌─────────────────────────────────────────────────────────────────────┐
│                      PARSER (Syntax Analysis)                        │
│  • LALR parser using PLY yacc                                        │
│  • Builds Abstract Syntax Tree (AST)                                 │
│  • Validates syntax rules                                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ AST
┌─────────────────────────────────────────────────────────────────────┐
│                   TYPE CHECKER (Semantic Analysis)                   │
│  • Validates type correctness                                        │
│  • Type inference for expressions                                    │
│  • Symbol table management                                           │
│  • Error detection and reporting                                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ Typed AST
┌─────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION PASSES (Optional)                    │
│  • Constant folding                                                  │
│  • Dead code elimination                                             │
│  • Ternary-specific optimizations                                    │
│  • Operation fusion                                                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼ Optimized AST
┌─────────────────────────────────────────────────────────────────────┐
│                    CODE GENERATOR (Code Emission)                    │
│  • Backend selection (PyTorch, Triton, etc.)                         │
│  • Template-based code generation                                    │
│  • Target-specific optimizations                                     │
│  • Output executable code                                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Target Code (PyTorch/CUDA)                     │
│  class MyTernaryLayer(nn.Module): ...                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Lexer (Tokenization)

### Purpose
The lexer converts raw source code into a stream of tokens, handling low-level details like whitespace, comments, and literal values.

### Implementation
Located in `compiler/lexer/triton_lexer.py`, the lexer uses PLY (Python Lex-Yacc) for tokenization.

### Token Categories

**Keywords:**
- Type keywords: `trit`, `int8`, `int32`, `float16`, `float32`, `tensor`, `TernaryTensor`
- Control keywords: `layer`, `let`, `fn`, `return`

**Operators:**
- Arithmetic: `+`, `-`, `*`, `@` (matmul)
- Assignment: `=`
- Type operators: `->`, `:`

**Literals:**
- Trit literals: `-1`, `0`, `1` (special handling for ternary values)
- Integer literals: sequences of digits
- Float literals: decimal numbers with point

**Delimiters:**
- Parentheses: `(`, `)`
- Braces: `{`, `}`
- Brackets: `[`, `]`
- Others: `,`, `;`, `...`

### Example Tokenization

```triton
let x: trit = 1
```

Token stream:
```
LET          "let"
IDENTIFIER   "x"
COLON        ":"
TRIT         "trit"
ASSIGN       "="
TRIT_LITERAL 1
```

### Special Handling

**Trit Literal Detection:**
The lexer specially handles `-1`, `0`, and `1` as `TRIT_LITERAL` tokens when they appear as standalone values, distinguishing them from general integers.

```python
def t_MINUS(t):
    r'-'
    next_pos = t.lexpos + 1
    if next_pos < len(t.lexer.lexdata) and t.lexer.lexdata[next_pos] == '1':
        # This is -1, consume the '1' as well
        t.value = '-1'
        t.type = 'TRIT_LITERAL'
        t.lexer.skip(1)
        t.value = -1
        return t
    return t
```

## Stage 2: Parser (Syntax Analysis)

### Purpose
The parser constructs an Abstract Syntax Tree (AST) from the token stream, validating syntactic correctness and building the hierarchical program structure.

### Grammar Overview

The Triton DSL uses an LALR(1) grammar defined with PLY yacc. Key production rules:

```
program         → statement_list

statement       → declaration
                | assignment
                | layer_def
                | return_stmt

declaration     → let IDENTIFIER : type = expression
                | let IDENTIFIER : type

layer_def       → layer IDENTIFIER ( params ) -> type { statement_list }

expression      → expression + expression
                | expression * expression
                | expression @ expression  (matrix multiply)
                | IDENTIFIER
                | literal
                | function_call

type            → trit | int8 | int32 | float16 | float32
                | TernaryTensor
                | tensor<type, [dimensions]>
```

### Operator Precedence

```python
precedence = (
    ("left", "PLUS", "MINUS"),      # Lower precedence
    ("left", "STAR", "MATMUL"),     # Higher precedence
    ("right", "UMINUS"),            # Unary minus (highest)
)
```

### AST Node Structure

All AST nodes inherit from `Node` base class with location tracking:

```python
@dataclass
class Node:
    lineno: int = field(default=0, kw_only=True)
    col_offset: int = field(default=0, kw_only=True)
    
    def accept(self, visitor: "Visitor") -> Any:
        """Accept a visitor for the visitor pattern."""
        return None
```

**Type Nodes:**
- `TritType`: Ternary type {-1, 0, 1}
- `IntType`: Integer types (8, 32 bits)
- `FloatType`: Floating point types (16, 32 bits)
- `TensorType`: Multi-dimensional arrays with element type

**Expression Nodes:**
- `TritLiteral`: Ternary value literal
- `IntLiteral`, `FloatLiteral`: Numeric literals
- `Identifier`: Variable reference
- `BinaryOp`: Binary operations (+, -, *, @)
- `UnaryOp`: Unary operations (-, not)
- `FunctionCall`: Function invocation
- `TernaryTensor`: Tensor literal with shape and values

**Statement Nodes:**
- `Assignment`: Variable assignment
- `Declaration`: Variable declaration with type
- `Return`: Return statement
- `LayerDef`: Layer/function definition

### Example AST

Source:
```triton
let x: trit = 1
```

AST:
```
Program(
  statements=[
    Declaration(
      name="x",
      type_annotation=TritType(),
      value=TritLiteral(value=1),
      lineno=1,
      col_offset=0
    )
  ]
)
```

### Error Recovery

The parser implements error recovery to continue parsing after syntax errors:

```python
def p_error(p):
    """Error handling with recovery."""
    if p:
        sys.stderr.write(f"Syntax error at token {p.type} ('{p.value}') at line {p.lineno}\n")
        parser.errok()  # Reset error state
    else:
        sys.stderr.write("Syntax error at EOF\n")
```

## Stage 3: Type Checker (Semantic Analysis)

### Purpose
The type checker validates semantic correctness, ensuring operations are type-safe and variables are properly declared.

### Type System

**Base Types:**
- `trit`: Ternary values {-1, 0, 1}
- `int8`, `int32`: Signed integers
- `float16`, `float32`: IEEE floating point
- `tensor<T, shape>`: Multi-dimensional arrays

**Type Inference:**
The type checker infers expression types through AST traversal:

```python
def infer_type(self, node: Expr) -> Optional[Type]:
    """Infer the type of an expression."""
    return node.accept(self)
```

### Validation Rules

**Trit Value Constraints:**
```python
def visit_trit_literal(self, node: TritLiteral) -> Any:
    if node.value not in {-1, 0, 1}:
        self.add_error(
            f"Trit literal must be -1, 0, or 1, got {node.value}",
            node
        )
    return TritType()
```

**Binary Operation Type Compatibility:**
```python
def visit_binary_op(self, node: BinaryOp) -> Optional[Type]:
    left_type = self.infer_type(node.left)
    right_type = self.infer_type(node.right)
    
    if not self.types_compatible(left_type, right_type):
        self.add_error(
            f"Binary operation '{node.op}' requires compatible types",
            node
        )
```

**Matrix Multiplication Dimension Checking:**
```python
# For 2D: (m, n) @ (n, p) -> (m, p)
if left_type.shape[-1] != right_type.shape[-2]:
    self.add_error(
        f"Matrix multiplication dimension mismatch: "
        f"{left_type.shape} cannot multiply with {right_type.shape}",
        node
    )
```

### Symbol Table Management

The type checker maintains symbol tables for:
- **Variables**: Maps identifiers to their types
- **Functions**: Maps function names to (param_types, return_type)

```python
class TypeChecker(Visitor):
    def __init__(self) -> None:
        self.errors: List[TypeError] = []
        self.symbol_table: Dict[str, Type] = {}
        self.function_table: Dict[str, Tuple[List[Type], Optional[Type]]] = {}
```

### Error Reporting

Type errors include location information for precise diagnostics:

```python
@dataclass
class TypeError:
    message: str
    lineno: int = 0
    col_offset: int = 0
    
    def __str__(self) -> str:
        return f"Line {lineno}, Col {col_offset}: {message}"
```

## Stage 4: Optimization Passes (Optional)

### Purpose
Optimize the AST before code generation to improve runtime performance and reduce code size.

### Pass Categories

**1. Constant Folding**
Evaluate constant expressions at compile time:
```triton
let x: int32 = 2 + 3  →  let x: int32 = 5
```

**2. Dead Code Elimination**
Remove unreachable or unused code:
```triton
let x: trit = 1
let y: trit = 0  # unused
return x
```
→ Remove `y` declaration

**3. Ternary-Specific Optimizations**
- Multiplication by 0 → 0
- Multiplication by 1 → identity
- Multiplication by -1 → negation

```triton
x * 0  →  0
x * 1  →  x
x * -1 →  -x
```

**4. Operation Fusion**
Combine multiple operations into single kernels for GPU backends:
```triton
y = x @ w
z = activation(y)
```
→ Fused matmul+activation kernel

### Pass Infrastructure

Optimization passes implement the `Visitor` pattern:

```python
class OptimizationPass(Visitor):
    def optimize(self, ast: Node) -> Node:
        """Apply optimization and return transformed AST."""
        return ast.accept(self)
```

Multiple passes can be chained:
```python
def optimize_ast(ast: Node) -> Node:
    ast = ConstantFoldingPass().optimize(ast)
    ast = DeadCodeEliminationPass().optimize(ast)
    ast = TernaryOptimizationPass().optimize(ast)
    return ast
```

## Stage 5: Code Generator

### Purpose
Convert the optimized AST into executable code for a target backend (PyTorch, Triton GPU, etc.).

### Backend Architecture

```
                    ┌─────────────────────┐
                    │   Code Generator    │
                    │   (Abstract Base)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
    │ PyTorch Backend │ │ Triton Backend│ │Custom Backend│
    │  (Jinja2 templ) │ │ (GPU kernels) │ │ (User-def)   │
    └─────────────────┘ └──────────────┘ └──────────────┘
```

### Template-Based Generation

The PyTorch backend uses Jinja2 templates for code generation:

```python
class PyTorchCodeGenerator:
    def __init__(self):
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def generate_module(self, layer_def: LayerDef) -> str:
        template = self.env.get_template('module.py.jinja')
        return template.render(
            class_name=layer_def.name,
            parameters=self._extract_parameters(layer_def),
            forward_body=self._generate_forward_body(layer_def)
        )
```

### Code Generation Example

**Input AST:**
```python
LayerDef(
    name="TernaryLinear",
    params=[Param("x", TensorType(TritType(), [128]))],
    return_type=TensorType(TritType(), [64]),
    body=[...]
)
```

**Generated PyTorch Code:**
```python
import torch
import torch.nn as nn

class TernaryLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 128))
        self._ternarize_weights()
    
    def _ternarize_weights(self):
        with torch.no_grad():
            self.weight.data = torch.sign(self.weight.data)
    
    def forward(self, x):
        return torch.matmul(x, self.weight.t())
```

## Data Flow and IR Representations

### Token Stream (Lexer Output)
```
Token(type='LET', value='let', lineno=1, pos=0)
Token(type='IDENTIFIER', value='x', lineno=1, pos=4)
...
```

### Abstract Syntax Tree (Parser Output)
```
Program
└── Declaration
    ├── name: "x"
    ├── type: TritType()
    └── value: TritLiteral(1)
```

### Typed AST (Type Checker Output)
Same structure as AST but with validated type annotations and symbol table information.

### Optimized AST (Optimization Output)
Transformed AST with constant expressions evaluated, dead code removed, and operations fused.

### Target Code (Code Generator Output)
Executable code in target language (Python, CUDA, etc.)

## Pipeline Control Flow

### Basic Compilation

```python
def compile_triton(source_code: str, backend='pytorch') -> str:
    # Stage 1: Tokenization
    lexer = TernaryLexer()
    lexer.input(source_code)
    
    # Stage 2: Parsing
    from compiler.parser.triton_parser import parse
    ast = parse(source_code)
    
    # Stage 3: Type Checking
    type_checker = TypeChecker()
    errors = type_checker.validate(ast)
    if errors:
        raise CompilationError(errors)
    
    # Stage 4: Optimization (optional)
    if optimize:
        ast = optimize_ast(ast)
    
    # Stage 5: Code Generation
    if backend == 'pytorch':
        generator = PyTorchCodeGenerator()
        return generator.generate_module(ast)
    elif backend == 'triton':
        generator = TritonGPUGenerator()
        return generator.generate_kernel(ast)
```

### Error Handling

Each stage can report errors with location information:

```python
try:
    ast = parse(source_code)
except SyntaxError as e:
    print(f"Syntax error at line {e.lineno}: {e.message}")

errors = type_checker.validate(ast)
for error in errors:
    print(f"Type error at line {error.lineno}: {error.message}")
```

## Performance Considerations

### Compilation Speed

**Lexer:** O(n) where n is source code length
**Parser:** O(n) for LALR parsing
**Type Checker:** O(nodes) single AST traversal
**Optimizer:** O(nodes * passes) multiple traversals
**Code Generator:** O(nodes) template instantiation

### Memory Usage

- AST nodes: ~100-200 bytes per node
- Symbol table: ~50 bytes per entry
- Token stream: discarded after parsing
- Generated code: string in memory before file write

### Optimization Opportunities

1. **Incremental Compilation:** Cache parsed ASTs, only recompile changed modules
2. **Parallel Type Checking:** Check independent functions in parallel
3. **JIT Compilation:** Generate code on-demand at runtime
4. **AOT Compilation:** Pre-compile common patterns

## Extension Points

The compiler pipeline can be extended at multiple points:

1. **Custom Lexer Tokens:** Add new keywords or operators
2. **Grammar Extensions:** Add new language constructs
3. **Custom Type Rules:** Define new type checking constraints
4. **Optimization Passes:** Add custom optimization passes
5. **Backend Targets:** Implement new code generation backends

## Best Practices

### For Compiler Developers

1. **Preserve Location Information:** Always include `lineno` and `col_offset`
2. **Use Visitor Pattern:** Keep traversal logic separate from node definitions
3. **Immutable AST:** Prefer creating new nodes over modifying existing ones
4. **Comprehensive Testing:** Test each stage independently
5. **Clear Error Messages:** Provide actionable error messages with context

### For DSL Users

1. **Type Annotations:** Always annotate variable types for clarity
2. **Explicit Shapes:** Specify tensor shapes when possible
3. **Modular Code:** Break large layers into smaller functions
4. **Comments:** Document complex ternary logic

## Debugging Tools

### AST Visualization

```python
def print_ast(node: Node, indent=0):
    """Pretty-print AST structure."""
    print("  " * indent + node.__class__.__name__)
    for field, value in node.__dict__.items():
        if isinstance(value, Node):
            print("  " * (indent + 1) + f"{field}:")
            print_ast(value, indent + 2)
        elif isinstance(value, list) and value and isinstance(value[0], Node):
            print("  " * (indent + 1) + f"{field}:")
            for item in value:
                print_ast(item, indent + 2)
```

### Compilation Trace

```python
def compile_with_trace(source_code: str):
    print("=== LEXER ===")
    tokens = list(lex_source(source_code))
    print(f"Generated {len(tokens)} tokens")
    
    print("\n=== PARSER ===")
    ast = parse(source_code)
    print(f"AST has {count_nodes(ast)} nodes")
    
    print("\n=== TYPE CHECKER ===")
    errors = type_check(ast)
    print(f"Found {len(errors)} type errors")
    
    print("\n=== CODE GENERATOR ===")
    code = generate_code(ast)
    print(f"Generated {len(code)} characters of code")
```

## Summary

The Triton DSL compiler pipeline transforms high-level ternary neural network descriptions into efficient executable code through five stages: lexical analysis, parsing, type checking, optimization, and code generation. Each stage is designed to be modular, extensible, and testable, enabling robust compilation of ternary computing primitives for various hardware backends.

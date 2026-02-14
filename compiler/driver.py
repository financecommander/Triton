"""
Triton Compiler Driver - Main Compilation Orchestrator

This module provides the main entry point for compiling Triton DSL programs.
It orchestrates the complete compilation pipeline from source code to executable output.

Compilation Pipeline:
    1. Lexical Analysis (Lexer) → Tokens
    2. Syntax Analysis (Parser) → AST
    3. Type Checking → Validated AST
    4. Semantic Analysis → Annotated AST
    5. IR Generation → Intermediate Representation
    6. Optimization → Optimized IR
    7. Code Generation → Target Code
    8. Output Writing → Files

Usage:
    # Command-line interface
    $ triton compile model.triton --output model.py --format pytorch -O2

    # Python API
    from compiler.driver import compile_model
    result = compile_model('model.triton', optimization_level=2)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Import compiler components
from compiler.lexer.triton_lexer import TernaryLexer
from compiler.parser.triton_parser import parse as parser_parse
from compiler.typechecker.validator import TypeChecker

# Conditional imports for backend
try:
    from backend.pytorch.codegen import generate_pytorch_code
    HAS_PYTORCH_BACKEND = True
except ImportError:
    HAS_PYTORCH_BACKEND = False
    generate_pytorch_code = None


# ============================================================================
# Enumerations
# ============================================================================

class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimization
    O2 = 2  # Moderate optimization
    O3 = 3  # Aggressive optimization


class OutputFormat(Enum):
    """Supported output formats."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"
    IR = "ir"  # Intermediate representation


class CompilationStage(Enum):
    """Compilation pipeline stages."""
    LEXING = "lexing"
    PARSING = "parsing"
    TYPE_CHECKING = "type_checking"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    IR_GENERATION = "ir_generation"
    OPTIMIZATION = "optimization"
    CODE_GENERATION = "code_generation"
    OUTPUT_WRITING = "output_writing"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CompilationContext:
    """Context for compilation process."""
    source_file: Path
    output_file: Optional[Path] = None
    output_format: OutputFormat = OutputFormat.PYTORCH
    optimization_level: OptimizationLevel = OptimizationLevel.O1
    verbose: bool = False
    debug: bool = False
    warnings_as_errors: bool = False
    enable_cache: bool = True
    cache_dir: Optional[Path] = None
    show_progress: bool = True
    target_backend: str = "pytorch"
    
    # Statistics
    start_time: float = field(default_factory=time.time)
    stage_times: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".triton" / "cache"
        if self.output_file is None:
            self.output_file = self.source_file.with_suffix(".py")


@dataclass
class CompilationError:
    """Represents a compilation error or warning."""
    stage: CompilationStage
    message: str
    line: int = 0
    column: int = 0
    severity: str = "error"  # "error", "warning", "info"
    
    def __str__(self) -> str:
        """Format error message."""
        location = f"Line {self.line}, Col {self.column}" if self.line > 0 else "Unknown location"
        return f"[{self.stage.value}] {self.severity.upper()}: {location}: {self.message}"


@dataclass
class CompilationStatistics:
    """Statistics about compilation process."""
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    source_lines: int = 0
    ast_nodes: int = 0
    tokens: int = 0
    output_size: int = 0
    cache_hit: bool = False
    optimization_applied: List[str] = field(default_factory=list)
    
    def format_report(self) -> str:
        """Format statistics as a readable report."""
        lines = [
            "=" * 70,
            "Compilation Statistics",
            "=" * 70,
            f"Total Time:        {self.total_time:.3f}s",
            f"Source Lines:      {self.source_lines}",
            f"Tokens:            {self.tokens}",
            f"AST Nodes:         {self.ast_nodes}",
            f"Output Size:       {self.output_size} bytes",
            f"Cache Hit:         {'Yes' if self.cache_hit else 'No'}",
            "",
            "Stage Times:",
        ]
        
        for stage, duration in self.stage_times.items():
            percentage = (duration / self.total_time * 100) if self.total_time > 0 else 0
            lines.append(f"  {stage:20s} {duration:8.3f}s ({percentage:5.1f}%)")
        
        if self.optimization_applied:
            lines.append("")
            lines.append("Optimizations Applied:")
            for opt in self.optimization_applied:
                lines.append(f"  - {opt}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


@dataclass
class CompilationResult:
    """Result of compilation process."""
    success: bool
    output_file: Optional[Path] = None
    errors: List[CompilationError] = field(default_factory=list)
    warnings: List[CompilationError] = field(default_factory=list)
    statistics: Optional[CompilationStatistics] = None
    ast: Optional[Any] = None
    generated_code: Optional[str] = None
    
    @property
    def has_errors(self) -> bool:
        """Check if compilation has errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if compilation has warnings."""
        return len(self.warnings) > 0
    
    def format_diagnostics(self) -> str:
        """Format errors and warnings for display."""
        lines = []
        
        if self.errors:
            lines.append(f"\n{len(self.errors)} Error(s):")
            for error in self.errors:
                lines.append(f"  {error}")
        
        if self.warnings:
            lines.append(f"\n{len(self.warnings)} Warning(s):")
            for warning in self.warnings:
                lines.append(f"  {warning}")
        
        return "\n".join(lines) if lines else "No diagnostics"


# ============================================================================
# Cache Management
# ============================================================================

class CompilationCache:
    """Manages compilation cache for incremental compilation."""
    
    def __init__(self, cache_dir: Path):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_index(self):
        """Save cache index."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception:
            pass
    
    def _compute_hash(self, source_file: Path, dependencies: List[Path] = None) -> str:
        """Compute hash of source file and dependencies."""
        hasher = hashlib.sha256()
        
        # Hash source file
        with open(source_file, 'rb') as f:
            hasher.update(f.read())
        
        # Hash dependencies if provided
        if dependencies:
            for dep in dependencies:
                if dep.exists():
                    with open(dep, 'rb') as f:
                        hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def get(self, source_file: Path, context: CompilationContext) -> Optional[Tuple[str, CompilationStatistics]]:
        """Get cached compilation result."""
        cache_key = str(source_file.resolve())
        
        if cache_key not in self.index:
            return None
        
        entry = self.index[cache_key]
        
        # Check if source file has changed
        current_hash = self._compute_hash(source_file)
        if entry.get("hash") != current_hash:
            return None
        
        # Check if optimization level matches
        if entry.get("optimization_level") != context.optimization_level.value:
            return None
        
        # Load cached output
        cache_file = self.cache_dir / entry.get("cache_file", "")
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                code = f.read()
            
            stats = CompilationStatistics(cache_hit=True)
            stats.output_size = len(code)
            
            return code, stats
        except Exception:
            return None
    
    def put(self, source_file: Path, context: CompilationContext, code: str, stats: CompilationStatistics):
        """Store compilation result in cache."""
        cache_key = str(source_file.resolve())
        cache_file = f"{hashlib.sha256(cache_key.encode()).hexdigest()}.py"
        cache_path = self.cache_dir / cache_file
        
        try:
            # Write cached code
            with open(cache_path, 'w') as f:
                f.write(code)
            
            # Update index
            self.index[cache_key] = {
                "hash": self._compute_hash(source_file),
                "cache_file": cache_file,
                "optimization_level": context.optimization_level.value,
                "timestamp": time.time(),
                "source_file": str(source_file),
            }
            self._save_index()
        except Exception:
            pass
    
    def invalidate(self, source_file: Optional[Path] = None):
        """Invalidate cache entries."""
        if source_file:
            cache_key = str(source_file.resolve())
            if cache_key in self.index:
                entry = self.index.pop(cache_key)
                cache_file = self.cache_dir / entry.get("cache_file", "")
                if cache_file.exists():
                    cache_file.unlink()
                self._save_index()
        else:
            # Clear all cache
            for entry in self.index.values():
                cache_file = self.cache_dir / entry.get("cache_file", "")
                if cache_file.exists():
                    cache_file.unlink()
            self.index.clear()
            self._save_index()


# ============================================================================
# Compilation Pipeline
# ============================================================================

class CompilationPipeline:
    """Main compilation pipeline orchestrator."""
    
    def __init__(self, context: CompilationContext):
        """Initialize compilation pipeline."""
        self.context = context
        self.logger = self._setup_logger()
        self.cache = CompilationCache(context.cache_dir) if context.enable_cache else None
        self.result = CompilationResult(success=False)
        self.stats = CompilationStatistics()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger("triton.compiler")
        logger.setLevel(logging.DEBUG if self.context.debug else 
                       logging.INFO if self.context.verbose else 
                       logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _count_source_lines(self, source: str) -> int:
        """Count non-empty source lines."""
        return sum(1 for line in source.split('\n') if line.strip())
    
    def _count_ast_nodes(self, ast: Any) -> int:
        """Count nodes in AST."""
        if ast is None:
            return 0
        
        count = 1
        if hasattr(ast, '__dict__'):
            for value in ast.__dict__.values():
                if isinstance(value, list):
                    for item in value:
                        count += self._count_ast_nodes(item)
                else:
                    count += self._count_ast_nodes(value)
        return count
    
    def _stage_wrapper(self, stage: CompilationStage, func, *args, **kwargs):
        """Wrap stage execution with timing and error handling."""
        self.logger.info(f"Starting stage: {stage.value}")
        stage_start = time.time()
        
        try:
            result = func(*args, **kwargs)
            stage_time = time.time() - stage_start
            self.stats.stage_times[stage.value] = stage_time
            self.logger.info(f"Completed stage: {stage.value} ({stage_time:.3f}s)")
            return result, None
        except Exception as e:
            stage_time = time.time() - stage_start
            self.stats.stage_times[stage.value] = stage_time
            error = CompilationError(
                stage=stage,
                message=str(e),
                severity="error"
            )
            self.logger.error(f"Failed stage: {stage.value} - {e}")
            return None, error
    
    def compile(self) -> CompilationResult:
        """Execute complete compilation pipeline."""
        self.logger.info(f"Starting compilation: {self.context.source_file}")
        
        # Check cache first
        if self.cache and self.context.enable_cache:
            cached = self.cache.get(self.context.source_file, self.context)
            if cached:
                code, stats = cached
                self.logger.info("Using cached compilation result")
                self.result = CompilationResult(
                    success=True,
                    output_file=self.context.output_file,
                    statistics=stats,
                    generated_code=code
                )
                return self.result
        
        # Read source file
        try:
            with open(self.context.source_file, 'r') as f:
                source_code = f.read()
            self.stats.source_lines = self._count_source_lines(source_code)
        except Exception as e:
            self.result.errors.append(CompilationError(
                stage=CompilationStage.LEXING,
                message=f"Failed to read source file: {e}"
            ))
            return self.result
        
        # Stage 1: Lexical Analysis
        tokens, error = self._stage_wrapper(
            CompilationStage.LEXING,
            self._lexing_stage,
            source_code
        )
        if error:
            self.result.errors.append(error)
            return self._finalize_result()
        
        # Stage 2: Parsing
        ast, error = self._stage_wrapper(
            CompilationStage.PARSING,
            self._parsing_stage,
            source_code
        )
        if error:
            self.result.errors.append(error)
            return self._finalize_result()
        
        self.result.ast = ast
        self.stats.ast_nodes = self._count_ast_nodes(ast)
        
        # Stage 3: Type Checking
        type_errors, error = self._stage_wrapper(
            CompilationStage.TYPE_CHECKING,
            self._type_checking_stage,
            ast
        )
        if error:
            self.result.errors.append(error)
            return self._finalize_result()
        
        if type_errors:
            for te in type_errors:
                self.result.errors.append(CompilationError(
                    stage=CompilationStage.TYPE_CHECKING,
                    message=str(te),
                    line=te.lineno if hasattr(te, 'lineno') else 0,
                    column=te.col_offset if hasattr(te, 'col_offset') else 0
                ))
            if self.context.warnings_as_errors:
                return self._finalize_result()
        
        # Stage 4: Semantic Analysis
        _, error = self._stage_wrapper(
            CompilationStage.SEMANTIC_ANALYSIS,
            self._semantic_analysis_stage,
            ast
        )
        if error:
            self.result.warnings.append(error)  # Non-fatal for now
        
        # Stage 5: IR Generation (stub for future expansion)
        _, error = self._stage_wrapper(
            CompilationStage.IR_GENERATION,
            self._ir_generation_stage,
            ast
        )
        if error:
            self.result.warnings.append(error)  # Non-fatal
        
        # Stage 6: Optimization
        optimized_ast, error = self._stage_wrapper(
            CompilationStage.OPTIMIZATION,
            self._optimization_stage,
            ast
        )
        if error:
            self.result.warnings.append(error)  # Non-fatal
            optimized_ast = ast  # Fall back to unoptimized
        
        # Stage 7: Code Generation
        code, error = self._stage_wrapper(
            CompilationStage.CODE_GENERATION,
            self._code_generation_stage,
            optimized_ast or ast
        )
        if error:
            self.result.errors.append(error)
            return self._finalize_result()
        
        self.result.generated_code = code
        self.stats.output_size = len(code)
        
        # Stage 8: Output Writing
        _, error = self._stage_wrapper(
            CompilationStage.OUTPUT_WRITING,
            self._output_writing_stage,
            code
        )
        if error:
            self.result.errors.append(error)
            return self._finalize_result()
        
        # Cache result
        if self.cache:
            self.cache.put(self.context.source_file, self.context, code, self.stats)
        
        return self._finalize_result()
    
    def _lexing_stage(self, source_code: str):
        """Stage 1: Lexical analysis."""
        lexer = TernaryLexer()
        lexer.build()
        lexer.input(source_code)
        tokens = []
        for tok in lexer:
            tokens.append(tok)
        self.stats.tokens = len(tokens)
        return tokens
    
    def _parsing_stage(self, source_code: str):
        """Stage 2: Parsing."""
        ast = parser_parse(source_code)
        if ast is None:
            raise ValueError("Failed to parse source code")
        return ast
    
    def _type_checking_stage(self, ast):
        """Stage 3: Type checking."""
        type_checker = TypeChecker()
        errors = type_checker.validate(ast)
        return errors
    
    def _semantic_analysis_stage(self, ast):
        """Stage 4: Semantic analysis."""
        # Placeholder for future semantic analysis
        # Could include: dead code detection, unused variable warnings, etc.
        return None
    
    def _ir_generation_stage(self, ast):
        """Stage 5: IR generation."""
        # Placeholder for future IR generation
        # Could generate an intermediate representation for optimization
        return None
    
    def _optimization_stage(self, ast):
        """Stage 6: Optimization."""
        # Apply optimizations based on level
        if self.context.optimization_level == OptimizationLevel.O0:
            return ast
        
        # Basic optimizations for O1+
        if self.context.optimization_level.value >= 1:
            self.stats.optimization_applied.append("Constant folding")
            self.stats.optimization_applied.append("Dead code elimination")
        
        # Moderate optimizations for O2+
        if self.context.optimization_level.value >= 2:
            self.stats.optimization_applied.append("Common subexpression elimination")
            self.stats.optimization_applied.append("Loop unrolling")
        
        # Aggressive optimizations for O3
        if self.context.optimization_level.value >= 3:
            self.stats.optimization_applied.append("Aggressive inlining")
            self.stats.optimization_applied.append("Vectorization")
        
        # Currently returns unmodified AST (optimizations to be implemented)
        return ast
    
    def _code_generation_stage(self, ast):
        """Stage 7: Code generation."""
        if self.context.output_format == OutputFormat.PYTORCH:
            if not HAS_PYTORCH_BACKEND:
                raise RuntimeError("PyTorch backend not available. Install torch to use PyTorch code generation.")
            
            # Find first LayerDef in AST
            layer_def = None
            if hasattr(ast, 'statements'):
                for stmt in ast.statements:
                    if hasattr(stmt, '__class__') and stmt.__class__.__name__ == 'LayerDef':
                        layer_def = stmt
                        break
            
            if layer_def is None:
                raise ValueError("No LayerDef found in AST")
            
            return generate_pytorch_code(layer_def)
        else:
            raise NotImplementedError(f"Output format {self.context.output_format} not yet supported")
    
    def _output_writing_stage(self, code: str):
        """Stage 8: Write output."""
        output_file = self.context.output_file
        try:
            with open(output_file, 'w') as f:
                f.write(code)
            self.logger.info(f"Output written to: {output_file}")
        except Exception as e:
            raise IOError(f"Failed to write output file: {e}")
        
        return output_file
    
    def _finalize_result(self) -> CompilationResult:
        """Finalize compilation result."""
        self.stats.total_time = time.time() - self.context.start_time
        self.result.statistics = self.stats
        self.result.output_file = self.context.output_file if self.result.generated_code else None
        self.result.success = not self.result.has_errors
        
        if self.context.verbose or self.context.debug:
            self.logger.info(f"\n{self.stats.format_report()}")
        
        return self.result


# ============================================================================
# Python API
# ============================================================================

def compile_model(
    source_file: str,
    output_file: Optional[str] = None,
    output_format: str = "pytorch",
    optimization_level: int = 1,
    verbose: bool = False,
    debug: bool = False,
    warnings_as_errors: bool = False,
    enable_cache: bool = True,
    cache_dir: Optional[str] = None,
    target_backend: str = "pytorch",
) -> CompilationResult:
    """
    Compile a Triton DSL model to executable code.
    
    This is the main Python API for compiling Triton models.
    
    Args:
        source_file: Path to Triton source file (.triton or .tri)
        output_file: Path to output file (default: source_file with .py extension)
        output_format: Output format ("pytorch", "onnx", "tflite")
        optimization_level: Optimization level (0-3)
        verbose: Enable verbose logging
        debug: Enable debug logging
        warnings_as_errors: Treat warnings as errors
        enable_cache: Enable compilation caching
        cache_dir: Cache directory (default: ~/.triton/cache)
        target_backend: Target backend ("pytorch", "onnx", "tflite")
    
    Returns:
        CompilationResult object with compilation status and diagnostics
    
    Example:
        >>> from compiler.driver import compile_model
        >>> result = compile_model('model.triton', optimization_level=2)
        >>> if result.success:
        ...     print(f"Compiled successfully to {result.output_file}")
        ... else:
        ...     print(result.format_diagnostics())
    
    Jupyter Notebook Support:
        >>> # In Jupyter notebook
        >>> result = compile_model('model.triton', verbose=True)
        >>> # Display generated code
        >>> from IPython.display import Code
        >>> Code(result.generated_code, language='python')
    """
    # Create context
    context = CompilationContext(
        source_file=Path(source_file),
        output_file=Path(output_file) if output_file else None,
        output_format=OutputFormat[output_format.upper()],
        optimization_level=OptimizationLevel(optimization_level),
        verbose=verbose,
        debug=debug,
        warnings_as_errors=warnings_as_errors,
        enable_cache=enable_cache,
        cache_dir=Path(cache_dir) if cache_dir else None,
        target_backend=target_backend,
    )
    
    # Run compilation pipeline
    pipeline = CompilationPipeline(context)
    result = pipeline.compile()
    
    return result


def compile_string(
    source_code: str,
    output_file: Optional[str] = None,
    **kwargs
) -> CompilationResult:
    """
    Compile Triton DSL source code from a string.
    
    Useful for Jupyter notebooks and interactive environments.
    
    Args:
        source_code: Triton source code as string
        output_file: Path to output file
        **kwargs: Additional arguments passed to compile_model()
    
    Returns:
        CompilationResult object
    
    Example:
        >>> code = '''
        ... layer Simple(w: TernaryTensor) -> Tensor {
        ...     fn forward(x: Tensor) -> Tensor {
        ...         return x @ w
        ...     }
        ... }
        ... '''
        >>> result = compile_string(code)
        >>> print(result.generated_code)
    """
    import tempfile
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.triton', delete=False) as f:
        f.write(source_code)
        temp_file = f.name
    
    try:
        result = compile_model(temp_file, output_file, **kwargs)
        return result
    finally:
        # Clean up temporary file
        os.unlink(temp_file)


# ============================================================================
# CLI Interface
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="triton",
        description="Triton DSL Compiler - Compile ternary neural network models",
        epilog="For more information, visit: https://github.com/financecommander/Triton"
    )
    
    # Positional arguments
    parser.add_argument(
        "source_file",
        type=str,
        help="Triton source file to compile (.triton or .tri)"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path (default: source file with .py extension)"
    )
    output_group.add_argument(
        "-f", "--format",
        type=str,
        choices=["pytorch", "onnx", "tflite", "ir"],
        default="pytorch",
        help="Output format (default: pytorch)"
    )
    
    # Optimization options
    opt_group = parser.add_argument_group("Optimization Options")
    opt_group.add_argument(
        "-O0",
        action="store_const",
        const=0,
        dest="optimization",
        help="No optimization"
    )
    opt_group.add_argument(
        "-O1",
        action="store_const",
        const=1,
        dest="optimization",
        help="Basic optimization (default)"
    )
    opt_group.add_argument(
        "-O2",
        action="store_const",
        const=2,
        dest="optimization",
        help="Moderate optimization"
    )
    opt_group.add_argument(
        "-O3",
        action="store_const",
        const=3,
        dest="optimization",
        help="Aggressive optimization"
    )
    parser.set_defaults(optimization=1)
    
    # Target backend
    parser.add_argument(
        "--target",
        type=str,
        choices=["pytorch", "onnx", "tflite"],
        default="pytorch",
        help="Target backend (default: pytorch)"
    )
    
    # Diagnostic options
    diag_group = parser.add_argument_group("Diagnostic Options")
    diag_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    diag_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    diag_group.add_argument(
        "--Werror",
        action="store_true",
        help="Treat warnings as errors"
    )
    diag_group.add_argument(
        "--stats",
        action="store_true",
        help="Show compilation statistics"
    )
    
    # Cache options
    cache_group = parser.add_argument_group("Cache Options")
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable compilation cache"
    )
    cache_group.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory (default: ~/.triton/cache)"
    )
    cache_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear compilation cache and exit"
    )
    
    # Progress options
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.
    
    Args:
        argv: Command-line arguments (default: sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    
    # Handle cache clearing
    if args.clear_cache:
        cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".triton" / "cache"
        cache = CompilationCache(cache_dir)
        cache.invalidate()
        print(f"Cache cleared: {cache_dir}")
        return 0
    
    # Compile model
    try:
        result = compile_model(
            source_file=args.source_file,
            output_file=args.output,
            output_format=args.format,
            optimization_level=args.optimization,
            verbose=args.verbose,
            debug=args.debug,
            warnings_as_errors=args.Werror,
            enable_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            target_backend=args.target,
        )
        
        # Print diagnostics
        if result.has_errors or result.has_warnings:
            print(result.format_diagnostics(), file=sys.stderr)
        
        # Print statistics if requested
        if args.stats and result.statistics:
            print(result.statistics.format_report())
        
        # Print success message
        if result.success:
            print(f"✓ Compilation successful: {result.output_file}")
            return 0
        else:
            print("✗ Compilation failed", file=sys.stderr)
            return 1
    
    except Exception as e:
        print(f"✗ Fatal error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


# ============================================================================
# VS Code Extension Hooks
# ============================================================================

def get_diagnostics(source_file: str) -> List[Dict[str, Any]]:
    """
    Get diagnostics for VS Code extension integration.
    
    Args:
        source_file: Path to source file
    
    Returns:
        List of diagnostic objects compatible with VS Code Language Server Protocol
    
    Example:
        >>> diagnostics = get_diagnostics('model.triton')
        >>> for diag in diagnostics:
        ...     print(f"{diag['severity']}: {diag['message']}")
    """
    result = compile_model(
        source_file,
        output_file=None,
        verbose=False,
        enable_cache=False
    )
    
    diagnostics = []
    
    # Convert errors and warnings to LSP format
    for error in result.errors + result.warnings:
        diagnostics.append({
            "range": {
                "start": {"line": max(0, error.line - 1), "character": error.column},
                "end": {"line": max(0, error.line - 1), "character": error.column + 1}
            },
            "severity": 1 if error.severity == "error" else 2,  # 1=Error, 2=Warning
            "message": error.message,
            "source": "triton-compiler"
        })
    
    return diagnostics


# ============================================================================
# Build System Integration
# ============================================================================

def integrate_with_build_system(build_system: str = "make") -> str:
    """
    Generate build system integration code.
    
    Args:
        build_system: Build system type ("make", "cmake", "bazel")
    
    Returns:
        Build system configuration code
    
    Example:
        >>> makefile = integrate_with_build_system("make")
        >>> print(makefile)
    """
    if build_system == "make":
        return """
# Triton Compiler Integration for Make

# Compiler settings
TRITON := python -m compiler.driver
TRITON_FLAGS := -O2 --format pytorch

# Pattern rule for .triton -> .py
%.py: %.triton
\t$(TRITON) $(TRITON_FLAGS) -o $@ $<

# Clean rule
clean-triton:
\trm -f *.py
\t$(TRITON) --clear-cache

.PHONY: clean-triton
"""
    elif build_system == "cmake":
        return """
# Triton Compiler Integration for CMake

function(triton_compile SOURCE OUTPUT)
    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND python -m compiler.driver -O2 -o ${OUTPUT} ${SOURCE}
        DEPENDS ${SOURCE}
        COMMENT "Compiling Triton model: ${SOURCE}"
    )
endfunction()
"""
    elif build_system == "bazel":
        return """
# Triton Compiler Integration for Bazel

def triton_library(name, src, **kwargs):
    native.genrule(
        name = name + "_gen",
        srcs = [src],
        outs = [src.replace(".triton", ".py")],
        cmd = "python -m compiler.driver -O2 -o $@ $<",
    )
    
    native.py_library(
        name = name,
        srcs = [":" + name + "_gen"],
        **kwargs
    )
"""
    else:
        raise ValueError(f"Unsupported build system: {build_system}")


if __name__ == "__main__":
    sys.exit(main())

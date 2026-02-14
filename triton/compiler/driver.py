"""
Triton Compiler Driver - Main Compilation Orchestrator

This module provides the main entry point for compiling Triton DSL programs.
It orchestrates the entire compilation pipeline from source code to executable output.

Pipeline Stages:
1. Lexical Analysis (Lexer)
2. Syntax Analysis (Parser → AST)
3. Type Checking
4. Semantic Analysis
5. IR Generation
6. Optimization
7. Code Generation
8. Output Writing

Usage:
    # CLI
    triton compile model.triton --output model.py --O2 --target pytorch
    
    # Python API
    from triton.compiler.driver import compile_model
    result = compile_model('model.triton', optimization_level=2)
"""

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Progress bar support
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
# Import compiler components
from compiler.lexer.triton_lexer import lexer
from compiler.parser.triton_parser import parse
from compiler.typechecker.validator import TypeChecker
from compiler.ast.nodes import Node, Program


# ============================================================================
# Enumerations and Constants
# ============================================================================

class OptimizationLevel(Enum):
    """Optimization levels for compilation."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimization
    O2 = 2  # Aggressive optimization
    O3 = 3  # Maximum optimization


class TargetBackend(Enum):
    """Supported compilation target backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TFLITE = "tflite"
    PYTHON = "python"


class CompilationStage(Enum):
    """Stages in the compilation pipeline."""
    LEXER = "lexer"
    PARSER = "parser"
    TYPE_CHECKER = "type_checker"
    SEMANTIC_ANALYZER = "semantic_analyzer"
    IR_GENERATION = "ir_generation"
    OPTIMIZATION = "optimization"
    CODE_GENERATION = "code_generation"
    OUTPUT_WRITING = "output_writing"


class OutputFormat(Enum):
    """Output format options."""
    PYTHON = "py"
    ONNX = "onnx"
    JSON = "json"
    BINARY = "bin"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CompilationError:
    """Represents an error that occurred during compilation."""
    stage: CompilationStage
    message: str
    lineno: int = 0
    col_offset: int = 0
    source_file: str = ""
    is_warning: bool = False
    
    def __str__(self) -> str:
        location = f"{self.source_file}:" if self.source_file else ""
        if self.lineno > 0:
            location += f"{self.lineno}:{self.col_offset}: "
        severity = "warning" if self.is_warning else "error"
        return f"{location}{severity}: [{self.stage.value}] {self.message}"


@dataclass
class CompilationStatistics:
    """Statistics collected during compilation."""
    total_time: float = 0.0
    lexer_time: float = 0.0
    parser_time: float = 0.0
    type_checker_time: float = 0.0
    semantic_analyzer_time: float = 0.0
    ir_generation_time: float = 0.0
    optimization_time: float = 0.0
    code_generation_time: float = 0.0
    output_writing_time: float = 0.0
    
    lines_of_code: int = 0
    ast_nodes: int = 0
    optimization_passes: int = 0
    
    memory_peak_mb: float = 0.0
    cache_hit: bool = False
    
    def __str__(self) -> str:
        lines = [
            "=== Compilation Statistics ===",
            f"Total Time: {self.total_time:.3f}s",
            f"  - Lexer: {self.lexer_time:.3f}s",
            f"  - Parser: {self.parser_time:.3f}s",
            f"  - Type Checker: {self.type_checker_time:.3f}s",
            f"  - Semantic Analyzer: {self.semantic_analyzer_time:.3f}s",
            f"  - IR Generation: {self.ir_generation_time:.3f}s",
            f"  - Optimization: {self.optimization_time:.3f}s ({self.optimization_passes} passes)",
            f"  - Code Generation: {self.code_generation_time:.3f}s",
            f"  - Output Writing: {self.output_writing_time:.3f}s",
            f"",
            f"Lines of Code: {self.lines_of_code}",
            f"AST Nodes: {self.ast_nodes}",
            f"Peak Memory: {self.memory_peak_mb:.2f} MB",
            f"Cache Hit: {'Yes' if self.cache_hit else 'No'}",
        ]
        return "\n".join(lines)


@dataclass
class CompilationOptions:
    """Options for compilation."""
    source_file: str
    output_file: Optional[str] = None
    optimization_level: OptimizationLevel = OptimizationLevel.O1
    target_backend: TargetBackend = TargetBackend.PYTORCH
    output_format: OutputFormat = OutputFormat.PYTHON
    
    verbose: bool = False
    debug: bool = False
    warnings_as_errors: bool = False
    show_statistics: bool = False
    show_optimization_report: bool = False
    
    use_cache: bool = True
    cache_dir: Optional[str] = None
    force_recompile: bool = False
    
    profile_memory: bool = False
    profile_performance: bool = False
    
    # Progress reporting
    show_progress: bool = True


@dataclass
class CompilationResult:
    """Result of a compilation."""
    success: bool
    output_file: Optional[str] = None
    ast: Optional[Node] = None
    ir: Optional[Any] = None
    
    errors: List[CompilationError] = field(default_factory=list)
    warnings: List[CompilationError] = field(default_factory=list)
    
    statistics: CompilationStatistics = field(default_factory=CompilationStatistics)
    optimization_report: Dict[str, Any] = field(default_factory=dict)
    
    def has_errors(self) -> bool:
        """Check if compilation has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if compilation has warnings."""
        return len(self.warnings) > 0


# ============================================================================
# Cache Management
# ============================================================================

class CompilationCache:
    """Manages compilation cache for incremental builds."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache manager."""
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".triton", "cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache metadata: {e}")
    
    def _compute_hash(self, source_file: str, options: CompilationOptions) -> str:
        """Compute hash for cache key."""
        hasher = hashlib.sha256()
        
        # Hash source file content
        with open(source_file, 'rb') as f:
            hasher.update(f.read())
        
        # Hash compilation options
        options_str = f"{options.optimization_level}:{options.target_backend}:{options.output_format}"
        hasher.update(options_str.encode())
        
        return hasher.hexdigest()
    
    def get(self, source_file: str, options: CompilationOptions) -> Optional[CompilationResult]:
        """Get cached compilation result."""
        cache_key = self._compute_hash(source_file, options)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid
        if cache_key in self.metadata:
            cache_time = self.metadata[cache_key].get("timestamp", 0)
            source_mtime = os.path.getmtime(source_file)
            
            if source_mtime > cache_time:
                # Source file modified after cache
                return None
        
        # Load cached result
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                result.statistics.cache_hit = True
                return result
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}")
            return None
    
    def put(self, source_file: str, options: CompilationOptions, result: CompilationResult):
        """Store compilation result in cache."""
        cache_key = self._compute_hash(source_file, options)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Update metadata
            self.metadata[cache_key] = {
                "source_file": source_file,
                "timestamp": time.time(),
                "optimization_level": options.optimization_level.value,
                "target_backend": options.target_backend.value,
            }
            self._save_metadata()
        except Exception as e:
            logging.warning(f"Failed to save to cache: {e}")
    
    def clear(self):
        """Clear all cached compilation results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass
        self.metadata = {}
        self._save_metadata()
    
    def invalidate(self, source_file: str):
        """Invalidate cache for a specific source file."""
        # Remove all cache entries for this source file
        to_remove = []
        for cache_key, meta in self.metadata.items():
            if meta.get("source_file") == source_file:
                to_remove.append(cache_key)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        
        for key in to_remove:
            del self.metadata[key]
        
        self._save_metadata()


# ============================================================================
# Compiler Driver
# ============================================================================

class TritonCompiler:
    """Main compiler driver orchestrating the compilation pipeline."""
    
    def __init__(self, options: CompilationOptions):
        """Initialize compiler with options."""
        self.options = options
        self.logger = self._setup_logging()
        self.cache = CompilationCache(options.cache_dir) if options.use_cache else None
        self.result = CompilationResult(success=False)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("triton.compiler")
        logger.setLevel(logging.DEBUG if self.options.debug else 
                       logging.INFO if self.options.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _count_ast_nodes(self, node: Node) -> int:
        """Count total AST nodes."""
        count = 1
        for attr_name in dir(node):
            if attr_name.startswith('_'):
                continue
            attr = getattr(node, attr_name, None)
            if isinstance(attr, Node):
                count += self._count_ast_nodes(attr)
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Node):
                        count += self._count_ast_nodes(item)
        return count
    
    def _stage_wrapper(self, stage: CompilationStage, func, *args, **kwargs):
        """Wrapper for compilation stages with timing and error handling."""
        self.logger.info(f"Starting stage: {stage.value}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Update statistics
            stage_time_attr = f"{stage.value}_time"
            if hasattr(self.result.statistics, stage_time_attr):
                setattr(self.result.statistics, stage_time_attr, elapsed)
            
            self.logger.info(f"Completed stage: {stage.value} ({elapsed:.3f}s)")
            return result, None
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            if self.options.debug:
                error_msg += f"\n{traceback.format_exc()}"
            
            error = CompilationError(
                stage=stage,
                message=error_msg,
                source_file=self.options.source_file
            )
            
            self.logger.error(f"Failed stage: {stage.value} ({elapsed:.3f}s) - {error_msg}")
            return None, error
    
    def _read_source(self) -> Tuple[Optional[str], Optional[CompilationError]]:
        """Read source file."""
        try:
            with open(self.options.source_file, 'r') as f:
                source = f.read()
            
            # Count lines
            self.result.statistics.lines_of_code = len(source.splitlines())
            return source, None
            
        except FileNotFoundError:
            return None, CompilationError(
                stage=CompilationStage.LEXER,
                message=f"Source file not found: {self.options.source_file}",
                source_file=self.options.source_file
            )
        except Exception as e:
            return None, CompilationError(
                stage=CompilationStage.LEXER,
                message=f"Failed to read source file: {e}",
                source_file=self.options.source_file
            )
    
    def _run_lexer(self, source: str) -> Tuple[Optional[Any], Optional[CompilationError]]:
        """Run lexical analysis."""
        def tokenize():
            lexer.input(source)
            tokens = []
            while True:
                tok = lexer.token()
                if not tok:
                    break
                tokens.append(tok)
            return tokens
        
        return self._stage_wrapper(CompilationStage.LEXER, tokenize)
    
    def _run_parser(self, source: str) -> Tuple[Optional[Program], Optional[CompilationError]]:
        """Run syntax analysis to generate AST."""
        def parse_source():
            ast = parse(source)
            if ast is None:
                raise RuntimeError("Parser returned None (syntax errors present)")
            return ast
        
        result, error = self._stage_wrapper(CompilationStage.PARSER, parse_source)
        
        if result:
            # Count AST nodes
            self.result.statistics.ast_nodes = self._count_ast_nodes(result)
            self.result.ast = result
        
        return result, error
    
    def _run_type_checker(self, ast: Program) -> Tuple[Optional[bool], Optional[CompilationError]]:
        """Run type checking."""
        def check_types():
            checker = TypeChecker()
            errors = checker.validate(ast)
            
            if errors:
                # Convert type errors to compilation errors/warnings
                for type_error in errors:
                    comp_error = CompilationError(
                        stage=CompilationStage.TYPE_CHECKER,
                        message=str(type_error),
                        lineno=getattr(type_error, 'lineno', 0),
                        col_offset=getattr(type_error, 'col_offset', 0),
                        source_file=self.options.source_file,
                        is_warning=False
                    )
                    self.result.errors.append(comp_error)
                
                if not self.options.warnings_as_errors:
                    # Continue compilation with warnings
                    return True
                else:
                    raise RuntimeError(f"Type checking failed with {len(errors)} error(s)")
            
            return True
        
        return self._stage_wrapper(CompilationStage.TYPE_CHECKER, check_types)
    
    def _run_semantic_analyzer(self, ast: Program) -> Tuple[Optional[bool], Optional[CompilationError]]:
        """Run semantic analysis."""
        def analyze():
            # Placeholder for semantic analysis
            # This would include:
            # - Symbol resolution
            # - Scope checking
            # - Control flow analysis
            # - Dead code detection
            self.logger.info("Semantic analysis (placeholder)")
            return True
        
        return self._stage_wrapper(CompilationStage.SEMANTIC_ANALYZER, analyze)
    
    def _generate_ir(self, ast: Program) -> Tuple[Optional[Any], Optional[CompilationError]]:
        """Generate intermediate representation."""
        def generate():
            # Placeholder for IR generation
            # This would convert AST to an intermediate representation
            # suitable for optimization and code generation
            self.logger.info("IR generation (placeholder)")
            ir = {
                "type": "ir",
                "ast": ast,
                "version": "0.1.0"
            }
            return ir
        
        result, error = self._stage_wrapper(CompilationStage.IR_GENERATION, generate)
        
        if result:
            self.result.ir = result
        
        return result, error
    
    def _optimize_ir(self, ir: Any) -> Tuple[Optional[Any], Optional[CompilationError]]:
        """Run optimization passes on IR."""
        def optimize():
            # Placeholder for optimization
            # This would include:
            # - Constant folding
            # - Dead code elimination
            # - Common subexpression elimination
            # - Loop optimization
            # - Inline expansion
            
            optimization_passes = {
                OptimizationLevel.O0: [],
                OptimizationLevel.O1: ["constant_folding", "dead_code_elimination"],
                OptimizationLevel.O2: ["constant_folding", "dead_code_elimination", 
                                       "common_subexpression_elimination", "inline_expansion"],
                OptimizationLevel.O3: ["constant_folding", "dead_code_elimination",
                                       "common_subexpression_elimination", "inline_expansion",
                                       "loop_optimization", "aggressive_inline"],
            }
            
            passes = optimization_passes.get(self.options.optimization_level, [])
            self.result.statistics.optimization_passes = len(passes)
            
            for pass_name in passes:
                self.logger.debug(f"Running optimization pass: {pass_name}")
            
            # Store optimization report
            self.result.optimization_report = {
                "level": self.options.optimization_level.name,
                "passes": passes,
                "transformations": []
            }
            
            return ir
        
        return self._stage_wrapper(CompilationStage.OPTIMIZATION, optimize)
    
    def _generate_code(self, ir: Any) -> Tuple[Optional[str], Optional[CompilationError]]:
        """Generate target code from IR."""
        def generate():
            # Placeholder for code generation
            # This would generate Python/PyTorch code from IR
            
            backend_name = self.options.target_backend.value
            self.logger.info(f"Generating code for backend: {backend_name}")
            
            # Generate simple Python code
            if self.options.target_backend == TargetBackend.PYTORCH:
                code = self._generate_pytorch_code(ir)
            elif self.options.target_backend == TargetBackend.ONNX:
                code = self._generate_onnx_code(ir)
            elif self.options.target_backend == TargetBackend.PYTHON:
                code = self._generate_python_code(ir)
            else:
                code = f"# Generated code for {backend_name}\n# (placeholder)\n"
            
            return code
        
        return self._stage_wrapper(CompilationStage.CODE_GENERATION, generate)
    
    def _generate_pytorch_code(self, ir: Any) -> str:
        """Generate PyTorch code."""
        return """# Generated PyTorch code
import torch
import torch.nn as nn
from backend.pytorch.ternary_tensor import TernaryTensor

class GeneratedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Model initialization
        pass
    
    def forward(self, x):
        # Forward pass implementation
        return x

# Instantiate model
model = GeneratedModel()
"""
    
    def _generate_onnx_code(self, ir: Any) -> str:
        """Generate ONNX export code."""
        return """# Generated ONNX export code
import torch
import torch.onnx

# Export model to ONNX format
def export_to_onnx(model, output_path='model.onnx'):
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True)
"""
    
    def _generate_python_code(self, ir: Any) -> str:
        """Generate pure Python code."""
        return """# Generated Python code
import numpy as np

def forward(x):
    # Forward pass implementation
    return x
"""
    
    def _write_output(self, code: str) -> Tuple[Optional[str], Optional[CompilationError]]:
        """Write generated code to output file."""
        def write():
            output_file = self.options.output_file
            
            if output_file is None:
                # Derive output filename from source
                source_path = Path(self.options.source_file)
                output_file = str(source_path.with_suffix(f'.{self.options.output_format.value}'))
            
            try:
                with open(output_file, 'w') as f:
                    f.write(code)
                
                self.logger.info(f"Output written to: {output_file}")
                return output_file
                
            except Exception as e:
                raise RuntimeError(f"Failed to write output: {e}")
        
        return self._stage_wrapper(CompilationStage.OUTPUT_WRITING, write)
    
    def compile(self) -> CompilationResult:
        """Run the complete compilation pipeline."""
        start_time = time.time()
        
        # Check cache
        if self.cache and not self.options.force_recompile:
            cached_result = self.cache.get(self.options.source_file, self.options)
            if cached_result:
                self.logger.info("Using cached compilation result")
                return cached_result
        
        # Setup progress tracking
        stages = [
            ("Reading source", lambda: self._read_source()),
            ("Lexical analysis", lambda src: self._run_lexer(src)),
            ("Parsing", lambda src: self._run_parser(src)),
            ("Type checking", lambda ast: self._run_type_checker(ast)),
            ("Semantic analysis", lambda ast: self._run_semantic_analyzer(ast)),
            ("IR generation", lambda ast: self._generate_ir(ast)),
            ("Optimization", lambda ir: self._optimize_ir(ir)),
            ("Code generation", lambda ir: self._generate_code(ir)),
            ("Writing output", lambda code: self._write_output(code)),
        ]
        
        if HAS_TQDM and self.options.show_progress:
            progress = tqdm(total=len(stages), desc="Compiling", unit="stage")
        else:
            progress = None
        
        # Execute pipeline
        try:
            # Read source
            source, error = self._read_source()
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Lexer
            tokens, error = self._run_lexer(source)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Parser
            ast, error = self._run_parser(source)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Type Checker
            _, error = self._run_type_checker(ast)
            if error:
                self.result.errors.append(error)
                if self.options.warnings_as_errors:
                    self.result.success = False
                    return self.result
            if progress:
                progress.update(1)
            
            # Semantic Analyzer
            _, error = self._run_semantic_analyzer(ast)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # IR Generation
            ir, error = self._generate_ir(ast)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Optimization
            ir, error = self._optimize_ir(ir)
            if error:
                self.result.errors.append(error)
                # Continue with unoptimized IR
            if progress:
                progress.update(1)
            
            # Code Generation
            code, error = self._generate_code(ir)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Write Output
            output_file, error = self._write_output(code)
            if error:
                self.result.errors.append(error)
                self.result.success = False
                return self.result
            if progress:
                progress.update(1)
            
            # Success!
            self.result.success = True
            self.result.output_file = output_file
            
        finally:
            if progress:
                progress.close()
            
            # Finalize statistics
            self.result.statistics.total_time = time.time() - start_time
            
            # Cache result if successful
            if self.result.success and self.cache:
                self.cache.put(self.options.source_file, self.options, self.result)
        
        return self.result


# ============================================================================
# Python API
# ============================================================================

def compile_model(
    source_file: str,
    output_file: Optional[str] = None,
    optimization_level: int = 1,
    target: str = "pytorch",
    verbose: bool = False,
    use_cache: bool = True,
    **kwargs
) -> CompilationResult:
    """
    Compile a Triton DSL model.
    
    Args:
        source_file: Path to the source .triton file
        output_file: Optional output file path
        optimization_level: Optimization level (0-3)
        target: Target backend ('pytorch', 'onnx', 'tflite', 'python')
        verbose: Enable verbose output
        use_cache: Enable compilation caching
        **kwargs: Additional compilation options
    
    Returns:
        CompilationResult with compilation status and output
    
    Example:
        >>> result = compile_model('model.triton', optimization_level=2)
        >>> if result.success:
        ...     print(f"Compiled to: {result.output_file}")
    """
    # Map string target to enum
    target_map = {
        "pytorch": TargetBackend.PYTORCH,
        "onnx": TargetBackend.ONNX,
        "tflite": TargetBackend.TFLITE,
        "python": TargetBackend.PYTHON,
    }
    target_backend = target_map.get(target.lower(), TargetBackend.PYTORCH)
    
    # Map optimization level to enum
    opt_levels = {
        0: OptimizationLevel.O0,
        1: OptimizationLevel.O1,
        2: OptimizationLevel.O2,
        3: OptimizationLevel.O3,
    }
    opt_level = opt_levels.get(optimization_level, OptimizationLevel.O1)
    
    # Create options
    options = CompilationOptions(
        source_file=source_file,
        output_file=output_file,
        optimization_level=opt_level,
        target_backend=target_backend,
        verbose=verbose,
        use_cache=use_cache,
        **kwargs
    )
    
    # Compile
    compiler = TritonCompiler(options)
    return compiler.compile()


# ============================================================================
# CLI Interface
# ============================================================================

def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="triton",
        description="Triton DSL Compiler - Compile ternary neural network models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic compilation
  triton compile model.triton
  
  # With optimization
  triton compile model.triton --O2
  
  # Specify output and target
  triton compile model.triton -o output.py --target pytorch
  
  # Enable verbose mode
  triton compile model.triton -v --statistics
  
  # Clear cache
  triton cache clear
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compile command
    compile_parser = subparsers.add_parser(
        "compile",
        help="Compile a Triton DSL source file"
    )
    compile_parser.add_argument(
        "source",
        help="Source file to compile (.triton)"
    )
    compile_parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    compile_parser.add_argument(
        "--format",
        choices=["py", "onnx", "json", "bin"],
        default="py",
        help="Output format (default: py)"
    )
    
    # Optimization options
    opt_group = compile_parser.add_mutually_exclusive_group()
    opt_group.add_argument(
        "--O0",
        action="store_const",
        const=0,
        dest="optimization_level",
        help="No optimization"
    )
    opt_group.add_argument(
        "--O1",
        action="store_const",
        const=1,
        dest="optimization_level",
        help="Basic optimization (default)"
    )
    opt_group.add_argument(
        "--O2",
        action="store_const",
        const=2,
        dest="optimization_level",
        help="Aggressive optimization"
    )
    opt_group.add_argument(
        "--O3",
        action="store_const",
        const=3,
        dest="optimization_level",
        help="Maximum optimization"
    )
    compile_parser.set_defaults(optimization_level=1)
    
    # Target backend
    compile_parser.add_argument(
        "--target",
        choices=["pytorch", "onnx", "tflite", "python"],
        default="pytorch",
        help="Target backend (default: pytorch)"
    )
    
    # Verbosity
    compile_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    compile_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    compile_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    # Error handling
    compile_parser.add_argument(
        "--Werror",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    # Caching
    compile_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable compilation cache"
    )
    compile_parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompilation (ignore cache)"
    )
    compile_parser.add_argument(
        "--cache-dir",
        help="Custom cache directory"
    )
    
    # Diagnostics
    compile_parser.add_argument(
        "--statistics",
        action="store_true",
        help="Show compilation statistics"
    )
    compile_parser.add_argument(
        "--optimization-report",
        action="store_true",
        help="Show optimization report"
    )
    compile_parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Cache management command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage compilation cache"
    )
    cache_action = cache_parser.add_subparsers(dest="cache_action")
    cache_action.add_parser("clear", help="Clear all cached compilations")
    cache_action.add_parser("info", help="Show cache information")
    
    # Version command
    subparsers.add_parser(
        "version",
        help="Show version information"
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle version command
    if args.command == "version":
        print("Triton DSL Compiler v0.1.0")
        return 0
    
    # Handle cache command
    if args.command == "cache":
        cache = CompilationCache()
        
        if args.cache_action == "clear":
            cache.clear()
            print("Cache cleared successfully")
            return 0
        
        elif args.cache_action == "info":
            print(f"Cache directory: {cache.cache_dir}")
            print(f"Cache entries: {len(cache.metadata)}")
            total_size = sum(f.stat().st_size for f in cache.cache_dir.glob("*.pkl"))
            print(f"Total size: {total_size / (1024*1024):.2f} MB")
            return 0
        
        else:
            parser.print_help()
            return 1
    
    # Handle compile command
    if args.command == "compile":
        # Check if source file exists
        if not os.path.exists(args.source):
            print(f"Error: Source file not found: {args.source}", file=sys.stderr)
            return 1
        
        # Create compilation options
        options = CompilationOptions(
            source_file=args.source,
            output_file=args.output,
            optimization_level=OptimizationLevel(args.optimization_level),
            target_backend=TargetBackend(args.target),
            output_format=OutputFormat(args.format),
            verbose=args.verbose,
            debug=args.debug,
            warnings_as_errors=args.Werror,
            show_statistics=args.statistics,
            show_optimization_report=args.optimization_report,
            use_cache=not args.no_cache,
            cache_dir=args.cache_dir,
            force_recompile=args.force,
            profile_performance=args.profile,
            show_progress=not args.quiet,
        )
        
        # Compile
        compiler = TritonCompiler(options)
        result = compiler.compile()
        
        # Print errors
        if result.errors:
            for error in result.errors:
                print(error, file=sys.stderr)
        
        # Print warnings
        if result.warnings:
            for warning in result.warnings:
                print(warning, file=sys.stderr)
        
        # Print statistics
        if args.statistics and result.success:
            print()
            print(result.statistics)
        
        # Print optimization report
        if args.optimization_report and result.success:
            print()
            print("=== Optimization Report ===")
            print(json.dumps(result.optimization_report, indent=2))
        
        # Print result
        if result.success:
            print(f"\nCompilation successful!")
            print(f"Output: {result.output_file}")
            return 0
        else:
            print(f"\nCompilation failed with {len(result.errors)} error(s)", file=sys.stderr)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

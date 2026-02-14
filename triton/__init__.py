"""
Triton DSL - Domain-Specific Language for Ternary Neural Networks

This package provides the main API for compiling and working with Triton DSL programs.
"""

from triton.compiler.driver import compile_model, CompilationResult, CompilationOptions

__version__ = "0.1.0"
__all__ = ["compile_model", "CompilationResult", "CompilationOptions"]

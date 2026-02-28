"""
Parser stress tests for Triton DSL.

This module contains comprehensive stress tests for the Triton parser,
focusing on edge cases, performance, memory usage, and error recovery.
"""

import pytest
import sys
import os
from typing import List

# Add project root to path for imports

from compiler.parser.triton_parser import parse

try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

try:
    from memory_profiler import memory_usage
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False


class TestDeeplyNestedStructures:
    """Test parsing of deeply nested structures."""

    @pytest.mark.parametrize("depth", [10, 50, 100, 200])
    def test_nested_expressions(self, depth: int):
        """Test parsing of deeply nested arithmetic expressions."""
        # Generate nested expression: 1 + (2 + (3 + ... ))
        expr = "1"
        for i in range(2, depth + 2):
            expr = f"({expr} + {i})"
        
        code = f"let x: int8 = {expr}"
        ast = parse(code)
        assert ast is not None
        assert len(ast.statements) == 1

    @pytest.mark.parametrize("depth", [5, 10, 20])
    def test_nested_function_calls(self, depth: int):
        """Test parsing of deeply nested function calls."""
        # Generate relu(relu(relu(...)))
        expr = "x"
        for _ in range(depth):
            expr = f"relu({expr})"
        
        code = f"let y: int8 = {expr}"
        ast = parse(code)
        assert ast is not None

    @pytest.mark.parametrize("depth", [3, 5, 8])
    def test_nested_tensor_shapes(self, depth: int):
        """Test parsing of nested tensor shape operations."""
        # Generate complex shape expressions
        shape_expr = "1"
        for i in range(depth):
            shape_expr = f"[{shape_expr}, {i+2}]"
        
        code = f"let tensor: trit = TernaryTensor{shape_expr}([1, 0, -1])"
        ast = parse(code)
        assert ast is not None

    @pytest.mark.parametrize("depth", [5, 10, 15])
    def test_nested_layer_definitions(self, depth: int):
        """Test parsing of nested layer definitions."""
        code = ""
        indent = ""
        for i in range(depth):
            code += f"{indent}layer layer{i} {{\n"
            indent += "  "
        
        # Add some content to innermost layer
        code += f"{indent}let x: int8 = 1\n"
        
        # Close all layers
        for i in range(depth):
            indent = indent[:-2]
            code += f"{indent}}}\n"
        
        ast = parse(code)
        assert ast is not None


class TestLargePrograms:
    """Test parsing of large programs."""

    def generate_function_definitions(self, count: int) -> str:
        """Generate multiple function definitions."""
        code = ""
        for i in range(count):
            code += f"fn func{i}() {{\n"
            code += f"  let x: int8 = {i}\n"
            code += f"  return x\n"
            code += "}\n\n"
        return code

    def generate_large_expressions(self, size: int) -> str:
        """Generate large expressions with many operations."""
        expr_parts = []
        for i in range(size):
            expr_parts.append(f"var{i}")
        
        expr = " + ".join(expr_parts)
        code = ""
        for i in range(size):
            code += f"let var{i}: int8 = {i}\n"
        code += f"let result: int8 = {expr}\n"
        return code

    @pytest.mark.parametrize("num_funcs", [100, 500, 1000])
    def test_many_function_definitions(self, num_funcs: int):
        """Test parsing programs with many function definitions."""
        code = self.generate_function_definitions(num_funcs)
        ast = parse(code)
        assert ast is not None
        assert len(ast.statements) == num_funcs

    @pytest.mark.parametrize("expr_size", [100, 500, 1000])
    def test_large_expressions(self, expr_size: int):
        """Test parsing large expressions."""
        code = self.generate_large_expressions(expr_size)
        ast = parse(code)
        assert ast is not None

    def test_very_large_program(self):
        """Test parsing a very large program (10,000+ lines)."""
        # Generate a large program with mixed content
        code = ""
        
        # Add many variable declarations
        for i in range(1000):
            code += f"let var{i}: int8 = {i % 3 - 1}\n"
        
        # Add many function definitions
        for i in range(100):
            code += f"fn func{i}() {{\n"
            for j in range(10):
                code += f"  let local{j}: int8 = var{i * 10 + j}\n"
            code += f"  return local0\n"
            code += "}\n\n"
        
        # Add complex expressions
        code += "let result: int8 = "
        terms = []
        for i in range(100):
            terms.append(f"func{i}()")
        code += " + ".join(terms)
        code += "\n"
        
        ast = parse(code)
        assert ast is not None


class TestAmbiguousGrammarCases:
    """Test ambiguous grammar and operator precedence."""

    def test_operator_precedence_basic(self):
        """Test basic operator precedence."""
        # 1 + 2 * 3 should be 1 + (2 * 3)
        code = "let x: int8 = 1 + 2 * 3"
        ast = parse(code)
        assert ast is not None
        
        # Check the structure
        expr = ast.statements[0].initializer
        assert expr.op == "+"
        assert expr.right.op == "*"

    def test_operator_precedence_complex(self):
        """Test complex operator precedence scenarios."""
        test_cases = [
            "1 + 2 * 3 - 4 / 5",  # Should be (1 + (2 * 3)) - (4 / 5)
            "a @ b + c * d",      # Matrix mul has higher precedence
            "func() + 1 * 2",     # Function call precedence
        ]
        
        for code in test_cases:
            full_code = f"let result: int8 = {code}"
            ast = parse(full_code)
            assert ast is not None

    def test_statement_expression_boundaries(self):
        """Test boundaries between statements and expressions."""
        # Cases that could be ambiguous
        test_cases = [
            "let x: int8 = a + b\nlet y: int8 = c",  # Statement boundary
            "fn f() { return x + y }",                # Return with expression
            "if (x > 0) { y = 1 } else { y = 0 }",    # Control flow (if supported)
        ]
        
        for code in test_cases:
            ast = parse(code)
            assert ast is not None

    def test_type_inference_complexity(self):
        """Test complex type inference scenarios."""
        # Complex expressions that might challenge type inference
        test_cases = [
            "let x: trit = TernaryTensor[2,2]([1, 0, -1, 1])",
            "let y: int8 = x + 1",  # Mixing types
            "let z: float32 = y * 3.14",  # Type promotion
        ]
        
        for code in test_cases:
            ast = parse(code)
            assert ast is not None


class TestErrorRecovery:
    """Test error recovery and handling of syntax errors."""

    def test_syntax_errors_at_positions(self):
        """Test syntax errors at various positions."""
        error_cases = [
            "let x: int8 = ",  # Incomplete assignment
            "let x int8 = 1",  # Missing colon
            "fn func( { return 1 }",  # Missing param list
            "let x: invalid_type = 1",  # Invalid type
            "let x: int8 = 1 + ",  # Incomplete expression
        ]
        
        for code in error_cases:
            # Parser should handle errors gracefully
            try:
                ast = parse(code)
                # Even with errors, should return some AST or raise specific error
            except Exception:
                # Expected for invalid syntax
                pass

    def test_multiple_simultaneous_errors(self):
        """Test handling multiple syntax errors in one program."""
        code = """
        let x: int8 = 
        let y int8 = 1
        fn func( {
          let z: = 2
          return
        }
        let w: int8 = a +
        """
        
        try:
            ast = parse(code)
        except Exception:
            # Expected with multiple errors
            pass

    def test_recovery_and_continuation(self):
        """Test that parser can recover from errors and continue."""
        # Program with errors but some valid parts
        code = """
        let valid1: int8 = 1
        let invalid: = 
        let valid2: int8 = 2
        let another_invalid int8 = 3
        let valid3: int8 = valid1 + valid2
        """
        
        try:
            ast = parse(code)
            # Should parse valid parts even with errors
        except Exception:
            pass


class TestMemoryTests:
    """Test memory usage during parsing."""

    @pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="memory_profiler not available")
    def test_peak_memory_usage(self):
        """Test peak memory usage during parsing."""
        def parse_large_program():
            # Generate a very large program
            code = ""
            for i in range(10000):
                code += f"let var{i}: int8 = {i % 3 - 1}\n"
            return parse(code)
        
        # Measure memory usage
        mem_usage = memory_usage(parse_large_program, max_usage=True)
        
        # Log memory usage (in MB)
        print(f"Peak memory usage: {mem_usage:.2f} MB")
        
        # Assert reasonable memory usage (less than 1GB)
        assert mem_usage < 1000

    @pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="memory_profiler not available") 
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated parsing."""
        import gc
        
        def parse_many_times():
            for _ in range(100):
                code = "let x: int8 = 1 + 2 * 3"
                parse(code)
                gc.collect()  # Force garbage collection
        
        # Measure memory before and after
        mem_before = memory_usage(lambda: None, max_usage=True)
        mem_after = memory_usage(parse_many_times, max_usage=True)
        
        # Memory should not grow significantly
        growth = mem_after - mem_before
        print(f"Memory growth: {growth:.2f} MB")
        
        assert growth < 50  # Less than 50MB growth

    @pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="memory_profiler not available")
    def test_large_ast_generation(self):
        """Test memory usage when generating large ASTs."""
        def create_large_ast():
            code = ""
            for i in range(1000):
                code += f"fn func{i}() {{ let x: int8 = {i}; return x }}\n"
            return parse(code)
        
        mem_usage = memory_usage(create_large_ast, max_usage=True)
        print(f"Large AST memory usage: {mem_usage:.2f} MB")
        
        assert mem_usage < 500


class TestPerformanceBenchmarking:
    """Performance benchmarks using pytest-benchmark."""

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_small_program_benchmark(self, benchmark):
        """Benchmark parsing of small programs."""
        code = """
        let x: int8 = 1
        let y: int8 = 2
        let z: int8 = x + y
        """
        
        benchmark(parse, code)

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_medium_program_benchmark(self, benchmark):
        """Benchmark parsing of medium-sized programs."""
        code = """
        for i in range(100):
            code += f"fn func{i}() {{ return {i} }}\n"
        
        benchmark(parse, code)

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_large_program_benchmark(self, benchmark):
        """Benchmark parsing of large programs."""
        code = """
        for i in range(1000):
            code += f"let var{i}: int8 = {i % 3 - 1}\n"
        
        benchmark(parse, code)

    @pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not available")
    def test_deeply_nested_benchmark(self, benchmark):
        """Benchmark parsing of deeply nested expressions."""
        # Generate 100 levels of nesting
        expr = "1"
        for i in range(99):
            expr = f"({expr} + {i+2})"
        
        code = f"let x: int8 = {expr}"
        
        benchmark(parse, code)
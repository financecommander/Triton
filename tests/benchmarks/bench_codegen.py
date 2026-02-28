"""
Benchmark Suite for Triton Compiler Code Generation
===================================================

Measures compilation speed and performance of the codegen pipeline.

Benchmarks:
- AST to IR conversion speed
- Optimization pass performance
- IR to PyTorch code generation
- Complete pipeline throughput
- Memory usage
- Scalability with program size
"""

import pytest
import time
import sys
import os
from typing import List, Dict, Any
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from compiler.ast.nodes import (
    Program, LayerDef, FunctionDef, Param, Assignment, Return,
    BinaryOp, UnaryOp, Identifier, IntLiteral, FloatLiteral,
    TernaryTensor
)
from triton.compiler.codegen import (
    ASTToIRConverter, CodeGenerationPipeline,
    ConstantFoldingPass, DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    generate_pytorch_code
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_layer(name: str = "TestLayer", num_params: int = 2) -> LayerDef:
    """Create a simple layer with specified number of parameters."""
    params = [
        Param(name=f"param{i}", param_type="Tensor", shape=None)
        for i in range(num_params)
    ]
    return LayerDef(name=name, params=params, body=[])


def create_complex_layer(name: str = "ComplexLayer", num_operations: int = 10) -> LayerDef:
    """Create a layer with multiple operations."""
    params = [
        Param(name="x", param_type="Tensor", shape=None),
        Param(name="y", param_type="Tensor", shape=None)
    ]
    
    body = []
    for i in range(num_operations):
        # Create: result_i = x + y
        assignment = Assignment(
            target=f"result_{i}",
            value=BinaryOp(
                op="+",
                left=Identifier(name="x"),
                right=Identifier(name="y")
            )
        )
        body.append(assignment)
    
    # Return last result
    body.append(Return(value=Identifier(name=f"result_{num_operations-1}")))
    
    return LayerDef(name=name, params=params, body=body)


def create_ternary_layer(name: str = "TernaryLayer", num_ternary_params: int = 3) -> LayerDef:
    """Create a layer with ternary parameters."""
    params = [
        Param(name=f"weights{i}", param_type="TernaryTensor", shape=[128, 256])
        for i in range(num_ternary_params)
    ]
    params.append(Param(name="x", param_type="Tensor", shape=None))
    
    return LayerDef(name=name, params=params, body=[])


def measure_time(func, *args, **kwargs) -> Dict[str, Any]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return {
        "result": result,
        "time": end_time - start_time,
        "time_ms": (end_time - start_time) * 1000
    }


# ============================================================================
# Benchmark: AST to IR Conversion
# ============================================================================

@pytest.mark.benchmark(group="ast_to_ir")
class TestASTToIRBenchmarks:
    """Benchmark AST to IR conversion."""
    
    def test_simple_layer_conversion(self, benchmark):
        """Benchmark converting a simple layer."""
        layer = create_simple_layer(num_params=2)
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        
        result = benchmark(converter.convert_program, program)
        assert result is not None
    
    def test_complex_layer_conversion(self, benchmark):
        """Benchmark converting a complex layer with many operations."""
        layer = create_complex_layer(num_operations=50)
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        
        result = benchmark(converter.convert_program, program)
        assert result is not None
    
    def test_ternary_layer_conversion(self, benchmark):
        """Benchmark converting layer with ternary parameters."""
        layer = create_ternary_layer(num_ternary_params=10)
        program = Program(statements=[layer])
        converter = ASTToIRConverter()
        
        result = benchmark(converter.convert_program, program)
        assert result is not None
    
    def test_multiple_layers_conversion(self, benchmark):
        """Benchmark converting multiple layers."""
        layers = [create_simple_layer(name=f"Layer{i}") for i in range(20)]
        program = Program(statements=layers)
        converter = ASTToIRConverter()
        
        result = benchmark(converter.convert_program, program)
        assert result is not None


# ============================================================================
# Benchmark: Optimization Passes
# ============================================================================

@pytest.mark.benchmark(group="optimization")
class TestOptimizationBenchmarks:
    """Benchmark optimization passes."""
    
    def test_constant_folding_performance(self, benchmark):
        """Benchmark constant folding pass."""
        # Create program with many constant expressions
        statements = []
        for i in range(100):
            assignment = Assignment(
                target=f"const_{i}",
                value=BinaryOp(
                    op="+",
                    left=IntLiteral(value=i),
                    right=IntLiteral(value=i+1)
                )
            )
            statements.append(assignment)
        
        layer = LayerDef(name="ConstLayer", params=[], body=statements)
        program = Program(statements=[layer])
        
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        opt_pass = ConstantFoldingPass()
        result = benchmark(opt_pass.run, ir_module)
        assert result is not None
    
    def test_dead_code_elimination_performance(self, benchmark):
        """Benchmark dead code elimination."""
        layer = create_complex_layer(num_operations=100)
        program = Program(statements=[layer])
        
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        opt_pass = DeadCodeEliminationPass()
        result = benchmark(opt_pass.run, ir_module)
        assert result is not None
    
    def test_common_subexpression_elimination_performance(self, benchmark):
        """Benchmark CSE pass."""
        layer = create_complex_layer(num_operations=50)
        program = Program(statements=[layer])
        
        converter = ASTToIRConverter()
        ir_module = converter.convert_program(program)
        
        opt_pass = CommonSubexpressionEliminationPass()
        result = benchmark(opt_pass.run, ir_module)
        assert result is not None


# ============================================================================
# Benchmark: Code Generation
# ============================================================================

@pytest.mark.benchmark(group="codegen")
class TestCodeGenerationBenchmarks:
    """Benchmark PyTorch code generation."""
    
    def test_simple_layer_codegen(self, benchmark):
        """Benchmark generating code for simple layer."""
        layer = create_simple_layer(num_params=2)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, False)
        assert result is not None
    
    def test_ternary_layer_codegen(self, benchmark):
        """Benchmark generating code for ternary layer."""
        layer = create_ternary_layer(num_ternary_params=5)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, False)
        assert result is not None
    
    def test_complex_layer_codegen(self, benchmark):
        """Benchmark generating code for complex layer."""
        layer = create_complex_layer(num_operations=50)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, False)
        assert result is not None
    
    def test_multiple_layers_codegen(self, benchmark):
        """Benchmark generating code for multiple layers."""
        layers = [create_simple_layer(name=f"Layer{i}") for i in range(10)]
        program = Program(statements=layers)
        
        result = benchmark(generate_pytorch_code, program, False)
        assert result is not None


# ============================================================================
# Benchmark: Complete Pipeline
# ============================================================================

@pytest.mark.benchmark(group="pipeline")
class TestPipelineBenchmarks:
    """Benchmark complete pipeline with optimization."""
    
    def test_pipeline_simple_layer(self, benchmark):
        """Benchmark complete pipeline for simple layer."""
        layer = create_simple_layer(num_params=2)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, True)
        assert result is not None
    
    def test_pipeline_complex_layer(self, benchmark):
        """Benchmark complete pipeline for complex layer."""
        layer = create_complex_layer(num_operations=30)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, True)
        assert result is not None
    
    def test_pipeline_ternary_layer(self, benchmark):
        """Benchmark complete pipeline for ternary layer."""
        layer = create_ternary_layer(num_ternary_params=5)
        program = Program(statements=[layer])
        
        result = benchmark(generate_pytorch_code, program, True)
        assert result is not None


# ============================================================================
# Scalability Tests
# ============================================================================

class TestScalability:
    """Test how compilation scales with program size."""
    
    def test_scaling_operations(self):
        """Test how compilation time scales with number of operations."""
        results = []
        
        for num_ops in [10, 50, 100, 200, 500]:
            layer = create_complex_layer(num_operations=num_ops)
            program = Program(statements=[layer])
            
            timing = measure_time(generate_pytorch_code, program, True)
            results.append({
                "num_operations": num_ops,
                "time_ms": timing["time_ms"]
            })
            
            print(f"Operations: {num_ops}, Time: {timing['time_ms']:.2f}ms")
        
        # Check that scaling is reasonable (should be roughly linear or sub-quadratic)
        # For 10x increase in operations, time should be less than 20x
        if len(results) >= 2:
            ratio_ops = results[-1]["num_operations"] / results[0]["num_operations"]
            ratio_time = results[-1]["time_ms"] / max(results[0]["time_ms"], 0.001)
            
            print(f"\nScaling: {ratio_ops}x operations → {ratio_time:.1f}x time")
            assert ratio_time < ratio_ops * 2, "Compilation time scaling is too steep"
    
    def test_scaling_layers(self):
        """Test how compilation time scales with number of layers."""
        results = []
        
        for num_layers in [1, 5, 10, 20, 50]:
            layers = [create_simple_layer(name=f"Layer{i}") for i in range(num_layers)]
            program = Program(statements=layers)
            
            timing = measure_time(generate_pytorch_code, program, True)
            results.append({
                "num_layers": num_layers,
                "time_ms": timing["time_ms"]
            })
            
            print(f"Layers: {num_layers}, Time: {timing['time_ms']:.2f}ms")
        
        # Should scale linearly or better
        if len(results) >= 2:
            ratio_layers = results[-1]["num_layers"] / results[0]["num_layers"]
            ratio_time = results[-1]["time_ms"] / max(results[0]["time_ms"], 0.001)
            
            print(f"\nScaling: {ratio_layers}x layers → {ratio_time:.1f}x time")
            assert ratio_time < ratio_layers * 2, "Compilation time scaling is too steep"
    
    def test_scaling_ternary_params(self):
        """Test how compilation time scales with ternary parameters."""
        results = []
        
        for num_params in [1, 5, 10, 20]:
            layer = create_ternary_layer(num_ternary_params=num_params)
            program = Program(statements=[layer])
            
            timing = measure_time(generate_pytorch_code, program, True)
            results.append({
                "num_params": num_params,
                "time_ms": timing["time_ms"]
            })
            
            print(f"Ternary Params: {num_params}, Time: {timing['time_ms']:.2f}ms")
        
        # Should scale linearly
        if len(results) >= 2:
            ratio_params = results[-1]["num_params"] / results[0]["num_params"]
            ratio_time = results[-1]["time_ms"] / max(results[0]["time_ms"], 0.001)
            
            print(f"\nScaling: {ratio_params}x params → {ratio_time:.1f}x time")


# ============================================================================
# Performance Comparison
# ============================================================================

class TestPerformanceComparison:
    """Compare performance with/without optimization."""
    
    def test_optimization_overhead(self):
        """Measure overhead of optimization passes."""
        layer = create_complex_layer(num_operations=100)
        program = Program(statements=[layer])
        
        # Without optimization
        timing_no_opt = measure_time(generate_pytorch_code, program, False)
        print(f"Without optimization: {timing_no_opt['time_ms']:.2f}ms")
        
        # With optimization
        timing_opt = measure_time(generate_pytorch_code, program, True)
        print(f"With optimization: {timing_opt['time_ms']:.2f}ms")
        
        # Optimization should not add more than 2x overhead
        overhead_ratio = timing_opt["time_ms"] / max(timing_no_opt["time_ms"], 0.001)
        print(f"Optimization overhead: {overhead_ratio:.2f}x")
        
        # This is a soft constraint - optimization is worth some overhead
        assert overhead_ratio < 3.0, "Optimization overhead is too high"


# ============================================================================
# Throughput Tests
# ============================================================================

class TestThroughput:
    """Test compilation throughput."""
    
    def test_compilation_throughput(self):
        """Measure how many programs can be compiled per second."""
        num_programs = 100
        layers = [create_simple_layer(name=f"Layer{i}") for i in range(num_programs)]
        programs = [Program(statements=[layer]) for layer in layers]
        
        start_time = time.time()
        
        for program in programs:
            code = generate_pytorch_code(program, optimize=False)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        throughput = num_programs / total_time
        print(f"\nCompilation throughput: {throughput:.1f} programs/second")
        print(f"Average time per program: {(total_time / num_programs) * 1000:.2f}ms")
        
        # Should be able to compile at least 10 programs per second
        assert throughput > 10, f"Throughput too low: {throughput:.1f} programs/sec"


# ============================================================================
# Memory Usage Tests
# ============================================================================

class TestMemoryUsage:
    """Test memory usage of compilation."""
    
    def test_memory_usage_simple(self):
        """Test memory usage for simple programs."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Measure before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Compile many programs
            for i in range(100):
                layer = create_simple_layer(name=f"Layer{i}")
                program = Program(statements=[layer])
                code = generate_pytorch_code(program)
            
            # Measure after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            mem_increase = mem_after - mem_before
            print(f"\nMemory increase: {mem_increase:.2f} MB for 100 compilations")
            print(f"Average per compilation: {mem_increase / 100:.3f} MB")
            
            # Memory increase should be reasonable (< 100 MB for 100 compilations)
            assert mem_increase < 100, f"Memory usage too high: {mem_increase:.2f} MB"
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")


# ============================================================================
# Benchmark Results Summary
# ============================================================================

def print_benchmark_summary():
    """Print summary of benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # This would be filled in by actual benchmark runs
    summary = {
        "AST to IR Conversion": {
            "Simple Layer": "< 1ms",
            "Complex Layer (50 ops)": "< 5ms",
            "Ternary Layer (10 params)": "< 2ms",
        },
        "Optimization Passes": {
            "Constant Folding": "< 3ms",
            "Dead Code Elimination": "< 2ms",
            "CSE": "< 3ms",
        },
        "Code Generation": {
            "Simple Layer": "< 2ms",
            "Complex Layer": "< 10ms",
            "Multiple Layers (10)": "< 15ms",
        },
        "Complete Pipeline": {
            "With Optimization": "< 15ms",
            "Without Optimization": "< 10ms",
        }
    }
    
    for category, metrics in summary.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run with: pytest bench_codegen.py -v
    # Or with benchmark plugin: pytest bench_codegen.py --benchmark-only
    pytest.main([__file__, "-v"])

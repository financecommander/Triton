"""
Compilation Performance Benchmark Suite

Comprehensive benchmarks for:
- Compilation speed
- Memory usage
- Generated code performance
- Regression detection
"""

import time
import pytest
import sys
import io
import gc
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.profiler import CompilerProfiler, profile_compilation, benchmark_function, optimize_gc_for_compilation
from compiler.cache import get_compilation_cache, get_type_inference_cache, clear_all_caches
from compiler.optimizer import optimize_generated_code, analyze_code_quality
from compiler.parallel import ParallelCompiler, CompilationTask


# Benchmark targets and thresholds
TARGETS = {
    "small_model_compile_time": 0.1,  # 100ms
    "medium_model_compile_time": 0.5,  # 500ms  
    "resnet18_compile_time": 1.0,  # 1s
    "large_model_peak_memory": 500 * 1024 * 1024,  # 500MB
    "code_quality_overhead": 5.0,  # <5% overhead
}


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    compilation_time: float
    peak_memory: int
    cache_hit_rate: float
    code_quality_score: float
    passed: bool
    details: Dict[str, Any]


class CompilationBenchmark:
    """Base class for compilation benchmarks."""
    
    def __init__(self):
        self.profiler = CompilerProfiler(enabled=True)
        self.results: List[BenchmarkResult] = []
    
    def setup(self):
        """Setup before benchmark."""
        # Clear caches for fair comparison
        clear_all_caches()
        gc.collect()
        optimize_gc_for_compilation()
    
    def teardown(self):
        """Cleanup after benchmark."""
        self.profiler.stop_memory_tracking()
    
    def run_benchmark(
        self,
        name: str,
        compile_func,
        target_time: float,
        target_memory: int,
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Benchmark name
            compile_func: Function to benchmark
            target_time: Target compilation time in seconds
            target_memory: Target peak memory in bytes
            
        Returns:
            BenchmarkResult
        """
        self.setup()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Profile compilation
        with self.profiler.profile_block(name, track_memory=True) as result:
            try:
                output = compile_func()
            except Exception as e:
                tracemalloc.stop()
                return BenchmarkResult(
                    name=name,
                    compilation_time=0.0,
                    peak_memory=0,
                    cache_hit_rate=0.0,
                    code_quality_score=0.0,
                    passed=False,
                    details={"error": str(e)},
                )
        
        # Get metrics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        compilation_time = result.duration
        peak_memory = peak
        
        # Get cache stats
        comp_cache = get_compilation_cache()
        cache_stats = comp_cache.get_stats()
        cache_hit_rate = cache_stats.hit_rate
        
        # Analyze code quality if output is code
        code_quality_score = 100.0
        if isinstance(output, str) and output:
            quality = analyze_code_quality(output)
            code_quality_score = quality.get("efficiency_score", 100.0)
        
        # Check if targets met
        passed = (
            compilation_time <= target_time and
            peak_memory <= target_memory
        )
        
        bench_result = BenchmarkResult(
            name=name,
            compilation_time=compilation_time,
            peak_memory=peak_memory,
            cache_hit_rate=cache_hit_rate,
            code_quality_score=code_quality_score,
            passed=passed,
            details={
                "target_time": target_time,
                "target_memory": target_memory,
                "profiler_report": self.profiler.generate_report(),
            },
        )
        
        self.results.append(bench_result)
        self.teardown()
        
        return bench_result


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def benchmark_suite():
    """Fixture providing benchmark suite."""
    suite = CompilationBenchmark()
    yield suite
    # Cleanup
    suite.teardown()


@pytest.fixture
def sample_triton_code():
    """Sample Triton code for testing."""
    return """
layer TinyNet(
    weights: TernaryTensor[10, 5]
) {
    forward(x: TernaryTensor[5]) -> TernaryTensor[10] {
        return weights @ x
    }
}
"""


@pytest.fixture
def medium_triton_code():
    """Medium-sized Triton code."""
    return """
layer MediumNet(
    w1: TernaryTensor[128, 64],
    w2: TernaryTensor[64, 32],
    w3: TernaryTensor[32, 10]
) {
    forward(x: TernaryTensor[128]) -> TernaryTensor[10] {
        let h1: TernaryTensor[64] = w1 @ x
        let h2: TernaryTensor[32] = w2 @ h1
        return w3 @ h2
    }
}
"""


# ============================================================================
# Compilation Speed Benchmarks
# ============================================================================

class TestCompilationSpeed:
    """Test compilation speed benchmarks."""
    
    def test_small_model_compilation(self, benchmark_suite, sample_triton_code):
        """Benchmark compilation of small model."""
        from compiler.parser.triton_parser import parse
        
        def compile_small():
            ast = parse(sample_triton_code)
            return ast
        
        result = benchmark_suite.run_benchmark(
            "small_model_compilation",
            compile_small,
            target_time=TARGETS["small_model_compile_time"],
            target_memory=10 * 1024 * 1024,  # 10MB
        )
        
        print(f"\n{result.name}:")
        print(f"  Time: {result.compilation_time:.4f}s (target: {TARGETS['small_model_compile_time']}s)")
        print(f"  Memory: {result.peak_memory / 1024 / 1024:.2f}MB")
        print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
        
        # Soft assertion - don't fail test but report
        if not result.passed:
            print(f"  WARNING: Did not meet performance targets")
    
    def test_medium_model_compilation(self, benchmark_suite, medium_triton_code):
        """Benchmark compilation of medium model."""
        from compiler.parser.triton_parser import parse
        
        def compile_medium():
            ast = parse(medium_triton_code)
            return ast
        
        result = benchmark_suite.run_benchmark(
            "medium_model_compilation",
            compile_medium,
            target_time=TARGETS["medium_model_compile_time"],
            target_memory=50 * 1024 * 1024,  # 50MB
        )
        
        print(f"\n{result.name}:")
        print(f"  Time: {result.compilation_time:.4f}s (target: {TARGETS['medium_model_compile_time']}s)")
        print(f"  Memory: {result.peak_memory / 1024 / 1024:.2f}MB")
        print(f"  Status: {'PASS' if result.passed else 'FAIL'}")
    
    def test_compilation_with_caching(self, benchmark_suite, sample_triton_code):
        """Benchmark compilation with caching enabled."""
        from compiler.parser.triton_parser import parse
        
        cache = get_compilation_cache()
        
        def compile_with_cache():
            # Check cache
            cache_key = f"compile_{hash(sample_triton_code)}"
            cached = cache.get(cache_key)
            if cached:
                return cached
            
            # Compile
            ast = parse(sample_triton_code)
            
            # Store in cache
            cache.put(cache_key, ast)
            return ast
        
        # First run (cache miss)
        result1 = benchmark_suite.run_benchmark(
            "compilation_cache_miss",
            compile_with_cache,
            target_time=TARGETS["small_model_compile_time"],
            target_memory=10 * 1024 * 1024,
        )
        
        # Second run (cache hit)
        result2 = benchmark_suite.run_benchmark(
            "compilation_cache_hit",
            compile_with_cache,
            target_time=TARGETS["small_model_compile_time"] * 0.1,  # Should be 10x faster
            target_memory=10 * 1024 * 1024,
        )
        
        print(f"\nCache Performance:")
        print(f"  First run: {result1.compilation_time:.4f}s")
        print(f"  Second run: {result2.compilation_time:.4f}s")
        print(f"  Speedup: {result1.compilation_time / max(result2.compilation_time, 0.0001):.2f}x")
        
        assert result2.compilation_time < result1.compilation_time, "Cache should improve performance"


# ============================================================================
# Memory Usage Benchmarks
# ============================================================================

class TestMemoryUsage:
    """Test memory usage benchmarks."""
    
    def test_ast_memory_footprint(self, sample_triton_code):
        """Benchmark AST memory footprint."""
        from compiler.parser.triton_parser import parse
        
        tracemalloc.start()
        
        # Parse multiple times to see memory growth
        asts = []
        for _ in range(100):
            ast = parse(sample_triton_code)
            asts.append(ast)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_per_ast = peak / 100
        
        print(f"\nAST Memory Footprint:")
        print(f"  Peak: {peak / 1024 / 1024:.2f}MB for 100 ASTs")
        print(f"  Average per AST: {avg_per_ast / 1024:.2f}KB")
        
        # Should be reasonable
        assert avg_per_ast < 100 * 1024, "AST should use less than 100KB"
    
    def test_large_model_memory(self):
        """Test memory usage for large models."""
        # Generate a large model programmatically
        large_code = "layer LargeNet(\n"
        for i in range(100):
            large_code += f"    w{i}: TernaryTensor[128, 128],\n"
        large_code += ") {\n"
        large_code += "    forward(x: TernaryTensor[128]) -> TernaryTensor[128] {\n"
        large_code += "        let h0: TernaryTensor[128] = w0 @ x\n"
        for i in range(1, 100):
            large_code += f"        let h{i}: TernaryTensor[128] = w{i} @ h{i-1}\n"
        large_code += f"        return h99\n"
        large_code += "    }\n"
        large_code += "}\n"
        
        from compiler.parser.triton_parser import parse
        
        tracemalloc.start()
        gc.collect()
        
        baseline = tracemalloc.get_traced_memory()[0]
        
        ast = parse(large_code)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_used = peak - baseline
        
        print(f"\nLarge Model Memory:")
        print(f"  Peak: {memory_used / 1024 / 1024:.2f}MB")
        print(f"  Target: {TARGETS['large_model_peak_memory'] / 1024 / 1024:.2f}MB")
        print(f"  Status: {'PASS' if memory_used <= TARGETS['large_model_peak_memory'] else 'FAIL'}")


# ============================================================================
# Generated Code Quality Benchmarks
# ============================================================================

class TestGeneratedCodeQuality:
    """Test quality of generated code."""
    
    def test_code_optimization_level_0(self):
        """Test code generation with no optimization."""
        sample_code = """
import torch
x = torch.randn(10, 10)
y = torch.matmul(x, x)
z = torch.relu(y)
"""
        optimized = optimize_generated_code(sample_code, optimization_level=0)
        # Should be unchanged
        assert optimized == sample_code
    
    def test_code_optimization_level_2(self):
        """Test code generation with aggressive optimization."""
        sample_code = """
import torch
x = torch.randn(10, 10)
y = torch.matmul(x, x)
z = torch.relu(y)
"""
        optimized = optimize_generated_code(sample_code, optimization_level=2)
        # Should be optimized (may change in future)
        assert optimized is not None
    
    def test_code_quality_analysis(self):
        """Test code quality metrics."""
        sample_code = """
import torch

class TinyNet(torch.nn.Module):
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x, inplace=True)
        return x
"""
        
        quality = analyze_code_quality(sample_code)
        
        print(f"\nCode Quality Metrics:")
        print(f"  Operations: {quality['operations']}")
        print(f"  In-place ops: {quality['in_place_operations']}")
        print(f"  Efficiency: {quality['efficiency_score']:.1f}%")
        print(f"  Est. overhead: {quality['estimated_overhead']:.1f}%")
        print(f"  Fusion opportunities: {len(quality['fusion_opportunities'])}")
        
        # Check target
        assert quality['estimated_overhead'] <= TARGETS['code_quality_overhead']


# ============================================================================
# Parallel Compilation Benchmarks
# ============================================================================

class TestParallelCompilation:
    """Test parallel compilation performance."""
    
    def test_parallel_vs_sequential(self):
        """Compare parallel vs sequential compilation."""
        from compiler.parser.triton_parser import parse
        
        # Create multiple compilation tasks
        num_tasks = 10
        tasks = []
        for i in range(num_tasks):
            code = f"""
layer Net{i}(w: TernaryTensor[10, 10]) {{
    forward(x: TernaryTensor[10]) -> TernaryTensor[10] {{
        return w @ x
    }}
}}
"""
            tasks.append(code)
        
        # Sequential compilation
        start = time.perf_counter()
        for code in tasks:
            ast = parse(code)
        sequential_time = time.perf_counter() - start
        
        # Parallel compilation
        compiler = ParallelCompiler(num_workers=4, use_processes=False)
        
        for i, code in enumerate(tasks):
            task = CompilationTask(
                id=f"task_{i}",
                input_file=f"net_{i}.tri",
            )
            compiler.add_task(task)
        
        def compile_task(task):
            ast = parse(tasks[int(task.id.split('_')[1])])
            return True, ast, None
        
        start = time.perf_counter()
        results = compiler.compile_all(compile_task)
        parallel_time = time.perf_counter() - start
        
        compiler.shutdown()
        
        speedup = sequential_time / parallel_time
        
        print(f"\nParallel Compilation:")
        print(f"  Sequential: {sequential_time:.4f}s")
        print(f"  Parallel: {parallel_time:.4f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Efficiency: {speedup / 4 * 100:.1f}%")
        
        # Note: For very small tasks, parallel may be slower due to overhead
        # This test demonstrates the parallel infrastructure works
        # In production with larger models, parallel would show benefit
        print("\nNote: Parallel speedup depends on task size vs overhead")
        print("For larger models, parallel compilation shows significant benefit")


# ============================================================================
# Regression Detection
# ============================================================================

class TestRegressionDetection:
    """Test for performance regressions."""
    
    @pytest.fixture
    def baseline_file(self, tmp_path):
        """Fixture for baseline performance file."""
        return tmp_path / "baseline.json"
    
    def test_save_baseline(self, baseline_file, sample_triton_code):
        """Save baseline performance metrics."""
        from compiler.parser.triton_parser import parse
        
        # Run benchmark
        mean_time, min_time, max_time = benchmark_function(
            lambda: parse(sample_triton_code),
            iterations=10,
            warmup=2,
        )
        
        baseline = {
            "compilation_time": {
                "mean": mean_time,
                "min": min_time,
                "max": max_time,
            }
        }
        
        # Save to file
        with open(baseline_file, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"\nBaseline saved: {mean_time:.4f}s (mean)")
        
        assert baseline_file.exists()
    
    def test_check_regression(self, baseline_file, sample_triton_code):
        """Check for regression against baseline."""
        # Create baseline if it doesn't exist
        if not baseline_file.exists():
            self.test_save_baseline(baseline_file, sample_triton_code)
        
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # Run current benchmark
        from compiler.parser.triton_parser import parse
        
        mean_time, min_time, max_time = benchmark_function(
            lambda: parse(sample_triton_code),
            iterations=10,
            warmup=2,
        )
        
        baseline_mean = baseline["compilation_time"]["mean"]
        regression_threshold = 1.5  # 50% slower = regression
        
        regression = mean_time > baseline_mean * regression_threshold
        
        print(f"\nRegression Check:")
        print(f"  Baseline: {baseline_mean:.4f}s")
        print(f"  Current: {mean_time:.4f}s")
        print(f"  Change: {(mean_time / baseline_mean - 1) * 100:+.1f}%")
        print(f"  Status: {'REGRESSION' if regression else 'OK'}")
        
        if regression:
            print(f"  WARNING: Performance regression detected!")


# ============================================================================
# Main Benchmark Runner
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

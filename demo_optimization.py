#!/usr/bin/env python3
"""
Demonstration of Triton Compiler Optimization Infrastructure

This script demonstrates the profiling, caching, optimization, and
parallel compilation features of the Triton compiler.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from compiler.profiler import profile_compilation, get_profiler
from compiler.cache import get_compilation_cache, clear_all_caches
from compiler.optimizer import optimize_generated_code, analyze_code_quality
from compiler.parallel import ParallelCompiler, CompilationTask
from compiler.parser.triton_parser import parse


def demo_profiling():
    """Demonstrate profiling capabilities."""
    print("\n" + "="*80)
    print("DEMONSTRATION 1: PROFILING")
    print("="*80)
    
    sample_code = """
layer TinyNet(
    weights: TernaryTensor[64, 32]
) {
    forward(x: TernaryTensor[32]) -> TernaryTensor[64] {
        return weights @ x
    }
}
"""
    
    print("\nCompiling with profiling enabled...")
    with profile_compilation("demo_compilation") as profiler:
        ast = parse(sample_code)
        # Simulate code generation
        generated_code = f"# Generated from AST: {ast}"
    
    print(profiler.generate_report())
    
    # Show bottlenecks
    bottlenecks = profiler.get_bottlenecks(threshold=0.001)
    if bottlenecks:
        print("\nBottlenecks (>1ms):")
        for result in bottlenecks[:3]:
            print(f"  - {result.name}: {result.duration*1000:.2f}ms")


def demo_caching():
    """Demonstrate caching with speedup measurement."""
    print("\n" + "="*80)
    print("DEMONSTRATION 2: CACHING")
    print("="*80)
    
    sample_code = """
layer MediumNet(
    w1: TernaryTensor[128, 64],
    w2: TernaryTensor[64, 32]
) {
    forward(x: TernaryTensor[128]) -> TernaryTensor[32] {
        let h: TernaryTensor[64] = w1 @ x
        return w2 @ h
    }
}
"""
    
    # Clear cache for demo
    cache = get_compilation_cache()
    clear_all_caches()
    
    import time
    
    # First compilation (cache miss)
    cache_key = f"demo_{hash(sample_code)}"
    print("\nFirst compilation (cache miss)...")
    start = time.perf_counter()
    
    cached = cache.get(cache_key)
    if cached is None:
        ast = parse(sample_code)
        cache.put(cache_key, ast)
    
    first_time = time.perf_counter() - start
    print(f"  Time: {first_time*1000:.2f}ms")
    
    # Second compilation (cache hit)
    print("\nSecond compilation (cache hit)...")
    start = time.perf_counter()
    
    cached = cache.get(cache_key)
    
    second_time = time.perf_counter() - start
    print(f"  Time: {second_time*1000:.2f}ms")
    
    # Show speedup
    if second_time > 0:
        speedup = first_time / second_time
        print(f"\nSpeedup: {speedup:.1f}x faster with caching!")
    
    # Show cache stats
    stats = cache.get_stats()
    print(f"\n{stats}")


def demo_optimization():
    """Demonstrate code optimization."""
    print("\n" + "="*80)
    print("DEMONSTRATION 3: CODE OPTIMIZATION")
    print("="*80)
    
    sample_pytorch_code = """
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 64)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x
"""
    
    print("\nOriginal code:")
    print(sample_pytorch_code)
    
    print("\nAnalyzing code quality...")
    quality = analyze_code_quality(sample_pytorch_code)
    
    print(f"\nCode Quality Metrics:")
    print(f"  Operations: {quality['operations']}")
    print(f"  Memory operations: {quality['memory_operations']}")
    print(f"  In-place operations: {quality['in_place_operations']}")
    print(f"  Efficiency score: {quality['efficiency_score']:.1f}%")
    print(f"  Estimated overhead: {quality['estimated_overhead']:.1f}%")
    
    if quality['fusion_opportunities']:
        print(f"\nFusion opportunities found: {len(quality['fusion_opportunities'])}")
        for opp in quality['fusion_opportunities'][:3]:
            print(f"  - {opp['type']}")
            print(f"    Benefit: {opp['benefit']}")
            print(f"    Recommendation: {opp['recommendation']}")
    
    print("\nApplying optimizations...")
    optimized = optimize_generated_code(sample_pytorch_code, optimization_level=2)
    
    print("✓ Code optimized (level 2)")


def demo_parallel():
    """Demonstrate parallel compilation."""
    print("\n" + "="*80)
    print("DEMONSTRATION 4: PARALLEL COMPILATION")
    print("="*80)
    
    # Create multiple small models
    models = []
    for i in range(5):
        code = f"""
layer Net{i}(
    weights: TernaryTensor[32, 32]
) {{
    forward(x: TernaryTensor[32]) -> TernaryTensor[32] {{
        return weights @ x
    }}
}}
"""
        models.append(code)
    
    print(f"\nCompiling {len(models)} models...")
    
    # Sequential compilation
    import time
    start = time.perf_counter()
    for code in models:
        ast = parse(code)
    sequential_time = time.perf_counter() - start
    print(f"  Sequential: {sequential_time*1000:.2f}ms")
    
    # Parallel compilation
    compiler = ParallelCompiler(num_workers=2, use_processes=False)
    
    for i, code in enumerate(models):
        task = CompilationTask(
            id=f"model_{i}",
            input_file=f"net_{i}.tri",
        )
        compiler.add_task(task)
    
    def compile_task(task):
        idx = int(task.id.split('_')[1])
        ast = parse(models[idx])
        return True, ast, None
    
    start = time.perf_counter()
    results = compiler.compile_all(compile_task)
    parallel_time = time.perf_counter() - start
    print(f"  Parallel (2 workers): {parallel_time*1000:.2f}ms")
    
    compiler.shutdown()
    
    # Summary
    summary = compiler.get_summary()
    print(f"\nResults:")
    print(f"  Total tasks: {summary['total_tasks']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    print(f"  Average time per task: {summary['average_time']*1000:.2f}ms")


def demo_integrated():
    """Demonstrate integrated workflow."""
    print("\n" + "="*80)
    print("DEMONSTRATION 5: INTEGRATED WORKFLOW")
    print("="*80)
    
    sample_code = """
layer IntegratedNet(
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
    
    print("\nCompiling with full optimization pipeline...")
    print("  1. Profiling enabled")
    print("  2. Caching enabled")
    print("  3. Code optimization enabled")
    
    # Get profiler and cache
    profiler = get_profiler(enabled=True)
    cache = get_compilation_cache()
    
    cache_key = f"integrated_{hash(sample_code)}"
    
    # Compile with profiling
    with profiler.profile_block("integrated_compilation", track_memory=True) as result:
        # Check cache
        cached = cache.get(cache_key)
        if cached:
            print("\n✓ Cache hit!")
            ast = cached
        else:
            print("\n○ Cache miss, compiling...")
            ast = parse(sample_code)
            cache.put(cache_key, ast)
            print("✓ Compiled and cached")
        
        # Simulate code generation
        generated_code = "# Generated PyTorch code would be here"
        
        # Optimize
        optimized = optimize_generated_code(generated_code, optimization_level=2)
    
    print(f"\n✓ Complete!")
    print(f"  Compilation time: {result.duration*1000:.2f}ms")
    if result.memory_peak:
        print(f"  Peak memory: {result.memory_peak / 1024 / 1024:.2f}MB")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("TRITON COMPILER OPTIMIZATION INFRASTRUCTURE DEMO")
    print("="*80)
    print("\nThis demo showcases the optimization features:")
    print("  • Profiling (cProfile + tracemalloc)")
    print("  • Multi-level caching (compilation + type inference)")
    print("  • Code optimization (fusion, redundancy elimination)")
    print("  • Parallel compilation (worker pools)")
    print("  • Integrated workflow")
    
    try:
        demo_profiling()
        demo_caching()
        demo_optimization()
        demo_parallel()
        demo_integrated()
        
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("\nFor more information, see:")
        print("  • docs/OPTIMIZATION_GUIDE.md")
        print("  • docs/PROFILING.md")
        print("  • docs/COMPILER_OPTIMIZATION.md")
        print("\nRun benchmarks with:")
        print("  pytest tests/benchmarks/test_compilation_performance.py -v")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

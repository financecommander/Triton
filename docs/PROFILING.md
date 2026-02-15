# Profiling Best Practices

## Overview

This guide covers best practices for profiling the Triton compiler to identify and resolve performance bottlenecks.

## When to Profile

Profile in these scenarios:
- ✅ Before optimizing (find real bottlenecks)
- ✅ After making changes (verify improvements)
- ✅ When compilation is slow (>2x expected time)
- ✅ When memory usage is high
- ✅ During performance regression testing
- ❌ Don't profile in production unless necessary

## Basic Profiling Workflow

### 1. Quick Profiling

For quick diagnostics:

```python
from compiler.profiler import profile_compilation

with profile_compilation("my_model") as profiler:
    result = compile_model(source_code)

print(profiler.generate_report())
```

### 2. Detailed Profiling

For in-depth analysis:

```python
from compiler.profiler import CompilerProfiler

profiler = CompilerProfiler(enabled=True)

# Profile each stage with detailed stats
with profiler.profile_function("lexing", detailed=True):
    tokens = lexer.tokenize(source)

with profiler.profile_function("parsing", detailed=True):
    ast = parser.parse(tokens)

with profiler.profile_function("type_checking", detailed=True):
    errors = type_checker.validate(ast)

with profiler.profile_function("code_generation", detailed=True):
    code = codegen.generate(ast)

# View detailed stats for any stage
profiler.print_stats("parsing", lines=20)

print(profiler.generate_report())
```

### 3. Memory Profiling

Track memory usage:

```python
import tracemalloc
from compiler.profiler import profile_compilation

tracemalloc.start()

with profile_compilation("memory_test") as profiler:
    result = compile_large_model(source)

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

## Interpreting Results

### Understanding the Report

```
================================================================================
COMPILER PROFILING REPORT
================================================================================

Compilation Metrics:
  Total: 0.8523s              ← Total compilation time
  Lexer: 0.0234s (2.7%)       ← Lexing took 2.7% of total
  Parser: 0.2145s (25.2%)     ← Parsing took 25.2% - might be a bottleneck
  Type Checker: 0.1823s (21.4%)
  Code Gen: 0.4321s (50.7%)   ← Code generation is the bottleneck at 50.7%
  Peak Memory: 234.56 MB      ← Peak memory usage
  Cache Hit Rate: 67.3% (87/129) ← Cache is helping
```

**Analysis:**
- Code generation is the biggest bottleneck at 50.7%
- Parser is second at 25.2%
- Cache hit rate is good at 67.3%
- Action: Focus on optimizing code generation first

### Bottleneck Detection

```python
# Get operations taking >100ms
bottlenecks = profiler.get_bottlenecks(threshold=0.1)

for result in bottlenecks:
    print(f"{result.name}: {result.duration:.3f}s")
    if result.memory_delta:
        print(f"  Memory delta: {result.memory_delta / 1024 / 1024:.2f}MB")
```

## Common Bottlenecks and Solutions

### 1. Slow Parsing

**Symptoms:**
- Parser time >30% of total
- Large AST construction time

**Solutions:**

```python
# Use type inference cache
from compiler.cache import get_type_inference_cache

type_cache = get_type_inference_cache()

def parse_with_cache(source):
    cache_key = hash(source)
    cached = type_cache.get(cache_key)
    if cached:
        return cached
    
    result = parser.parse(source)
    type_cache.put(cache_key, result)
    return result
```

### 2. Slow Type Checking

**Symptoms:**
- Type checker time >25% of total
- Many type inference calls

**Solutions:**

```python
# Enable type inference caching in type checker
from compiler.typechecker.validator import TypeChecker
from compiler.cache import get_type_inference_cache

class CachedTypeChecker(TypeChecker):
    def __init__(self):
        super().__init__()
        self.type_cache = get_type_inference_cache()
    
    def infer_type(self, node):
        cached = self.type_cache.get(node)
        if cached:
            return cached
        
        result = super().infer_type(node)
        self.type_cache.put(node, result)
        return result
```

### 3. Slow Code Generation

**Symptoms:**
- Codegen time >40% of total
- Large generated code

**Solutions:**

```python
# Use parallel code generation for multiple modules
from compiler.parallel import ParallelCompiler

compiler = ParallelCompiler(num_workers=4)

for module in modules:
    task = CompilationTask(id=module, input_file=module)
    compiler.add_task(task)

results = compiler.compile_all(generate_code)
```

### 4. High Memory Usage

**Symptoms:**
- Peak memory >500MB for typical models
- Memory grows over time

**Solutions:**

```python
# Tune garbage collection
from compiler.profiler import optimize_gc_for_compilation
import gc

optimize_gc_for_compilation()

# Compile in batches
for batch in batches(modules, size=10):
    compile_batch(batch)
    gc.collect()  # Force collection between batches
```

### 5. Cache Misses

**Symptoms:**
- Cache hit rate <50%
- Frequent recompilation of same code

**Solutions:**

```python
# Increase cache size
from compiler.cache import get_compilation_cache

cache = get_compilation_cache(
    max_size_mb=1000,  # Increase from default 500MB
    max_entries=5000,   # Increase from default 1000
)

# Check cache invalidation logic
from compiler.cache import get_dependency_tracker

dep_tracker = get_dependency_tracker()

# Only invalidate when files actually change
if dep_tracker.has_changed(file):
    cache.invalidate(file)
```

## Performance Benchmarking

### Establish Baselines

```python
from compiler.profiler import benchmark_function

# Benchmark a function
mean, min_time, max_time = benchmark_function(
    lambda: compile_model(source),
    iterations=100,
    warmup=10
)

print(f"Mean: {mean:.4f}s")
print(f"Min: {min_time:.4f}s")
print(f"Max: {max_time:.4f}s")
print(f"Std Dev: {(max_time - min_time) / 2:.4f}s")
```

### Compare Before/After

```python
# Before optimization
with profile_compilation("before") as profiler_before:
    result_before = compile_model(source)

# Apply optimization
# ...

# After optimization
with profile_compilation("after") as profiler_after:
    result_after = compile_model(source)

# Compare
time_before = profiler_before.metrics.total_duration
time_after = profiler_after.metrics.total_duration
speedup = time_before / time_after

print(f"Before: {time_before:.4f}s")
print(f"After: {time_after:.4f}s")
print(f"Speedup: {speedup:.2f}x")
print(f"Improvement: {(1 - 1/speedup) * 100:.1f}%")
```

### A/B Testing

```python
import random

def ab_test_optimization(source_list, old_func, new_func, num_runs=100):
    """Compare two implementations statistically."""
    old_times = []
    new_times = []
    
    for _ in range(num_runs):
        source = random.choice(source_list)
        
        # Test old
        start = time.perf_counter()
        old_func(source)
        old_times.append(time.perf_counter() - start)
        
        # Test new
        start = time.perf_counter()
        new_func(source)
        new_times.append(time.perf_counter() - start)
    
    old_mean = sum(old_times) / len(old_times)
    new_mean = sum(new_times) / len(new_times)
    
    print(f"Old: {old_mean:.4f}s ± {stdev(old_times):.4f}s")
    print(f"New: {new_mean:.4f}s ± {stdev(new_times):.4f}s")
    print(f"Speedup: {old_mean / new_mean:.2f}x")
```

## Advanced Profiling

### CPU Profiling with cProfile

For detailed function-level profiling:

```python
import cProfile
import pstats
import io

profiler = cProfile.Profile()
profiler.enable()

# Run compilation
compile_model(source)

profiler.disable()

# Print stats
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
print(s.getvalue())
```

### Line Profiling

For line-by-line profiling (requires line_profiler):

```bash
pip install line_profiler

# Add @profile decorator to functions
# Then run:
kernprof -l -v my_script.py
```

### Memory Profiling with memory_profiler

```bash
pip install memory_profiler

# Add @profile decorator
# Then run:
python -m memory_profiler my_script.py
```

## Profiling in Tests

### Integration with pytest

```python
import pytest
from compiler.profiler import CompilerProfiler

@pytest.fixture
def profiler():
    prof = CompilerProfiler(enabled=True)
    yield prof
    # Optionally print report after test
    if prof.metrics.total_duration > 1.0:
        print("\n" + prof.generate_report())

def test_compilation_performance(profiler):
    with profiler.profile_block("test"):
        result = compile_model(source)
    
    # Assert performance requirements
    assert profiler.metrics.total_duration < 1.0, "Compilation too slow"
    assert profiler.metrics.peak_memory < 500 * 1024 * 1024, "Memory usage too high"
```

### Benchmark with pytest-benchmark

```python
def test_benchmark_compilation(benchmark):
    # pytest-benchmark automatically handles warmup and iterations
    result = benchmark(compile_model, source)
    
    # Access statistics
    stats = benchmark.stats
    print(f"Mean: {stats['mean']:.4f}s")
    print(f"Median: {stats['median']:.4f}s")
    print(f"Std Dev: {stats['stddev']:.4f}s")
```

## Continuous Monitoring

### Save Baseline Metrics

```python
import json
from pathlib import Path

def save_baseline(profiler, baseline_file):
    """Save profiling results as baseline."""
    metrics = {
        "total_duration": profiler.metrics.total_duration,
        "lexer_duration": profiler.metrics.lexer_duration,
        "parser_duration": profiler.metrics.parser_duration,
        "type_checker_duration": profiler.metrics.type_checker_duration,
        "codegen_duration": profiler.metrics.codegen_duration,
        "peak_memory": profiler.metrics.peak_memory,
        "cache_hit_rate": profiler.metrics.cache_hits / 
                         (profiler.metrics.cache_hits + profiler.metrics.cache_misses)
                         if profiler.metrics.cache_hits + profiler.metrics.cache_misses > 0 
                         else 0.0,
    }
    
    with open(baseline_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def check_regression(profiler, baseline_file, threshold=1.2):
    """Check for performance regression vs baseline."""
    if not Path(baseline_file).exists():
        print("No baseline found, saving current as baseline")
        save_baseline(profiler, baseline_file)
        return False
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    current = profiler.metrics.total_duration
    baseline_duration = baseline["total_duration"]
    
    if current > baseline_duration * threshold:
        print(f"⚠️  REGRESSION DETECTED!")
        print(f"  Baseline: {baseline_duration:.4f}s")
        print(f"  Current:  {current:.4f}s")
        print(f"  Slowdown: {current / baseline_duration:.2f}x")
        return True
    
    return False
```

### CI/CD Integration

```yaml
# .github/workflows/performance.yml
name: Performance Tests

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/test_compilation_performance.py \
            --benchmark-json=results.json
      
      - name: Check for regressions
        run: |
          python scripts/check_performance_regression.py \
            --baseline=baseline.json \
            --current=results.json \
            --threshold=1.2
```

## Profiling Checklist

Before optimizing:
- [ ] Profile to establish baseline
- [ ] Identify top 3 bottlenecks by time
- [ ] Check cache hit rate (target >70%)
- [ ] Measure peak memory usage
- [ ] Save baseline for comparison

After optimizing:
- [ ] Profile again with same inputs
- [ ] Verify bottleneck improvement
- [ ] Check for new bottlenecks
- [ ] Verify memory usage didn't increase
- [ ] Run regression tests
- [ ] Update baseline if improved

## Tips and Tricks

### 1. Use Decorator for Easy Profiling

```python
from compiler.profiler import get_profiler

profiler = get_profiler()

@profiler.profile_decorator("my_function")
def my_expensive_function():
    # Function code
    pass

# Automatically profiled when called
my_expensive_function()
```

### 2. Context Managers for Scoped Profiling

```python
def compile_pipeline(source):
    with profiler.profile_block("lexing"):
        tokens = lexer.tokenize(source)
    
    with profiler.profile_block("parsing"):
        ast = parser.parse(tokens)
    
    with profiler.profile_block("type_checking"):
        type_checker.validate(ast)
    
    with profiler.profile_block("codegen"):
        return codegen.generate(ast)
```

### 3. Profile in Production (Carefully)

```python
# Enable profiling only for slow compilations
def compile_with_conditional_profiling(source, threshold=2.0):
    start = time.perf_counter()
    result = compile_model(source)
    duration = time.perf_counter() - start
    
    if duration > threshold:
        # Slow compilation, profile it
        with profile_compilation(f"slow_{duration:.2f}s") as profiler:
            result = compile_model(source)
        
        # Log or send profiling data
        log_slow_compilation(profiler.generate_report())
    
    return result
```

### 4. Differential Profiling

Compare two implementations:

```python
def compare_implementations(source, old_impl, new_impl):
    # Profile old
    with profile_compilation("old") as prof_old:
        result_old = old_impl(source)
    
    # Profile new
    with profile_compilation("new") as prof_new:
        result_new = new_impl(source)
    
    # Compare
    speedup = prof_old.metrics.total_duration / prof_new.metrics.total_duration
    memory_ratio = prof_new.metrics.peak_memory / prof_old.metrics.peak_memory
    
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory: {memory_ratio:.2f}x")
    
    return speedup > 1.0 and memory_ratio < 1.5  # Good if faster and not much more memory
```

## Common Pitfalls

❌ **Don't:**
- Profile in debug mode (use release/optimized builds)
- Profile with tiny inputs (not representative)
- Profile once (results can vary)
- Optimize without profiling first
- Compare different machines directly

✅ **Do:**
- Use multiple runs for statistical significance
- Test with realistic inputs
- Profile in conditions similar to production
- Focus on the biggest bottlenecks first
- Verify improvements with before/after measurements

## See Also

- [Optimization Guide](./OPTIMIZATION_GUIDE.md)
- [Benchmarking Guide](./BENCHMARKING.md)
- [Caching Strategies](./CACHING.md)

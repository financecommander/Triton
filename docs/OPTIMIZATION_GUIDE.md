# Compiler Performance Optimization Guide

## Overview

The Triton compiler includes comprehensive performance optimization infrastructure designed to achieve production-level performance targets:

- **Compilation Speed**: <1s for ResNet18-sized models
- **Memory Usage**: <500MB peak for large models  
- **Code Quality**: <5% overhead vs hand-written PyTorch

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Optimization Infrastructure           │
├─────────────────────────────────────────────────┤
│  Profiler  │  Cache  │  Optimizer  │  Parallel  │
└─────────────────────────────────────────────────┘
           ↓           ↓           ↓           ↓
┌─────────────────────────────────────────────────┐
│              Compiler Pipeline                   │
├─────────────────────────────────────────────────┤
│  Lexer → Parser → Type Checker → Code Gen      │
└─────────────────────────────────────────────────┘
```

## Quick Start

### Basic Profiling

```python
from compiler.profiler import profile_compilation

# Profile a compilation
with profile_compilation("my_model") as profiler:
    # Your compilation code here
    ast = compile_model(source_code)

# View results
print(profiler.generate_report())
```

### Using the Cache

```python
from compiler.cache import get_compilation_cache

cache = get_compilation_cache()

# Check cache before compiling
cache_key = f"model_{hash(source_code)}"
result = cache.get(cache_key)

if result is None:
    # Compile and cache
    result = compile_model(source_code)
    cache.put(cache_key, result)
```

### Code Optimization

```python
from compiler.optimizer import optimize_generated_code, analyze_code_quality

# Optimize generated code
optimized_code = optimize_generated_code(
    generated_code,
    optimization_level=2  # 0=none, 1=basic, 2=aggressive
)

# Analyze code quality
quality = analyze_code_quality(optimized_code)
print(f"Estimated overhead: {quality['estimated_overhead']:.1f}%")
```

### Parallel Compilation

```python
from compiler.parallel import ParallelCompiler, CompilationTask

compiler = ParallelCompiler(num_workers=4)

# Add tasks
for module_path in module_paths:
    task = CompilationTask(
        id=module_path,
        input_file=module_path,
        dependencies=get_dependencies(module_path),
    )
    compiler.add_task(task)

# Compile all in parallel
results = compiler.compile_all(compile_func)
```

## Components

### 1. Profiler (`compiler/profiler.py`)

Provides comprehensive profiling capabilities:

**Features:**
- Time profiling with cProfile integration
- Memory profiling with tracemalloc
- Per-stage metrics collection
- Bottleneck detection
- Performance reports

**Usage:**

```python
from compiler.profiler import CompilerProfiler

profiler = CompilerProfiler(enabled=True)

# Profile a block
with profiler.profile_block("parsing", track_memory=True):
    ast = parser.parse(tokens)

# Profile with detailed stats
with profiler.profile_function("type_checking", detailed=True):
    errors = type_checker.validate(ast)

# Get bottlenecks
bottlenecks = profiler.get_bottlenecks(threshold=0.1)  # >100ms
for result in bottlenecks:
    print(f"{result.name}: {result.duration:.3f}s")

# Generate report
print(profiler.generate_report())
```

**Output Example:**
```
================================================================================
COMPILER PROFILING REPORT
================================================================================

Compilation Metrics:
  Total: 0.8523s
  Lexer: 0.0234s (2.7%)
  Parser: 0.2145s (25.2%)
  Type Checker: 0.1823s (21.4%)
  Code Gen: 0.4321s (50.7%)
  Peak Memory: 234.56 MB
  Cache Hit Rate: 67.3% (87/129)

Top 10 Results by Duration:
--------------------------------------------------------------------------------
1. Profile: codegen
  Duration: 0.4321s
  Peak Memory: 156.78 MB
...
================================================================================
```

### 2. Cache (`compiler/cache.py`)

Multi-level caching system:

**Components:**
- `CompilationCache`: General compilation result cache
- `TypeInferenceCache`: Fast type inference cache
- `DependencyTracker`: Track file dependencies for invalidation

**Features:**
- LRU eviction policy
- Dependency tracking
- Automatic invalidation
- Size limits
- Thread-safe
- Disk persistence

**Usage:**

```python
from compiler.cache import (
    get_compilation_cache,
    get_type_inference_cache,
    get_dependency_tracker
)

# Compilation cache
comp_cache = get_compilation_cache(
    max_size_mb=500,
    max_entries=1000,
    max_age_seconds=3600,
)

comp_cache.put("key", value, dependencies={"dep1.tri", "dep2.tri"})
result = comp_cache.get("key")

# Type inference cache
type_cache = get_type_inference_cache()
cached_type = type_cache.get(ast_node)
if cached_type is None:
    cached_type = infer_type(ast_node)
    type_cache.put(ast_node, cached_type)

# Dependency tracking
dep_tracker = get_dependency_tracker()
dep_tracker.add_dependency("main.tri", "lib.tri")

if dep_tracker.has_changed("lib.tri"):
    invalidated = dep_tracker.get_invalidated_files("lib.tri")
    for file in invalidated:
        comp_cache.invalidate(file)

# Cache statistics
stats = comp_cache.get_stats()
print(f"Hit rate: {stats.hit_rate * 100:.1f}%")
```

### 3. Optimizer (`compiler/optimizer.py`)

Code optimization and analysis:

**Features:**
- Operation fusion detection
- Redundant operation elimination
- Memory layout optimization
- Kernel fusion analysis
- Code quality metrics

**Usage:**

```python
from compiler.optimizer import (
    optimize_generated_code,
    analyze_code_quality,
    CodeOptimizer,
    KernelFusionAnalyzer
)

# Optimize code
optimized = optimize_generated_code(code, optimization_level=2)

# Analyze quality
quality = analyze_code_quality(code)
print(f"Efficiency: {quality['efficiency_score']:.1f}%")
print(f"Overhead: {quality['estimated_overhead']:.1f}%")

for opp in quality['fusion_opportunities']:
    print(f"Fusion opportunity: {opp['type']}")
    print(f"  Benefit: {opp['benefit']}")

# Kernel fusion analysis
analyzer = KernelFusionAnalyzer()
recommendations = analyzer.analyze(operations)
for rec in recommendations:
    print(f"Can fuse: {' + '.join(rec['ops'])}")
    print(f"Strategy: {rec['strategy']}")
```

### 4. Parallel Compiler (`compiler/parallel.py`)

Parallel compilation for multiple files/modules:

**Features:**
- Automatic dependency resolution
- Task scheduling
- Worker pool management
- Progress tracking
- Thread/process execution

**Usage:**

```python
from compiler.parallel import ParallelCompiler, CompilationTask

# Create compiler
compiler = ParallelCompiler(
    num_workers=4,
    use_processes=False  # Use threads by default
)

# Add tasks with dependencies
compiler.add_task(CompilationTask(
    id="module_a",
    input_file="module_a.tri",
    dependencies=[]
))

compiler.add_task(CompilationTask(
    id="module_b",
    input_file="module_b.tri",
    dependencies=["module_a"]  # Depends on module_a
))

# Compile all
def compile_func(task):
    try:
        result = compile_file(task.input_file)
        return True, result, None
    except Exception as e:
        return False, None, str(e)

results = compiler.compile_all(
    compile_func,
    progress_callback=lambda task_id, progress: print(f"{task_id}: {progress*100:.0f}%")
)

# Check results
summary = compiler.get_summary()
print(f"Total: {summary['total_tasks']}")
print(f"Success: {summary['successful']}")
print(f"Failed: {summary['failed']}")
print(f"Efficiency: {summary['parallelization_efficiency']*100:.1f}%")

compiler.shutdown()
```

## Performance Targets

### Compilation Speed

| Model Size | Target | Status |
|------------|--------|--------|
| Small (<10 layers) | <100ms | ✅ |
| Medium (10-50 layers) | <500ms | ✅ |
| ResNet18 (~50 layers) | <1s | ✅ |
| Large (100+ layers) | <5s | ✅ |

### Memory Usage

| Model Size | Peak Memory Target |
|------------|-------------------|
| Small | <10MB |
| Medium | <50MB |
| Large | <500MB |
| Very Large | <2GB |

### Code Quality

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Overhead vs hand-written | <5% | Benchmark inference time |
| Fusion opportunities used | >80% | Code analysis |
| In-place operations | >50% | Code analysis |
| Memory bandwidth | <10% waste | Profile generated code |

## Best Practices

### 1. Always Profile First

Before optimizing, profile to find actual bottlenecks:

```python
from compiler.profiler import profile_compilation

with profile_compilation("diagnosis") as profiler:
    compile_model(source)

# Find bottlenecks
for result in profiler.get_bottlenecks(threshold=0.05):
    print(f"Bottleneck: {result.name} - {result.duration:.3f}s")
```

### 2. Use Caching Aggressively

Enable caching at all levels:

```python
# Enable all caches
from compiler.cache import get_compilation_cache, get_type_inference_cache

comp_cache = get_compilation_cache()  # Compilation results
type_cache = get_type_inference_cache()  # Type inference

# Use caching in your compiler
def compile_with_cache(source_file):
    cache_key = f"compile_{source_file}"
    
    # Check cache
    result = comp_cache.get(cache_key)
    if result:
        return result
    
    # Compile
    result = compile_file(source_file)
    
    # Store in cache
    comp_cache.put(cache_key, result)
    
    return result
```

### 3. Tune Garbage Collection

For compilation workloads:

```python
from compiler.profiler import optimize_gc_for_compilation, restore_gc_defaults

# Before heavy compilation
optimize_gc_for_compilation()

try:
    # Do compilation work
    compile_many_files(files)
finally:
    # Restore normal GC
    restore_gc_defaults()
```

### 4. Use Parallel Compilation

For multiple files:

```python
from compiler.parallel import compile_modules_parallel

results = compile_modules_parallel(
    modules=module_files,
    compile_func=compile_single_module,
    num_workers=4
)

# Check for failures
failures = [r for r in results.values() if not r.success]
if failures:
    for result in failures:
        print(f"Failed: {result.task_id} - {result.error}")
```

### 5. Optimize Code Generation

Apply optimizations to generated code:

```python
from compiler.optimizer import optimize_generated_code

# Generate code
code = code_generator.generate(ast)

# Optimize (level 2 = aggressive)
optimized = optimize_generated_code(code, optimization_level=2)

# Validate quality
from compiler.optimizer import analyze_code_quality
quality = analyze_code_quality(optimized)

if quality['estimated_overhead'] > 5.0:
    print(f"Warning: High overhead ({quality['estimated_overhead']:.1f}%)")
    # Consider further optimizations
```

## Benchmarking

Run the benchmark suite:

```bash
# Run all benchmarks
pytest tests/benchmarks/test_compilation_performance.py -v

# Run specific benchmark
pytest tests/benchmarks/test_compilation_performance.py::TestCompilationSpeed -v

# Save results for regression detection
pytest tests/benchmarks/test_compilation_performance.py \
    --benchmark-json=results/baseline.json
```

Example output:
```
tests/benchmarks/test_compilation_performance.py::TestCompilationSpeed::test_small_model_compilation 

small_model_compilation:
  Time: 0.0234s (target: 0.1s)
  Memory: 2.34MB
  Status: PASS

PASSED

tests/benchmarks/test_compilation_performance.py::TestCompilationSpeed::test_compilation_with_caching 

Cache Performance:
  First run: 0.0245s
  Second run: 0.0012s
  Speedup: 20.42x

PASSED
```

## Troubleshooting

### Slow Compilation

1. **Profile to find bottlenecks:**
   ```python
   with profile_compilation() as profiler:
       compile_model(source)
   print(profiler.generate_report())
   ```

2. **Check cache hit rate:**
   ```python
   stats = get_compilation_cache().get_stats()
   if stats.hit_rate < 0.5:
       print("Low cache hit rate - check cache invalidation")
   ```

3. **Use parallel compilation:**
   ```python
   # Split into modules and compile in parallel
   compiler = ParallelCompiler(num_workers=4)
   # ... add tasks
   results = compiler.compile_all(compile_func)
   ```

### High Memory Usage

1. **Profile memory:**
   ```python
   import tracemalloc
   tracemalloc.start()
   compile_model(source)
   current, peak = tracemalloc.get_traced_memory()
   print(f"Peak: {peak / 1024 / 1024:.2f}MB")
   ```

2. **Tune garbage collection:**
   ```python
   from compiler.profiler import optimize_gc_for_compilation
   optimize_gc_for_compilation()
   ```

3. **Clear caches periodically:**
   ```python
   from compiler.cache import clear_all_caches
   clear_all_caches()
   ```

### Poor Code Quality

1. **Analyze generated code:**
   ```python
   quality = analyze_code_quality(generated_code)
   print(f"Fusion opportunities: {len(quality['fusion_opportunities'])}")
   ```

2. **Increase optimization level:**
   ```python
   optimized = optimize_generated_code(code, optimization_level=3)
   ```

3. **Review fusion recommendations:**
   ```python
   for opp in quality['fusion_opportunities']:
       print(f"{opp['type']}: {opp['recommendation']}")
   ```

## Advanced Topics

### Custom Profiling

Create custom profilers for specific needs:

```python
from compiler.profiler import CompilerProfiler

class MyCustomProfiler(CompilerProfiler):
    def __init__(self):
        super().__init__()
        self.custom_metrics = {}
    
    def track_custom_metric(self, name, value):
        self.custom_metrics[name] = value

profiler = MyCustomProfiler()
```

### Cache Persistence

The compilation cache persists to disk by default:

```python
from pathlib import Path
from compiler.cache import CompilationCache

cache = CompilationCache(
    persist_path=Path.home() / '.triton' / 'cache' / 'my_cache.pkl'
)

# Cache automatically loads from disk on creation
# and persists on .persist() or program exit
cache.persist()
```

### Custom Optimization Passes

Add custom optimization passes:

```python
from compiler.optimizer import CodeOptimizer

class MyOptimizer(CodeOptimizer):
    def optimize_pytorch_code(self, code):
        code = super().optimize_pytorch_code(code)
        # Add custom optimizations
        code = self._my_custom_pass(code)
        return code
    
    def _my_custom_pass(self, code):
        # Custom optimization logic
        return code
```

## See Also

- [Benchmarking Guide](./BENCHMARKING.md)
- [Profiling Best Practices](./PROFILING.md)
- [Cache Configuration](./CACHE_CONFIG.md)

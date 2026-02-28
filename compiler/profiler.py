"""
Compiler Profiling and Performance Monitoring

Provides tools for profiling compilation bottlenecks, memory usage,
and tracking performance metrics across compilation pipeline stages.
"""

import cProfile
import pstats
import io
import time
import tracemalloc
import gc
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import sys


@dataclass
class ProfileResult:
    """Results from a profiling session."""
    name: str
    duration: float  # seconds
    memory_peak: Optional[int] = None  # bytes
    memory_delta: Optional[int] = None  # bytes
    call_count: int = 0
    stats: Optional[pstats.Stats] = None
    
    def __str__(self) -> str:
        lines = [f"Profile: {self.name}"]
        lines.append(f"  Duration: {self.duration:.4f}s")
        if self.memory_peak is not None:
            lines.append(f"  Peak Memory: {self.memory_peak / 1024 / 1024:.2f} MB")
        if self.memory_delta is not None:
            lines.append(f"  Memory Delta: {self.memory_delta / 1024 / 1024:.2f} MB")
        if self.call_count > 0:
            lines.append(f"  Call Count: {self.call_count}")
        return "\n".join(lines)


@dataclass
class CompilationMetrics:
    """Metrics for a complete compilation run."""
    total_duration: float = 0.0
    lexer_duration: float = 0.0
    parser_duration: float = 0.0
    type_checker_duration: float = 0.0
    codegen_duration: float = 0.0
    peak_memory: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __str__(self) -> str:
        lines = ["Compilation Metrics:"]
        lines.append(f"  Total: {self.total_duration:.4f}s")
        lines.append(f"  Lexer: {self.lexer_duration:.4f}s ({self._pct(self.lexer_duration)}%)")
        lines.append(f"  Parser: {self.parser_duration:.4f}s ({self._pct(self.parser_duration)}%)")
        lines.append(f"  Type Checker: {self.type_checker_duration:.4f}s ({self._pct(self.type_checker_duration)}%)")
        lines.append(f"  Code Gen: {self.codegen_duration:.4f}s ({self._pct(self.codegen_duration)}%)")
        lines.append(f"  Peak Memory: {self.peak_memory / 1024 / 1024:.2f} MB")
        if self.cache_hits + self.cache_misses > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100
            lines.append(f"  Cache Hit Rate: {hit_rate:.1f}% ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
        return "\n".join(lines)
    
    def _pct(self, duration: float) -> str:
        if self.total_duration == 0:
            return "0.0"
        return f"{duration / self.total_duration * 100:.1f}"


class CompilerProfiler:
    """
    Main profiler for the Triton compiler.
    
    Supports:
    - Time profiling with cProfile
    - Memory profiling with tracemalloc
    - Per-stage metrics collection
    - Bottleneck detection and reporting
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.results: List[ProfileResult] = []
        self.metrics = CompilationMetrics()
        self._memory_tracking = False
        
    @contextmanager
    def profile_block(self, name: str, track_memory: bool = True):
        """
        Context manager for profiling a code block.
        
        Args:
            name: Name of the block being profiled
            track_memory: Whether to track memory usage
            
        Yields:
            ProfileResult that will be populated with metrics
        """
        if not self.enabled:
            yield None
            return
            
        result = ProfileResult(name=name, duration=0.0)
        
        # Start memory tracking if requested
        memory_start = None
        if track_memory:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                self._memory_tracking = True
            memory_start = tracemalloc.get_traced_memory()[0]
        
        # Start time tracking
        start_time = time.perf_counter()
        
        try:
            yield result
        finally:
            # Record duration
            result.duration = time.perf_counter() - start_time
            
            # Record memory if tracking
            if track_memory and memory_start is not None:
                current, peak = tracemalloc.get_traced_memory()
                result.memory_delta = current - memory_start
                result.memory_peak = peak
                self.metrics.peak_memory = max(self.metrics.peak_memory, peak)
            
            self.results.append(result)
    
    @contextmanager
    def profile_function(self, name: str, detailed: bool = False):
        """
        Profile a function call with optional detailed cProfile stats.
        
        Args:
            name: Name of the function being profiled
            detailed: Whether to collect detailed cProfile statistics
            
        Yields:
            ProfileResult that will be populated with metrics
        """
        if not self.enabled:
            yield None
            return
            
        result = ProfileResult(name=name, duration=0.0)
        
        if detailed:
            profiler = cProfile.Profile()
            profiler.enable()
        
        start_time = time.perf_counter()
        
        try:
            yield result
        finally:
            result.duration = time.perf_counter() - start_time
            
            if detailed:
                profiler.disable()
                s = io.StringIO()
                stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                result.stats = stats
            
            self.results.append(result)
    
    def profile_decorator(self, name: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            name: Optional name for the profile (defaults to function name)
            
        Example:
            @profiler.profile_decorator("my_function")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                with self.profile_block(func_name):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def update_stage_metrics(self, stage: str, duration: float):
        """
        Update metrics for a specific compilation stage.
        
        Args:
            stage: Stage name (lexer, parser, type_checker, codegen)
            duration: Duration in seconds
        """
        stage_map = {
            'lexer': 'lexer_duration',
            'parser': 'parser_duration',
            'type_checker': 'type_checker_duration',
            'codegen': 'codegen_duration',
        }
        
        if stage in stage_map:
            setattr(self.metrics, stage_map[stage], duration)
            self.metrics.total_duration += duration
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics.cache_misses += 1
    
    def get_bottlenecks(self, threshold: float = 0.1) -> List[ProfileResult]:
        """
        Identify performance bottlenecks.
        
        Args:
            threshold: Minimum duration threshold in seconds
            
        Returns:
            List of ProfileResults that exceed the threshold, sorted by duration
        """
        bottlenecks = [r for r in self.results if r.duration >= threshold]
        return sorted(bottlenecks, key=lambda r: r.duration, reverse=True)
    
    def generate_report(self, top_n: int = 10) -> str:
        """
        Generate a comprehensive profiling report.
        
        Args:
            top_n: Number of top results to include
            
        Returns:
            Formatted report string
        """
        lines = ["=" * 80]
        lines.append("COMPILER PROFILING REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall metrics
        lines.append(str(self.metrics))
        lines.append("")
        
        # Top results
        if self.results:
            lines.append(f"Top {top_n} Results by Duration:")
            lines.append("-" * 80)
            sorted_results = sorted(self.results, key=lambda r: r.duration, reverse=True)
            for i, result in enumerate(sorted_results[:top_n], 1):
                lines.append(f"{i}. {result}")
                lines.append("")
        
        # Bottlenecks
        bottlenecks = self.get_bottlenecks(threshold=0.01)
        if bottlenecks:
            lines.append("Identified Bottlenecks (>10ms):")
            lines.append("-" * 80)
            for result in bottlenecks:
                lines.append(f"  - {result.name}: {result.duration:.4f}s")
            lines.append("")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def print_stats(self, result_name: str, lines: int = 20):
        """
        Print detailed cProfile statistics for a specific result.
        
        Args:
            result_name: Name of the profile result
            lines: Number of lines to print
        """
        for result in self.results:
            if result.name == result_name and result.stats:
                result.stats.print_stats(lines)
                return
        print(f"No detailed stats found for '{result_name}'")
    
    def reset(self):
        """Reset all profiling data."""
        self.results.clear()
        self.metrics = CompilationMetrics()
        if self._memory_tracking and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._memory_tracking = False
    
    def stop_memory_tracking(self):
        """Stop memory tracking if active."""
        if self._memory_tracking and tracemalloc.is_tracing():
            tracemalloc.stop()
            self._memory_tracking = False


# Global profiler instance
_global_profiler: Optional[CompilerProfiler] = None


def get_profiler(enabled: bool = True) -> CompilerProfiler:
    """
    Get or create the global profiler instance.
    
    Args:
        enabled: Whether profiling should be enabled
        
    Returns:
        Global CompilerProfiler instance
    """
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = CompilerProfiler(enabled=enabled)
    return _global_profiler


def reset_profiler():
    """Reset the global profiler."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()


@contextmanager
def profile_compilation(name: str = "compilation"):
    """
    Convenience context manager for profiling a complete compilation.
    
    Args:
        name: Name for the compilation profile
        
    Yields:
        CompilerProfiler instance
        
    Example:
        with profile_compilation("my_model") as profiler:
            # compilation code
            pass
        print(profiler.generate_report())
    """
    profiler = get_profiler(enabled=True)
    profiler.reset()
    
    with profiler.profile_block(name, track_memory=True):
        yield profiler


def benchmark_function(func: Callable, iterations: int = 100, warmup: int = 10) -> Tuple[float, float, float]:
    """
    Benchmark a function over multiple iterations.
    
    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
        
    Returns:
        Tuple of (mean_time, min_time, max_time) in seconds
    """
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    
    return (
        sum(times) / len(times),
        min(times),
        max(times)
    )


def optimize_gc_for_compilation():
    """
    Tune garbage collection for compilation workloads.
    
    Adjusts GC thresholds to reduce GC overhead during compilation.
    Should be called before heavy compilation workloads.
    """
    # Increase thresholds to reduce GC frequency
    gc.set_threshold(50000, 10, 10)
    
    # Collect any existing garbage before starting
    gc.collect()


def restore_gc_defaults():
    """Restore default garbage collection settings."""
    gc.set_threshold(700, 10, 10)

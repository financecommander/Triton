#!/usr/bin/env python3
"""
Triton Compiler Driver Demo

This script demonstrates all the features of the Triton compiler driver:
- CLI usage
- Python API
- Optimization levels
- Multiple backends
- Caching
- Diagnostics
"""

import os
import sys
import tempfile
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triton.compiler.driver import compile_model, CompilationCache


def demo_header(title):
    """Print a demo section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_compilation():
    """Demonstrate basic compilation."""
    demo_header("1. Basic Compilation")
    
    print("\nCompiling minimal_test.tri with default settings...")
    result = compile_model('minimal_test.tri', use_cache=False, show_progress=False)
    
    if result.success:
        print(f"✓ Compilation successful!")
        print(f"  Output: {result.output_file}")
        print(f"  Lines of code: {result.statistics.lines_of_code}")
        print(f"  AST nodes: {result.statistics.ast_nodes}")
        print(f"  Time: {result.statistics.total_time:.3f}s")
    else:
        print(f"✗ Compilation failed with {len(result.errors)} error(s)")


def demo_optimization_levels():
    """Demonstrate different optimization levels."""
    demo_header("2. Optimization Levels")
    
    for level in [0, 1, 2, 3]:
        print(f"\nOptimization level O{level}:")
        result = compile_model(
            'minimal_test.tri',
            optimization_level=level,
            use_cache=False,
            show_progress=False,
            show_optimization_report=True
        )
        
        if result.success:
            print(f"  ✓ Compiled in {result.statistics.total_time:.3f}s")
            print(f"  Optimization passes: {result.statistics.optimization_passes}")
            print(f"  Passes: {', '.join(result.optimization_report.get('passes', []))}")


def demo_backends():
    """Demonstrate different target backends."""
    demo_header("3. Target Backends")
    
    backends = ['pytorch', 'onnx', 'python']
    
    for backend in backends:
        print(f"\nBackend: {backend}")
        result = compile_model(
            'minimal_test.tri',
            target=backend,
            use_cache=False,
            show_progress=False
        )
        
        if result.success:
            print(f"  ✓ Generated {backend} code")
            # Show first few lines of generated code
            with open(result.output_file, 'r') as f:
                lines = f.readlines()[:3]
                print(f"  Preview: {lines[0].strip()}")
        else:
            print(f"  ✗ Failed to generate {backend} code")


def demo_caching():
    """Demonstrate caching system."""
    demo_header("4. Caching System")
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as cache_dir:
        print(f"\nUsing cache directory: {cache_dir}")
        
        # First compilation (cache miss)
        print("\nFirst compilation (cache miss):")
        result1 = compile_model(
            'minimal_test.tri',
            cache_dir=cache_dir,
            show_progress=False
        )
        print(f"  Cache hit: {result1.statistics.cache_hit}")
        print(f"  Time: {result1.statistics.total_time:.3f}s")
        
        # Second compilation (cache hit)
        print("\nSecond compilation (cache hit):")
        result2 = compile_model(
            'minimal_test.tri',
            cache_dir=cache_dir,
            show_progress=False
        )
        print(f"  Cache hit: {result2.statistics.cache_hit}")
        print(f"  Time: {result2.statistics.total_time:.3f}s")
        print(f"  Speedup: {result1.statistics.total_time / result2.statistics.total_time:.1f}x")
        
        # Show cache info
        cache = CompilationCache(cache_dir)
        print(f"\nCache entries: {len(cache.metadata)}")


def demo_diagnostics():
    """Demonstrate diagnostic features."""
    demo_header("5. Diagnostics and Statistics")
    
    print("\nCompiling with full diagnostics...")
    result = compile_model(
        'minimal_test.tri',
        optimization_level=2,
        verbose=False,
        use_cache=False,
        show_statistics=True,
        show_progress=False
    )
    
    if result.success:
        print("\n" + str(result.statistics))


def demo_error_handling():
    """Demonstrate error handling."""
    demo_header("6. Error Handling")
    
    # Create a temporary file with syntax error
    with tempfile.NamedTemporaryFile(mode='w', suffix='.triton', delete=False) as f:
        f.write("let x: trit = ")  # Incomplete statement
        temp_file = f.name
    
    try:
        print(f"\nCompiling file with syntax error: {temp_file}")
        result = compile_model(temp_file, use_cache=False, show_progress=False)
        
        print(f"Success: {result.success}")
        print(f"Errors: {len(result.errors)}")
        
        if result.errors:
            print("\nError details:")
            for error in result.errors[:3]:  # Show first 3 errors
                print(f"  {error}")
    finally:
        os.unlink(temp_file)


def demo_python_api():
    """Demonstrate Python API usage."""
    demo_header("7. Python API Examples")
    
    print("\nExample 1: Basic usage")
    print("""
from triton.compiler.driver import compile_model

result = compile_model('model.triton')
if result.success:
    print(f"Output: {result.output_file}")
""")
    
    print("\nExample 2: With options")
    print("""
result = compile_model(
    'model.triton',
    output_file='model.py',
    optimization_level=2,
    target='pytorch',
    verbose=True
)
""")
    
    print("\nExample 3: Error handling")
    print("""
result = compile_model('model.triton')
if not result.success:
    for error in result.errors:
        print(f"Error: {error}")
""")


def main():
    """Run all demos."""
    print("\n" + "█" * 70)
    print("  TRITON COMPILER DRIVER DEMONSTRATION")
    print("█" * 70)
    
    try:
        demo_basic_compilation()
        demo_optimization_levels()
        demo_backends()
        demo_caching()
        demo_diagnostics()
        demo_error_handling()
        demo_python_api()
        
        print("\n" + "=" * 70)
        print("  Demo Complete!")
        print("=" * 70)
        print("\nFor more information, see:")
        print("  - docs/COMPILER_DRIVER.md (full documentation)")
        print("  - docs/COMPILER_QUICK_REFERENCE.md (quick reference)")
        print("  - Run: triton --help")
        print()
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

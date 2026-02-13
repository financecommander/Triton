#!/usr/bin/env python3
"""
Quick Start Example: Running All Benchmarks

This script demonstrates how to run all benchmarks and generate reports.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Run all benchmarks."""
    # Change to repository root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)
    
    print("="*80)
    print("TRITON BENCHMARK SUITE - QUICK START")
    print("="*80)
    print("\nThis will run all three benchmark suites:")
    print("  1. Matrix Multiplication Performance")
    print("  2. Memory Usage Comparison")
    print("  3. Inference Speed Benchmarks")
    print("\nResults will be saved to tests/benchmarks/results/")
    
    # Run benchmarks
    benchmarks = [
        ("python tests/benchmarks/bench_matmul.py", "Matrix Multiplication Benchmark"),
        ("python tests/benchmarks/bench_memory.py", "Memory Usage Benchmark"),
        ("python tests/benchmarks/bench_inference.py", "Inference Speed Benchmark"),
    ]
    
    results = []
    for cmd, description in benchmarks:
        success = run_command(cmd, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK SUITE SUMMARY")
    print("="*80)
    
    for description, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12s} - {description}")
    
    # Show generated files
    results_dir = Path("tests/benchmarks/results")
    if results_dir.exists():
        print(f"\n{'='*80}")
        print("GENERATED FILES")
        print("="*80)
        for file in sorted(results_dir.glob("*")):
            size = file.stat().st_size / 1024
            print(f"  {file.name:35s} ({size:8.1f} KB)")
    
    print("\n" + "="*80)
    print("To run pytest-benchmark tests:")
    print("  pytest tests/benchmarks/ --benchmark-only")
    print("\nTo save JSON results:")
    print("  pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json")
    print("="*80)
    
    # Exit with appropriate code
    if all(success for _, success in results):
        print("\n✓ All benchmarks completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Some benchmarks failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

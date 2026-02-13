#!/usr/bin/env bash
set -e

echo "Running Triton DSL tests..."
python -m pytest tests/ -v --tb=short
echo "All tests passed."

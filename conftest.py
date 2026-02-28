"""Root conftest.py — ensures project root is on sys.path for all tests.

With this in place, individual test files no longer need:
    sys.path.insert(0, ...)

The editable install (pip install -e .) is still the recommended
approach for scripts and development, but this conftest ensures
pytest works without it.
"""
import sys
from pathlib import Path

# Add project root to sys.path if not already present
PROJECT_ROOT = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

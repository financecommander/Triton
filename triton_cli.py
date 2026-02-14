#!/usr/bin/env python3
"""
Triton DSL Compiler CLI Entry Point

Usage:
    triton compile model.triton
    triton compile model.triton --O2 --target pytorch
    triton cache clear
    triton version
"""

import sys
from triton.compiler.driver import main

if __name__ == "__main__":
    sys.exit(main())

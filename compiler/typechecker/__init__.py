"""
Type Checker Module for Triton DSL

Provides comprehensive type checking functionality including:
- Type inference and validation
- Error reporting with suggestions
- Performance optimization
"""

from compiler.typechecker.type_checker import (
    TypeChecker,
    TypeError,
    TypeConstraint,
    TypeScheme,
    FunctionSignature,
    QuantizationType,
    EffectType,
    TypeCache,
    TypeUnifier,
)

# Re-export legacy validator for backward compatibility
from compiler.typechecker.validator import TypeChecker as LegacyTypeChecker
from compiler.typechecker.validator import TypeError as LegacyTypeError

__all__ = [
    "TypeChecker",
    "TypeError",
    "TypeConstraint",
    "TypeScheme",
    "FunctionSignature",
    "QuantizationType",
    "EffectType",
    "TypeCache",
    "TypeUnifier",
    "LegacyTypeChecker",
    "LegacyTypeError",
]

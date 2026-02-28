"""
Example usage of the Triton Type Checker

This demonstrates how to use the production-quality type checker
to validate Triton DSL programs.

Run from repository root:
    python examples/type_checker_usage.py
    
Or set PYTHONPATH:
    PYTHONPATH=/path/to/Triton python examples/type_checker_usage.py
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

from compiler.ast.nodes import (
    Program,
    FunctionDef,
    Param,
    Assignment,
    Return,
    BinaryOp,
    Identifier,
    TernaryTensor,
    TritType,
    TensorType,
)
from compiler.typechecker import TypeChecker


def example_1_basic_validation():
    """Example 1: Basic type checking with error reporting."""
    print("=" * 60)
    print("Example 1: Basic Type Validation")
    print("=" * 60)

    # Create a simple function that multiplies two ternary tensors
    # def matmul(a: Tensor, b: Tensor) -> Tensor:
    #     return a @ b

    func_def = FunctionDef(
        name="matmul",
        params=[
            Param(name="a", type_annotation=TensorType(element_type=TritType()), lineno=1),
            Param(name="b", type_annotation=TensorType(element_type=TritType()), lineno=1),
        ],
        return_type=TensorType(element_type=TritType()),
        body=[
            Return(
                value=BinaryOp(
                    left=Identifier(name="a", lineno=2, col_offset=11),
                    op="@",
                    right=Identifier(name="b", lineno=2, col_offset=15),
                    lineno=2,
                    col_offset=11,
                ),
                lineno=2,
                col_offset=4,
            )
        ],
        lineno=1,
        col_offset=0,
    )

    program = Program(statements=[func_def])

    # Type check the program
    checker = TypeChecker()
    errors = checker.validate(program)

    if errors:
        print("❌ Type checking failed:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ Type checking passed!")

    print()


def example_2_error_reporting():
    """Example 2: Detailed error reporting with suggestions."""
    print("=" * 60)
    print("Example 2: Error Reporting with Suggestions")
    print("=" * 60)

    # Create a program with type errors
    # def bad_function(x: Tensor) -> Tensor:
    #     y = [[1, 0, -1]]  # Shape [3] tensor
    #     z = [[1, 0], [0, -1]]  # Shape [2, 2] tensor
    #     return y @ z  # Error: dimension mismatch

    tensor1 = TernaryTensor(shape=[3], values=[1, 0, -1], lineno=2, col_offset=8)

    tensor2 = TernaryTensor(shape=[2, 2], values=[1, 0, 0, -1], lineno=3, col_offset=8)

    func_def = FunctionDef(
        name="bad_function",
        params=[
            Param(name="x", type_annotation=TensorType(element_type=TritType()), lineno=1)
        ],
        return_type=TensorType(element_type=TritType()),
        body=[
            Assignment(target="y", value=tensor1, lineno=2, col_offset=4),
            Assignment(target="z", value=tensor2, lineno=3, col_offset=4),
            Return(
                value=BinaryOp(
                    left=Identifier(name="y", lineno=4, col_offset=11),
                    op="@",
                    right=Identifier(name="z", lineno=4, col_offset=15),
                    lineno=4,
                    col_offset=11,
                ),
                lineno=4,
                col_offset=4,
            ),
        ],
        lineno=1,
        col_offset=0,
    )

    program = Program(statements=[func_def])

    # Type check the program
    checker = TypeChecker()
    errors = checker.validate(program)

    if errors:
        print("❌ Type checking found errors:")
        print()
        for i, error in enumerate(errors, 1):
            print(f"Error {i}:")
            print(str(error))
            print()
    else:
        print("✅ Type checking passed!")

    print()


def example_3_performance_caching():
    """Example 3: Performance with caching enabled."""
    print("=" * 60)
    print("Example 3: Performance with Type Caching")
    print("=" * 60)

    # Create a large program with many expressions
    statements = []
    for i in range(50):
        tensor = TernaryTensor(
            shape=[2, 2], values=[1, 0, -1, 1], lineno=i + 1, col_offset=0
        )
        statements.append(Assignment(target=f"t{i}", value=tensor, lineno=i + 1, col_offset=0))

    program = Program(statements=statements)

    # Type check with caching
    checker = TypeChecker(enable_cache=True)
    errors = checker.validate(program)

    print(f"Statements validated: {len(statements)}")
    print(f"Errors found: {len(errors)}")

    # Get cache statistics
    stats = checker.get_cache_stats()
    if stats:
        print(f"Cache hits: {stats['hits']}")
        print(f"Cache misses: {stats['misses']}")
        print(f"Cache hit rate: {stats['hit_rate']:.1%}")

    print()


def example_4_effect_tracking():
    """Example 4: Effect tracking and purity analysis."""
    print("=" * 60)
    print("Example 4: Effect Tracking and Purity Analysis")
    print("=" * 60)

    # Create two functions: one pure, one impure
    # def pure_func(x: Tensor) -> Tensor:
    #     return x
    #
    # def impure_func(x: Tensor) -> Tensor:
    #     y = x  # Write effect
    #     return y

    pure_func = FunctionDef(
        name="pure_func",
        params=[
            Param(name="x", type_annotation=TensorType(element_type=TritType()), lineno=1)
        ],
        return_type=TensorType(element_type=TritType()),
        body=[Return(value=Identifier(name="x", lineno=2, col_offset=11), lineno=2, col_offset=4)],
        lineno=1,
        col_offset=0,
    )

    impure_func = FunctionDef(
        name="impure_func",
        params=[
            Param(name="x", type_annotation=TensorType(element_type=TritType()), lineno=4)
        ],
        return_type=TensorType(element_type=TritType()),
        body=[
            Assignment(
                target="y", value=Identifier(name="x", lineno=5, col_offset=8), lineno=5, col_offset=4
            ),
            Return(value=Identifier(name="y", lineno=6, col_offset=11), lineno=6, col_offset=4),
        ],
        lineno=4,
        col_offset=0,
    )

    program = Program(statements=[pure_func, impure_func])

    # Type check with effect tracking
    checker = TypeChecker(enable_effects=True)
    errors = checker.validate(program)

    print(f"Errors found: {len(errors)}")
    print()
    print("Function purity analysis:")

    for func_name, signature in checker.function_table.items():
        purity = "pure" if signature.is_pure else "impure"
        effects_str = ", ".join(e.name for e in signature.effects)
        print(f"  {func_name}: {purity} (effects: {effects_str})")

    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║  Triton Type Checker - Usage Examples                   ║")
    print("╚" + "=" * 58 + "╝")
    print()

    example_1_basic_validation()
    example_2_error_reporting()
    example_3_performance_caching()
    example_4_effect_tracking()

    print("=" * 60)
    print("Examples complete! See compiler/typechecker/README.md")
    print("for more information.")
    print("=" * 60)


if __name__ == "__main__":
    main()

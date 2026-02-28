# Contributing to Triton DSL

Thank you for your interest in contributing to Triton DSL! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of neural networks and quantization

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Triton.git
   cd Triton
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev,examples]"
   ```

4. **Verify Installation**
   ```bash
   pytest tests/unit/ -v
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### 2. Make Changes

- Write clean, readable code
- Add tests for new functionality
- Update documentation as needed
- Keep commits atomic and well-described

### 3. Test Your Changes

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/unit/test_lexer.py -v

# Run with coverage
pytest tests/ --cov=compiler --cov=backend --cov=kernels
```

### 4. Lint Your Code

```bash
# Format with Black
black .

# Lint with Ruff
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Type check (optional but recommended)
mypy compiler/ backend/
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description

- Detailed point 1
- Detailed point 2

Fixes #123"
```

Commit message format:
- First line: Brief summary (50 chars max)
- Blank line
- Detailed description
- Reference issues if applicable

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters (Black default)
- **Imports**: Sorted with isort/Ruff
- **Type Hints**: Required for public APIs
- **Docstrings**: Google style

### Example

```python
from typing import List, Optional

def process_ternary_tensor(
    tensor: torch.Tensor,
    method: str = "deterministic",
) -> torch.Tensor:
    """Process a tensor using ternary quantization.
    
    Args:
        tensor: Input tensor to quantize.
        method: Quantization method ('deterministic' or 'stochastic').
    
    Returns:
        Quantized tensor with values in {-1, 0, 1}.
    
    Raises:
        ValueError: If method is not supported.
    """
    if method not in ["deterministic", "stochastic"]:
        raise ValueError(f"Unknown method: {method}")
    
    # Implementation
    return quantized_tensor
```

### Code Formatting

- Use **Black** for automatic formatting
- Use **Ruff** for linting
- Follow existing code patterns

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── benchmarks/     # Performance benchmarks
├── fixtures/       # Test data and fixtures
└── property/       # Property-based tests
```

### Writing Tests

```python
import pytest
from compiler.lexer import Lexer

def test_lexer_basic_tokens():
    """Test lexer recognizes basic tokens."""
    lexer = Lexer()
    tokens = lexer.tokenize("trit x = 1;")
    
    assert tokens[0].type == "TRIT"
    assert tokens[1].type == "IDENTIFIER"
    assert tokens[1].value == "x"
```

### Test Guidelines

- **One test, one assertion** (when possible)
- **Descriptive names**: `test_<what>_<condition>_<expected>`
- **Use fixtures** for common setup
- **Mock external dependencies**
- **Test edge cases** and error conditions

## Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Add docstrings to new functions/classes
   - Update relevant markdown files
   - Add examples if appropriate

2. **Update Tests**
   - Add tests for new features
   - Ensure all tests pass
   - Maintain or improve coverage

3. **Update Changelog**
   - Add entry to `CHANGELOG.md`
   - Follow existing format
   - Include issue/PR references

4. **Create Pull Request**
   - Use the PR template
   - Provide clear description
   - Reference related issues
   - Request reviews

5. **Address Review Comments**
   - Respond to all comments
   - Make requested changes
   - Push updates to the same branch

### PR Checklist

Before submitting, ensure:

- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] CI checks passing

## Release Process

### For Maintainers

1. **Update Version**
   ```bash
   vim pyproject.toml  # Update version
   vim CHANGELOG.md    # Update changelog
   ```

2. **Commit and Tag**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git push
   
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

3. **Automated Release**
   - CI runs tests
   - Package builds
   - Publishes to PyPI
   - Creates GitHub release

## Areas for Contribution

### High Priority

- [ ] CUDA kernel optimizations
- [ ] Additional backend support
- [ ] Compiler optimizations
- [ ] Documentation improvements
- [ ] Tutorial examples

### Good First Issues

Look for issues labeled `good-first-issue` or `help-wanted`.

### Feature Requests

Check existing issues or create a new one to discuss your idea before implementing.

## Getting Help

- **Documentation**: Check [docs/](docs/)
- **Issues**: Browse [existing issues](https://github.com/financecommander/Triton/issues)
- **Discussions**: Ask in [GitHub Discussions](https://github.com/financecommander/Triton/discussions)
- **CI/CD**: See [CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md
- Release notes

Thank you for contributing to Triton DSL! 🎉

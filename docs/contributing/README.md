# Contributing Documentation

Welcome to the Triton DSL contributing guide! This directory contains comprehensive documentation for contributors.

## Quick Links

- **[Development Setup](development_setup.md)** - Set up your development environment
- **[Code Style Guide](code_style.md)** - Follow our coding standards
- **[Testing Guide](testing.md)** - Write and run tests
- **[PR Process](pr_process.md)** - Submit and review pull requests

## Getting Started

### First-time Contributors

1. **Setup**: Read [Development Setup](development_setup.md) to configure your environment
2. **Standards**: Review [Code Style Guide](code_style.md) to understand our conventions
3. **Testing**: Learn our testing requirements in [Testing Guide](testing.md)
4. **Submit**: Follow [PR Process](pr_process.md) to contribute code

### Quick Start

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Triton.git
cd Triton

# 2. Set up environment
python3.10 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 3. Create branch
git checkout -b feature/my-feature

# 4. Make changes and test
black .
ruff check . --fix
pytest tests/ -v

# 5. Submit PR
git push origin feature/my-feature
gh pr create
```

## Documentation Overview

### [Development Setup](development_setup.md) (15 KB)

Complete guide to setting up your development environment:

- **Prerequisites**: Python 3.10+, Git, CUDA (optional)
- **Repository Setup**: Fork, clone, configure Git
- **Virtual Environment**: venv or conda setup
- **Development Dependencies**: Install all required packages
- **IDE Configuration**: VS Code and PyCharm setup with recommended extensions
- **Build and Test Setup**: Running tests, code quality checks, pre-commit hooks
- **Docker Development**: Containerized development environment
- **Troubleshooting**: Common issues and solutions

**Key Topics**:
- Installing Python 3.10+ and dependencies
- Configuring VS Code with Black, Ruff, mypy, pytest
- Setting up PyCharm with formatters and linters
- Creating Docker development environment
- Running tests and coverage reports

### [Code Style Guide](code_style.md) (20 KB)

Comprehensive coding standards and best practices:

- **Python Style**: PEP 8, Black (100 char lines), Ruff linting
- **DSL Code Style**: Triton DSL naming and formatting conventions
- **Documentation Standards**: Google-style docstrings, module docs, inline comments
- **Naming Conventions**: Modules, classes, functions, variables, constants
- **Type Annotations**: Required type hints, complex types, mypy configuration
- **Examples and Anti-patterns**: Good practices vs. common mistakes

**Key Topics**:
- Black formatter configuration (line-length=100)
- Import ordering and organization
- Docstring format with examples
- Type hints for all public APIs
- Common anti-patterns to avoid (mutable defaults, bare except, etc.)

### [Testing Guide](testing.md) (23 KB)

Complete testing requirements and procedures:

- **Test Organization**: Directory structure for unit, integration, benchmark tests
- **Unit Testing**: pytest framework, fixtures, parametrization, mocking
- **Integration Testing**: Multi-component tests, CUDA integration
- **Benchmark Testing**: Performance tests, regression detection
- **Coverage Requirements**: ≥85% overall, ≥95% for critical paths
- **CI/CD Testing**: GitHub Actions workflows, pre-commit/pre-push checks

**Key Topics**:
- Writing pytest tests with proper structure
- Using fixtures and parametrize for clean tests
- Mocking external dependencies
- Running benchmarks with pytest-benchmark
- Achieving and maintaining code coverage
- CI/CD integration

### [PR Process](pr_process.md) (17 KB)

Step-by-step pull request workflow:

- **Before You Start**: Check issues, sync with upstream
- **Branch Naming**: Convention-based naming (feature/, fix/, docs/, etc.)
- **Commit Messages**: Conventional Commits format with examples
- **PR Description Template**: Comprehensive template with checklist
- **Review Process**: Automated checks, human review, addressing feedback
- **Merge Requirements**: Required checks, code quality, functional requirements
- **Post-merge Procedures**: Clean up, close issues, update documentation

**Key Topics**:
- Creating well-named branches (feature/fix/docs/refactor)
- Writing conventional commit messages
- Using PR template with comprehensive checklist
- Responding to review feedback
- Squashing and rebasing commits
- Post-merge cleanup

## Contribution Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Contribution Workflow                         │
└─────────────────────────────────────────────────────────────────┘

1. Find/Create Issue
   └─> Check existing issues or create new one

2. Setup Environment (Development Setup Guide)
   ├─> Fork and clone repository
   ├─> Install dependencies
   ├─> Configure IDE
   └─> Verify setup with tests

3. Create Branch (PR Process Guide)
   └─> git checkout -b feature/my-feature

4. Make Changes (Code Style Guide)
   ├─> Follow Python style (Black, Ruff)
   ├─> Add type hints
   ├─> Write docstrings
   └─> Add inline comments for complex logic

5. Write Tests (Testing Guide)
   ├─> Unit tests for new functions/classes
   ├─> Integration tests if multiple components affected
   ├─> Maintain coverage ≥85%
   └─> Run tests locally

6. Quality Checks
   ├─> black . (format code)
   ├─> ruff check . --fix (lint)
   ├─> mypy compiler/ backend/ kernels/ (type check)
   └─> pytest tests/ -v (run tests)

7. Commit Changes (PR Process Guide)
   └─> feat(component): descriptive commit message

8. Create Pull Request
   ├─> Use PR template
   ├─> Link related issue
   ├─> Describe changes thoroughly
   └─> Complete checklist

9. Address Review Feedback
   ├─> Respond to comments
   ├─> Make requested changes
   ├─> Push updates
   └─> Resolve conversations

10. Merge & Cleanup
    ├─> PR merged by maintainer
    ├─> Delete feature branch
    └─> Close related issues
```

## Code Quality Standards

All contributions must meet these standards:

### Required Checks

- ✅ **Black**: All code formatted with Black (line-length=100)
- ✅ **Ruff**: No linting errors (E, F, W, I, N rules)
- ✅ **mypy**: All type hints valid, no type errors
- ✅ **pytest**: All tests pass
- ✅ **Coverage**: ≥85% code coverage (≥90% for new code)

### Best Practices

- ✅ Clear, descriptive variable and function names
- ✅ Google-style docstrings for all public APIs
- ✅ Type hints for all function signatures
- ✅ Unit tests for all new functionality
- ✅ Comments explaining non-obvious logic
- ✅ Atomic commits with conventional messages

### Run All Checks

```bash
# One command to rule them all
black . && \
ruff check . --fix && \
mypy compiler/ backend/ kernels/ && \
pytest tests/ -v --cov=compiler --cov=backend --cov=kernels --cov-fail-under=85

# Or use pre-commit
pre-commit run --all-files
```

## Getting Help

### Resources

- **Documentation**: Check `docs/` directory for comprehensive guides
- **Examples**: See `examples/` for working code samples
- **Tests**: Look at `tests/` for testing examples
- **Issues**: Search existing issues for similar problems

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
  - https://github.com/financecommander/Triton/issues
- **GitHub Discussions**: General questions, ideas
  - https://github.com/financecommander/Triton/discussions
- **Pull Requests**: Code review, technical discussions

### Asking Good Questions

When asking for help:

1. **Be specific**: Describe the problem clearly
2. **Show context**: Include relevant code, error messages, environment details
3. **Show effort**: Explain what you've tried
4. **Be patient**: Maintainers are volunteers

Example good question:
```markdown
I'm trying to implement ternary batch normalization but getting a type error.

**Environment**:
- Python 3.10.6
- PyTorch 2.1.0
- Triton DSL main branch (commit abc123)

**Code**:
\`\`\`python
layer = TernaryBatchNorm(num_features=128)
\`\`\`

**Error**:
\`\`\`
TypeError: expected Tensor but got TernaryTensor
\`\`\`

**What I've tried**:
- Checked TernaryTensor inherits from torch.Tensor ✅
- Verified num_features is int ✅
- Searched existing issues (found #123 but different error)

**Question**: Should TernaryBatchNorm accept TernaryTensor, or should I convert to regular Tensor first?
```

## Contributing Beyond Code

Not all contributions are code! We welcome:

- 📝 **Documentation**: Improve guides, fix typos, add examples
- 🐛 **Bug Reports**: Detailed issue reports with reproduction steps
- 💡 **Feature Ideas**: Proposals for new features
- 🧪 **Testing**: Add test cases, improve coverage
- 📊 **Benchmarks**: Performance testing and optimization
- 👀 **Code Review**: Review PRs, provide feedback
- ❓ **Support**: Answer questions in discussions
- 📢 **Advocacy**: Write blog posts, give talks, share the project

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Assume good intentions

### Review Etiquette

**As an author**:
- Be open to feedback
- Don't take criticism personally
- Explain your reasoning clearly
- Thank reviewers for their time

**As a reviewer**:
- Be kind and constructive
- Explain the "why" behind suggestions
- Distinguish between requirements and preferences
- Praise good work

## Maintainer Notes

For maintainers reviewing PRs:

### Review Checklist

- [ ] Code follows style guide (Black, Ruff pass)
- [ ] Type hints present and correct (mypy passes)
- [ ] Tests added and passing
- [ ] Coverage maintained/improved
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] No merge conflicts
- [ ] CI checks pass
- [ ] Breaking changes documented

### Merge Guidelines

- Use "Squash and Merge" for multi-commit feature branches
- Use "Rebase and Merge" for clean, logical commit history
- Ensure PR title follows conventional commit format
- Update CHANGELOG.md for significant changes
- Close related issues automatically (use "Closes #123" in PR description)

## License

By contributing to Triton DSL, you agree that your contributions will be licensed under the MIT License.

---

**Ready to contribute?** Start with [Development Setup](development_setup.md)! 🚀

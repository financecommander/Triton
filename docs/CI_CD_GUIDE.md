# CI/CD Pipeline Guide

This document describes the CI/CD pipeline for the Triton DSL project.

## Overview

The Triton DSL project uses GitHub Actions for continuous integration and deployment. The pipeline includes:

1. **Automated Testing** - Run tests on every push and pull request
2. **Code Quality** - Linting, formatting, and type checking
3. **Package Building** - Build distribution packages
4. **PyPI Publishing** - Automatically publish to PyPI on version tags
5. **Documentation** - Build and deploy documentation to GitHub Pages

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger via workflow_dispatch

**Jobs:**

#### Lint
- Runs Black, Ruff, and MyPy
- Checks code formatting and style
- Python 3.11 on Ubuntu

#### Test
- Runs on multiple Python versions (3.10, 3.11, 3.12)
- Tests on Ubuntu, Windows, and macOS
- Generates coverage report (Ubuntu 3.11 only)
- Uploads coverage to Codecov

#### Integration
- Runs integration tests
- Requires lint and test to pass

#### Build
- Builds Python package
- Validates package with twine
- Uploads artifacts for 7 days

#### Status
- Final check ensuring all jobs passed

### 2. PyPI Publishing Workflow (`.github/workflows/publish.yml`)

**Triggers:**
- Push of version tags (e.g., `v1.0.0`)
- Manual trigger with version input

**Jobs:**

#### Check Version
- Validates version tag format
- Ensures version matches pyproject.toml

#### Build
- Builds source and wheel distributions
- Uploads artifacts

#### Test Install
- Tests installation on multiple platforms
- Validates imports work correctly

#### Publish to PyPI
- Uses trusted publishing (no tokens needed)
- Publishes to https://pypi.org/project/triton-dsl/

#### Publish to TestPyPI
- Only on manual triggers
- For testing releases

#### Create Release
- Creates GitHub release
- Generates changelog
- Attaches distribution files

### 3. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers:**
- Push to `main` branch (docs changes)
- Pull requests with doc changes
- Manual trigger

**Jobs:**

#### Build
- Generates API documentation
- Builds MkDocs site
- Uploads artifacts

#### Deploy
- Deploys to GitHub Pages (main branch only)
- Available at: https://financecommander.github.io/Triton

#### Check Links
- Validates markdown links
- Prevents broken documentation links

## Setup Requirements

### Repository Secrets

No secrets are required for basic CI/CD! The workflows use:
- GitHub's built-in `GITHUB_TOKEN` for releases
- PyPI's trusted publishing (configured in PyPI settings)

### PyPI Trusted Publishing Setup

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new pending publisher:
   - PyPI Project Name: `triton-dsl`
   - Owner: `financecommander`
   - Repository name: `Triton`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

### GitHub Pages Setup

1. Go to repository Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`

### Branch Protection (Recommended)

1. Go to repository Settings → Branches
2. Add rule for `main`:
   - Require status checks before merging
   - Require "Lint" and "Test" to pass
   - Require up-to-date branches

## Release Process

### Creating a New Release

1. **Update Version**
   ```bash
   # Update version in pyproject.toml
   vim pyproject.toml
   # Commit the change
   git add pyproject.toml
   git commit -m "Bump version to 1.0.0"
   git push
   ```

2. **Create Tag**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. **Automated Process**
   - CI tests run automatically
   - Package builds and publishes to PyPI
   - GitHub release created with changelog
   - Documentation updates

### Manual Testing (TestPyPI)

To test a release before publishing to PyPI:

1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Enter version (e.g., `v1.0.0-rc1`)
4. Package publishes to TestPyPI
5. Test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ triton-dsl
   ```

## Development Workflow

### Running Tests Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=compiler --cov=backend --cov=kernels
```

### Code Quality Checks

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy compiler/ backend/
```

### Building Documentation Locally

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## CI/CD Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/financecommander/Triton/workflows/CI/badge.svg)](https://github.com/financecommander/Triton/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/triton-dsl)](https://pypi.org/project/triton-dsl/)
[![Python](https://img.shields.io/pypi/pyversions/triton-dsl)](https://pypi.org/project/triton-dsl/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://financecommander.github.io/Triton)
[![codecov](https://codecov.io/gh/financecommander/Triton/branch/main/graph/badge.svg)](https://codecov.io/gh/financecommander/Triton)
```

## Troubleshooting

### Tests Failing in CI but Pass Locally

- Check Python version differences
- Verify all dependencies are in pyproject.toml
- Check platform-specific issues (Windows/macOS)

### PyPI Publishing Fails

- Verify version in pyproject.toml matches tag
- Check PyPI trusted publishing is configured
- Ensure environment name matches (`pypi`)

### Documentation Not Deploying

- Check GitHub Pages settings
- Verify gh-pages branch exists
- Check workflow logs for errors

### Coverage Too Low

- Add more tests
- Remove test exclusions
- Check coverage configuration in pyproject.toml

## Monitoring

### CI Dashboard

Monitor all workflows at:
https://github.com/financecommander/Triton/actions

### Key Metrics

- **Test Pass Rate**: Should be 100%
- **Coverage**: Target 80%+
- **Build Time**: < 10 minutes
- **Documentation**: Always up-to-date

## Future Enhancements

Planned improvements:
- [ ] Security scanning (CodeQL, Dependabot)
- [ ] Performance benchmarking
- [ ] Docker image publishing
- [ ] Nightly builds
- [ ] Release automation bot

## Support

For CI/CD issues:
1. Check workflow logs
2. Review this guide
3. Open an issue with the `ci/cd` label
4. Contact maintainers

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [Codecov Integration](https://docs.codecov.com/docs)

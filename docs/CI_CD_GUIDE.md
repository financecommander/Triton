# CI/CD Workflow Documentation

## Overview

The Triton DSL project uses a comprehensive GitHub Actions CI/CD pipeline that ensures code quality, security, and automated deployment. This document explains how the workflow operates and how to work with it.

## Workflow File

Location: `.github/workflows/ci.yml`

## Triggers

The workflow runs automatically on:

1. **Push Events**: Commits to `main` or `develop` branches
2. **Pull Requests**: All PRs targeting `main` or `develop`
3. **Tags**: Version tags following the pattern `v*.*.*` (e.g., `v1.0.0`)
4. **Schedule**: Daily at 2 AM UTC
5. **Manual**: Via GitHub Actions UI (workflow_dispatch)

## Jobs

### 1. Lint & Code Quality

**Purpose**: Enforce consistent code style and catch potential issues

**Tools Used**:
- `black` - Code formatting (line length: 100)
- `ruff` - Fast Python linter
- `flake8` - Additional linting checks
- `mypy` - Static type checking

**Run locally**:
```bash
black --check .
ruff check .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
mypy compiler backend kernels --ignore-missing-imports
```

### 2. Security Scanning

**Purpose**: Identify security vulnerabilities in code and dependencies

**Tools Used**:
- `bandit` - Python security scanner
- `safety` - Dependency vulnerability checker

**Outputs**: Security reports uploaded as artifacts

**Run locally**:
```bash
bandit -r compiler backend kernels
safety check
```

### 3. Tests (Matrix)

**Purpose**: Validate functionality across multiple environments

**Matrix Dimensions**:
- **Python versions**: 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu, Windows, macOS
- **PyTorch versions**: 2.0, 2.1, 2.2

**Matrix Optimizations**:
- Excludes incompatible combinations (e.g., Python 3.12 + PyTorch 2.0)
- Reduces Windows/macOS combinations to save CI minutes
- Full matrix on Ubuntu for comprehensive testing

**Coverage Requirements**:
- Minimum 80% code coverage enforced
- Reports uploaded to Codecov
- Coverage artifacts saved for all matrix combinations

**Run locally**:
```bash
# Install specific PyTorch version
pip install torch==2.2.2 torchvision==0.17.2

# Run tests with coverage
pytest tests/unit -v --cov=. --cov-report=xml --cov-report=term-missing --cov-fail-under=80
```

### 4. Integration Tests

**Purpose**: Test component interactions and end-to-end scenarios

**Runs**: After lint and unit tests pass

**Includes**:
- Integration test suite (`tests/integration`)
- Stress tests (non-slow variants)

**Run locally**:
```bash
pytest tests/integration -v
pytest tests/stress -v -k "not slow"
```

### 5. Documentation Build

**Purpose**: Verify documentation can be built successfully

**Process**:
1. Checks for Sphinx configuration (`docs/conf.py`)
2. Builds HTML documentation if Sphinx is configured
3. Falls back to simple Markdown conversion if no Sphinx config
4. Uploads documentation artifact

**Run locally**:
```bash
# If using Sphinx
sphinx-build -b html docs docs/_build/html

# Simple build
mkdir -p docs/_build/html
cp README.md docs/_build/html/index.html
```

### 6. Build Distribution Packages

**Purpose**: Create wheel and source distribution packages

**Requirements**: Passes after tests and integration tests

**Outputs**:
- Wheel files (`.whl`)
- Source distribution (`.tar.gz`)
- Artifacts uploaded for deployment

**Run locally**:
```bash
python -m build
twine check dist/*
```

### 7. Build Docker Image

**Purpose**: Create containerized version of Triton DSL

**Triggers**: Only on pushes to `main` or version tags

**Image Tags**:
- Branch name (e.g., `main`)
- Semantic version (e.g., `1.0.0`, `1.0`)
- Git SHA

**Registry**: Docker Hub (requires secrets)

**Run locally**:
```bash
docker build -t triton-dsl:local .
docker run -it triton-dsl:local python
```

## Deployments

### PyPI Deployment

**Trigger**: Version tags (e.g., `v1.0.0`)

**Requirements**:
- All tests must pass
- Security scans complete
- Wheels built successfully

**Process**:
1. Downloads built wheel artifacts
2. Publishes to PyPI using trusted publishing (OIDC)
3. Skips existing versions

**Required Secrets**: `PYPI_API_TOKEN`

### Documentation Deployment

**Trigger**: Pushes to `main` branch

**Target**: GitHub Pages

**Process**:
1. Downloads built documentation
2. Deploys to `gh-pages` branch
3. Accessible at `https://financecommander.github.io/Triton/`

**Permissions**: Requires `contents: write`

### Docker Hub Deployment

**Trigger**: Pushes to `main` or version tags

**Process**:
1. Builds multi-platform Docker image
2. Pushes to Docker Hub with appropriate tags
3. Uses layer caching for efficiency

**Required Secrets**:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

## Caching Strategy

The workflow implements aggressive caching to reduce CI time:

1. **Pip Cache**: Python packages cached by `actions/setup-python`
2. **PyTorch Cache**: Large PyTorch installations cached per version
3. **Docker Cache**: Docker layers cached using GitHub Actions cache

## Status Badges

The following badges are available in README.md:

1. **CI/CD Pipeline**: Build status
2. **Codecov**: Code coverage percentage
3. **PyPI Version**: Latest published version
4. **Python Support**: Supported Python versions
5. **License**: Project license

## Required Secrets

Configure these in GitHub repository settings (Settings → Secrets and variables → Actions):

| Secret | Purpose | Required For |
|--------|---------|--------------|
| `CODECOV_TOKEN` | Upload coverage to Codecov | Coverage reporting |
| `PYPI_API_TOKEN` | Publish to PyPI | PyPI deployment |
| `DOCKER_USERNAME` | Docker Hub login | Docker image publishing |
| `DOCKER_PASSWORD` | Docker Hub password | Docker image publishing |

## Environment Protection

The workflow uses GitHub Environments for sensitive deployments:

- **pypi**: PyPI deployment (requires approval for first-time setup)

Configure in: Settings → Environments

## Troubleshooting

### Coverage Below 80%

If tests fail due to insufficient coverage:

```bash
# Run locally to see uncovered lines
pytest tests/unit --cov=. --cov-report=term-missing

# Add tests for uncovered code paths
# Or add # pragma: no cover for truly untestable code
```

### Matrix Job Failures

If a specific matrix combination fails:

1. Check the job logs for the specific Python/PyTorch/OS combination
2. Run tests locally with that exact configuration
3. May indicate compatibility issues requiring exclusion

### Docker Build Failures

Common issues:

1. **Missing secrets**: Configure `DOCKER_USERNAME` and `DOCKER_PASSWORD`
2. **Dockerfile not found**: Workflow creates it automatically
3. **Build context issues**: Ensure all required files are committed

### PyPI Deployment Failures

Common issues:

1. **Missing token**: Add `PYPI_API_TOKEN` secret
2. **Version already exists**: Increment version in `pyproject.toml`
3. **Invalid package**: Run `twine check dist/*` locally

## Local Development Workflow

### Before Committing

```bash
# Format code
black .

# Run linters
ruff check .
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Run type checker
mypy compiler backend kernels --ignore-missing-imports

# Run security scan
bandit -r compiler backend kernels

# Run tests with coverage
pytest tests/unit -v --cov=. --cov-fail-under=80
pytest tests/integration -v
```

### Testing a Release

To test the release process locally:

```bash
# Build packages
python -m build

# Check packages
twine check dist/*

# Test upload to TestPyPI (optional)
twine upload --repository testpypi dist/*

# Build Docker image
docker build -t triton-dsl:test .
docker run -it triton-dsl:test python -c "import torch; print(torch.__version__)"
```

## Release Process

To create a new release:

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Commit changes**: `git commit -am "Release v1.0.0"`
4. **Create tag**: `git tag v1.0.0`
5. **Push tag**: `git push origin v1.0.0`
6. **Monitor workflow**: Check GitHub Actions for deployment progress
7. **Verify deployment**: Check PyPI and Docker Hub

The workflow will automatically:
- Run all tests
- Build packages
- Deploy to PyPI
- Build and push Docker image

## Workflow Customization

### Adding Python Versions

Edit `.github/workflows/ci.yml`:

```yaml
matrix:
  python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]  # Add 3.13
```

### Adding OS Platforms

```yaml
matrix:
  os: [ubuntu-latest, windows-latest, macos-latest, ubuntu-20.04]  # Add ubuntu-20.04
```

### Modifying Coverage Threshold

```yaml
# Change from 80% to 85%
pytest tests/unit -v --cov=. --cov-fail-under=85
```

### Changing Schedule

```yaml
schedule:
  # Change from daily 2 AM to weekly Sunday 3 AM
  - cron: '0 3 * * 0'
```

## Performance Metrics

Typical workflow execution times:

- **Lint**: 2-3 minutes
- **Security**: 2-3 minutes
- **Test (single matrix)**: 5-8 minutes
- **Integration Tests**: 3-5 minutes
- **Documentation**: 1-2 minutes
- **Build Wheels**: 1-2 minutes
- **Build Docker**: 3-5 minutes (with cache)

**Total parallel time**: ~15-20 minutes for full pipeline

## Best Practices

1. **Run checks locally** before pushing to catch issues early
2. **Keep dependencies updated** to avoid security vulnerabilities
3. **Write tests** to maintain coverage above threshold
4. **Use semantic versioning** for releases (MAJOR.MINOR.PATCH)
5. **Document breaking changes** in CHANGELOG.md
6. **Monitor CI costs** - matrix testing can consume many CI minutes
7. **Review security reports** from Bandit and Safety regularly

## Support

For issues with the CI/CD workflow:

1. Check [GitHub Actions logs](https://github.com/financecommander/Triton/actions)
2. Review this documentation
3. Open an issue with the `ci/cd` label
4. Tag maintainers for urgent issues

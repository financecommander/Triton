# CI/CD Quick Reference

## 🚀 Quick Actions

### Run Tests Locally
```bash
pip install -e ".[dev]"
pytest tests/
```

### Format Code
```bash
black .
ruff check .
```

### Create a Release
```bash
# 1. Update version in pyproject.toml
vim pyproject.toml

# 2. Commit and tag
git add pyproject.toml
git commit -m "Bump version to X.Y.Z"
git push

# 3. Create and push tag
git tag vX.Y.Z
git push origin vX.Y.Z

# CI will automatically:
# - Run tests
# - Build package
# - Publish to PyPI
# - Create GitHub release
```

## 📊 Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push, PR | Run tests, linting, build |
| `publish.yml` | Tag `v*` | Publish to PyPI + release |
| `docs.yml` | Push to main | Deploy documentation |

## ✅ CI Checks

- **Lint**: Black, Ruff, MyPy
- **Test**: Python 3.10, 3.11, 3.12
- **Platforms**: Ubuntu, Windows, macOS
- **Coverage**: Reports generated for Ubuntu 3.11

## 📦 Publishing

### Automatic (Recommended)
```bash
git tag v1.0.0
git push origin v1.0.0
```

### Manual Test (TestPyPI)
1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Enter version (e.g., `v1.0.0-rc1`)

## 📚 Documentation

### Local Preview
```bash
pip install mkdocs mkdocs-material
mkdocs serve
# Visit http://localhost:8000
```

### Automatic Deploy
- Push to `main` branch
- Docs deploy to: https://financecommander.github.io/Triton

## 🔧 Setup Required

### PyPI Trusted Publishing
1. Visit: https://pypi.org/manage/account/publishing/
2. Add publisher:
   - Project: `triton-dsl`
   - Owner: `financecommander`
   - Repo: `Triton`
   - Workflow: `publish.yml`
   - Environment: `pypi`

### GitHub Pages
1. Settings → Pages
2. Source: `gh-pages` branch
3. Root directory

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests fail in CI | Check Python version, dependencies |
| PyPI publish fails | Verify version matches tag |
| Docs not deploying | Check gh-pages branch exists |
| Coverage too low | Add more tests, check config |

## 📖 Full Documentation

See [CI_CD_GUIDE.md](CI_CD_GUIDE.md) for complete documentation.

## 🎯 Status Badges

```markdown
[![CI](https://github.com/financecommander/Triton/workflows/CI/badge.svg)](https://github.com/financecommander/Triton/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/triton-dsl)](https://pypi.org/project/triton-dsl/)
[![Python](https://img.shields.io/pypi/pyversions/triton-dsl)](https://pypi.org/project/triton-dsl/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://financecommander.github.io/Triton)
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/financecommander/Triton/issues)
- **Discussions**: [GitHub Discussions](https://github.com/financecommander/Triton/discussions)
- **CI Logs**: [GitHub Actions](https://github.com/financecommander/Triton/actions)

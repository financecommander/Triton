# Release Checklist — Triton DSL v1.0.0

## Pre-Release

- [ ] All tests passing (100%)
- [ ] Code coverage ≥ 90%
- [ ] No critical security issues (pip-audit, Bandit, CodeQL)
- [ ] Performance benchmarks met
- [ ] Documentation complete and reviewed

## Version Updates

- [x] `pyproject.toml` version set to `1.0.0`
- [x] `compiler/__init__.py` `__version__` set to `1.0.0`
- [x] `tests/benchmarks/__init__.py` `__version__` set to `1.0.0`
- [x] Development Status classifier: `Production/Stable`
- [x] Python 3.12 added to classifiers
- [x] CHANGELOG.md updated with v1.0.0 entry

## Packaging

- [x] `MANIFEST.in` verified
- [x] `pyproject.toml` finalized
- [ ] Build source distribution: `python -m build --sdist`
- [ ] Build wheel: `python -m build --wheel`
- [ ] Test install from built package
- [ ] Upload to Test PyPI: `twine upload --repository testpypi dist/*`
- [ ] Verify Test PyPI install: `pip install -i https://test.pypi.org/simple/ triton-dsl`
- [ ] Upload to PyPI: `twine upload dist/*`

## Documentation

- [x] CHANGELOG.md updated
- [x] MIGRATION.md created
- [ ] API reference finalized
- [ ] All docs reviewed for accuracy
- [ ] Versioned docs tagged (v1.0)

## Release Assets

- [ ] Create GitHub release with tag `v1.0.0`
- [ ] Attach wheel and sdist to release
- [ ] Publish release notes

## Post-Release

- [ ] Verify PyPI package installs correctly
- [ ] Verify GitHub release is accessible
- [ ] Announce release

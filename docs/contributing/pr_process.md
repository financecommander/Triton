# Pull Request Process

This guide explains how to contribute code to Triton DSL through pull requests (PRs).

## Table of Contents

- [Before You Start](#before-you-start)
- [Branch Naming](#branch-naming)
- [Commit Messages](#commit-messages)
- [PR Description Template](#pr-description-template)
- [Review Process](#review-process)
- [Merge Requirements](#merge-requirements)
- [Post-merge Procedures](#post-merge-procedures)
- [Quick Reference](#quick-reference)

## Before You Start

### 1. Check for Existing Issues

Before creating a PR, ensure there's an issue describing the problem or feature:

```bash
# Search existing issues
gh issue list --search "your feature"

# Or visit: https://github.com/financecommander/Triton/issues
```

If no issue exists, create one:
```bash
gh issue create --title "Add support for ternary batch normalization" \
                --body "Description of feature..."
```

### 2. Sync with Upstream

Always start from the latest `main` branch:

```bash
# Fetch latest changes
git fetch upstream

# Update your local main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

### 3. Create a Feature Branch

```bash
# Create and checkout new branch
git checkout -b feature/ternary-batchnorm

# Or for bug fixes
git checkout -b fix/parser-memory-leak
```

## Branch Naming

### Naming Convention

Follow this pattern: `<type>/<short-description>`

**Types**:
- `feature/` - New features or enhancements
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring (no functional changes)
- `test/` - Adding or updating tests
- `perf/` - Performance improvements
- `chore/` - Maintenance tasks (dependencies, build config)

**Examples**:
```bash
# Good branch names
feature/ternary-batchnorm
feature/cuda-kernel-optimization
fix/parser-memory-leak
fix/quantization-gradient-bug
docs/add-tutorial-advanced-features
refactor/simplify-type-checker
test/add-integration-tests-cuda
perf/optimize-matmul-sparse
chore/update-pytorch-dependency

# Bad branch names
my-changes
fix-bug
update
feature123
john-dev-branch
```

### Creating Branches

```bash
# Feature branch
git checkout -b feature/my-feature

# Bug fix branch
git checkout -b fix/issue-123-parser-crash

# From specific issue
gh issue develop 123 --checkout
```

## Commit Messages

### Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, missing semicolons, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding/updating tests
- `build:` - Build system or dependencies
- `ci:` - CI configuration
- `chore:` - Other maintenance

### Scopes (Optional)

Specify the component affected:
- `lexer` - Lexer changes
- `parser` - Parser changes
- `typechecker` - Type checker changes
- `codegen` - Code generation
- `backend` - Backend changes
- `kernels` - CUDA/Triton kernels
- `examples` - Example code
- `tests` - Test infrastructure

### Examples

**Good commit messages**:

```bash
# Simple feature
git commit -m "feat(lexer): add support for ternary operators"

# With body
git commit -m "feat(backend): implement CUDA kernel for ternary matmul

Add optimized CUDA kernel for ternary matrix multiplication with
zero-skipping. Achieves 2.5x speedup over naive implementation.

Closes #123"

# Bug fix
git commit -m "fix(parser): handle empty input gracefully

Previously crashed with IndexError when parsing empty source.
Now returns empty Program node.

Fixes #456"

# Breaking change
git commit -m "feat(api)!: change TernaryLayer interface

BREAKING CHANGE: TernaryLayer constructor now requires explicit
in_features and out_features arguments. The previous shape inference
is removed for clarity.

Migration guide:
  # Old
  layer = TernaryLayer()
  
  # New
  layer = TernaryLayer(in_features=128, out_features=256)"

# Documentation
git commit -m "docs: add tutorial for custom quantization methods"

# Multiple changes (should be separate commits)
# ❌ Bad
git commit -m "fix bugs and add features"

# ✅ Good - make separate commits
git commit -m "fix(parser): handle missing semicolons"
git commit -m "feat(parser): add support for multi-line strings"
```

**Commit message template**:

```bash
# Set up commit message template
cat > ~/.gitmessage << 'EOF'
# <type>[optional scope]: <description>
# |<----  Using a Maximum Of 50 Characters  ---->|

# [optional body]
# |<----   Using a Maximum Of 72 Characters   ---->|

# [optional footer(s)]
# Example: Closes #123, Fixes #456

# Types:
#   feat:     New feature
#   fix:      Bug fix
#   docs:     Documentation
#   style:    Formatting
#   refactor: Refactoring
#   perf:     Performance
#   test:     Tests
#   build:    Build system
#   ci:       CI/CD
#   chore:    Maintenance
EOF

git config --global commit.template ~/.gitmessage
```

### Commit Best Practices

**1. Atomic Commits**: One logical change per commit

```bash
# ✅ Good: Separate logical changes
git add compiler/lexer.py
git commit -m "feat(lexer): add ternary literal support"

git add compiler/parser.py
git commit -m "feat(parser): parse ternary literals"

git add tests/unit/test_lexer.py tests/unit/test_parser.py
git commit -m "test: add tests for ternary literals"

# ❌ Bad: Everything in one commit
git add -A
git commit -m "add ternary literals"
```

**2. Commit Often**: Small, frequent commits are easier to review

```bash
# Work in progress commits
git commit -m "feat(lexer): add token definitions for ternary ops (WIP)"
git commit -m "feat(lexer): implement ternary tokenization logic"
git commit -m "feat(lexer): add error handling for invalid ternary values"

# Squash before PR if needed
git rebase -i HEAD~3
```

**3. Write Descriptive Messages**:

```bash
# ❌ Bad: Vague
git commit -m "update code"
git commit -m "fix bug"
git commit -m "changes"

# ✅ Good: Specific
git commit -m "fix(parser): prevent IndexError on empty input"
git commit -m "refactor(typechecker): extract type inference to separate method"
git commit -m "perf(quantize): use torch.compile for 30% speedup"
```

## PR Description Template

### Template

Use this template when creating a PR:

```markdown
## Description

Brief description of changes (2-3 sentences).

Closes #<issue-number>

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Test addition/update

## Changes Made

Detailed list of changes:

- Changed X to Y because Z
- Added new module `foo.py` for handling bar
- Refactored `baz()` function to improve readability
- Updated documentation in `README.md`

## Testing

Describe testing performed:

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing performed

**Test coverage**: 87% → 89% (+2%)

### Manual Testing Steps

1. Step 1
2. Step 2
3. Expected result

### Test Results

```
$ pytest tests/ -v
================================ test session starts =================================
collected 247 items

tests/unit/test_lexer.py::test_ternary_literal PASSED                          [ 45%]
tests/unit/test_parser.py::test_parse_ternary PASSED                           [ 46%]
...
================================ 247 passed in 12.34s ================================
```

## Performance Impact

For performance-related changes:

**Before**: X ms/op
**After**: Y ms/op
**Improvement**: Z% faster

## Breaking Changes

If applicable, describe breaking changes and migration path:

### Breaking Change: API modification

**Old API**:
```python
layer = TernaryLayer()
```

**New API**:
```python
layer = TernaryLayer(in_features=128, out_features=256)
```

**Migration**: Update all TernaryLayer instantiations to include explicit dimensions.

## Screenshots/Examples

For UI changes or new features, provide screenshots or examples:

```python
# Example usage
from compiler import TritonCompiler

compiler = TritonCompiler()
result = compiler.compile("model.tri")
```

## Checklist

- [ ] Code follows project style guidelines (Black, Ruff)
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] Documentation updated (if applicable)
- [ ] No new warnings generated
- [ ] Tests added/updated and passing
- [ ] Coverage maintained/improved
- [ ] Manual testing performed
- [ ] Benchmark tests run (for performance changes)
- [ ] Breaking changes documented

## Additional Context

Any additional information, context, or screenshots.
```

### Creating a PR

```bash
# Push your branch
git push origin feature/my-feature

# Create PR via GitHub CLI
gh pr create --title "feat(parser): add support for ternary operators" \
             --body-file pr-template.md \
             --base main \
             --head feature/my-feature

# Or create via web interface
# Visit: https://github.com/YOUR_USERNAME/Triton/pull/new/feature/my-feature
```

### PR Title Format

Follow commit message format:

```
<type>[optional scope]: <description>
```

**Examples**:
```
feat(lexer): add support for ternary operators
fix(parser): handle empty input gracefully
docs: add tutorial for custom quantization
perf(kernels): optimize CUDA kernel for 30% speedup
refactor(backend): simplify PyTorch code generation
```

## Review Process

### 1. Automated Checks

After creating a PR, automated checks will run:

- ✅ **Black**: Code formatting
- ✅ **Ruff**: Linting
- ✅ **mypy**: Type checking
- ✅ **pytest**: Test suite
- ✅ **Coverage**: Code coverage check

All checks must pass before merge.

**If checks fail**:

```bash
# Fix formatting
black .

# Fix linting issues
ruff check . --fix

# Fix type errors
mypy compiler/ backend/ kernels/

# Run tests locally
pytest tests/ -v

# Push fixes
git add -A
git commit -m "fix: address review comments"
git push origin feature/my-feature
```

### 2. Human Review

At least one maintainer must approve the PR.

**Review timeline**:
- Simple fixes: 1-2 days
- Features: 3-5 days
- Complex changes: 1 week

**Addressing review comments**:

```bash
# Make requested changes
vim compiler/parser.py

# Commit changes
git add compiler/parser.py
git commit -m "refactor: simplify parsing logic per review"

# Push update
git push origin feature/my-feature

# Reply to comments on GitHub
# Use "Resolve conversation" when addressed
```

### 3. Responding to Feedback

**Good responses**:
```markdown
✅ Fixed in commit abc123

✅ Good catch! Updated to use `isinstance()` instead

✅ Added tests as requested in test_parser.py

✅ You're right, this can be simplified. Refactored in commit def456

🤔 I kept it this way because X, but open to alternatives

❓ Can you clarify what you mean by "simplify this logic"?
```

**Bad responses**:
```markdown
❌ No

❌ That's how I like it

❌ This is fine

❌ [No response]
```

### 4. Updating Your PR

**After review comments**:

```bash
# Make changes
vim file.py

# Commit
git commit -m "refactor: address review feedback"

# Push
git push origin feature/my-feature
```

**Squashing commits (if requested)**:

```bash
# Interactive rebase
git rebase -i HEAD~5

# In editor, mark commits to squash:
# pick abc123 feat(parser): add feature
# squash def456 fix typo
# squash ghi789 address review comments
# squash jkl012 fix tests

# Force push (⚠️ only on your branch!)
git push --force-with-lease origin feature/my-feature
```

**Rebasing on main**:

```bash
# Fetch latest main
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Resolve conflicts if any
# Edit conflicted files
git add resolved-file.py
git rebase --continue

# Force push
git push --force-with-lease origin feature/my-feature
```

## Merge Requirements

Before a PR can be merged, it must meet these requirements:

### Required Checks

- ✅ All CI tests pass
- ✅ Code coverage ≥ 85% (and not decreased)
- ✅ No merge conflicts with `main`
- ✅ At least 1 approving review from maintainer
- ✅ All conversations resolved

### Code Quality

- ✅ Follows style guide (Black, Ruff pass)
- ✅ Type hints present (mypy passes)
- ✅ Documentation updated (if applicable)
- ✅ Tests added for new code
- ✅ No new warnings or errors

### Functional

- ✅ Feature works as described
- ✅ No regressions introduced
- ✅ Performance acceptable (benchmarks pass)
- ✅ Breaking changes documented

### Merge Methods

**Squash and Merge** (Default):
- Combines all commits into one
- Use for feature branches with many small commits
- Keeps `main` history clean

**Rebase and Merge**:
- Preserves individual commits
- Use for branches with clean, logical commits
- Requires pre-squashing if needed

**Merge Commit**:
- Not typically used
- Special cases only

## Post-merge Procedures

### 1. Clean Up

After your PR is merged:

```bash
# Switch to main
git checkout main

# Pull latest
git pull upstream main

# Delete local branch
git branch -d feature/my-feature

# Delete remote branch
git push origin --delete feature/my-feature

# Or use GitHub CLI
gh pr close <PR-NUMBER> --delete-branch
```

### 2. Close Related Issues

If not auto-closed:

```bash
gh issue close <ISSUE-NUMBER> --comment "Fixed in #<PR-NUMBER>"
```

### 3. Update Documentation

If your changes require documentation updates:

```bash
# Create follow-up PR for docs
git checkout -b docs/update-feature-docs
# Edit docs
git commit -m "docs: update documentation for new feature"
gh pr create
```

### 4. Announce (Optional)

For significant features:
- Update `CHANGELOG.md`
- Post in discussions
- Update project roadmap

## Quick Reference

### Workflow Summary

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# 2. Create branch
git checkout -b feature/my-feature

# 3. Make changes
# ... edit files ...

# 4. Commit
git add file.py
git commit -m "feat(component): add feature"

# 5. Push
git push origin feature/my-feature

# 6. Create PR
gh pr create --title "feat(component): add feature" --body "..."

# 7. Address review feedback
# ... make changes ...
git commit -m "refactor: address review comments"
git push origin feature/my-feature

# 8. After merge
git checkout main
git pull upstream main
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

### Common Git Commands

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Amend last commit message
git commit --amend -m "new message"

# Stash changes
git stash
git stash pop

# View diff
git diff
git diff --staged

# View log
git log --oneline --graph

# Cherry-pick commit
git cherry-pick <commit-hash>

# Interactive rebase (squash commits)
git rebase -i HEAD~3
```

### GitHub CLI Commands

```bash
# Create PR
gh pr create

# List PRs
gh pr list

# View PR
gh pr view <PR-NUMBER>

# Checkout PR locally
gh pr checkout <PR-NUMBER>

# Review PR
gh pr review <PR-NUMBER> --approve
gh pr review <PR-NUMBER> --request-changes

# Merge PR
gh pr merge <PR-NUMBER> --squash

# Close PR
gh pr close <PR-NUMBER>
```

## Tips for Success

### ✅ Do

- Start small - make focused, incremental changes
- Write clear commit messages
- Add tests for new code
- Update documentation
- Respond promptly to review feedback
- Ask questions if unclear
- Run tests locally before pushing
- Keep PR description up-to-date

### ❌ Don't

- Create massive PRs (>500 lines)
- Mix unrelated changes in one PR
- Push directly to `main`
- Force push to `main` (forbidden)
- Ignore review feedback
- Leave conversations unresolved
- Merge your own PR (except maintainers)
- Add unnecessary dependencies

## Getting Help

If you're stuck or have questions:

1. **Review documentation**: Check docs/contributing/
2. **Search issues**: Someone may have asked before
3. **Ask in PR comments**: Tag maintainers if needed
4. **GitHub Discussions**: For general questions
5. **Discord/Slack**: Real-time help (if available)

## See Also

- [Development Setup](development_setup.md)
- [Code Style Guide](code_style.md)
- [Testing Guide](testing.md)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [GitHub Flow](https://guides.github.com/introduction/flow/)

---

Thank you for contributing to Triton DSL! 🎉

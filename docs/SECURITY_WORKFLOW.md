# Security Workflow Documentation

This document describes the comprehensive security scanning workflow implemented for the Triton repository.

## Overview

The security workflow (`security.yml`) provides automated security scanning across multiple dimensions:

- **Dependency Scanning**: Detects vulnerabilities in Python dependencies
- **Code Security**: Identifies security issues in Python code
- **Secret Scanning**: Prevents accidental commits of sensitive data
- **Container Security**: Scans Docker images for vulnerabilities (when present)
- **CodeQL Analysis**: Advanced semantic code analysis for security issues

## Workflow Triggers

The security workflow runs on:

1. **Pull Requests**: All PRs to `main` and `develop` branches
2. **Push Events**: Direct pushes to `main` and `develop` branches
3. **Scheduled**: Daily at 2 AM UTC
4. **Manual**: Via workflow_dispatch

## Security Scanning Components

### 1. Dependency Scanning

**Tools Used:**
- `pip-audit`: Official PyPA tool for detecting known vulnerabilities
- `Safety`: Additional vulnerability database checking

**Severity Thresholds:**
- **Critical**: Blocks PR and creates issue immediately
- **High**: Creates warning, requires review
- **Medium/Low**: Logged for reference

**Reports Generated:**
- `pip-audit-report.json`: Detailed vulnerability report
- `safety-report.json`: Safety check results

### 2. Python Code Security (Bandit)

**Tool:** Bandit - Python AST-based security linter

**Scanned Directories:**
- `backend/`
- `compiler/`
- `models/`
- `kernels/`

**Severity Levels:**
- **High**: Blocks PR, creates issue
- **Medium**: Warning, requires review
- **Low**: Informational only

**Common Issues Detected:**
- Hardcoded passwords/secrets
- Unsafe functions (pickle, eval, exec)
- SQL injection risks
- Command injection vulnerabilities
- Insecure cryptography usage

### 3. CodeQL Analysis

**Tool:** GitHub CodeQL

**Query Suites:**
- `security-extended`: Extended security queries
- `security-and-quality`: Combined security and code quality

**Language:** Python

**Excluded Paths:**
- `tests/`
- `examples/`
- `docs/`

**Features:**
- Semantic code analysis
- Data flow analysis
- Control flow analysis
- Results uploaded to GitHub Security tab

### 4. Secret Scanning

**Tools:**
- **TruffleHog**: Scans for verified secrets
- **GitLeaks**: Additional secret pattern detection

**Scope:** Only runs on pull requests

**Detection:**
- API keys
- Authentication tokens
- Passwords
- Private keys
- AWS credentials
- Database connection strings

### 5. Container Security (Trivy)

**Tool:** Aqua Security Trivy

**Scope:** Runs if Dockerfiles are detected

**Scan Types:**
- Filesystem scanning
- Vulnerability detection
- Misconfigurations

**Severity Levels:**
- CRITICAL
- HIGH
- MEDIUM

**Output Formats:**
- SARIF (uploaded to GitHub Security)
- Table (workflow logs)

## Security Gate

The security gate job enforces security policies on pull requests:

### Blocking Conditions

A PR will be **blocked** if:
- Critical dependency vulnerabilities found
- High severity Python security issues detected
- CodeQL analysis finds security problems

### PR Comments

The security gate automatically adds a comment to PRs with:
- ✅/❌ Status indicators
- Summary of scan results
- Links to detailed reports
- Action items if issues found

## Automated Issue Creation

Issues are automatically created for:

1. **Critical Dependency Vulnerabilities** (scheduled scans)
   - Label: `security`, `critical`, `dependencies`
   - Includes scan date and workflow run link

2. **High Severity Code Issues** (scheduled/push)
   - Label: `security`, `code-quality`
   - Lists detected issues and remediation guidance

## Notifications

### Workflow Annotations

- **Errors**: Critical issues that block PRs
- **Warnings**: Medium severity issues requiring attention
- **Notices**: Informational messages

### GitHub UI Integration

- Security tab: CodeQL and Trivy results
- Checks tab: All scan results
- PR comments: Summary and status

## Dependabot Configuration

### Python Dependencies

**Schedule:** Weekly on Mondays at 2 AM
**Max Open PRs:** 10

**Grouped Updates:**
1. Development dependencies (pytest, black, ruff, mypy)
2. Torch ecosystem (torch, triton)

**Update Types:**
- Patch and minor versions grouped
- Major versions reviewed individually
- Security updates created immediately

### GitHub Actions

**Schedule:** Weekly on Mondays at 2 AM
**Max Open PRs:** 5

**Updates:** All actions grouped for minor/patch versions

## Usage Guide

### For Contributors

#### Before Creating a PR

```bash
# Install security tools locally
pip install pip-audit safety bandit

# Run dependency audit
pip-audit

# Run code security scan
bandit -r backend/ compiler/ models/ kernels/ -ll

# Check for secrets (optional)
pip install detect-secrets
detect-secrets scan
```

#### Responding to Security Issues

1. **Review the scan results** in the workflow run
2. **Download artifacts** for detailed reports
3. **Fix identified issues** in your PR
4. **Re-run checks** to verify fixes
5. **Request review** once all checks pass

### For Maintainers

#### Reviewing Security Issues

1. Navigate to **Security** tab → **Code scanning alerts**
2. Review CodeQL findings
3. Check Dependabot alerts
4. Triage and prioritize based on severity

#### Handling Security PRs

1. Review Dependabot PRs promptly
2. Verify changes don't break functionality
3. Run tests before merging
4. Merge security updates quickly

#### Manual Workflow Execution

1. Go to **Actions** tab
2. Select **Security Scanning** workflow
3. Click **Run workflow**
4. Select branch and run

## Auto-Fix Suggestions

### Dependency Updates

When vulnerabilities are found:
```bash
# Update specific package
pip install --upgrade package-name

# Update all packages (carefully)
pip install --upgrade -r requirements.txt

# Update pyproject.toml
# Edit version constraints in pyproject.toml
pip install -e .
```

### Code Security Issues

Common fixes:

1. **Hardcoded Secrets**
   ```python
   # ❌ Bad
   api_key = "sk-1234567890abcdef"
   
   # ✅ Good
   import os
   api_key = os.environ.get('API_KEY')
   ```

2. **Unsafe pickle**
   ```python
   # ❌ Bad
   import pickle
   data = pickle.load(file)
   
   # ✅ Good
   import json
   data = json.load(file)
   ```

3. **Command Injection**
   ```python
   # ❌ Bad
   os.system(f"rm {user_input}")
   
   # ✅ Good
   import shutil
   shutil.rmtree(safe_path)
   ```

4. **SQL Injection**
   ```python
   # ❌ Bad
   cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
   
   # ✅ Good
   cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
   ```

## Severity Matrix

| Severity | Dependency Scan | Code Scan | Action |
|----------|----------------|-----------|--------|
| Critical | Block PR | Block PR | Immediate fix required |
| High | Block PR | Block PR | Fix before merge |
| Medium | Warning | Warning | Should fix, can defer |
| Low | Info | Info | Optional fix |

## Troubleshooting

### Workflow Fails to Run

1. Check branch protection rules
2. Verify workflow permissions
3. Check YAML syntax

### False Positives

1. Review the specific finding
2. If valid false positive, add to ignore list
3. Document reason for ignoring

### Performance Issues

1. Workflows run in parallel where possible
2. Caching enabled for pip dependencies
3. Scheduled runs use separate job queue

## Best Practices

1. **Keep dependencies updated**: Review Dependabot PRs weekly
2. **Fix security issues promptly**: Prioritize critical/high severity
3. **Don't ignore warnings**: Even medium severity issues matter
4. **Use environment variables**: Never hardcode secrets
5. **Review scan reports**: Don't just dismiss failures
6. **Test after fixes**: Verify security fixes don't break functionality
7. **Document exceptions**: If ignoring a finding, document why

## Additional Resources

- [GitHub Code Scanning](https://docs.github.com/en/code-security/code-scanning)
- [Dependabot](https://docs.github.com/en/code-security/dependabot)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [Trivy Documentation](https://aquasecurity.github.io/trivy/)

## Support

For security issues or questions:
1. Create an issue with `security` label
2. For sensitive security issues, use private security advisories
3. Contact maintainers directly for critical vulnerabilities

---

**Last Updated:** February 2026
**Maintainer:** Finance Commander

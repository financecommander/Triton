# Security Checklist for Contributors

## Before Creating a Pull Request

### Local Security Checks

Run these checks before pushing your code:

```bash
# 1. Install security tools
pip install pip-audit safety bandit

# 2. Check dependencies for vulnerabilities
pip-audit

# 3. Run Safety check
safety check

# 4. Scan code for security issues
bandit -r backend/ compiler/ models/ kernels/ -ll

# 5. Check for accidentally committed secrets (optional)
pip install detect-secrets
detect-secrets scan --all-files
```

### Common Security Issues to Avoid

#### ❌ Don't Do This:

**1. Hardcoded Secrets**
```python
API_KEY = "sk-1234567890abcdef"
PASSWORD = "my_secret_password"
```

**2. Unsafe Deserialization**
```python
import pickle
data = pickle.load(open('data.pkl', 'rb'))
```

**3. Command Injection**
```python
import os
os.system(f"rm -rf {user_input}")
```

**4. SQL Injection**
```python
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
```

**5. Eval/Exec Usage**
```python
eval(user_input)
exec(untrusted_code)
```

#### ✅ Do This Instead:

**1. Environment Variables**
```python
import os
API_KEY = os.environ.get('API_KEY')
PASSWORD = os.environ.get('PASSWORD')
```

**2. Safe Serialization**
```python
import json
data = json.load(open('data.json', 'r'))
```

**3. Safe Command Execution**
```python
import subprocess
subprocess.run(['rm', '-rf', safe_path], check=True)
```

**4. Parameterized Queries**
```python
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

**5. Avoid Dynamic Code Execution**
```python
# Use safe alternatives or validate input thoroughly
# Consider using ast.literal_eval() for simple cases
import ast
result = ast.literal_eval(trusted_string)
```

## Understanding Security Scan Results

### Severity Levels

| Level | Action Required | Timeline |
|-------|----------------|----------|
| 🔴 Critical | Fix immediately | Block PR |
| 🟠 High | Fix before merge | Block PR |
| 🟡 Medium | Should fix | Review required |
| 🟢 Low | Optional | Informational |

### Common Bandit Issue IDs

- **B101**: `assert` used - Can be disabled in production
- **B102**: `exec` used - Dangerous if input is untrusted
- **B103**: `chmod` with insecure permissions
- **B201**: Flask `debug=True` - Never use in production
- **B301**: `pickle` module - Use JSON or other safe formats
- **B303**: Insecure hash functions (MD5, SHA1)
- **B311**: Random without cryptographic strength
- **B403**: `import pickle` - Insecure deserialization
- **B404**: `import subprocess` - Use carefully
- **B501**: Weak SSL/TLS protocol version
- **B601-B609**: SQL injection vulnerabilities
- **B701**: Using `jinja2` with autoescape=False

### Handling False Positives

If a security finding is a false positive:

1. **Document why** it's safe in your code comments
2. **Add a Bandit skip** with justification:
   ```python
   # nosec B101 - Assert is used only in tests
   assert value > 0
   ```

3. **Create a `.bandit` configuration** (if needed):
   ```yaml
   # .bandit
   exclude_dirs:
     - tests/
     - examples/
   
   skips:
     - B101  # assert_used (safe in our context)
   ```

## Responding to Security Workflow Failures

### Step 1: Check the Workflow Run

1. Go to the **Actions** tab in GitHub
2. Click on the failed **Security Scanning** workflow
3. Review which job failed

### Step 2: Download Reports

1. Scroll to **Artifacts** section
2. Download relevant reports:
   - `dependency-scan-reports`
   - `bandit-report`

### Step 3: Fix Issues Locally

```bash
# Update vulnerable dependencies
pip install --upgrade package-name

# Fix code issues identified by Bandit
# Edit the relevant files

# Re-run checks locally
pip-audit
bandit -r backend/ compiler/ models/ kernels/ -ll
```

### Step 4: Push Fixes

```bash
git add .
git commit -m "fix: address security vulnerabilities"
git push
```

### Step 5: Verify

- Wait for security checks to re-run
- Verify all checks pass
- Request code review

## Security Best Practices

### For All Code

- ✅ Use environment variables for secrets
- ✅ Validate and sanitize all user input
- ✅ Use parameterized queries for databases
- ✅ Keep dependencies up to date
- ✅ Use HTTPS/TLS for network communication
- ✅ Implement proper error handling
- ✅ Log security-relevant events
- ✅ Follow principle of least privilege

### For Dependencies

- ✅ Review Dependabot PRs promptly
- ✅ Pin major versions, allow patch updates
- ✅ Audit new dependencies before adding
- ✅ Prefer well-maintained packages
- ✅ Check for known vulnerabilities

### For API Keys and Secrets

- ✅ Never commit secrets to git
- ✅ Use `.env` files (add to `.gitignore`)
- ✅ Use secret management services
- ✅ Rotate secrets regularly
- ✅ Use different secrets for dev/prod

### For File Operations

- ✅ Validate file paths
- ✅ Use safe file permissions
- ✅ Avoid path traversal vulnerabilities
- ✅ Check file size limits
- ✅ Validate file types

## Getting Help

### Security Questions

For security-related questions:
1. Create an issue with `security` label
2. Tag maintainers if urgent
3. Follow responsible disclosure for vulnerabilities

### Reporting Security Vulnerabilities

For sensitive security issues:
1. **DO NOT** create a public issue
2. Use GitHub Security Advisories (private)
3. Or email maintainers directly
4. Provide detailed reproduction steps
5. Wait for maintainer response before disclosure

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [pip-audit Documentation](https://github.com/pypa/pip-audit)
- [Security Workflow Documentation](docs/SECURITY_WORKFLOW.md)

---

**Remember:** Security is everyone's responsibility. When in doubt, ask for review!

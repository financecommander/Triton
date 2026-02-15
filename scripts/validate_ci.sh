#!/usr/bin/env bash
# CI/CD Workflow Validation Script
# This script validates the CI/CD setup locally before pushing

set -e

echo "🔍 CI/CD Workflow Validation"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if required tools are installed
check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}✗${NC} $1 is not installed"
        return 1
    fi
}

echo "1. Checking required tools..."
check_tool python
check_tool git
check_tool pip
echo ""

# Validate YAML syntax
echo "2. Validating YAML syntax..."
python -c "
import yaml
import sys

files = [
    '.github/workflows/ci.yml',
    '.github/workflows/publish.yml',
    '.github/workflows/docs.yml',
    '.github/dependabot.yml',
    '.github/ISSUE_TEMPLATE/bug_report.yml',
    '.github/ISSUE_TEMPLATE/feature_request.yml',
    '.github/ISSUE_TEMPLATE/config.yml',
]

errors = 0
for file in files:
    try:
        with open(file, 'r') as f:
            yaml.safe_load(f)
        print(f'✓ {file}')
    except Exception as e:
        print(f'✗ {file}: {e}')
        errors += 1
        
sys.exit(errors)
"
echo ""

# Check if pyproject.toml exists and is valid
echo "3. Checking pyproject.toml..."
if [ -f "pyproject.toml" ]; then
    echo -e "${GREEN}✓${NC} pyproject.toml exists"
    
    # Extract version
    VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
    echo "   Current version: ${VERSION}"
else
    echo -e "${RED}✗${NC} pyproject.toml not found"
    exit 1
fi
echo ""

# Summary
echo "=============================="
echo -e "${GREEN}✓${NC} CI/CD validation complete!"
echo ""

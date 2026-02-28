"""Tests for version consistency and release readiness."""

import re
import sys
import os
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestVersionConsistency:
    """Test that version strings are consistent across the project."""

    def test_compiler_init_has_version(self):
        """Test that compiler/__init__.py defines __version__."""
        from compiler import __version__

        assert __version__, "__version__ should be defined in compiler/__init__.py"
        assert re.match(r"^\d+\.\d+\.\d+", __version__), (
            f"Version '{__version__}' should follow semantic versioning"
        )

    def test_pyproject_toml_version(self):
        """Test that pyproject.toml has a version."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        assert match, "pyproject.toml should contain a version"
        version = match.group(1)
        assert re.match(r"^\d+\.\d+\.\d+", version), (
            f"Version '{version}' should follow semantic versioning"
        )

    def test_versions_match(self):
        """Test that all version strings are consistent."""
        from compiler import __version__ as compiler_version

        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
        pyproject_version = match.group(1)

        assert compiler_version == pyproject_version, (
            f"compiler/__init__.py ({compiler_version}) != pyproject.toml ({pyproject_version})"
        )

    def test_changelog_has_current_version(self):
        """Test that CHANGELOG.md has an entry for the current version."""
        from compiler import __version__

        changelog = PROJECT_ROOT / "CHANGELOG.md"
        assert changelog.exists(), "CHANGELOG.md should exist"
        content = changelog.read_text()
        assert f"[{__version__}]" in content, (
            f"CHANGELOG.md should have an entry for v{__version__}"
        )


class TestReleaseArtifacts:
    """Test that release artifacts exist."""

    def test_manifest_in_exists(self):
        """Test that MANIFEST.in exists for source distributions."""
        manifest = PROJECT_ROOT / "MANIFEST.in"
        assert manifest.exists(), "MANIFEST.in should exist"

    def test_migration_guide_exists(self):
        """Test that migration guide exists."""
        migration = PROJECT_ROOT / "MIGRATION.md"
        assert migration.exists(), "MIGRATION.md should exist"

    def test_release_checklist_exists(self):
        """Test that release checklist exists."""
        checklist = PROJECT_ROOT / "RELEASE_CHECKLIST.md"
        assert checklist.exists(), "RELEASE_CHECKLIST.md should exist"

    def test_release_script_exists(self):
        """Test that release automation script exists."""
        script = PROJECT_ROOT / "scripts" / "release.py"
        assert script.exists(), "scripts/release.py should exist"

    def test_pyproject_has_stable_classifier(self):
        """Test that pyproject.toml has Production/Stable classifier."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text()
        assert "Production/Stable" in content, (
            "pyproject.toml should have Production/Stable classifier"
        )

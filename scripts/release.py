#!/usr/bin/env python3
"""
Release automation script for Triton DSL.

Usage:
    python scripts/release.py check      # Validate release readiness
    python scripts/release.py build      # Build distribution packages
    python scripts/release.py publish    # Upload to PyPI (requires credentials)
"""

import argparse
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
MAX_OUTPUT_CHARS = 2000


def get_version() -> str:
    """Get the current version from compiler/__init__.py."""
    init_file = PROJECT_ROOT / "compiler" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print("ERROR: Could not find __version__ in compiler/__init__.py")
        sys.exit(1)
    return match.group(1)


def get_pyproject_version() -> str:
    """Get the version from pyproject.toml."""
    content = PYPROJECT_TOML.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("ERROR: Could not find version in pyproject.toml")
        sys.exit(1)
    return match.group(1)


def check_versions_consistent() -> bool:
    """Verify all version strings are consistent."""
    py_version = get_version()
    toml_version = get_pyproject_version()

    versions = {"compiler/__init__.py": py_version, "pyproject.toml": toml_version}

    bench_init = PROJECT_ROOT / "tests" / "benchmarks" / "__init__.py"
    if bench_init.exists():
        content = bench_init.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            versions["tests/benchmarks/__init__.py"] = match.group(1)

    unique = set(versions.values())
    if len(unique) > 1:
        print("ERROR: Version mismatch detected:")
        for loc, ver in versions.items():
            print(f"  {loc}: {ver}")
        return False

    print(f"All versions consistent: {py_version}")
    return True


def check_changelog(version: str) -> bool:
    """Verify CHANGELOG.md has an entry for the current version."""
    changelog = PROJECT_ROOT / "CHANGELOG.md"
    if not changelog.exists():
        print("ERROR: CHANGELOG.md not found")
        return False
    content = changelog.read_text()
    if f"[{version}]" not in content:
        print(f"ERROR: No changelog entry for v{version}")
        return False
    print(f"Changelog entry found for v{version}")
    return True


def check_manifest() -> bool:
    """Verify MANIFEST.in exists."""
    manifest = PROJECT_ROOT / "MANIFEST.in"
    if not manifest.exists():
        print("ERROR: MANIFEST.in not found")
        return False
    print("MANIFEST.in found")
    return True


def run_tests() -> bool:
    """Run the test suite."""
    print("Running tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: Tests failed")
        print(result.stdout[-MAX_OUTPUT_CHARS:] if len(result.stdout) > MAX_OUTPUT_CHARS else result.stdout)
        return False
    print("All tests passed")
    return True


def cmd_check(args: argparse.Namespace) -> None:
    """Validate release readiness."""
    version = get_version()
    print(f"Checking release readiness for v{version}...")
    print("=" * 50)

    checks = [
        ("Version consistency", check_versions_consistent),
        ("Changelog entry", lambda: check_changelog(version)),
        ("MANIFEST.in", check_manifest),
    ]

    if not args.skip_tests:
        checks.append(("Test suite", run_tests))

    results = []
    for name, check_fn in checks:
        print(f"\n--- {name} ---")
        passed = check_fn()
        results.append((name, passed))

    print("\n" + "=" * 50)
    print("Release Readiness Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\nv{version} is ready for release!")
    else:
        print(f"\nv{version} has issues that need to be resolved.")
        sys.exit(1)


def cmd_build(args: argparse.Namespace) -> None:
    """Build distribution packages."""
    version = get_version()
    print(f"Building Triton DSL v{version}...")

    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists():
        import shutil

        shutil.rmtree(dist_dir)

    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: Build failed")
        print(result.stderr)
        sys.exit(1)

    print("Build artifacts:")
    for f in dist_dir.iterdir():
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")
    print("Build successful!")


def cmd_publish(args: argparse.Namespace) -> None:
    """Upload to PyPI."""
    version = get_version()
    dist_dir = PROJECT_ROOT / "dist"

    if not dist_dir.exists() or not list(dist_dir.iterdir()):
        print("ERROR: No build artifacts found. Run 'build' first.")
        sys.exit(1)

    repo_flag = []
    if args.test:
        repo_flag = ["--repository", "testpypi"]
        print(f"Uploading Triton DSL v{version} to Test PyPI...")
    else:
        print(f"Uploading Triton DSL v{version} to PyPI...")

    result = subprocess.run(
        [sys.executable, "-m", "twine", "upload", *repo_flag, "dist/*"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("ERROR: Upload failed")
        sys.exit(1)
    print("Upload successful!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton DSL Release Tool")
    subparsers = parser.add_subparsers(dest="command", help="Release command")

    check_parser = subparsers.add_parser("check", help="Validate release readiness")
    check_parser.add_argument(
        "--skip-tests", action="store_true", help="Skip running test suite"
    )

    subparsers.add_parser("build", help="Build distribution packages")

    publish_parser = subparsers.add_parser("publish", help="Upload to PyPI")
    publish_parser.add_argument(
        "--test", action="store_true", help="Upload to Test PyPI instead"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "check": cmd_check,
        "build": cmd_build,
        "publish": cmd_publish,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()

"""The package version has one source and reaches every provenance record."""

import re
from pathlib import Path

import syncmoments
from syncmoments.model import _basis_core

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    assert syncmoments.__version__ == declared


def test_provenance_version_is_the_package_version_not_installed_metadata():
    # importlib.metadata reports whatever distribution is installed, which can
    # be a stale build or, under the old name, an unrelated PyPI project;
    # provenance must use the version of the imported package.
    assert _basis_core.package_version() == syncmoments.__version__

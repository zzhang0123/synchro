"""The package version has one source and reaches every provenance record."""

import re
from pathlib import Path

import synchro
from synchro.model import _basis_core

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    assert synchro.__version__ == declared


def test_provenance_version_is_the_package_version_not_installed_metadata():
    # importlib.metadata.version("synchro") can name an unrelated PyPI
    # distribution of the same name; provenance must use this package.
    assert _basis_core.package_version() == synchro.__version__

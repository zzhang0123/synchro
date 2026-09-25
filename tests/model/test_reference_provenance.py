"""The vendored manuscript reference is the code that produced its saved numbers."""

import hashlib
import json
from pathlib import Path

from _harmonic_oracles import REFERENCE

VALIDATION = Path(REFERENCE) / "validation"


def _sha256(name: str) -> str:
    return hashlib.sha256((VALIDATION / name).read_bytes()).hexdigest()


def test_saved_results_name_the_vendored_scripts():
    results = json.loads((VALIDATION / "full_response_results.json").read_text())
    assert results["script_sha256"] == _sha256("full_response.py")
    assert results["quadrature_script_sha256"] == _sha256("full_response_product.py")


def test_reference_path_is_inside_the_repository():
    tests_root = Path(__file__).resolve().parents[1]
    assert VALIDATION.resolve().is_relative_to(tests_root)
    assert (VALIDATION / "__init__.py").is_file()

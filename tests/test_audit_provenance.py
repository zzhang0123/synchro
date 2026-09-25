"""The numerical audit must measure the same checkout that it fingerprints."""

from pathlib import Path
import os
import subprocess
import sys


def test_radiation_audit_rejects_shadow_package_before_computing(tmp_path):
    shadow = tmp_path / "shadow"
    package = shadow / "syncmoments"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    sentinel = "raise AssertionError('wrong package numerical kernel was executed')"
    (package / "bessel.py").write_text(
        f"def bessel_jn_and_prime(*args, **kwargs):\n    {sentinel}\n"
        f"def bessel_kn(*args, **kwargs):\n    {sentinel}\n",
        encoding="utf-8",
    )
    (package / "ultrarel.py").write_text(
        f"def F(*args, **kwargs):\n    {sentinel}\n"
        f"def G(*args, **kwargs):\n    {sentinel}\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "should-not-exist.json"
    process = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "review_radiation.py"),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(shadow), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert process.returncode != 0
    assert "provenance mismatch" in process.stderr.lower()
    assert str(package.resolve()) in process.stderr
    assert "wrong package numerical kernel was executed" not in process.stderr
    assert not output.exists()

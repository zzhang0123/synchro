"""The ``slow`` gate of ``tests/conftest.py``: both sides of every switch.

A ``slow`` test runs with ``-m slow`` or ``SYNCMOMENTS_RUN_SLOW=1`` and is
skipped otherwise; ``-m 'not slow'`` deselects it. Checked by running pytest
in a subprocess on a copy of the conftest and a two-test module.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")
MODULE = """
import pytest

@pytest.mark.slow
def test_heavy():
    pass

def test_light():
    pass
"""
INI = "[pytest]\nmarkers =\n    slow: long-running tests\n"


def load_conftest():
    spec = importlib.util.spec_from_file_location("_root_conftest", CONFTEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "markexpr, env, expected",
    [
        ("", {}, False),
        ("not slow", {}, False),
        ("not  slow", {}, False),
        ("slow", {}, True),
        ("slow and not gpu", {}, True),
        ("", {"SYNCMOMENTS_RUN_SLOW": "1"}, True),
        ("", {"SYNCMOMENTS_RUN_SLOW": "0"}, False),
        ("", {"SYNCMOMENTS_RUN_SLOW": ""}, False),
        ("not slow", {"SYNCMOMENTS_RUN_SLOW": "1"}, True),
    ],
)
def test_slow_requested(markexpr, env, expected):
    assert load_conftest().slow_requested(markexpr, env) is expected


def run(tmp_path, *args, env_value=None):
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_gate.py").write_text(MODULE)
    (tmp_path / "pytest.ini").write_text(INI)
    env = {k: v for k, v in os.environ.items() if k != "SYNCMOMENTS_RUN_SLOW"}
    if env_value is not None:
        env["SYNCMOMENTS_RUN_SLOW"] = env_value
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "args, env_value, summary",
    [
        ((), None, "1 passed, 1 skipped"),
        (("-m", "not slow"), None, "1 passed, 1 deselected"),
        (("-m", "slow"), None, "1 passed, 1 deselected"),
        ((), "1", "2 passed"),
        ((), "0", "1 passed, 1 skipped"),
    ],
)
def test_slow_gate_in_a_subprocess(tmp_path, args, env_value, summary):
    assert run(tmp_path, *args, env_value=env_value).startswith(summary)

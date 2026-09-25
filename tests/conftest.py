"""Shared pytest settings live in pyproject.toml; this file gates ``slow`` tests.

The ``slow`` marker is registered in ``pyproject.toml``. A test marked
``slow`` runs when it is selected explicitly (``-m slow``, or any ``-m``
expression naming ``slow`` without ``not slow``) or when the environment
sets ``SYNCMOMENTS_RUN_SLOW=1``; otherwise it is skipped with that reason.
``-m 'not slow'`` deselects it.
"""

import os

import pytest

SLOW_ENV = "SYNCMOMENTS_RUN_SLOW"


def slow_requested(markexpr: str, environ=os.environ) -> bool:
    """True when the ``-m`` expression or ``SYNCMOMENTS_RUN_SLOW`` asks for slow tests."""
    if environ.get(SLOW_ENV, "").strip() not in ("", "0"):
        return True
    expr = " ".join(markexpr.split())
    return "slow" in expr and "not slow" not in expr


def pytest_collection_modifyitems(config, items):
    if slow_requested(config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason=f"slow: select with -m slow or set {SLOW_ENV}=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

"""R30 wording scan extended to ``syncmoments.model.fit`` (review item NEW2-5).

``test_remainder_probe_route.py`` scans ``syncmoments.model`` but skips the
``fit`` subpackage and a few modules; this file scans those, with the same
rule: a docstring may say what is *not* certified, never that a finite check
certifies something. It also pins the ``AssumptionRecord.hyper`` wording for
fitted closures (NaN placeholders).
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import syncmoments  # noqa: F401
import syncmoments.model.fit
from syncmoments.model.errors import AssumptionRecord
from test_remainder_probe_route import _positive_certification_claims


def _modules():
    for info in pkgutil.iter_modules(syncmoments.model.fit.__path__):
        yield importlib.import_module(f"syncmoments.model.fit.{info.name}")
    yield importlib.import_module("syncmoments.model.fit")
    for name in ("assumptions", "_project", "_refine"):
        yield importlib.import_module(f"syncmoments.model.{name}")


def _docstrings(module):
    texts = [module.__doc__ or ""]
    for value in vars(module).values():
        if getattr(value, "__module__", None) == module.__name__:
            texts.append(getattr(value, "__doc__", None) or "")
            if isinstance(value, type):
                texts += [
                    getattr(m, "__doc__", None) or "" for m in vars(value).values()
                ]
    return texts


@pytest.mark.parametrize("module", list(_modules()), ids=lambda m: m.__name__)
def test_no_positive_certification_claims_in_fit_docstrings(module):
    claims = [c for t in _docstrings(module) for c in _positive_certification_claims(t)]
    assert not claims, claims


def test_assumption_record_doc_states_nan_placeholders_for_fitted_hyper():
    doc = " ".join(AssumptionRecord.__doc__.split())
    assert "its concrete hyper-parameters" not in doc
    assert "NaN placeholders" in doc
    assert "(fitted)" in doc

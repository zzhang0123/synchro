"""Statements of the harmonic-kernel docstrings about derivatives and memory.

Round-2 review (T-004, items S00-S02). The checks are textual and pin what
the docstrings must and must not claim; the numbers behind them are tested
in ``test_b_analytic_derivatives.py`` (analytic against autodiff),
``test_memory_numerics.py`` and ``test_kernel_chunking.py`` (budgets).

S00: in affine-follow cells (``0 < lo < RHO_SWITCH hi``) and at an exact
channel-edge/line coincidence at ``t = 0`` the derivative of the rule is
not the derivative of the computed values; they differ by the rule's
quadrature error, which is estimated, not bounded.
S01: ``derivatives="autodiff"`` differentiates the shared-contour Bessel rule
(``bessel_jn_neighbours``); it has the v0.2.0 structure but is not
bit-identical to v0.2.0 and loses accuracy at ``|x| << m``.
S02: the ``chunk_budget`` guarantee covers ``channel_modes`` vmapped over
samples only through the direct average's ``samples_per_step`` cap.
"""

import re

from syncmoments.model import _direct, _harmonic_cells, _harmonic_taylor, harmonic


def flat(text):
    return re.sub(r"\s+", " ", text or "")


def test_taylor_docstring_does_not_claim_the_derivative_of_the_computed_values():
    doc = flat(_harmonic_taylor.__doc__)
    assert "derivative of the same computed rule" not in doc
    assert "same AD of the same program" in doc
    assert "affine" in doc and "not bounded" in doc
    assert "coincidence" in doc


def test_harmonic_module_docstring_derivative_paths():
    doc = flat(harmonic.__doc__)
    assert "keeps the v0.2.0 path" not in doc
    assert "same computed rule" not in doc
    assert "bessel_jn_neighbours" in doc
    assert "not bit-identical to v0.2.0" in doc
    assert "|x| << m" in doc
    assert "coincidence" in doc


def test_harmonic_lines_and_cells_docstrings():
    lines = flat(_harmonic_cells.harmonic_lines.__doc__)
    assert "(the v0.2.0 path)" not in lines
    assert "not bit-identical to v0.2.0" in lines
    cells = flat(_harmonic_cells.product_cells.__doc__)
    assert "coincidence" in cells and "not differentiable" in cells


def test_memory_statement_names_the_sample_vmapped_path():
    doc = flat(harmonic.__doc__)
    assert "samples_per_step" in doc
    assert "direct_channel_average" in doc
    kernel_doc = flat(harmonic.HarmonicKernel.samples_per_step.__doc__)
    assert "chunk_budget" in kernel_doc
    assert "samples_per_step" in flat(_direct.direct_channel_average.__doc__)

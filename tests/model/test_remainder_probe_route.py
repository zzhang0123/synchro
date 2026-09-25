"""Round-2 regressions of ``RemainderInputs.from_samples`` and of the R30 wording.

NEW-11: ``from_samples`` with ``phase=None`` probed the exact per-emitter
phase ``exp(i tau depth_n)`` even when the basis uses a screen route, so the
``P`` column of ``basis_remainder`` (``intrinsic_residual`` on screen routes)
described a route the basis does not use. ``phase=None`` now means
``basis.phase``. The test uses the harmonic kernel (not a polynomial one, on
which both routes sit at roundoff) with a Gaussian screen whose ``sigma``
damps the polarised weight, so the two routes differ by a visible factor.

R30: the docstrings no longer call finite numerical checks "certified".
"""

import importlib
import pkgutil
import re

import numpy as np
from numpy.testing import assert_allclose
import pytest

import syncmoments  # noqa: F401
import syncmoments.model
from syncmoments.model.basis import build_basis
from syncmoments.model.bounds import RemainderInputs
from syncmoments.model.index import Truncation
from syncmoments.model.moments import JointMoments
from syncmoments.model.phase import GaussianScreen, TaylorPhase
from syncmoments.model.predict import predict

from _predict_helpers import H_DEPTH_REF, H_S_DEPTH, harmonic_setup, samples_of


@pytest.fixture(scope="module")
def screen_case():
    kernel, channels, supp, ref, pop = harmonic_setup()
    screen = GaussianScreen(mean=H_DEPTH_REF, sigma=3.0 * H_S_DEPTH)
    basis = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1, depth_degree=0),
        ref,
        support=supp,
        phase=screen,
        convergence=False,
    )
    pop = dict(pop)
    pop["depth"] = pop["depth"] + 0.5 * H_S_DEPTH
    samples = samples_of(pop)
    return kernel, basis, ref, samples


def test_new11_default_probe_route_is_the_basis_route(screen_case):
    kernel, basis, ref, samples = screen_case
    default = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    explicit = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe", phase=basis.phase
    )
    for name in ("rho_ang", "H", "intrinsic_residual"):
        assert_allclose(
            np.asarray(getattr(default, name)),
            np.asarray(getattr(explicit, name)),
            rtol=1e-12,
            err_msg=name,
        )


def test_new11_screen_route_differs_from_the_per_emitter_phase(screen_case):
    """On this kernel the route choice changes the probed ``P`` input visibly."""
    kernel, basis, ref, samples = screen_case
    screen = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe"
    )
    exact = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe", phase=TaylorPhase(0)
    )
    a = np.asarray(screen.intrinsic_residual)
    b = np.asarray(exact.intrinsic_residual)
    assert np.all(a > 0) and np.all(b > 0)
    assert np.max(np.abs(a - b) / b) > 1e-2, (a, b)
    # I and V do not carry the phase: identical on both routes.
    assert_allclose(
        np.asarray(screen.rho_ang)[:, [0, 2]],
        np.asarray(exact.rho_ang)[:, [0, 2]],
        rtol=1e-12,
    )


def test_new11_predict_samples_path_uses_the_basis_route(screen_case):
    kernel, basis, ref, samples = screen_case
    moments = JointMoments.from_samples(samples, basis.index, ref)
    via_predict = predict(
        basis, moments, amplitude=1.0, samples=samples, kernel=kernel
    ).budget.basis_remainder
    right = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, angular_residual="probe", phase=basis.phase
    )
    direct = predict(basis, moments, amplitude=1.0, errors=right).budget
    assert_allclose(
        np.asarray(via_predict.value),
        np.asarray(direct.basis_remainder.value),
        rtol=1e-12,
    )


def test_new11_stub_basis_without_phase_keeps_the_exact_phase():
    """A basis without a ``phase`` field still probes the per-emitter phase."""
    from _basis_stub import build_stub_basis

    kernel, channels, supp, ref, pop = harmonic_setup()
    basis = build_stub_basis(kernel, channels, Truncation(1, 1, 1), ref, supp)
    samples = samples_of(pop)
    default = RemainderInputs.from_samples(samples, basis, kernel=kernel)
    exact = RemainderInputs.from_samples(
        samples, basis, kernel=kernel, phase=TaylorPhase(0)
    )
    assert_allclose(np.asarray(default.H), np.asarray(exact.H), rtol=1e-12)


# -- R30: finite checks are not called certificates ----------------------------------

# A positive "certif..." claim: the word not preceded by a negation.
_NEGATED = re.compile(
    r"(not|nothing|never|no)\b[\w\s,()`'-]{0,40}$", flags=re.IGNORECASE
)

_NEGATED_AFTER = re.compile(r"\s+(nothing|no)\b", flags=re.IGNORECASE)


def _positive_certification_claims(text):
    claims = []
    for match in re.finditer(r"\bcertif\w*", text, flags=re.IGNORECASE):
        word = match.group(0)
        if word.lower() == "certified_orders":
            continue
        before = text[max(0, match.start() - 60) : match.start()]
        before = before.split(".")[-1]
        after = text[match.end() : match.end() + 12]
        if not _NEGATED.search(before) and not _NEGATED_AFTER.match(after):
            claims.append(text[max(0, match.start() - 40) : match.end() + 20])
    return claims


def _owned_modules():
    skip = {"predict", "_budget_terms", "_direct", "assumptions", "_project", "_refine"}
    for info in pkgutil.iter_modules(syncmoments.model.__path__):
        if info.name in skip or info.name == "fit":
            continue
        yield importlib.import_module(f"syncmoments.model.{info.name}")


def test_r30_basis_docstring_does_not_call_finite_checks_certified():
    import syncmoments.model.basis as basis_module

    doc = " ".join(basis_module.__doc__.split())
    assert "Certified here" not in doc
    assert "finite checks, not certificates" in doc
    assert "finite-difference" in basis_module.SpectralBasis.__doc__


@pytest.mark.parametrize("module", list(_owned_modules()), ids=lambda m: m.__name__)
def test_r30_no_positive_certification_claims_in_docstrings(module):
    texts = [module.__doc__ or ""]
    for value in vars(module).values():
        if getattr(value, "__module__", None) == module.__name__:
            texts.append(getattr(value, "__doc__", None) or "")
            if isinstance(value, type):
                texts += [
                    getattr(m, "__doc__", None) or "" for m in vars(value).values()
                ]
    claims = [c for text in texts for c in _positive_certification_claims(text)]
    assert not claims, claims

"""Eager refusals of invalid weights and samples (T-004 round 3, H2).

The weight and sample checks of ``syncmoments.rm`` run inside one compiled
helper per shape. Compiled with ``jax.jit`` (round 2), an eager refusal raised
``jax.errors.JaxRuntimeError`` and JAX printed a "jax.pure_callback failed"
traceback to stderr; v0.2.0 raised ``equinox.EquinoxRuntimeError``. These
tests pin the v0.2.0 error type and a silent stderr at every public entry
point that reaches the helpers: ``syncmoments.rm``, ``syncmoments.faraday``,
``syncmoments.model.phase.EmpiricalScreen``, ``PopulationSamples`` (constructor,
``normalised_weights``, ``product``), the harmonic-tail probe and
``mixed_moments``.

``PopulationSamples.product`` validates all marginals in one compiled call;
its weights must stay bit-identical to per-marginal normalisation.
"""

import logging
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_array_equal
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.expansion import mixed_moments
from syncmoments.faraday import emission_polarisation, joint_faraday_moments
from syncmoments.model._harmonic_tail import _points
from syncmoments.model.moments import PopulationSamples
from syncmoments.model.phase import EmpiricalScreen
from syncmoments.rm import _screen_samples, gaussian_rm_cumulants, screen_polarisation

X = np.array([1.5, 2.0, 2.5, 3.0])
LAM = jnp.array([0.2])
BAD_WEIGHTS = {
    "zeros": np.zeros(4),
    "nan": np.array([np.nan, 1.0, 1.0, 1.0]),
    "negative": np.array([-1.0, 1.0, 1.0, 1.0]),
    "subnormal-max": np.full(4, 1e-320),
}
MESSAGES = {
    "zeros": "weights must be finite",
    "nan": "weights must be finite",
    "negative": "weights must be finite",
    "subnormal-max": "largest weight is below the smallest normal float64",
}


def _population(w):
    v = jnp.asarray(X)
    return PopulationSamples(v, v, 0.1 * v, 0.2 * v, v, v, weights=w)


def _product(w):
    ones = np.ones(1)
    return PopulationSamples.product(
        gamma=(X, w), B=ones, mu=0.1 * ones, eta=0.2 * ones, phi=ones, depth=ones
    )


def _tail_probe(w):
    samples = SimpleNamespace(gamma=X, B=X, mu=0.1 * X, eta=0.2 * X, weights=w)
    return _points(None, samples, None)[4]


WEIGHT_ENTRIES = {
    "rm_cumulants": lambda w: gaussian_rm_cumulants(X, w),
    "screen_polarisation": lambda w: screen_polarisation(1.0 + 0j, X, LAM, w),
    "emission_polarisation": lambda w: emission_polarisation(1.0, X, LAM, w),
    "joint_faraday_moments": lambda w: joint_faraday_moments(jnp.ones((4, 1)), X, 1, w),
    "empirical_screen": lambda w: EmpiricalScreen(jnp.asarray(X), jnp.asarray(w))(
        jnp.asarray([0.1])
    ),
    "population": _population,
    "population_product": _product,
    "tail_probe": _tail_probe,
    "mixed_moments": lambda w: mixed_moments(X[:, None], jnp.ones(4), w),
}
NONFINITE = np.array([1.0, np.inf, 2.0, 3.0])
SAMPLE_ENTRIES = {
    "rm_cumulants": lambda: gaussian_rm_cumulants(NONFINITE),
    "rm_cumulants_weighted": lambda: gaussian_rm_cumulants(NONFINITE, np.ones(4)),
    "screen_polarisation": lambda: screen_polarisation(1.0 + 0j, NONFINITE, LAM),
    "emission_polarisation": lambda: emission_polarisation(1.0, NONFINITE, LAM),
    "empirical_screen": lambda: EmpiricalScreen(jnp.asarray(NONFINITE))(
        jnp.asarray([0.1])
    ),
    "tail_probe": lambda: _points(
        None, SimpleNamespace(gamma=NONFINITE, B=X, mu=X, eta=X), None
    ),
}


def _assert_quiet_equinox_refusal(call, match, capfd, caplog):
    """``call`` raises ``EquinoxRuntimeError`` (not ``JaxRuntimeError``) and
    writes nothing to stderr or to the error log."""
    capfd.readouterr()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(eqx.EquinoxRuntimeError, match=match) as info:
            jax.block_until_ready(call())
    assert not isinstance(info.value, jax.errors.JaxRuntimeError)
    captured = capfd.readouterr()
    assert captured.err == "", captured.err[:400]
    assert "pure_callback" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("case", BAD_WEIGHTS)
@pytest.mark.parametrize("entry", WEIGHT_ENTRIES)
def test_eager_weight_refusal_is_equinox_error_without_stderr(
    entry, case, capfd, caplog
):
    w = BAD_WEIGHTS[case]
    _assert_quiet_equinox_refusal(
        lambda: WEIGHT_ENTRIES[entry](w), MESSAGES[case], capfd, caplog
    )


@pytest.mark.parametrize("entry", SAMPLE_ENTRIES)
def test_eager_nonfinite_sample_refusal_is_equinox_error_without_stderr(
    entry, capfd, caplog
):
    _assert_quiet_equinox_refusal(
        SAMPLE_ENTRIES[entry], "samples must be finite", capfd, caplog
    )


POPULATION_VALUES = {
    "gamma": (0, 0.5, "gamma must be >= 1"),
    "B": (1, -1.0, "B must be >= 0"),
    "mu": (2, 1.5, "mu must lie in"),
    "eta": (3, -1.5, "eta must lie in"),
    "phi": (4, np.nan, "phi must be finite"),
    "depth": (5, np.inf, "depth must be finite"),
}


@pytest.mark.parametrize("wrap", ["eager", "jit"])
@pytest.mark.parametrize("case", POPULATION_VALUES)
def test_population_value_refusals(case, wrap, capfd, caplog):
    """The constructor's value checks run in one compiled call; each refusal
    keeps its message, eagerly and under ``eqx.filter_jit``."""
    position, bad, message = POPULATION_VALUES[case]
    arrays = [np.full(3, 1.5), np.ones(3), np.zeros(3), np.zeros(3)]
    arrays += [np.zeros(3), np.zeros(3)]
    arrays[position] = np.array([arrays[position][0], bad, arrays[position][2]])

    def build(*a):  # return every array: unused checks are dropped under JIT
        return PopulationSamples(*a, weights=jnp.ones(3))

    call = build if wrap == "eager" else eqx.filter_jit(build)
    _assert_quiet_equinox_refusal(lambda: call(*arrays), message, capfd, caplog)


def test_product_weights_equal_per_marginal_normalisation_bitwise():
    """One validated call for all marginals gives the same bits as normalising
    each marginal on its own (``1e300``- and ``1e-300``-scale masses included)."""
    rng = np.random.default_rng(7)
    marginals = {
        "gamma": (rng.uniform(1, 3, 3), rng.uniform(0, 1e300, 3)),
        "B": rng.uniform(1, 2, 2),
        "mu": (rng.uniform(-1, 1, 4), rng.uniform(0, 1, 4)),
        "eta": np.array([0.3]),
        "phi": (np.array([0.0, 1.0]), np.array([1e-300, 3e-300])),
        "depth": (np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.0, 2.0])),
    }
    population = PopulationSamples.product(**marginals)
    expected = []
    for name in ("gamma", "B", "mu", "eta", "phi", "depth"):
        entry = marginals[name]
        values, w = entry if isinstance(entry, tuple) else (entry, np.ones_like(entry))
        expected.append(_screen_samples(jnp.asarray(values), jnp.asarray(w))[1])
    mass = jnp.ones(())
    for g in jnp.meshgrid(*expected, indexing="ij"):
        mass = mass * g
    assert_array_equal(np.asarray(population.weights), np.asarray(mass.ravel()))

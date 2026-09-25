"""``ParameterMap.flatten``/``unflatten`` under plain ``jax.jit``.

Regression: ``flatten`` used a traced boolean index for the imaginary parts of
complex (``e^{2i phi}``) free tables, which ``jax.jit`` rejects
(``NonConcreteBooleanIndexError``). Oracle: the eager layout (real parts,
then the imaginary parts of the complex entries, then hyper-parameters)
built with NumPy from ``labels()``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model.assumptions import (
    field_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
)
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference

REFERENCE = Reference(5.2, 2.1, 0.3, scales=(1.3, 0.7, 0.9))


def correlated_moments(index, n=9, seed=4):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, n)
    samples = PopulationSamples(
        5.0 + 1.5 * t,
        2.0 + 0.8 * t**2,
        np.clip(0.6 * t + 0.2 * rng.standard_normal(n), -0.95, 0.95),
        np.clip(-0.5 * t, -0.95, 0.95),
        0.3 + 1.7 * t,
        1.5 * t,
        weights=rng.uniform(0.2, 3.0, n),
    )
    return JointMoments.from_samples(samples, index, REFERENCE)


MAPS = {
    "no_assumption": lambda index: no_assumption(index),
    "isotropic_pitch": lambda index: isotropic_pitch(index),
    "field_independent": lambda index: field_independent(index),
    "independent_screen": lambda index: independent_screen(index),
    "gaussian_fit": lambda index: gaussian_screen(index, 0.2, 0.5, fit_hyper=True),
}


@pytest.mark.parametrize("truncation", [Truncation(1, 1, 2), Truncation(2, 2, 2)])
@pytest.mark.parametrize("name", list(MAPS))
def test_flatten_and_unflatten_are_jit_safe(name, truncation):
    index = MomentIndex.build(truncation)
    pm = MAPS[name](index)
    theta = pm.project(correlated_moments(index))
    eager = np.asarray(pm.flatten(theta))
    assert eager.shape == (pm.n_free(),) and eager.dtype == np.float64
    # layout oracle: "Im ..." labels hold the imaginary parts of complex entries
    labels = pm.labels()
    n_im = sum(label.startswith("Im ") for label in labels)
    if name != "gaussian_fit":
        assert n_im > 0  # the map has complex free tables (the regression case)
    expected, pos = [], 0
    for table in (np.asarray(t) for t in theta.tables):
        chunk = labels[pos : pos + table.size]
        complex_entries = [
            i for i, label in enumerate(chunk) if label.startswith("Re ")
        ]
        expected += [table.real, table.imag[complex_entries]]
        pos += table.size + len(complex_entries)
    expected += [np.asarray(h, float) for h in theta.hyper]
    assert_allclose(eager, np.concatenate(expected), atol=0)
    # the map is a traced argument (its closure hyper are array leaves)
    flatten = jax.jit(type(pm).flatten)
    jitted = flatten(pm, theta)
    assert_allclose(np.asarray(jitted), eager, atol=0)
    back = jax.jit(type(pm).unflatten)(pm, jitted)
    for a, b in zip(back.tables, theta.tables):
        assert_allclose(np.asarray(a), np.asarray(b), atol=1e-15)
    assert_allclose(np.asarray(jax.jit(lambda th: pm.flatten(th))(theta)), eager)
    # gradients flow through the jitted flatten into the imaginary parts
    grad = jax.grad(lambda th: jnp.sum(flatten(pm, th) ** 2))(theta)
    for g, t in zip(grad.tables, theta.tables):
        assert np.all(np.isfinite(np.asarray(g)))
        assert_allclose(np.asarray(g).real, 2 * np.asarray(t).real, atol=1e-14)

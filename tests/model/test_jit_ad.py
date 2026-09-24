"""JIT and autodiff contract of ``synchro.model`` (FINAL_DESIGN Section 11, "JIT/AD").

Checks, on a small harmonic configuration (``m_max = 10``, one 65 % bump
channel at ``2 nu_*``, ``Truncation(1, 1, 2)``):

* one trace (compile) per static configuration: ``predict`` under
  ``eqx.filter_jit`` retraces for a new ``MomentIndex`` only, and the jitted
  ``_build_core`` for a new index only, counted with a Python side effect that
  runs at trace time;
* ``jax.grad`` of ``pred.stokes.sum()`` with respect to ``m0`` (exact column
  sums of the response matrix), the Gaussian-screen closure hyper-parameters
  ``(mean, sigma)`` through ``ParameterMap.__call__`` and ``gamma0`` through
  ``_build_core``; each finite and equal to Richardson central differences;
* ``vmap`` over a batch of ``JointMoments`` (``eqx.filter_vmap`` with the
  moment leaves batched and the reference shared).

Not certified: compile counts under plain ``jax.jit`` (the basis and the
moments carry static fields, so ``eqx.filter_jit`` is the supported entry),
and gradients with respect to channel parameters.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from numpy.testing import assert_allclose

from synchro.model.assumptions import Parameters, gaussian_screen
from synchro.model.basis import _build_core, build_basis
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import JointMoments, PopulationSamples, Reference
from synchro.model.phase import TaylorPhase
from synchro.model.predict import predict

from _basis_fixtures import small_harmonic
from _harmonic_oracles import B0, GAMMA0, S_DEPTH

TRUNCATION = Truncation(1, 1, 2)


def population(seed=3, n=40):
    """Correlated atoms inside the small support (gamma 19.5..20.5, B 0.98..1.02)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, n)
    v = rng.uniform(-1, 1, n)
    return PopulationSamples(
        GAMMA0 * (1 + 0.02 * u),
        B0 * (1 + 0.015 * (0.6 * u + 0.4 * v)),
        np.clip(0.3 * u + 0.2 * v, -0.9, 0.9),
        np.clip(0.5 * v - 0.1 * u, -0.9, 0.9),
        0.2 + 0.4 * u + 0.2 * v,
        S_DEPTH * (3.0 + 1.5 * u + 0.5 * v),
        weights=1.0 + 0.5 * rng.uniform(0, 1, n),
    )


@pytest.fixture(scope="module")
def setup():
    kernel, channels, reference, support = small_harmonic()
    basis = build_basis(
        kernel, channels, TRUNCATION, reference, support=support, convergence=False
    )
    moments = JointMoments.from_samples(population(), basis.index, reference)
    return kernel, channels, reference, support, basis, moments


def richardson(f, x, h):
    """Central difference with one Richardson step (error ``O(h^4)``)."""
    d = lambda h: (f(x + h) - f(x - h)) / (2 * h)  # noqa: E731
    return (4 * d(h / 2) - d(h)) / 3


# -- compile counts ------------------------------------------------------------------------


def test_predict_traces_once_per_static_configuration(setup):
    kernel, channels, reference, support, basis, moments = setup
    traces = []

    @eqx.filter_jit
    def run(basis, moments, amplitude):
        traces.append(1)  # runs at trace time only
        return predict(basis, moments, amplitude=amplitude).stokes

    a = run(basis, moments, jnp.asarray(1.0))
    scaled = eqx.tree_at(lambda m: m.m0, moments, moments.m0 * 1.0 + 0.0)
    b = run(basis, scaled, jnp.asarray(2.0))
    assert len(traces) == 1
    assert_allclose(np.asarray(b), 2 * np.asarray(a), rtol=1e-14)
    # A different truncation is a different static configuration: one more trace.
    other = build_basis(
        kernel,
        channels,
        Truncation(1, 1, 1),
        reference,
        support=support,
        convergence=False,
    )
    other_moments = JointMoments.from_samples(population(), other.index, reference)
    run(other, other_moments, jnp.asarray(1.0))
    assert len(traces) == 2
    run(other, other_moments, jnp.asarray(3.0))
    assert len(traces) == 2


def test_build_core_traces_once_per_index(setup):
    kernel, channels, reference, support, basis, moments = setup
    traces = []

    def core(reference, *, index):
        traces.append(1)
        return _build_core(
            kernel,
            channels,
            reference,
            TaylorPhase(index.truncation.max_b()),
            index=index,
            quadrature=None,
        )

    run = eqx.filter_jit(core)
    I1, _, P1 = run(reference, index=basis.index)
    moved = Reference(
        jnp.asarray(20.2), B0, depth_ref=reference.depth_ref, scales=reference.scales
    )
    I2, _, _ = run(moved, index=basis.index)
    assert len(traces) == 1
    assert np.max(np.abs(np.asarray(I1) - np.asarray(I2))) > 0  # the leaves were traced
    assert_allclose(np.asarray(I1), np.asarray(basis.I_basis), rtol=1e-12, atol=1e-300)
    run(reference, index=MomentIndex.build(Truncation(1, 1, 1)))
    assert len(traces) == 2


# -- gradients -----------------------------------------------------------------------------


def test_grad_with_respect_to_m0_matches_response_matrix_and_differences(setup):
    *_, basis, moments = setup

    def total(m0):
        return predict(
            basis, eqx.tree_at(lambda m: m.m0, moments, m0), amplitude=1.5
        ).stokes.sum()

    g = np.asarray(jax.grad(total)(moments.m0))
    C = np.asarray(basis.response_matrix())
    expected = 1.5 * C[:, : basis.index.n0].sum(0)
    assert np.all(np.isfinite(g))
    assert_allclose(g, expected, rtol=1e-12, atol=1e-300)
    # Finite differences (the map is linear in m0, so one step is exact to roundoff).
    m0 = np.asarray(moments.m0)
    for a in (0, 3, basis.index.n0 - 1):
        e = np.zeros_like(m0)
        e[a] = 1.0
        fd = richardson(lambda t: float(total(jnp.asarray(m0 + t * e))), 0.0, 1e-3)
        assert_allclose(g[a], fd, rtol=1e-8, atol=1e-12 * np.max(np.abs(expected)))


def test_grad_with_respect_to_closure_hyper_matches_differences(setup):
    """``gaussian_screen(fit_hyper=True)``: ``d stokes.sum() / d (mean, sigma)``."""
    *_, reference, support, basis, moments = setup[2:]
    pm = gaussian_screen(basis.index, 4.5 * S_DEPTH, 0.7 * S_DEPTH, fit_hyper=True)
    theta = pm.project(moments)
    assert len(theta.hyper) == 1 and theta.hyper[0].shape == (2,)

    def total(hyper):
        params = Parameters(tables=theta.tables, hyper=(hyper,))
        return predict(basis, pm(params, reference), amplitude=1.0).stokes.sum()

    hyper0 = jnp.asarray([4.5 * S_DEPTH, 0.7 * S_DEPTH])
    g = np.asarray(jax.grad(total)(hyper0))
    assert g.shape == (2,) and np.all(np.isfinite(g))
    scale = np.abs(g) * S_DEPTH
    assert np.all(scale > 0)
    for i in range(2):
        e = np.zeros(2)
        e[i] = 1.0
        fd = richardson(lambda t: float(total(hyper0 + t * e)), 0.0, 1e-3 * S_DEPTH)
        assert_allclose(g[i], fd, rtol=1e-6)


def test_grad_with_respect_to_gamma0_through_build_core(setup):
    kernel, channels, reference, support, basis, moments = setup
    index = MomentIndex.build(Truncation(1, 1, 1))
    m = np.asarray(
        JointMoments.from_samples(population(), index, reference).to_vector()
    )
    phase = TaylorPhase(1)

    def response(gamma0):
        ref = Reference(
            gamma0, B0, depth_ref=reference.depth_ref, scales=reference.scales
        )
        I_b, V_b, P_b = _build_core(
            kernel, channels, ref, phase, index=index, quadrature=None
        )
        stokes = (
            I_b @ m[: index.n0]
            + V_b @ m[: index.n0]
            + jnp.real(
                P_b
                @ (m[index.n0 : index.n0 + index.n2] + 1j * m[index.n0 + index.n2 :])
            )
        )
        return stokes.sum()

    g = float(jax.grad(response)(jnp.asarray(GAMMA0)))
    assert np.isfinite(g) and g != 0
    fd = richardson(lambda x: float(response(jnp.asarray(x))), GAMMA0, 2e-2)
    assert_allclose(g, fd, rtol=1e-6)
    # Forward mode agrees with reverse mode.
    assert_allclose(float(jax.jacfwd(response)(jnp.asarray(GAMMA0))), g, rtol=1e-12)


# -- vmap ----------------------------------------------------------------------------------


def test_vmap_over_a_batch_of_joint_moments(setup):
    *_, reference, support, basis, moments = setup[2:]
    batch = [
        JointMoments.from_samples(population(seed), basis.index, reference)
        for seed in (3, 11, 42)
    ]
    # Every array leaf (m0, m2, m0_ext and the reference leaves, which are
    # equal across the batch) gains a leading batch axis; static fields are shared.
    stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *batch)
    assert stacked.m0.shape == (3, basis.index.n0)
    assert stacked.reference.gamma0.shape == (3,)

    def stokes(m):
        return predict(basis, m, amplitude=1.0).stokes

    batched = eqx.filter_vmap(stokes)(stacked)  # in_axes = if_array(0)
    assert batched.shape == (3, basis.n_ch, 4)
    for i, m in enumerate(batch):
        assert_allclose(np.asarray(batched[i]), np.asarray(stokes(m)), rtol=1e-13)
    # The plain jax.vmap form over the batched leaves.
    plain = jax.vmap(
        lambda m0, m2: stokes(JointMoments(basis.index, m0, m2, reference))
    )(jnp.stack([m.m0 for m in batch]), jnp.stack([m.m2 for m in batch]))
    assert_allclose(np.asarray(plain), np.asarray(batched), rtol=1e-13)
    assert batched.shape[0] != 1  # the batch axis is real, not a broadcast

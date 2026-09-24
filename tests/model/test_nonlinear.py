"""Tests of ``synchro.model.fit.nonlinear``: ``Transform`` and ``LogDensity``.

Oracles: finite-difference Jacobians and ``numpy.linalg.slogdet`` for the
log-determinants, a hand-written channel-major response matrix of the
polynomial test kernel, the affine pieces of the map and NumPy linear
algebra for the chi-square, central differences for gradients.
"""

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _nonlinear_oracles import (
    fd_gradient,
    polynomial_basis,
    random_nodes,
    samples_of,
)
from synchro.model.assumptions import (
    Parameters,
    gaussian_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
)
from synchro.model.fit.nonlinear import LogDensity, Transform
from synchro.model.fit.observation import StokesData
from synchro.model.index import MomentIndex, Truncation

# -- helpers ---------------------------------------------------------------------


def gaussian_map(index):
    return gaussian_screen(index, 0.8, 0.6, fit_hyper=True)


def random_z(transform, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(transform.n_params))


def constrained_np(transform, z):
    return np.asarray(transform.constrained(jnp.asarray(z)))


def fd_logdet(transform, z, h=1e-5):
    """``log |det d constrained / dz|`` from a central-difference Jacobian."""
    z = np.asarray(z, dtype=float)
    n = z.size
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        J[:, i] = (
            constrained_np(transform, z + e) - constrained_np(transform, z - e)
        ) / (2 * h)
    sign, logdet = np.linalg.slogdet(J)
    assert sign > 0
    return logdet


def make_data(
    basis, m, amplitude, n_ch, seed=0, *, noise=1e-2, mask=None, response=None
):
    C = np.asarray(basis.response_matrix())
    d = amplitude * C @ np.asarray(m)
    if response is not None:
        d = np.asarray(response) @ d
    n = d.size
    rng = np.random.default_rng(seed)
    if noise == "cov":
        A = rng.standard_normal((n, n))
        cov = A @ A.T / n + np.eye(n)
        shaped = d.reshape(-1) if response is not None else d.reshape(n_ch, 4)
        return StokesData(shaped, cov, mask=mask, response=response)
    return StokesData(
        d.reshape(-1) if response is not None else d.reshape(n_ch, 4),
        np.full(n, noise),
        mask=mask,
        response=response,
    )


# -- Transform: layout ---------------------------------------------------------------


def test_transform_kinds_and_labels(index_111):
    pm = gaussian_map(index_111)
    tr = Transform(pm)
    assert tr.n_params == pm.n_free() + 1
    assert tr.labels[0] == "log_amplitude" and tr.labels[1:] == pm.labels()
    assert tr.kinds[0] == "identity"
    assert tr.kinds[-1] == "log" and tr.labels[-1].endswith(":sigma")
    assert tr.kinds[-2] == "identity" and tr.labels[-2].endswith(":mean")
    assert set(tr.kinds[1:-2]) == {"identity"}
    boxed = Transform(pm, bounds={"hyper:gaussian_depth:mean": (-2.0, 3.0)})
    assert boxed.kinds[-2] == "box" and boxed.bounds[-2] == (-2.0, 3.0)
    plain = Transform(pm, amplitude=False)
    assert plain.n_params == pm.n_free() and plain.labels == pm.labels()
    hash(tr.kinds), hash(tr.bounds)


def test_transform_nodal_kinds(index_111):
    nodes = samples_of(random_nodes(1, 4))
    tr = Transform(nodal(index_111, nodes))
    assert tr.kinds == ("identity",) + ("logit",) * 4


def test_transform_rejects_bad_configuration(index_111):
    pm = gaussian_map(index_111)
    with pytest.raises(ValueError):
        Transform(object())
    with pytest.raises(ValueError):
        Transform(pm, amplitude=1)
    with pytest.raises(ValueError):
        Transform(pm, bounds={"nope": (0.0, 1.0)})
    with pytest.raises(ValueError):
        Transform(pm, bounds={"hyper:gaussian_depth:mean": (1.0, 1.0)})
    with pytest.raises(ValueError):
        Transform(pm, bounds={"hyper:gaussian_depth:mean": (0.0, np.inf)})
    with pytest.raises(ValueError):
        Transform(pm, bounds={"hyper:gaussian_depth:sigma": (0.0, 1.0)})
    with pytest.raises(ValueError):
        Transform(pm, positive=("unknown",))
    with pytest.raises(ValueError):
        Transform(pm).forward(
            Parameters(tables=pm.unflatten(jnp.zeros(pm.n_free())).tables)
        )
    with pytest.raises(ValueError):
        Transform(pm).inverse(jnp.zeros(3))
    with pytest.raises(ValueError):
        Transform(pm).log_det(jnp.zeros(pm.n_free() + 1) + 0j)


# -- Transform: round trips and Jacobians ----------------------------------------------


@pytest.mark.parametrize("boxed", [False, True])
def test_transform_round_trip(index_111, boxed):
    pm = gaussian_map(index_111)
    bounds = {"hyper:gaussian_depth:mean": (-2.0, 3.0)} if boxed else None
    tr = Transform(pm, bounds=bounds)
    for seed in (0, 1):
        z = random_z(tr, seed)
        theta = tr.inverse(z)
        assert theta.log_amplitude is not None and len(theta.hyper) == 1
        assert float(theta.hyper[0][1]) > 0
        if boxed:
            assert -2.0 < float(theta.hyper[0][0]) < 3.0
        assert_allclose(tr.forward(theta), z, atol=1e-12)
        again = tr.inverse(tr.forward(theta))
        assert_allclose(pm.flatten(again), pm.flatten(theta), atol=1e-12)
        assert_allclose(again.log_amplitude, theta.log_amplitude)
    # jit on both directions and the constrained map
    z = random_z(tr, 2)
    assert_allclose(jax.jit(lambda v: tr.forward(tr.inverse(v)))(z), z, atol=1e-12)
    assert_allclose(jax.jit(lambda v: tr.constrained(v))(z), tr.constrained(z))


@pytest.mark.parametrize("boxed", [False, True])
def test_transform_log_det_matches_jacobian(index_111, boxed):
    pm = gaussian_map(index_111)
    bounds = {"hyper:gaussian_depth:mean": (-2.0, 3.0), "<z_gamma>": (-4.0, 4.0)}
    tr = Transform(pm, bounds=bounds if boxed else None)
    for seed in (0, 3):
        z = random_z(tr, seed)
        assert_allclose(float(tr.log_det(z)), fd_logdet(tr, z), rtol=1e-7, atol=1e-8)
    # closed forms: amplitude slot identity, sigma slot log, box slot tanh
    z = jnp.zeros(tr.n_params)
    expected = 0.0
    for kind, pair in zip(tr.kinds, tr.bounds):
        if kind == "box":
            expected += np.log((pair[1] - pair[0]) / 2)
    assert_allclose(float(tr.log_det(z)), expected, atol=1e-13)
    grad = jax.jit(jax.grad(tr.log_det))(random_z(tr, 5))
    assert np.all(np.isfinite(grad))


def test_transform_logit_log_det_is_alr_jacobian(index_111):
    nodes = samples_of(random_nodes(2, 5))
    tr = Transform(nodal(index_111, nodes))
    rng = np.random.default_rng(7)
    z = rng.standard_normal(6)
    z[-1] = 0.0  # pin the gauge: ALR coordinates are z[1:-1]
    w = np.exp(z[1:]) / np.exp(z[1:]).sum()

    def alr(u):
        full = np.concatenate([u, [0.0]])
        return np.exp(full)[:-1] / np.exp(full).sum()

    h, n = 1e-6, 4
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        J[:, i] = (alr(z[1:-1] + e) - alr(z[1:-1] - e)) / (2 * h)
    expected = z[0] * 0.0 + np.linalg.slogdet(J)[1]
    assert_allclose(expected, np.sum(np.log(w)), rtol=1e-6)
    assert_allclose(float(tr.log_det(jnp.asarray(z))), np.sum(np.log(w)), rtol=1e-12)
    # gauge invariance and extreme logits +/- 40 stay finite
    shifted = z.copy()
    shifted[1:] += 3.7
    assert_allclose(float(tr.log_det(jnp.asarray(shifted))), float(tr.log_det(z)))
    extreme = jnp.asarray([0.0, 40.0, -40.0, 40.0, -40.0, 0.0])
    assert np.isfinite(float(tr.log_det(extreme)))
    assert np.all(np.isfinite(jax.grad(tr.log_det)(extreme)))


def test_transform_value_errors_are_runtime(index_111):
    pm = gaussian_map(index_111)
    tr = Transform(pm, bounds={"hyper:gaussian_depth:mean": (-2.0, 3.0)})
    theta = tr.inverse(random_z(tr, 0))
    bad_sigma = Parameters(
        log_amplitude=theta.log_amplitude,
        tables=theta.tables,
        hyper=(jnp.array([0.5, -0.1]),),
    )
    with pytest.raises(Exception, match="> 0"):
        tr.forward(bad_sigma)
    outside = Parameters(
        log_amplitude=theta.log_amplitude,
        tables=theta.tables,
        hyper=(jnp.array([5.0, 0.4]),),
    )
    with pytest.raises(Exception, match="inside"):
        tr.forward(outside)


# -- LogDensity ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_111():
    index = MomentIndex.build(Truncation(1, 1, 1))
    basis, C = polynomial_basis(index, n_ch=8, seed=11)
    return index, basis, C


def chi2_oracle(density, z):
    """NumPy chi-square: ``||L^-1 (d - R A C m)||^2`` with ``m`` from the map."""
    theta = density.transform.inverse(jnp.asarray(z))
    m = np.asarray(density.parameter_map(theta, density.basis.reference).to_vector())
    A = (
        np.exp(float(theta.log_amplitude))
        if density.amplitude is None
        else float(density.amplitude)
    )
    data = density.data
    C = np.asarray(density.basis.response_matrix())
    model = A * C @ m
    if data.response is not None:
        model = np.asarray(data.response) @ model
    kept = list(data.kept_rows())
    d = np.asarray(data.data_vector())[kept]
    noise = np.asarray(data.noise)
    cov = np.diag(noise[kept]) if noise.ndim == 1 else noise[np.ix_(kept, kept)]
    r = d - model[kept]
    return float(r @ np.linalg.solve(cov, r))


@pytest.mark.parametrize("name", ["isotropic_pitch", "gaussian", "no_assumption"])
def test_logdensity_matches_numpy_chi2(setup_111, name):
    index, basis, C = setup_111
    pm = {
        "isotropic_pitch": isotropic_pitch,
        "gaussian": gaussian_map,
        "no_assumption": no_assumption,
    }[name](index)
    truth = Transform(pm).inverse(random_z(Transform(pm), 1))
    m_true = pm(truth, basis.reference).to_vector()
    data = make_data(basis, m_true, float(jnp.exp(truth.log_amplitude)), 8)
    density = LogDensity(basis, data, pm)
    assert density.n_params == pm.n_free() + 1
    assert_allclose(np.asarray(density.design), C / 0.1, atol=1e-12)
    for seed in (0, 4):
        z = random_z(density.transform, seed)
        assert_allclose(float(density.chi2(z)), chi2_oracle(density, z), rtol=1e-10)
        expected = -0.5 * chi2_oracle(density, z) + float(density.transform.log_det(z))
        assert_allclose(float(density(z)), expected, rtol=1e-10)
    # the truth has zero residual
    z_true = density.transform.forward(truth)
    assert float(density.chi2(z_true)) < 1e-18


def test_logdensity_mask_response_and_covariance(setup_111):
    index, basis, _ = setup_111
    pm = isotropic_pitch(index)
    tr = Transform(pm)
    truth = tr.inverse(random_z(tr, 2))
    m_true = pm(truth, basis.reference).to_vector()
    amp = float(jnp.exp(truth.log_amplitude))
    mask = np.ones((8, 4), dtype=bool)
    mask[2, 1] = mask[5, :] = False
    rng = np.random.default_rng(3)
    response = rng.standard_normal((20, 32))
    cases = [
        make_data(basis, m_true, amp, 8, mask=mask),
        make_data(basis, m_true, amp, 8, response=response),
        make_data(basis, m_true, amp, 8, noise="cov"),
        make_data(basis, m_true, amp, 8, noise="cov", mask=mask),
    ]
    for data in cases:
        density = LogDensity(basis, data, pm)
        assert density.design.shape == (data.n_kept(), index.n_real)
        z = random_z(tr, 9)
        assert_allclose(float(density.chi2(z)), chi2_oracle(density, z), rtol=1e-9)
        assert float(density.chi2(tr.forward(truth))) < 1e-16


def test_logdensity_gradient_matches_finite_differences(setup_111):
    index, basis, _ = setup_111
    pm = gaussian_map(index)
    tr = Transform(pm, bounds={"hyper:gaussian_depth:mean": (-3.0, 3.0)})
    truth = tr.inverse(random_z(tr, 1))
    data = make_data(basis, pm(truth, basis.reference).to_vector(), 2.0, 8)
    # data need not match the truth here; the gradient check is generic

    def prior(theta):
        return -0.5 * jnp.sum(jnp.abs(theta.tables[0]) ** 2) - theta.log_amplitude**2

    density = LogDensity(basis, data, pm, prior, tr)
    z = random_z(tr, 6) * 0.3
    grad = jax.jit(jax.grad(density))(z)
    fd = fd_gradient(lambda v: float(density(jnp.asarray(v))), np.asarray(z), h=1e-3)
    assert_allclose(np.asarray(grad), fd, rtol=1e-6, atol=1e-6)
    assert np.all(np.isfinite(jax.hessian(density)(z)))
    # prior and log-determinant enter additively
    flat = LogDensity(basis, data, pm, None, tr)
    assert_allclose(
        float(density(z)) - float(flat(z)), float(prior(tr.inverse(z))), rtol=1e-9
    )


def test_logdensity_vmap_jit_and_fixed_amplitude(setup_111):
    index, basis, _ = setup_111
    pm = isotropic_pitch(index)
    tr = Transform(pm)
    truth = tr.inverse(random_z(tr, 8))
    amp = float(jnp.exp(truth.log_amplitude))
    data = make_data(basis, pm(truth, basis.reference).to_vector(), amp, 8)
    density = LogDensity(basis, data, pm)
    batch = jnp.stack([random_z(tr, s) for s in range(4)])
    values = jax.jit(jax.vmap(density))(batch)
    assert_allclose(values, [float(density(z)) for z in batch], rtol=1e-12)
    fixed_tr = Transform(pm, amplitude=False)
    fixed = LogDensity(basis, data, pm, transform=fixed_tr, amplitude=amp)
    z_free = tr.forward(truth)
    assert float(fixed.chi2(z_free[1:])) < 1e-16
    assert_allclose(
        float(fixed.chi2(z_free[1:])), float(density.chi2(z_free)), atol=1e-16
    )
    assert fixed.n_params == pm.n_free()
    residual = density.whitened_residual(z_free)
    assert residual.shape == (32,)


def test_logdensity_rejects_bad_configuration(setup_111):
    index, basis, _ = setup_111
    pm = isotropic_pitch(index)
    data = make_data(
        basis,
        pm(Transform(pm).inverse(random_z(Transform(pm))), basis.reference).to_vector(),
        1.0,
        8,
    )
    with pytest.raises(ValueError):
        LogDensity(basis, data, object())
    with pytest.raises(ValueError):
        LogDensity(basis, data, pm, transform=Transform(no_assumption(index)))
    with pytest.raises(ValueError):
        LogDensity(basis, data, pm, prior=3.0)
    with pytest.raises(ValueError):
        LogDensity(basis, data, pm, transform=Transform(pm), amplitude=2.0)
    with pytest.raises(ValueError):
        LogDensity(basis, data, pm, transform=Transform(pm, amplitude=False))
    assert LogDensity(basis, data, pm, amplitude=2.0).n_params == pm.n_free()
    other = MomentIndex.build(Truncation(1, 1, 2))
    with pytest.raises(ValueError):
        LogDensity(basis, data, isotropic_pitch(other))
    wrong_data = StokesData(np.zeros((5, 4)), np.ones(20))
    with pytest.raises(ValueError):
        LogDensity(basis, wrong_data, pm)

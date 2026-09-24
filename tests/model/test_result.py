"""Tests of ``synchro.model.fit.result`` (``FitResult`` and the builders that
wrap ``fit_bfgs``/``fit_nodal`` outcomes).

Oracles: NumPy Gauss-Newton Fisher ``J^T J`` from central-difference
Jacobians of the whitened model in the fit coordinates, ``numpy.linalg.pinv``,
``scipy.special.eval_legendre`` node features (through
``_nonlinear_oracles.node_features``), and ``json`` round trips.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from _nonlinear_oracles import node_features, polynomial_basis, random_nodes, samples_of
from synchro.model.assumptions import (
    Parameters,
    gaussian_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
)
from synchro.model.errors import ErrorTerm, Provenance
from synchro.model.fit.linear import fit_linear
from synchro.model.fit.nonlinear import LogDensity, Transform, fit_bfgs, fit_nodal
from synchro.model.fit.observation import StokesData
from synchro.model.fit.result import MAX_FISHER_PARAMS, FitResult
from synchro.model.index import MomentIndex, Truncation
from synchro.model.predict import Prediction

# -- helpers ---------------------------------------------------------------------


def setup(t=(1, 1, 1), n_ch=8, seed=11):
    index = MomentIndex.build(Truncation(*t))
    basis, C = polynomial_basis(index, n_ch=n_ch, seed=seed)
    return index, basis, C


def data_of(C, m, amplitude, *, noise=1e-2, seed=None):
    d = amplitude * C @ np.asarray(m)
    if seed is not None:
        d = d + np.sqrt(noise) * np.random.default_rng(seed).standard_normal(d.size)
    return StokesData(d.reshape(-1, 4), np.full(d.size, noise)), d


def fd_jacobian(f, x, h=1e-5):
    x = np.asarray(x, dtype=float)
    cols = []
    for a in range(x.size):
        e = np.zeros_like(x)
        e[a] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.stack(cols, axis=1)


def round_trip(d):
    text = json.dumps(d)
    assert json.loads(text) == d
    return text


# -- fit_bfgs -> FitResult ----------------------------------------------------------


@pytest.mark.parametrize("name", ["isotropic_pitch", "gaussian_fit"])
def test_fit_bfgs_returns_fit_result_with_gauss_newton_fisher(name):
    if name == "gaussian_fit":
        index, basis, C = setup((1, 1, 2), n_ch=16, seed=22)
        pm = gaussian_screen(index, 0.8, 0.6, fit_hyper=True)

        def prior(theta):
            return -jnp.log(theta.hyper[0][1])

    else:
        index, basis, C = setup((1, 1, 1), n_ch=10, seed=21)
        pm = isotropic_pitch(index)
        prior = None
    tr = Transform(pm)
    rng = np.random.default_rng(1)
    z_true = jnp.asarray(0.5 * rng.standard_normal(tr.n_params))
    truth = tr.inverse(z_true)
    m_true = pm(truth, basis.reference).to_vector()
    data, d = data_of(C, m_true, float(jnp.exp(z_true[0])))
    density = LogDensity(basis, data, pm, prior)
    z0 = z_true + 0.2 * jnp.asarray(rng.standard_normal(tr.n_params))
    result = fit_bfgs(density, z0, maxiter=3000, tol=1e-9)
    assert isinstance(result, FitResult)
    assert result.method == "bfgs" and result.converged and result.n_iter > 0
    assert result.labels == tr.labels and result.z.shape == (tr.n_params,)
    assert_allclose(np.asarray(result.z), np.asarray(z_true), atol=1e-8)
    assert_allclose(float(result.fun), -float(density(result.z)), rtol=1e-9, atol=1e-12)
    assert float(result.chi2) < 1e-12 and result.weights is None
    assert_allclose(float(result.amplitude), float(jnp.exp(z_true[0])), rtol=1e-8)
    assert_allclose(
        np.asarray(result.moments.to_vector()), np.asarray(m_true), atol=1e-8
    )
    # Gauss-Newton Fisher J^T J in z against central differences of the whitened model
    z = np.asarray(result.z)

    def model(v):
        return -np.asarray(density.whitened_residual(jnp.asarray(v)))

    J = fd_jacobian(model, z)
    F = J.T @ J
    assert_allclose(
        np.asarray(result.fisher), F, rtol=1e-5, atol=1e-9 * np.abs(F).max()
    )
    n = tr.n_params
    if name == "gaussian_fit":
        # sigma enters through b = 2 at N = 2: full rank
        assert result.rank == n and result.covariance is not None
        cov = np.linalg.inv(F)
        assert_allclose(
            np.asarray(result.covariance), cov, rtol=1e-4, atol=1e-9 * np.abs(cov).max()
        )
    else:
        assert result.rank == n and result.covariance is not None
    assert result.dof == data.n_kept() - result.rank
    assert result.identifiability is not None
    assert result.identifiability.labels == ("amplitude",) + pm.labels()
    assert result.feasibility is not None
    assert isinstance(result.prediction, Prediction)
    assert_allclose(
        np.asarray(result.prediction.stokes).ravel(), d, atol=1e-7 * abs(d).max()
    )
    # NEW-4: no data.discrepancy is declared, so the moment bias is not bounded
    # and statistical_input is unbounded, as for fit_linear (the one-sigma noise
    # part is quoted in the note); the linearised bias is tested in
    # test_nonlinear_bias_regressions.py
    stat = result.prediction.budget.statistical_input
    assert stat.kind == "unbounded" and "bias" in stat.note
    assert result.bias_bound.kind == "unbounded"
    # NEW-1 (reverts R15): the record is theta-independent (NaN placeholders for
    # fitted hyper); the provenance notes carry the fitted values
    assert pm.record() in result.provenance.assumptions
    fitted_notes = [n for n in result.provenance.notes if "fitted hyper" in n]
    if name == "gaussian_fit":
        mean, sigma = np.asarray(result.theta.hyper[0], dtype=float)
        assert fitted_notes == [
            "fitted hyper-parameters of 'gaussian_screen' (the AssumptionRecord "
            "holds NaN placeholders for them): "
            f"hyper:gaussian_depth:mean={mean:.10g}, "
            f"hyper:gaussian_depth:sigma={sigma:.10g}"
        ]
        assert result.provenance.assumptions[-1].closure_kind.endswith("(fitted)")
        # the NaN placeholders become null in the strict-JSON summary
        assert result.to_dict()["provenance"]["assumptions"][-1]["hyper"] == [
            None,
            None,
        ]
    else:
        assert fitted_notes == []
    assert any("bfgs" in note for note in result.provenance.notes)
    round_trip(result.to_dict())


def test_fit_bfgs_fixed_amplitude_result():
    index, basis, C = setup((1, 1, 1), n_ch=8, seed=4)
    pm = isotropic_pitch(index)
    tr = Transform(pm, amplitude=False)
    rng = np.random.default_rng(6)
    z_true = jnp.asarray(0.3 * rng.standard_normal(tr.n_params))
    m_true = pm(tr.inverse(z_true), basis.reference).to_vector()
    A = 2.0
    data, _ = data_of(C, m_true, A)
    density = LogDensity(basis, data, pm, transform=tr, amplitude=A)
    result = fit_bfgs(density, z_true + 0.1, maxiter=2000)
    assert result.converged and float(result.amplitude) == A
    assert result.labels == pm.labels() and result.theta.log_amplitude is not None
    assert_allclose(float(result.theta.log_amplitude), np.log(A))
    assert_allclose(np.asarray(result.z), np.asarray(z_true), atol=1e-7)
    assert result.identifiability.labels == pm.labels()
    # the linear route with the same fixed amplitude agrees
    linear = fit_linear(basis, data, pm, amplitude=A)
    assert_allclose(
        np.asarray(pm.flatten(linear.theta)), np.asarray(result.z), atol=1e-7
    )
    assert_allclose(np.asarray(linear.fisher), np.asarray(result.fisher), rtol=1e-6)


def test_fit_bfgs_unconverged_result_is_still_wrapped():
    index, basis, C = setup((1, 1, 1), n_ch=6, seed=2)
    pm = no_assumption(index)
    tr = Transform(pm)
    m = pm(tr.inverse(jnp.zeros(tr.n_params)), basis.reference).to_vector()
    data, _ = data_of(C, m, 1.0)
    density = LogDensity(basis, data, pm)
    z0 = jnp.asarray(np.random.default_rng(3).standard_normal(tr.n_params))
    result = fit_bfgs(density, z0, maxiter=1)
    assert isinstance(result, FitResult) and not result.converged
    assert result.n_iter <= 1 and np.all(np.isfinite(np.asarray(result.z)))
    assert result.rank <= data.n_kept() and result.covariance is None
    round_trip(result.to_dict())


# -- fit_nodal -> FitResult ----------------------------------------------------------


def nodal_setup(n_nodes=5, n_ch=6, seed=7):
    index, basis, C = setup((1, 1, 1), n_ch=n_ch, seed=seed)
    pop = random_nodes(seed, n_nodes)
    rng = np.random.default_rng(seed + 1)
    w_true = rng.uniform(0.2, 1.0, n_nodes)
    w_true /= w_true.sum()
    X = node_features(pop, index)
    A_true = 2.5
    d = A_true * C @ (X.T @ w_true)
    data = StokesData(d.reshape(-1, 4), np.full(d.size, 1e-4))
    return index, basis, C, pop, X, w_true, A_true, data, d


def test_fit_nodal_returns_fit_result_with_closed_form_fisher():
    index, basis, C, pop, X, w_true, A_true, data, d = nodal_setup()
    nodes = samples_of(pop)
    term = ErrorTerm(jnp.full((6, 4), 1e-3), "bound", "extra eq: nodal bound test")
    result = fit_nodal(basis, data, nodes, iters=5000, step=0.1, discretisation=term)
    assert isinstance(result, FitResult) and result.method == "nodal"
    assert result.converged and result.discretisation is term
    w = np.asarray(result.weights)
    assert_allclose(w, w_true, atol=1e-7)
    assert_allclose(float(result.amplitude), A_true, rtol=1e-7)
    assert result.labels == ("log_amplitude",) + nodal(index, nodes).labels()
    assert_allclose(float(result.chi2), 2 * float(result.fun), rtol=1e-12)
    # Fisher in (log A, logits): J^T J of the whitened model, central differences
    S = w.size
    Linv = 1 / np.sqrt(np.asarray(data.noise))

    def model(v):
        wv = np.exp(v[1:]) / np.exp(v[1:]).sum()
        return np.exp(v[0]) * (C @ (X.T @ wv)) * Linv

    z = np.asarray(result.z)
    J = fd_jacobian(model, z, h=1e-6)
    assert_allclose(
        np.asarray(result.fisher), J.T @ J, rtol=1e-5, atol=1e-6 * (J.T @ J).max()
    )
    # the softmax gauge is a null direction: rank S, covariance None
    assert result.rank == S and result.covariance is None
    assert result.dof == data.n_kept() - S
    assert result.identifiability is None
    assert result.feasibility is not None and result.feasibility.failed() == ()
    pred = result.prediction
    assert isinstance(pred, Prediction)
    assert_allclose(np.asarray(pred.stokes).ravel(), d, rtol=1e-6)
    assumption = dict(pred.budget.assumption)
    assert assumption["nodal"].kind == "bound"
    assert_allclose(np.asarray(assumption["nodal"].value), 1e-3)
    assert "nodal bound" in assumption["nodal"].note
    round_trip(result.to_dict())


def test_fit_nodal_without_discretisation_leaves_assumption_unbounded():
    index, basis, C, pop, X, w_true, A_true, data, d = nodal_setup(seed=9)
    result = fit_nodal(basis, data, samples_of(pop, w_true), iters=50)
    assert result.discretisation is None
    assert dict(result.prediction.budget.assumption)["nodal"].kind == "unbounded"
    assert result.prediction.budget.total().kind == "unbounded"


def test_fit_nodal_many_nodes_skips_fisher():
    n = MAX_FISHER_PARAMS  # S + 1 = MAX + 1 > MAX: fisher not formed
    index, basis, C, pop, X, w_true, A_true, data, d = nodal_setup(
        n_nodes=n, n_ch=3, seed=13
    )
    result = fit_nodal(basis, data, samples_of(pop), iters=5, step=0.1)
    assert result.fisher is None and result.covariance is None
    assert result.rank == 0 and result.dof == data.n_kept() - (n + 1)
    assert any("fisher" in note for note in result.provenance.notes)
    assert result.weights.shape == (n,)
    below = nodal_setup(n_nodes=n - 1, n_ch=3, seed=14)
    result2 = fit_nodal(below[1], below[7], samples_of(below[3]), iters=5, step=0.1)
    assert result2.fisher is not None and result2.fisher.shape == (n, n)


# -- FitResult container ------------------------------------------------------------


def test_fit_result_to_dict_round_trip_complex_tables():
    index, basis, C = setup((1, 1, 1), n_ch=8, seed=15)
    pm = no_assumption(index)
    rng = np.random.default_rng(5)
    theta = pm.unflatten(jnp.asarray(0.3 * rng.standard_normal(pm.n_free())))
    m = pm(theta, basis.reference).to_vector()
    data, _ = data_of(C, m, 1.5, seed=1)
    result = fit_linear(basis, data, pm)
    assert jnp.iscomplexobj(result.theta.tables[0])
    d = result.to_dict()
    text = round_trip(d)
    assert len(text) > 100
    assert d["label"] and d["method"] == "linear" and d["labels"] == list(result.labels)
    assert d["theta"]["tables"][0]["real"] and d["theta"]["tables"][0]["imag"]
    assert set(d["parameters"]) == set(pm.labels())
    assert_allclose(
        [d["parameters"][k] for k in pm.labels()], np.asarray(pm.flatten(result.theta))
    )
    assert d["rank"] == result.rank == pm.n_free() + 1 and d["dof"] == result.dof == 4
    assert len(d["covariance"]) == 28 and d["bias_bound"]["kind"] == "unbounded"
    assert d["identifiability"]["null_dim"] == result.identifiability.null_dim
    assert d["feasibility"]["checks"] and d["prediction"]["stokes"]
    assert d["provenance"]["assumptions"][-1]["name"] == "no_assumption"
    assert d["fisher"] is not None and len(d["fisher"]) == pm.n_free() + 1


def test_fit_result_validation_and_immutability():
    index, basis, C = setup((1, 1, 1), n_ch=8, seed=15)
    pm = isotropic_pitch(index)
    theta = pm.unflatten(jnp.zeros(pm.n_free()))
    m = pm(theta, basis.reference).to_vector()
    data, _ = data_of(C, m, 1.0)
    result = fit_linear(basis, data, pm, diagnostics=False)
    with pytest.raises(Exception):
        result.rank = 3  # equinox modules are frozen
    kwargs = dict(
        theta=result.theta,
        moments=result.moments,
        amplitude=result.amplitude,
        fisher=result.fisher,
        covariance=result.covariance,
        rank=result.rank,
        chi2=result.chi2,
        dof=result.dof,
        converged=True,
        n_iter=0,
        bias_bound=None,
        identifiability=None,
        feasibility=None,
        prediction=None,
        provenance=result.provenance,
    )
    FitResult(**kwargs)
    for bad in ({"rank": -1}, {"rank": 1.5}, {"converged": 1}, {"n_iter": -2}):
        with pytest.raises(ValueError):
            FitResult(**{**kwargs, **bad})
    with pytest.raises(ValueError):
        FitResult(**{**kwargs, "provenance": ()})
    with pytest.raises(ValueError):
        FitResult(**{**kwargs, "bias_bound": jnp.zeros(3)})
    with pytest.raises(ValueError):
        FitResult(**{**kwargs, "fisher": jnp.zeros((2, 3))})
    assert isinstance(result.provenance, Provenance)
    assert isinstance(result.theta, Parameters)
    # a plain jax.jit over the array leaves works through equinox partitioning
    total = jax.jit(lambda r: r.chi2 + jnp.sum(r.fisher))(result)
    assert np.isfinite(float(total))

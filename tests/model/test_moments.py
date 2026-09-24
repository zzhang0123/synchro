"""Exact discrete-population oracles for ``synchro.model.moments``.

Oracles are NumPy/SciPy (``eval_legendre``, explicit weighted sums) written
independently of the JAX path. The manuscript toy population of
``eq: channel toy population`` is rebuilt from ``validation/full_response.py``
(latent 16 x 16 Gauss-Legendre nodes, 64 x 64 angular nodes).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

from synchro.model.assumptions import field_independent
from synchro.model.index import MomentIndex, Truncation
from synchro.model.moments import (
    UNIT_MOMENT_TOL,
    JointMoments,
    PopulationSamples,
    Reference,
    Support,
)

E_ESU, M_E, C_CGS = 4.803204712570263e-10, 9.1093837139e-28, 2.99792458e10
C_SI_M = 2.99792458e8


# -- oracles -----------------------------------------------------------------


def oracle_rows(pop, rows, h, reference):
    """``M^{(h)}_{lk;rsb}`` of a discrete population by explicit sums."""
    gamma0, B0, depth_ref, (sg, sB, sd) = reference
    w = np.asarray(pop["w"], float)
    w = w / w.sum()
    zg = (np.asarray(pop["gamma"]) - gamma0) / sg
    zB = (np.asarray(pop["B"]) - B0) / sB
    zd = (np.asarray(pop["depth"]) - depth_ref) / sd
    out = []
    for l, k, r, s, b in rows:
        f = (
            eval_legendre(l, pop["mu"])
            * eval_legendre(k, pop["eta"])
            * zg**r
            * zB**s
            * zd**b
        )
        if h == 2:
            f = f * np.exp(2j * np.asarray(pop["phi"]))
        out.append(np.sum(w * f))
    return np.array(out)


def seven_atoms(seed=7):
    """Seven correlated atoms: every variable depends on a common latent."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, 7)
    return dict(
        gamma=5.0 + 1.5 * t + 0.3 * rng.standard_normal(7),
        B=2.0 + 0.8 * t**2 + 0.1 * rng.standard_normal(7),
        mu=np.clip(0.6 * t + 0.2 * rng.standard_normal(7), -0.95, 0.95),
        eta=np.clip(-0.5 * t + 0.3 * rng.standard_normal(7), -0.95, 0.95),
        phi=0.3 + 1.7 * t + 0.4 * rng.standard_normal(7),
        depth=1.5 * t + 0.2 * rng.standard_normal(7),
        w=rng.uniform(0.2, 3.0, 7),
    )


REF7 = (5.2, 2.1, 0.1, (1.3, 0.7, 0.9))


def samples_of(pop, weights=True):
    return PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=pop["w"] if weights else None,
    )


def reference_of(ref):
    gamma0, B0, depth_ref, scales = ref
    return Reference(gamma0, B0, depth_ref, scales=scales)


# -- Reference / Support / PopulationSamples --------------------------------


def test_reference_z_and_describe():
    ref = Reference(10.0, 2.0, 0.5, scales=(2.0, 4.0, 0.25))
    z = ref.z(jnp.array([12.0, 8.0]), jnp.array([6.0, 2.0]), jnp.array([1.0, 0.5]))
    assert z.shape == (2, 3)
    assert_allclose(z, [[1.0, 1.0, 2.0], [-1.0, 0.0, 0.0]], atol=1e-15)
    assert ref.z(10.0, 2.0, 0.5).shape == (3,)
    described = dict(ref.describe())
    assert described["gamma0"] == 10.0 and described["scales"] == (2.0, 4.0, 0.25)
    default = Reference(3.0, 1.0)
    assert_allclose(default.z(4.0, 3.0, 2.0), [1.0, 2.0, 2.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gamma0=1.0, B0=1.0),
        dict(gamma0=0.5, B0=1.0),
        dict(gamma0=2.0, B0=0.0),
        dict(gamma0=2.0, B0=-1.0),
        dict(gamma0=np.nan, B0=1.0),
        dict(gamma0=2.0, B0=1.0, scales=(1.0, 0.0, 1.0)),
        dict(gamma0=2.0, B0=1.0, scales=(1.0, 1.0, -1.0)),
        dict(gamma0=2.0, B0=1.0, depth_ref=np.inf),
    ],
)
def test_reference_rejects_invalid_values(kwargs):
    with pytest.raises(Exception):
        Reference(**kwargs)


def test_reference_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Reference(jnp.array([2.0, 3.0]), 1.0)
    with pytest.raises(ValueError):
        Reference(2.0, 1.0, scales=(1.0, 1.0))
    with pytest.raises(ValueError):
        Reference(2.0 + 0j, 1.0)


def test_reference_is_traceable():
    def f(g0):
        return Reference(g0, 1.0, scales=(2.0, 1.0, 1.0)).z(4.0, 1.0, 0.0)[0]

    assert_allclose(jax.jit(f)(2.0), 1.0)
    assert_allclose(jax.grad(f)(2.0), -0.5)


@pytest.mark.parametrize("truncated", [None, False, True])
def test_support_truncated_tri_state(truncated):
    support = Support((2.0, 30.0), (0.5, 5.0), (-1.0, 3.0), truncated=truncated)
    assert support.truncated is truncated
    expected = {None: "unbounded", False: "declared_zero", True: "required_input"}
    assert support.tail_kind() == expected[truncated]
    assert support.gamma_max() == 30.0 and support.B_min() == 0.5
    described = dict(support.describe())
    assert described["truncated"] is truncated
    assert described["gamma"] == (2.0, 30.0) and described["depth"] == (-1.0, 3.0)


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (((2.0, 30.0), (0.5, 5.0), (-1.0, 3.0)), dict(truncated=1)),
        (((2.0, 30.0), (0.5, 5.0), (-1.0, 3.0)), dict(truncated="yes")),
        (((2.0, 30.0, 40.0), (0.5, 5.0), (-1.0, 3.0)), {}),
        (((30.0, 2.0), (0.5, 5.0), (-1.0, 3.0)), {}),
        (((1.0, 30.0), (0.5, 5.0), (-1.0, 3.0)), {}),
        (((2.0, 30.0), (0.0, 5.0), (-1.0, 3.0)), {}),
        (((2.0, 30.0), (0.5, 5.0), (3.0, -1.0)), {}),
        (((2.0, np.inf), (0.5, 5.0), (-1.0, 3.0)), {}),
    ],
)
def test_support_rejects_invalid(args, kwargs):
    with pytest.raises(Exception):
        Support(*args, **kwargs)


def test_support_allows_degenerate_interval():
    support = Support((5.0, 5.0), (1.0, 1.0), (0.0, 0.0))
    assert support.gamma_max() == 5.0 and support.B_min() == 1.0


def test_population_samples_contract():
    pop = seven_atoms()
    samples = samples_of(pop)
    assert samples.size == 7
    assert_allclose(samples.normalised_weights(), pop["w"] / pop["w"].sum())
    uniform = samples_of(pop, weights=False)
    assert_allclose(uniform.normalised_weights(), np.full(7, 1 / 7))
    with pytest.raises(ValueError):
        PopulationSamples(
            pop["gamma"][:3], *(pop[k] for k in "B mu eta phi depth".split())
        )
    with pytest.raises(ValueError):
        PopulationSamples(
            *(pop[k] for k in "gamma B mu eta phi depth".split()), weights=pop["w"][:2]
        )
    with pytest.raises(ValueError):
        PopulationSamples(
            *(pop[k].reshape(1, 7) for k in "gamma B mu eta phi depth".split())
        )
    for bad in [
        dict(mu=1.5),
        dict(eta=-1.5),
        dict(gamma=0.5),
        dict(B=-1.0),
        dict(depth=np.nan),
    ]:
        broken = {**pop, **{k: np.full(7, v) for k, v in bad.items()}}
        with pytest.raises(Exception):
            samples_of(broken)
    with pytest.raises(Exception):
        samples_of({**pop, "w": -pop["w"]})
    with pytest.raises(Exception):
        samples_of({**pop, "w": np.zeros(7)})


def test_population_product_outer_measure():
    samples = PopulationSamples.product(
        gamma=(np.array([4.0, 6.0]), np.array([1.0, 3.0])),
        B=np.array([1.0, 2.0, 3.0]),
        mu=(np.array([0.5]), np.array([2.0])),
        eta=np.array([-0.5, 0.5]),
        phi=np.array([0.1]),
        depth=(np.array([0.0, 1.0]), np.array([0.25, 0.75])),
    )
    assert samples.size == 2 * 3 * 1 * 2 * 1 * 2
    w = np.asarray(samples.normalised_weights())
    expected = np.einsum(
        "a,b,c,d,e,f->abcdef",
        [0.25, 0.75],
        np.full(3, 1 / 3),
        [1.0],
        [0.5, 0.5],
        [1.0],
        [0.25, 0.75],
    ).ravel()
    assert_allclose(w, expected, atol=1e-16)
    grid = np.asarray(samples.gamma).reshape(2, 3, 1, 2, 1, 2)
    assert np.all(grid[0] == 4.0) and np.all(grid[1] == 6.0)
    assert np.all(np.asarray(samples.depth).reshape(2, 3, 1, 2, 1, 2)[..., 1] == 1.0)
    with pytest.raises(ValueError):
        PopulationSamples.product(gamma=np.array([4.0]))
    with pytest.raises(ValueError):
        PopulationSamples.product(
            gamma=np.array([4.0]),
            B=np.array([1.0]),
            mu=np.array([0.0]),
            eta=np.array([0.0]),
            phi=np.array([0.0]),
            depth=np.array([0.0]),
            extra=np.array([1.0]),
        )


# -- JointMoments.from_samples -------------------------------------------------


@pytest.mark.parametrize(
    "truncation",
    [
        Truncation(2, 2, 2),
        Truncation(1, 1, 1),
        Truncation(0, 0, 2),
        Truncation(2, 2, 2, depth_degree=3),
        Truncation(2, 1, 0),
    ],
)
def test_from_samples_matches_oracle_on_all_rows(truncation):
    index = MomentIndex.build(truncation)
    pop = seven_atoms()
    moments = JointMoments.from_samples(samples_of(pop), index, reference_of(REF7))
    assert moments.m0.shape == (index.n0,) and moments.m2.shape == (index.n2,)
    assert jnp.iscomplexobj(moments.m2) and not jnp.iscomplexobj(moments.m0)
    assert moments.m0_ext.shape == (len(index.h0_ext),)
    assert_allclose(moments.m0[0], 1.0, atol=1e-15)
    assert_allclose(
        moments.m0, oracle_rows(pop, index.h0, 0, REF7), atol=1e-14, rtol=1e-14
    )
    assert_allclose(
        moments.m0_ext, oracle_rows(pop, index.h0_ext, 0, REF7), atol=1e-14, rtol=1e-14
    )
    assert_allclose(
        moments.m2, oracle_rows(pop, index.h2, 2, REF7), atol=1e-14, rtol=1e-14
    )
    vector = moments.to_vector()
    assert vector.shape == (index.n_real,)
    assert_allclose(vector[: index.n0], moments.m0)
    assert_allclose(vector[index.n0 : index.n0 + index.n2], moments.m2.real)
    assert_allclose(vector[index.n0 + index.n2 :], moments.m2.imag)


def test_from_samples_without_V_and_without_parity(index_222):
    pop = seven_atoms(3)
    no_V = MomentIndex.build(Truncation(2, 2, 2), components=("I", "Q"))
    full = MomentIndex.build(Truncation(2, 2, 2), parity=False)
    for index in (no_V, full):
        moments = JointMoments.from_samples(samples_of(pop), index, reference_of(REF7))
        assert_allclose(moments.m0, oracle_rows(pop, index.h0, 0, REF7), atol=1e-14)
        assert_allclose(moments.m2, oracle_rows(pop, index.h2, 2, REF7), atol=1e-14)
    assert no_V.n0 < index_222.n0 and full.n2 > index_222.n2


def test_get_and_from_vector_roundtrip(index_222):
    pop = seven_atoms(11)
    moments = JointMoments.from_samples(samples_of(pop), index_222, reference_of(REF7))
    rebuilt = JointMoments.from_vector(
        index_222, moments.to_vector(), moments.reference
    )
    assert_allclose(rebuilt.m0, moments.m0, atol=1e-16)
    assert_allclose(rebuilt.m2, moments.m2, atol=1e-16)
    assert rebuilt.m0_ext is None
    for row in [(1, 2, 1, 0, 0), (2, 2, 0, 2, 0), (0, 0, 0, 0, 0)]:
        assert_allclose(
            moments.get(0, *row), oracle_rows(pop, [row], 0, REF7)[0], atol=1e-14
        )
    for row in [(0, 2, 0, 1, 1), (1, 1, 1, 0, 1), (2, 0, 0, 0, 2)]:
        assert_allclose(
            moments.get(2, *row), oracle_rows(pop, [row], 2, REF7)[0], atol=1e-14
        )
        assert_allclose(
            moments.get(0, *row), oracle_rows(pop, [row], 0, REF7)[0], atol=1e-14
        )
    assert moments.get(2, 0, 2, 0, 1, 1).dtype == moments.m2.dtype
    with pytest.raises(ValueError):
        rebuilt.get(0, 0, 2, 0, 1, 1)  # m0_ext absent
    with pytest.raises(ValueError):
        moments.get(0, 1, 2, 0, 0, 1)  # odd parity row not in h0_ext
    with pytest.raises(ValueError):
        moments.get(2, 3, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        moments.get(1, 0, 0, 0, 0, 0)
    assert len(moments.index.labels()) == index_222.n_real


def test_from_vector_rejects_unit_moment_and_shape(index_111):
    ref = Reference(4.0, 1.0)
    good = np.zeros(index_111.n_real)
    good[0] = 1.0
    JointMoments.from_vector(index_111, good, ref)
    bad = good.copy()
    bad[0] = 1.0 + 1e-3
    with pytest.raises(Exception):
        JointMoments.from_vector(index_111, bad, ref)
    with pytest.raises(Exception):
        eqx.filter_jit(JointMoments.from_vector)(
            index_111, jnp.asarray(bad), ref
        ).to_vector()
    assert_allclose(
        JointMoments.from_vector(index_111, bad, ref, tol=1e-2).to_vector(), bad
    )
    with pytest.raises(ValueError):
        JointMoments.from_vector(index_111, good[:-1], ref)
    with pytest.raises(ValueError):
        JointMoments.from_vector(index_111, good.astype(complex), ref)
    with pytest.raises(ValueError):
        JointMoments(
            index_111, jnp.zeros(index_111.n0 + 1), jnp.zeros(index_111.n2), ref
        )
    with pytest.raises(ValueError):
        JointMoments(
            index_111, jnp.zeros(index_111.n0), jnp.zeros(index_111.n2 - 1), ref
        )


def test_from_vector_default_tolerance_accepts_sample_roundoff(index_222):
    """Default ``tol = UNIT_MOMENT_TOL = 1e-12``: both sides of the threshold."""
    assert UNIT_MOMENT_TOL == 1e-12
    pop = seven_atoms(3)
    pop["w"] = np.random.default_rng(0).uniform(0.1, 7.0, 7) * 1e-3
    moments = JointMoments.from_samples(samples_of(pop), index_222, reference_of(REF7))
    m = np.asarray(moments.to_vector())
    rebuilt = JointMoments.from_vector(index_222, m, moments.reference)
    assert_allclose(rebuilt.to_vector(), m, atol=0)
    ref = moments.reference
    for offset in (1e-13, -1e-13, 0.9e-12):
        v = m.copy()
        v[0] = 1.0 + offset
        JointMoments.from_vector(index_222, v, ref)
    for offset in (1.1e-12, -1e-11, 1e-6):
        v = m.copy()
        v[0] = 1.0 + offset
        with pytest.raises(Exception, match="m\\[0\\] must equal 1"):
            JointMoments.from_vector(index_222, v, ref)
    v = m.copy()
    v[0] = 1.0 - 4e-16
    JointMoments.from_vector(index_222, v, ref)
    with pytest.raises(Exception, match="must equal 1"):
        JointMoments.from_vector(index_222, v, ref, tol=0.0)
    jitted = eqx.filter_jit(JointMoments.from_vector)(index_222, jnp.asarray(v), ref)
    assert_allclose(jitted.to_vector(), v, atol=0)


def test_to_raw_displacements_consistency(index_222):
    pop = seven_atoms(5)
    scaled = JointMoments.from_samples(samples_of(pop), index_222, reference_of(REF7))
    raw_ref = (REF7[0], REF7[1], REF7[2], (1.0, 1.0, 1.0))
    raw = JointMoments.from_samples(samples_of(pop), index_222, reference_of(raw_ref))
    converted = scaled.to_raw_displacements()
    assert_allclose(converted.m0, raw.m0, rtol=1e-13, atol=1e-15)
    assert_allclose(converted.m2, raw.m2, rtol=1e-13, atol=1e-15)
    assert_allclose(converted.m0_ext, raw.m0_ext, rtol=1e-13, atol=1e-15)
    assert dict(converted.reference.describe())["scales"] == (1.0, 1.0, 1.0)
    assert converted.index is index_222
    # A unit-scale reference is a fixed point.
    again = converted.to_raw_displacements()
    assert_allclose(again.to_vector(), converted.to_vector(), atol=1e-16)


def test_extreme_weights_and_single_sample(index_222):
    pop = seven_atoms(2)
    ref = reference_of(REF7)
    base = JointMoments.from_samples(samples_of(pop), index_222, ref).to_vector()
    huge = JointMoments.from_samples(
        samples_of({**pop, "w": pop["w"] * 1e300}), index_222, ref
    ).to_vector()
    assert_allclose(huge, base, atol=1e-14, rtol=1e-14)
    one = {k: v[:1] for k, v in pop.items()}
    single = JointMoments.from_samples(samples_of(one), index_222, ref)
    assert single.index is index_222
    assert_allclose(single.m0, oracle_rows(one, index_222.h0, 0, REF7), atol=1e-14)
    assert_allclose(single.m2, oracle_rows(one, index_222.h2, 2, REF7), atol=1e-14)
    assert_allclose(abs(single.get(2, 0, 0, 0, 0, 0)), 1.0, atol=1e-15)


def test_from_samples_jit_and_gradients(index_111):
    pop = seven_atoms(9)
    ref = reference_of(REF7)

    def vector(gamma, weights):
        samples = PopulationSamples(
            gamma,
            pop["B"],
            pop["mu"],
            pop["eta"],
            pop["phi"],
            pop["depth"],
            weights=weights,
        )
        return JointMoments.from_samples(samples, index_111, ref).to_vector()

    gamma, weights = jnp.asarray(pop["gamma"]), jnp.asarray(pop["w"])
    compiled = jax.jit(vector)
    assert_allclose(compiled(gamma, weights), vector(gamma, weights), atol=1e-15)
    probe = np.asarray(np.random.default_rng(0).standard_normal(index_111.n_real))

    def scalar(g, w):
        return jnp.dot(probe, vector(g, w))

    dg, dw = jax.jit(jax.grad(scalar, argnums=(0, 1)))(gamma, weights)
    assert np.all(np.isfinite(dg)) and np.all(np.isfinite(dw))
    step = 1e-5
    for n in range(7):
        for arg, grad_value in ((0, dg[n]), (1, dw[n])):
            plus = [np.asarray(gamma).copy(), np.asarray(weights).copy()]
            minus = [np.asarray(gamma).copy(), np.asarray(weights).copy()]
            plus[arg][n] += step
            minus[arg][n] -= step
            fd = (scalar(*plus) - scalar(*minus)) / (2 * step)
            assert_allclose(grad_value, fd, rtol=1e-6, atol=1e-8)


def test_vmap_over_batches_of_moment_vectors(index_111):
    ref = Reference(4.0, 1.0)
    batch = np.zeros((3, index_111.n_real))
    batch[:, 0] = 1.0
    batch[:, 1:] = np.random.default_rng(1).standard_normal((3, index_111.n_real - 1))

    def vec(m):
        return JointMoments.from_vector(index_111, m, ref).to_vector()

    assert_allclose(jax.vmap(vec)(jnp.asarray(batch)), batch)


def test_from_samples_rejects_bad_map_and_discrepancy_options(index_111):
    samples = samples_of(seven_atoms())
    ref = reference_of(REF7)
    with pytest.raises(ValueError, match="ParameterMap"):
        JointMoments.from_samples(samples, index_111, ref, parameter_map=object())
    with pytest.raises(ValueError, match="parameter_map"):
        JointMoments.from_samples(samples, index_111, ref, discrepancy="measured")
    with pytest.raises(ValueError):
        JointMoments.from_samples(samples, index_111, ref, discrepancy="bogus")


def test_from_samples_with_parameter_map_returns_factorised_moments(index_222):
    """Implemented phase-2 path: ``parameter_map`` projects the joint tensor and
    ``discrepancy="measured"`` stores ``|m_joint - m_fac|`` (kind ``measured``)."""
    pm = field_independent(index_222)
    samples, ref = samples_of(seven_atoms()), reference_of(REF7)
    joint = JointMoments.from_samples(samples, index_222, ref)
    fac = JointMoments.from_samples(samples, index_222, ref, parameter_map=pm)
    expected = pm(pm.project(joint), ref)
    assert_allclose(fac.to_vector(), expected.to_vector(), atol=1e-14)
    assert [r.name for r in fac.assumptions] == [pm.name]
    assert fac.discrepancy is None and joint.assumptions == ()
    measured = JointMoments.from_samples(
        samples, index_222, ref, parameter_map=pm, discrepancy="measured"
    )
    assert_allclose(measured.to_vector(), fac.to_vector(), atol=0)
    term = measured.discrepancy
    assert term.kind == "measured" and term.manuscript_term == "E_phys"
    gap = np.abs(np.asarray(joint.to_vector()) - np.asarray(fac.to_vector()))
    assert_allclose(term.value, gap, atol=0)
    assert gap.max() > 1e-3  # the correlated atoms violate the factorisation
    assert "7 supplied samples" in term.note and pm.name in term.note


def test_to_dict_layout(index_111):
    moments = JointMoments.from_samples(
        samples_of(seven_atoms()), index_111, reference_of(REF7)
    )
    d = moments.to_dict()
    assert set(d) >= {
        "index",
        "vector",
        "labels",
        "reference",
        "m0_ext",
        "assumptions",
        "discrepancy",
    }
    assert d["index"]["n_real"] == index_111.n_real and d["index"]["L_mu"] == 1
    assert len(d["vector"]) == len(d["labels"]) == index_111.n_real
    assert d["vector"][0] == 1.0 and d["labels"][0] == "M0[l=0,k=0;r=0,s=0]"
    assert d["assumptions"] == [] and d["discrepancy"] is None
    assert d["reference"]["gamma0"] == 5.2


# -- manuscript toy population (eq: channel toy population) -------------------


def toy_population(width, latent_nodes=16, angular_nodes=64, B0=1.0):
    """Discrete measure of validation/full_response.population, as arrays."""
    nodes, weights = leggauss(angular_nodes)
    latent, lw = leggauss(latent_nodes)
    u = latent[:, None, None, None]
    v = latent[None, :, None, None]
    mu = nodes[None, None, :, None]
    eta = nodes[None, None, None, :]
    rho = np.exp((2 + u) * mu + (1 + 0.5 * v) * eta + 0.75 * mu * eta)
    measure = weights[None, None, :, None] * weights[None, None, None, :] * rho
    measure = measure / measure.sum(axis=(2, 3), keepdims=True)
    w = lw[:, None, None, None] * lw[None, :, None, None] / 4 * measure
    q_gamma = width * 0.2 * u
    q_B = width * 0.2 * (0.6 * u + 0.4 * v)
    zeta = 4 + width * (2 * u + v)
    nu_star = E_ESU * B0 / (2 * np.pi * 20.0 * M_E * C_CGS)
    s_depth = 1.0 / (2 * (C_SI_M / nu_star) ** 2)
    shape = (latent_nodes, latent_nodes, angular_nodes, angular_nodes)
    full = lambda x: np.broadcast_to(x, shape).ravel()  # noqa: E731
    pop = dict(
        gamma=full(20.0 * (1 + q_gamma)),
        B=full(B0 * (1 + q_B)),
        mu=full(mu),
        eta=full(eta),
        phi=full(0.2 + 0.4 * mu + 0.2 * v),
        depth=full(zeta * s_depth),
        w=w.ravel(),
    )
    reference = (20.0, B0, 4.0 * s_depth, (20.0, B0, s_depth))
    return pop, reference, (latent, lw, nodes, weights)


def toy_angular_oracle(width, grids, L=2):
    """Angular blocks ``ang0``, ``ang2`` of full_response.population at r=s=b=0."""
    latent, lw, nodes, weights = grids
    poly = legvander(nodes, L)
    mu, eta = nodes[:, None], nodes[None, :]
    base = weights[:, None] * weights[None, :]
    ang0 = np.zeros((L + 1, L + 1))
    ang2 = np.zeros((L + 1, L + 1), complex)
    for iu, u in enumerate(latent):
        for iv, v in enumerate(latent):
            rho = np.exp((2 + u) * mu + (1 + 0.5 * v) * eta + 0.75 * mu * eta)
            measure = base * rho
            measure /= measure.sum()
            sky = np.exp(2j * (0.2 + 0.4 * mu + 0.2 * v))
            weight = lw[iu] * lw[iv] / 4
            ang0 += weight * np.einsum("ij,il,jk->lk", measure, poly, poly)
            ang2 += weight * np.einsum("ij,il,jk->lk", measure * sky, poly, poly)
    return ang0, ang2


@pytest.mark.parametrize("width", [0.5, 1.0])
def test_manuscript_toy_population_moment_pins(width, index_222):
    pop, reference, grids = toy_population(width)
    assert pop["gamma"].size == 16 * 16 * 64 * 64
    moments = JointMoments.from_samples(
        samples_of(pop), index_222, reference_of(reference)
    )
    assert_allclose(moments.get(0, 0, 0, 0, 0, 0), 1.0, atol=2e-14)
    assert_allclose(moments.get(0, 0, 0, 1, 1, 0), 0.008 * width**2, atol=2e-14)
    assert_allclose(moments.get(0, 0, 0, 1, 0, 1), (0.4 / 3) * width**2, atol=2e-14)
    assert_allclose(moments.get(0, 0, 0, 0, 1, 1), (0.32 / 3) * width**2, atol=2e-14)
    assert_allclose(
        moments.get(2, 0, 0, 1, 0, 1).real, moments.get(2, 0, 0, 1, 0, 1).real
    )
    ang0, ang2 = toy_angular_oracle(width, grids)
    for l in range(3):
        for k in range(3):
            assert_allclose(moments.get(0, l, k, 0, 0, 0), ang0[l, k], atol=2e-14)
            if (l + k) % 2 == 0:
                assert_allclose(moments.get(2, l, k, 0, 0, 0), ang2[l, k], atol=2e-14)
    # The population is correlated: the joint tensor is not a product.
    joint = moments.get(0, 1, 0, 1, 0, 0)
    product = moments.get(0, 1, 0, 0, 0, 0) * moments.get(0, 0, 0, 1, 0, 0)
    assert abs(joint - product) > 1e-4

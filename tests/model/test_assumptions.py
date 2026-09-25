"""Tests of ``syncmoments.model.assumptions``: factorised parameter maps.

Oracles are NumPy/SciPy: exact weighted sums over discrete populations
(``eval_legendre``), the product-of-marginals definition of a factorised
tensor (FINAL_DESIGN Section 7) evaluated term by term, an independent
free-parameter counter (the projection rule of the design), Gaussian raw
moments from the double-factorial formula and the Bell recursion for
cumulant closures.
"""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from numpy.testing import assert_allclose
import pytest
from scipy.special import eval_legendre

from syncmoments.model.assumptions import (
    VARS,
    Closure,
    Factorisation,
    ParameterMap,
    Parameters,
    azimuth_separable,
    field_independent,
    fully_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
    nodal,
)
from syncmoments.model.errors import AssumptionRecord, ErrorTerm
from syncmoments.model.index import MomentIndex, Truncation
from syncmoments.model.moments import JointMoments, PopulationSamples, Reference

REF = (5.2, 2.1, 0.3, (1.3, 0.7, 0.9))
POSITION = {"phi": 0, "mu": 1, "eta": 2, "gamma": 3, "B": 4, "depth": 5}
REST = {v: tuple(u for u in VARS if u != v) for v in VARS}


# -- population helpers ---------------------------------------------------------


def reference_of(ref=REF):
    gamma0, B0, depth_ref, scales = ref
    return Reference(gamma0, B0, depth_ref, scales=scales)


def samples_of(pop):
    return PopulationSamples(
        pop["gamma"],
        pop["B"],
        pop["mu"],
        pop["eta"],
        pop["phi"],
        pop["depth"],
        weights=pop["w"],
    )


def correlated_atoms(seed, n=7, keep=VARS):
    """Correlated atoms (every variable depends on one latent), restricted to ``keep``."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1, 1, n)
    pop = dict(
        gamma=5.0 + 1.5 * t + 0.3 * rng.standard_normal(n),
        B=2.0 + 0.8 * t**2 + 0.1 * rng.standard_normal(n),
        mu=np.clip(0.6 * t + 0.2 * rng.standard_normal(n), -0.95, 0.95),
        eta=np.clip(-0.5 * t + 0.3 * rng.standard_normal(n), -0.95, 0.95),
        phi=0.3 + 1.7 * t + 0.4 * rng.standard_normal(n),
        depth=1.5 * t + 0.2 * rng.standard_normal(n),
        w=rng.uniform(0.2, 3.0, n),
    )
    return {k: v for k, v in pop.items() if k in keep or k == "w"}


def marginal_atoms(seed, variable, n=4):
    rng = np.random.default_rng(seed)
    values = dict(
        gamma=4.0 + 3.0 * rng.uniform(size=n),
        B=1.0 + 2.5 * rng.uniform(size=n),
        mu=rng.uniform(-0.9, 0.9, n),
        eta=rng.uniform(-0.9, 0.9, n),
        phi=rng.uniform(0, 2 * np.pi, n),
        depth=rng.uniform(-2.0, 2.0, n),
    )
    return {variable: values[variable], "w": rng.uniform(0.5, 2.0, n)}


def outer(*parts):
    """Product measure of independent parts (dicts of arrays plus ``'w'``)."""
    grids = np.meshgrid(*[np.arange(len(p["w"])) for p in parts], indexing="ij")
    idx = [g.ravel() for g in grids]
    out, w = {}, np.ones(idx[0].size)
    for part, i in zip(parts, idx):
        for key, value in part.items():
            if key == "w":
                w = w * np.asarray(value)[i]
            else:
                out[key] = np.asarray(value)[i]
    out["w"] = w
    return out


def uniform_mu_part(n=6):
    nodes, weights = leggauss(n)
    return {"mu": nodes, "w": weights / 2}


def gaussian_depth_part(mean, sigma, n=4):
    x, w = hermegauss(n)
    return {"depth": mean + sigma * x, "w": w / w.sum()}


# -- NumPy oracles -----------------------------------------------------------------


def factor_np(variable, exponent, pop, ref):
    gamma0, B0, depth_ref, (sg, sB, sd) = ref
    if variable == "gamma":
        return ((pop["gamma"] - gamma0) / sg) ** exponent
    if variable == "B":
        return ((pop["B"] - B0) / sB) ** exponent
    if variable == "depth":
        return ((pop["depth"] - depth_ref) / sd) ** exponent
    if variable == "phi":
        return np.exp(1j * exponent * pop["phi"])
    return eval_legendre(exponent, pop[variable])


def marginal_np(pop, group, projection, ref):
    w = np.asarray(pop["w"], float)
    f = np.ones(w.size, dtype=complex)
    for v, e in zip(group, projection):
        f = f * factor_np(v, e, pop, ref)
    return np.sum(w * f) / w.sum()


def joint_np(pop, rows, ref):
    return np.array([marginal_np(pop, VARS, project(row, VARS), ref) for row in rows])


def project(row, group):
    return tuple(row[POSITION[v]] for v in group)


def all_rows(index):
    h0 = [(0, *r) for r in index.h0]
    ext = [(0, *r) for r in index.h0_ext]
    h2 = [(2, *r) for r in index.h2]
    return h0, ext, h2


def gaussian_raw_moment(mean, sigma, b):
    """``E[(mean + sigma x)^b]`` for standard normal ``x`` (double factorials)."""
    total = 0.0
    for k in range(0, b + 1, 2):
        dfact = math.prod(range(k - 1, 0, -2)) if k > 0 else 1
        total += math.comb(b, k) * mean ** (b - k) * sigma**k * dfact
    return total


def factorised_np(pop, index, ref, groups, closed):
    """Term-by-term product of group marginals for every retained row.

    ``closed`` maps a group to a callable ``projection -> factor``; other
    groups use the population marginal. Returns ``(m0, m0_ext, m2)``.
    """
    out = []
    for block in all_rows(index):
        values = []
        for row in block:
            value = 1.0 + 0j
            for g in groups:
                p = project(row, g)
                value *= closed[g](p) if g in closed else marginal_np(pop, g, p, ref)
            values.append(value)
        out.append(np.array(values))
    return out


def count_free_np(index, groups, closed_mu=False):
    """Independent free-count oracle: distinct nonconstant projections of the
    rows of ``m`` per group (complex ``h=2`` entries count twice), rows with
    ``l > 0`` dropped under ``uniform_mu``."""
    h0, _, h2 = all_rows(index)
    rows = [r for r in h0 + h2 if not (closed_mu and r[1] > 0)]
    total = 0
    for g in groups:
        if closed_mu and g == ("mu",):
            continue
        for p in sorted({project(r, g) for r in rows}):
            if any(p):
                total += 2 if ("phi" in g and p[g.index("phi")] == 2) else 1
    return total


def vector_of(fac):
    return np.asarray(fac.to_vector())


# -- maps under test ---------------------------------------------------------------

NAMED = {
    "no_assumption": no_assumption,
    "independent_screen": independent_screen,
    "field_independent": field_independent,
    "azimuth_separable": azimuth_separable,
    "isotropic_pitch": isotropic_pitch,
    "fully_independent": fully_independent,
}


def product_population(name, seed=3):
    """A population that satisfies the named assumption exactly."""
    if name == "independent_screen":
        return outer(
            correlated_atoms(seed, keep=REST["depth"]), marginal_atoms(seed, "depth")
        )
    if name == "field_independent":
        return outer(correlated_atoms(seed, keep=REST["B"]), marginal_atoms(seed, "B"))
    if name == "azimuth_separable":
        return outer(
            correlated_atoms(seed, keep=REST["phi"]), marginal_atoms(seed, "phi")
        )
    if name == "isotropic_pitch":
        return outer(correlated_atoms(seed, keep=REST["mu"]), uniform_mu_part())
    if name == "fully_independent":
        return outer(*[marginal_atoms(seed + i, v, 3) for i, v in enumerate(VARS)])
    if name == "screen_azimuth":
        rest = correlated_atoms(seed, keep=("gamma", "B", "mu", "eta"))
        return outer(
            rest, marginal_atoms(seed, "phi"), marginal_atoms(seed + 1, "depth")
        )
    if name == "gaussian_screen":
        rest = correlated_atoms(seed, keep=REST["depth"])
        return outer(rest, gaussian_depth_part(0.8, 0.6))
    return correlated_atoms(seed)


# -- free counts ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("no_assumption", 153),
        ("independent_screen", 115),
        ("field_independent", 88),
        ("azimuth_separable", 75),
        ("isotropic_pitch", 57),
        ("fully_independent", 12),
    ],
)
def test_free_counts_222(index_222, name, expected):
    pm = NAMED[name](index_222)
    assert pm.n_free() == expected
    assert len(pm.labels()) == expected
    assert len(set(pm.labels())) == expected
    assert pm.record().n_free == expected


def test_gaussian_and_combined_counts_222(index_222):
    assert gaussian_screen(index_222, 0.5, 0.2).n_free() == 113
    assert gaussian_screen(index_222, 0.5, 0.2, fit_hyper=True).n_free() == 115
    combined = independent_screen(index_222).assume(azimuth_separable(index_222))
    assert combined.n_free() == 57
    assert combined.factorisation.groups == (
        ("gamma", "B", "mu", "eta"),
        ("phi",),
        ("depth",),
    )
    table = Closure(("depth",), "fixed_table", (jnp.array([0.1, 0.2]),))
    assert independent_screen(index_222, screen=table).n_free() == 113


@pytest.mark.parametrize(
    "truncation",
    [
        Truncation(0, 0, 0),
        Truncation(0, 0, 2),
        Truncation(1, 1, 1),
        Truncation(2, 2, 2),
        Truncation(2, 1, 3),
        Truncation(2, 2, 2, depth_degree=3),
    ],
)
@pytest.mark.parametrize("name", list(NAMED))
def test_free_counts_match_independent_counter(truncation, name):
    index = MomentIndex.build(truncation)
    pm = NAMED[name](index)
    closed_mu = name == "isotropic_pitch"
    assert pm.n_free() == count_free_np(index, pm.factorisation.groups, closed_mu)
    if name == "fully_independent":
        L_mu, L_eta, N = truncation.L_mu, truncation.L_eta, truncation.N
        expected = 3 * N + L_mu + L_eta + 2
        if truncation.depth_degree is not None:
            expected += truncation.depth_degree - N
        assert pm.n_free() == expected


def test_no_V_index_counts():
    index = MomentIndex.build(Truncation(2, 2, 2), components=("I", "Q"))
    for name, pm in ((n, f(index)) for n, f in NAMED.items()):
        closed_mu = name == "isotropic_pitch"
        assert pm.n_free() == count_free_np(index, pm.factorisation.groups, closed_mu)


# -- static structure ----------------------------------------------------------------


def test_uniform_mu_drops_l_rows(index_222):
    pm = isotropic_pitch(index_222)
    h0, ext, h2 = all_rows(index_222)
    rows = h0 + ext + h2
    for i, row in enumerate(rows):
        assert pm.keep[i] == (row[1] == 0)
        if row[1] > 0:
            assert all(g[i] == -1 for g in pm.gathers)
    mu_group = pm.factorisation.groups.index(("mu",))
    assert pm.tables_spec[mu_group] == ()
    theta = pm.unflatten(np.random.default_rng(0).standard_normal(pm.n_free()))
    moments = pm(theta, reference_of())
    for i, row in enumerate(index_222.h0):
        if row[0] > 0:
            assert moments.m0[i] == 0.0
    for i, row in enumerate(index_222.h2):
        if row[0] > 0:
            assert moments.m2[i] == 0.0
    assert moments.m0_ext is None and not pm.ext_ok


def test_ext_reconstruction_flags(index_222):
    expected = {
        "no_assumption": False,
        "independent_screen": True,
        "field_independent": False,
        "azimuth_separable": True,
        "isotropic_pitch": False,
        "fully_independent": True,
    }
    for name, flag in expected.items():
        assert NAMED[name](index_222).ext_ok is flag


@pytest.mark.parametrize(
    "name,affine",
    [
        ("no_assumption", True),
        ("isotropic_pitch", True),
        ("independent_screen", False),
        ("field_independent", False),
        ("azimuth_separable", False),
        ("fully_independent", False),
    ],
)
def test_is_affine_both_sides(index_222, name, affine):
    assert NAMED[name](index_222).is_affine() is affine


def test_is_affine_with_closures(index_222):
    assert gaussian_screen(index_222, 0.1, 0.4).is_affine() is True
    assert gaussian_screen(index_222, 0.1, 0.4, fit_hyper=True).is_affine() is False
    fixed = Closure(("depth",), "fixed_table", (jnp.array([0.3, 0.5]),))
    assert independent_screen(index_222, screen=fixed).is_affine() is True
    with pytest.raises(ValueError):
        independent_screen(index_222).affine_pieces()


def test_labels_and_gathers_layout(index_111):
    pm = fully_independent(index_111)
    assert pm.labels() == (
        "<z_gamma>",
        "<z_B>",
        "<P_1(mu)>",
        "<P_1(eta)>",
        "Re <e^{2i phi}>",
        "Im <e^{2i phi}>",
        "<z_depth>",
    )
    h0, ext, h2 = all_rows(index_111)
    assert all(len(g) == len(h0) + len(ext) + len(h2) for g in pm.gathers)
    assert all(g[0] == 0 for g in pm.gathers)  # unit row is constant in every group


# -- product populations reproduce the joint tensor -------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "independent_screen",
        "field_independent",
        "azimuth_separable",
        "isotropic_pitch",
        "fully_independent",
        "screen_azimuth",
        "gaussian_screen",
    ],
)
def test_product_population_reproduces_joint_tensor(index_222, name):
    if name == "screen_azimuth":
        pm = independent_screen(index_222).assume(azimuth_separable(index_222))
    elif name == "gaussian_screen":
        pm = gaussian_screen(index_222, 0.8, 0.6)
    else:
        pm = NAMED[name](index_222)
    pop = product_population(name)
    ref = reference_of()
    joint = JointMoments.from_samples(samples_of(pop), index_222, ref)
    theta = pm.project(joint)
    fac = pm(theta, ref)
    assert_allclose(vector_of(fac), vector_of(joint), atol=1e-13, rtol=1e-13)
    assert np.max(np.abs(np.asarray(pm.constraint_residual(joint)))) < 1e-13
    if pm.ext_ok:
        assert_allclose(fac.m0_ext, joint.m0_ext, atol=1e-13, rtol=1e-13)
    assert fac.assumptions == (pm.record(),)
    measured = JointMoments.from_samples(
        samples_of(pop), index_222, ref, parameter_map=pm, discrepancy="measured"
    )
    assert measured.discrepancy.kind == "measured"
    assert np.max(np.asarray(measured.discrepancy.value)) < 1e-13
    assert_allclose(vector_of(measured), vector_of(fac), atol=1e-14)


@pytest.mark.parametrize(
    "name",
    [
        "independent_screen",
        "field_independent",
        "azimuth_separable",
        "isotropic_pitch",
        "fully_independent",
    ],
)
def test_correlated_population_differs_and_measured_discrepancy(index_222, name):
    pm = NAMED[name](index_222)
    pop = correlated_atoms(11)
    ref = reference_of()
    joint = JointMoments.from_samples(samples_of(pop), index_222, ref)
    fac = pm(pm.project(joint), ref)
    difference = np.abs(vector_of(joint) - vector_of(fac))
    assert np.max(difference) > 1e-3
    measured = JointMoments.from_samples(
        samples_of(pop), index_222, ref, parameter_map=pm, discrepancy="measured"
    )
    assert_allclose(measured.discrepancy.value, difference, atol=1e-15)
    assert measured.discrepancy.kind == "measured"
    assert measured.assumptions[0].name == name
    assert_allclose(vector_of(measured), vector_of(fac), atol=1e-15)
    plain = JointMoments.from_samples(samples_of(pop), index_222, ref, parameter_map=pm)
    assert plain.discrepancy is None
    with pytest.raises(ValueError):
        JointMoments.from_samples(
            samples_of(pop), index_222, ref, discrepancy="measured"
        )
    with pytest.raises(ValueError):
        JointMoments.from_samples(
            samples_of(pop), index_222, ref, parameter_map=object()
        )


def test_zero_covariance_without_independence():
    """``B = |z_gamma|`` with symmetric ``gamma``: the ``N = 2`` cross term
    agrees under ``field_independent`` but ``<P_2(mu) z_B>`` and the ``N = 3``
    term ``<z_gamma^2 z_B>`` do not."""
    ref = (5.0, 2.0, 0.0, (1.0, 1.0, 1.0))
    z = np.array([-1.5, -0.5, 0.5, 1.5])
    pop = dict(
        gamma=5.0 + z,
        B=2.0 + np.abs(z),
        mu=0.9 * z**2 / 2.25 - 0.4,
        eta=np.zeros(4),
        phi=np.zeros(4),
        depth=np.zeros(4),
        w=np.ones(4),
    )
    reference = reference_of(ref)
    for N in (2, 3):
        index = MomentIndex.build(Truncation(2, 2, N))
        pm = field_independent(index)
        joint = JointMoments.from_samples(samples_of(pop), index, reference)
        fac = pm(pm.project(joint), reference)
        cross = index.position(0, 0, 0, 1, 1, 0)
        assert_allclose(joint.m0[cross], 0.0, atol=1e-15)
        assert_allclose(fac.m0[cross], 0.0, atol=1e-15)
        angular = index.position(0, 2, 0, 0, 1, 0)
        assert abs(float(joint.m0[angular] - fac.m0[angular])) > 1e-2
        if N == 3:
            cubic = index.position(0, 0, 0, 2, 1, 0)
            assert abs(float(joint.m0[cubic] - fac.m0[cubic])) > 1e-2


# -- exact factorisation formulas, term by term -----------------------------------------


def closed_factors(name, ref):
    if name == "isotropic_pitch":
        return {("mu",): lambda p: 1.0 if p[0] == 0 else 0.0}
    if name == "gaussian_screen":
        _, _, depth_ref, (_, _, sd) = ref
        mean, sigma = (0.8 - depth_ref) / sd, 0.6 / sd
        return {("depth",): lambda p: gaussian_raw_moment(mean, sigma, p[0])}
    return {}


@pytest.mark.parametrize(
    "name",
    [
        "no_assumption",
        "independent_screen",
        "field_independent",
        "azimuth_separable",
        "isotropic_pitch",
        "fully_independent",
        "gaussian_screen",
    ],
)
@pytest.mark.parametrize("seed", [1, 2])
def test_factorised_tensor_matches_numpy_products(index_222, name, seed):
    pm = (
        gaussian_screen(index_222, 0.8, 0.6)
        if name == "gaussian_screen"
        else NAMED[name](index_222)
    )
    pop = correlated_atoms(seed)
    ref = reference_of()
    joint = JointMoments.from_samples(samples_of(pop), index_222, ref)
    fac = pm(pm.project(joint), ref)
    m0, ext, m2 = factorised_np(
        pop, index_222, REF, pm.factorisation.groups, closed_factors(name, REF)
    )
    assert_allclose(fac.m0, m0.real, atol=1e-14, rtol=1e-13)
    assert_allclose(fac.m2, m2, atol=1e-14, rtol=1e-13)
    if pm.ext_ok:
        assert_allclose(fac.m0_ext, ext.real, atol=1e-14, rtol=1e-13)
    if name == "no_assumption":
        assert_allclose(vector_of(fac), vector_of(joint), atol=1e-15)


# -- affine pieces, projection and flattening ----------------------------------------------


def random_theta(pm, seed=0):
    rng = np.random.default_rng(seed)
    return pm.unflatten(rng.standard_normal(pm.n_free()))


def affine_maps(index):
    fixed = Closure(("depth",), "fixed_table", (jnp.array([0.3, 0.5]),))
    delta = Closure(("depth",), "delta", (0.7,))
    return [
        no_assumption(index),
        isotropic_pitch(index),
        gaussian_screen(index, 0.8, 0.6),
        independent_screen(index, screen=fixed),
        independent_screen(index, screen=delta),
        gaussian_screen(index, 0.8, 0.6).assume(isotropic_pitch(index)),
    ]


@pytest.mark.parametrize("which", range(6))
def test_affine_pieces_reproduce_call(index_222, which):
    pm = affine_maps(index_222)[which]
    ref = reference_of()
    assert pm.is_affine()
    P, c = pm.affine_pieces(reference=ref)
    assert P.shape == (index_222.n_real, pm.n_free()) and c.shape == (index_222.n_real,)
    assert not jnp.iscomplexobj(P) and not jnp.iscomplexobj(c)
    for seed in (0, 1):
        theta = random_theta(pm, seed)
        expected = vector_of(pm(theta, ref))
        assert_allclose(
            np.asarray(P) @ np.asarray(pm.flatten(theta)) + np.asarray(c),
            expected,
            atol=1e-13,
        )
    assert_allclose(c[0], 1.0)  # unit moment


def test_affine_pieces_need_reference_for_depth_closures(index_222):
    with pytest.raises(ValueError):
        gaussian_screen(index_222, 0.8, 0.6).affine_pieces()
    P, c = isotropic_pitch(index_222).affine_pieces()
    assert P.shape[1] == 57


@pytest.mark.parametrize(
    "name",
    [
        "no_assumption",
        "independent_screen",
        "field_independent",
        "azimuth_separable",
        "isotropic_pitch",
        "fully_independent",
        "gaussian_fit",
    ],
)
def test_project_of_call_roundtrips(index_222, name):
    if name == "gaussian_fit":
        pm = gaussian_screen(index_222, 0.8, 0.6, fit_hyper=True)
        theta = random_theta(pm, 4)
        theta = Parameters(tables=theta.tables, hyper=(jnp.array([0.25, 0.9]),))
    else:
        pm = NAMED[name](index_222)
        theta = random_theta(pm, 4)
    ref = reference_of()
    recovered = pm.project(pm(theta, ref), amplitude=3.0)
    assert_allclose(pm.flatten(recovered), pm.flatten(theta), atol=1e-13, rtol=1e-13)
    assert_allclose(recovered.log_amplitude, np.log(3.0))
    assert len(recovered.tables) == len(theta.tables)


def test_flatten_unflatten_roundtrip_and_labels(index_222):
    for pm in (
        independent_screen(index_222),
        gaussian_screen(index_222, 0.1, 0.2, fit_hyper=True),
    ):
        vector = np.random.default_rng(5).standard_normal(pm.n_free())
        theta = pm.unflatten(vector)
        assert_allclose(pm.flatten(theta), vector, atol=1e-15)
        assert len(pm.labels()) == pm.n_free()
    pm = gaussian_screen(index_222, 0.1, 0.2, fit_hyper=True)
    assert pm.labels()[-2:] == (
        "hyper:gaussian_depth:mean",
        "hyper:gaussian_depth:sigma",
    )
    rest = independent_screen(index_222).labels()
    assert rest[0] == "Re <e^{2i phi}>"  # lexicographic in (gamma, B, mu, eta, phi)
    assert "<z_B>" in rest and "Im <z_B P_2(mu) P_2(eta) e^{2i phi}>" in rest
    assert rest[-2:] == ("<z_depth>", "<z_depth^2>")


# -- closures ---------------------------------------------------------------------------


def test_gaussian_closure_table_matches_double_factorials():
    index = MomentIndex.build(Truncation(0, 0, 4))
    ref = reference_of()
    pm = gaussian_screen(index, 0.8, 0.6)
    theta = pm.unflatten(np.zeros(pm.n_free()))
    moments = pm(theta, ref)
    _, _, depth_ref, (_, _, sd) = REF
    for b in range(1, 5):
        got = moments.m0_ext[index.ext_position(0, 0, 0, 0, b)]
        assert_allclose(
            got, gaussian_raw_moment((0.8 - depth_ref) / sd, 0.6 / sd, b), rtol=1e-13
        )


def test_cumulant_closure_matches_bell_recursion():
    index = MomentIndex.build(Truncation(0, 0, 4))
    ref = reference_of()
    kappa = np.array([0.4, 0.5, -0.2, 0.15])
    closure = Closure(("depth",), "cumulant_depth", (jnp.asarray(kappa),))
    pm = independent_screen(index, screen=closure)
    moments = pm(pm.unflatten(np.zeros(pm.n_free())), ref)
    _, _, depth_ref, (_, _, sd) = REF
    kz = [(kappa[0] - depth_ref) / sd] + [kappa[n] / sd ** (n + 1) for n in range(1, 4)]
    m = [1.0]
    for n in range(1, 5):
        m.append(
            sum(math.comb(n - 1, j - 1) * kz[j - 1] * m[n - j] for j in range(1, n + 1))
        )
    for b in range(1, 5):
        assert_allclose(
            moments.m0_ext[index.ext_position(0, 0, 0, 0, b)], m[b], rtol=1e-13
        )
    # Gaussian cumulants reproduce the Gaussian closure.
    gauss = Closure(("depth",), "cumulant_depth", (jnp.array([0.8, 0.36]),))
    a = independent_screen(index, screen=gauss)
    b = gaussian_screen(index, 0.8, 0.6)
    ta, tb = a.unflatten(np.zeros(a.n_free())), b.unflatten(np.zeros(b.n_free()))
    assert_allclose(a(ta, ref).m0_ext, b(tb, ref).m0_ext, rtol=1e-13)


def test_delta_and_fixed_table_closures(index_111):
    ref = reference_of()
    _, _, depth_ref, (_, _, sd) = REF
    delta = independent_screen(index_111, screen=Closure(("depth",), "delta", (0.7,)))
    theta = random_theta(delta, 2)
    moments = delta(theta, ref)
    slot = index_111.position(2, 0, 0, 0, 0, 1) - index_111.n0
    base = index_111.position(2, 0, 0, 0, 0, 0) - index_111.n0
    assert_allclose(
        moments.m2[slot], moments.m2[base] * (0.7 - depth_ref) / sd, rtol=1e-13
    )
    fixed = independent_screen(
        index_111, screen=Closure(("depth",), "fixed_table", (jnp.array([0.45]),))
    )
    moments = fixed(theta, ref)
    assert_allclose(moments.m2[slot], moments.m2[base] * 0.45, rtol=1e-13)
    # A delta closure on the angular pair reproduces Legendre values at the node.
    pair = Closure(("mu", "eta"), "delta", (0.3, -0.5))
    groups = (("gamma", "B", "phi", "depth"), ("mu", "eta"))
    pm = ParameterMap.build(index_111, Factorisation(groups, "angular_delta"), (pair,))
    moments = pm(random_theta(pm, 1), ref)
    assert_allclose(
        moments.m0[index_111.position(0, 1, 1, 0, 0, 0)], 0.3 * -0.5, rtol=1e-13
    )


def test_cumulant_hyper_projection_roundtrip():
    index = MomentIndex.build(Truncation(0, 0, 4))
    ref = reference_of()
    kappa = jnp.array([0.4, 0.5, -0.2, 0.15])
    closure = Closure(("depth",), "cumulant_depth", (kappa,))
    factorisation = Factorisation((REST["depth"], ("depth",)), "cumulant")
    pm = ParameterMap.build(index, factorisation, (closure,), free_hyper=(0,))
    assert pm.n_free() == count_free_np(index, pm.factorisation.groups) - 4 + 4
    theta = random_theta(pm, 3)
    theta = Parameters(tables=theta.tables, hyper=(jnp.array([0.1, 0.7, 0.3, -0.05]),))
    recovered = pm.project(pm(theta, ref))
    assert_allclose(recovered.hyper[0], theta.hyper[0], atol=1e-12)


# -- assume(): refinement and conflicts --------------------------------------------------


def test_assume_refinement(index_222):
    pm = independent_screen(index_222).assume(
        azimuth_separable(index_222), isotropic_pitch(index_222)
    )
    assert pm.factorisation.groups == (
        ("gamma", "B", "eta"),
        ("mu",),
        ("phi",),
        ("depth",),
    )
    assert pm.n_free() == count_free_np(
        index_222, pm.factorisation.groups, closed_mu=True
    )
    assert pm.closures[0].kind == "uniform_mu"
    assert "independent_screen" in pm.name and "isotropic_pitch" in pm.name
    same = isotropic_pitch(index_222).assume(isotropic_pitch(index_222))
    assert same.n_free() == 57
    six = no_assumption(index_222).assume(
        *[NAMED[n](index_222) for n in NAMED if n != "isotropic_pitch"]
    )
    assert six.n_free() == 12


def test_assume_conflicts(index_222):
    a = gaussian_screen(index_222, 0.1, 0.2)
    b = gaussian_screen(index_222, 0.3, 0.2)
    with pytest.raises(ValueError):
        a.assume(b)
    a.assume(gaussian_screen(index_222, 0.1, 0.2))  # equal hyper are not a conflict
    delta_mu = Closure(("mu",), "delta", (0.2,))
    groups = (REST["mu"], ("mu",))
    other = ParameterMap.build(
        index_222, Factorisation(groups, "delta_mu"), (delta_mu,)
    )
    with pytest.raises(ValueError):
        isotropic_pitch(index_222).assume(other)
    pair = Closure(("mu", "eta"), "delta", (0.3, -0.5))
    joint_angles = ParameterMap.build(
        index_222,
        Factorisation((("gamma", "B", "phi", "depth"), ("mu", "eta")), "pair"),
        (pair,),
    )
    with pytest.raises(ValueError):
        joint_angles.assume(isotropic_pitch(index_222))
    with pytest.raises(ValueError):
        no_assumption(index_222).assume(
            no_assumption(MomentIndex.build(Truncation(1, 1, 1)))
        )
    term = ErrorTerm(jnp.zeros(index_222.n_real), "bound", "x")
    with_term = ParameterMap.build(
        index_222, Factorisation((VARS,), "d"), discrepancy=term
    )
    with pytest.raises(ValueError):
        with_term.assume(with_term)
    assert with_term.assume(no_assumption(index_222)).discrepancy is term


# -- JIT, autodiff, records ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["independent_screen", "isotropic_pitch"])
def test_jit_and_grad_through_call(index_222, name):
    pm = NAMED[name](index_222)
    ref = reference_of()
    theta = random_theta(pm, 7)
    probe = np.random.default_rng(1).standard_normal(index_222.n_real)

    def scalar(tables):
        return jnp.dot(probe, pm(Parameters(tables=tables), ref).to_vector())

    compiled = jax.jit(scalar)
    assert_allclose(compiled(theta.tables), scalar(theta.tables), atol=1e-13)
    grads = jax.jit(jax.grad(scalar))(theta.tables)
    P, _ = pm.affine_pieces() if pm.is_affine() else (None, None)
    for j, g in enumerate(grads):
        assert np.all(np.isfinite(np.asarray(g)))
        step = 1e-6
        table = np.asarray(theta.tables[j])
        for i in range(min(3, table.size)):
            plus, minus = table.copy(), table.copy()
            plus[i] += step
            minus[i] -= step
            fd = (
                scalar(_replace(theta.tables, j, plus))
                - scalar(_replace(theta.tables, j, minus))
            ) / (2 * step)
            assert_allclose(np.real(g[i]), fd, rtol=1e-6, atol=1e-8)
    if P is not None:
        assert_allclose(
            jax.grad(scalar)(theta.tables)[0].real[:5],
            (probe @ np.asarray(P))[:5],
            atol=1e-10,
        )


def _replace(tables, j, value):
    return tuple(jnp.asarray(value) if i == j else t for i, t in enumerate(tables))


def test_grad_through_reference_and_hyper(index_111):
    pm = gaussian_screen(index_111, 0.8, 0.6, fit_hyper=True)
    theta = random_theta(pm, 2)
    theta = Parameters(tables=theta.tables, hyper=(jnp.array([0.5, 0.3]),))
    probe = np.random.default_rng(2).standard_normal(index_111.n_real)

    def scalar(hyper, depth_ref):
        ref = Reference(5.2, 2.1, depth_ref, scales=(1.3, 0.7, 0.9))
        return jnp.dot(
            probe, pm(Parameters(tables=theta.tables, hyper=(hyper,)), ref).to_vector()
        )

    g_hyper, g_ref = jax.jit(jax.grad(scalar, argnums=(0, 1)))(theta.hyper[0], 0.3)
    assert np.all(np.isfinite(g_hyper)) and np.isfinite(g_ref)
    step = 1e-6
    fd = (scalar(theta.hyper[0], 0.3 + step) - scalar(theta.hyper[0], 0.3 - step)) / (
        2 * step
    )
    assert_allclose(g_ref, fd, rtol=1e-6)
    for i in range(2):
        e = np.zeros(2)
        e[i] = step
        fd = (scalar(theta.hyper[0] + e, 0.3) - scalar(theta.hyper[0] - e, 0.3)) / (
            2 * step
        )
        assert_allclose(g_hyper[i], fd, rtol=1e-6)


def test_vmap_over_theta_batch(index_111):
    pm = isotropic_pitch(index_111)
    ref = reference_of()
    batch = np.random.default_rng(3).standard_normal((4, pm.n_free()))

    def vec(v):
        return pm(pm.unflatten(v), ref).to_vector()

    stacked = jax.vmap(vec)(jnp.asarray(batch))
    for i in range(4):
        assert_allclose(stacked[i], vec(batch[i]), atol=1e-14)


def test_record_json_roundtrip(index_222):
    for pm in (
        gaussian_screen(index_222, 0.8, 0.6),
        isotropic_pitch(index_222),
        no_assumption(index_222),
    ):
        record = pm.record()
        assert isinstance(record, AssumptionRecord)
        loaded = json.loads(json.dumps(record.to_dict()))
        rebuilt = AssumptionRecord(
            name=loaded["name"],
            label=loaded["label"],
            groups=tuple(tuple(g) for g in loaded["groups"]),
            closure_kind=loaded["closure_kind"],
            hyper=tuple(loaded["hyper"]),
            n_free=loaded["n_free"],
            discrepancy_kind=loaded["discrepancy_kind"],
        )
        assert rebuilt == record
        assert hash(record) == hash(rebuilt)
    record = gaussian_screen(index_222, 0.8, 0.6).record()
    assert record.hyper == (0.8, 0.6) and record.closure_kind == "gaussian_depth"
    assert (
        record.label == "eq: independent gaussian screen"
        and record.discrepancy_kind == "unbounded"
    )
    term = ErrorTerm(jnp.zeros(index_222.n_real), "estimate", "supplied")
    with_term = ParameterMap.build(
        index_222, Factorisation((VARS,), "d"), discrepancy=term
    )
    assert with_term.record().discrepancy_kind == "estimate"
    theta = random_theta(with_term, 0)
    assert with_term(theta, reference_of()).discrepancy is term
    assert isotropic_pitch(index_222).record().closure_kind == "uniform_mu"


# -- nodal map -----------------------------------------------------------------------------


def test_nodal_map(index_111):
    pop = correlated_atoms(4, n=5)
    nodes = samples_of({**pop, "w": np.ones(5)})
    pm = nodal(index_111, nodes)
    ref = reference_of()
    assert pm.n_free() == 5 and pm.labels() == tuple(f"logit[{j}]" for j in range(5))
    assert not pm.is_affine()
    logits = jnp.array([0.2, -1.0, 0.5, 1.5, -0.3])
    moments = pm(Parameters(logits=logits), ref)
    weights = np.exp(np.asarray(logits))
    expected = JointMoments.from_samples(
        samples_of({**pop, "w": weights}), index_111, ref
    )
    assert_allclose(vector_of(moments), vector_of(expected), atol=1e-14)
    assert_allclose(moments.m0_ext, expected.m0_ext, atol=1e-14)
    assert moments.assumptions[0].closure_kind == "nodal"
    grad = jax.jit(jax.grad(lambda z: pm(Parameters(logits=z), ref).to_vector().sum()))(
        logits
    )
    assert np.all(np.isfinite(grad))
    assert_allclose(pm.flatten(pm.unflatten(logits)), logits)
    with pytest.raises(ValueError):
        pm.project(expected)
    with pytest.raises(ValueError):
        pm.assume(no_assumption(index_111))
    with pytest.raises(ValueError):
        pm(Parameters(logits=jnp.zeros(4)), ref)
    with pytest.raises(ValueError):
        nodal(index_111, object())


# -- validation errors ---------------------------------------------------------------------


def test_build_rejects_bad_inputs(index_111):
    with pytest.raises(ValueError):
        ParameterMap.build(index_111, Factorisation((("gamma", "B"),), "x"))
    with pytest.raises(ValueError):
        ParameterMap.build(index_111, Factorisation((VARS, ("gamma",)), "x"))
    with pytest.raises(ValueError):
        ParameterMap.build(index_111, Factorisation((VARS[:5] + ("bogus",),), "x"))
    with pytest.raises(ValueError):
        ParameterMap.build(object(), Factorisation((VARS,), "x"))
    bad = [
        Closure(("mu",), "uniform_mu", (0.1,)),
        Closure(("eta",), "uniform_mu"),
        Closure(("depth",), "gaussian_depth", (0.1,)),
        Closure(("depth",), "cumulant_depth", (jnp.zeros(0),)),
        Closure(("depth",), "delta", (0.1, 0.2)),
        Closure(("depth",), "fixed_table", (jnp.zeros(3),)),
        Closure(("depth",), "nonsense"),
        Closure(("gamma", "depth"), "delta", (5.0, 0.1)),
    ]
    for closure in bad:
        with pytest.raises(ValueError):
            ParameterMap.build(
                index_111, Factorisation(tuple((v,) for v in VARS), "x"), (closure,)
            )
    with pytest.raises(ValueError):
        ParameterMap.build(index_111, Factorisation((VARS,), "x"), free_hyper=(0,))
    with pytest.raises(ValueError):
        ParameterMap.build(
            index_111,
            Factorisation((VARS,), "x"),
            discrepancy=ErrorTerm(jnp.zeros(3), "bound"),
        )
    with pytest.raises(ValueError):
        ParameterMap.build(index_111, Factorisation((VARS,), "x"), discrepancy=object())
    with pytest.raises(ValueError):
        independent_screen(index_111, screen=Closure(("mu",), "uniform_mu"))
    with pytest.raises(ValueError):
        ParameterMap.build(
            index_111,
            Factorisation((VARS[:5], ("depth",)), "x"),
            nodes=samples_of(correlated_atoms(1)),
        )


def test_call_rejects_bad_theta(index_111):
    ref = reference_of()
    pm = independent_screen(index_111)
    good = random_theta(pm)
    with pytest.raises(ValueError):
        pm(Parameters(tables=good.tables[:1]), ref)
    with pytest.raises(ValueError):
        pm(Parameters(tables=(good.tables[0][:-1], good.tables[1])), ref)
    with pytest.raises(ValueError):
        pm(Parameters(tables=(good.tables[0], good.tables[1].astype(complex))), ref)
    with pytest.raises(ValueError):
        pm(object(), ref)
    with pytest.raises(ValueError):
        pm.unflatten(np.zeros(pm.n_free() + 1))
    fit = gaussian_screen(index_111, 0.1, 0.2, fit_hyper=True)
    with pytest.raises(ValueError):
        fit(Parameters(tables=random_theta(fit).tables, hyper=(jnp.zeros(3),)), ref)
    with pytest.raises(ValueError):
        fit(Parameters(tables=random_theta(fit).tables), ref)
    other = MomentIndex.build(Truncation(2, 2, 2))
    with pytest.raises(ValueError):
        pm.project(
            JointMoments.from_samples(samples_of(correlated_atoms(1)), other, ref)
        )


def test_project_needs_ext_for_depth_marginals(index_111):
    pm = independent_screen(index_111)
    ref = reference_of()
    full = JointMoments.from_vector(index_111, np.eye(index_111.n_real)[0], ref)
    with pytest.raises(ValueError):
        pm.project(full)
    with pytest.raises(ValueError):
        gaussian_screen(index_111, 0.1, 0.2, fit_hyper=True).project(
            JointMoments.from_samples(samples_of(correlated_atoms(1)), index_111, ref)
        )


def test_complex_table_ignores_imaginary_part_of_real_entries(index_111):
    pm = no_assumption(index_111)
    ref = reference_of()
    theta = random_theta(pm, 9)
    table = np.asarray(theta.tables[0]).astype(complex)
    labels = pm.labels()
    real_entries = [i for i in range(table.size) if not labels[i].startswith("Re ")]
    assert len(real_entries) == 11  # h0 rows of (1,1,1) without the unit moment
    table[real_entries] += 1j * 3.0  # no e^{2i phi}: imaginary part is not free
    moments = pm(Parameters(tables=(jnp.asarray(table),)), ref)
    assert_allclose(vector_of(moments), vector_of(pm(theta, ref)), atol=1e-15)
    assert_allclose(
        pm.flatten(Parameters(tables=(jnp.asarray(table),))), pm.flatten(theta)
    )

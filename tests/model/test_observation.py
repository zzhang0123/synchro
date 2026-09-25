"""``StokesData`` against NumPy oracles: masking, whitening, response, dof, jit.

Conventions under test (``syncmoments/model/fit/observation.py``):
``stokes`` is ``(n_ch, 4)`` (channel-major flattening, Stokes order
I, Q, U, V) or, with a ``response`` ``(n_data, 4 n_ch)``, the data vector
``(n_data,)``; ``noise`` is ``(n_data,)`` variances or ``(n_data, n_data)``
covariance; ``mask`` marks the rows that are KEPT and broadcasts from
``(4,)`` to ``(n_ch, 4)``; masked rows are removed inside ``whitened()``
and from ``dof``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncmoments.model.errors import ErrorTerm
from syncmoments.model.fit.observation import StokesData

# ----------------------------------------------------------------------
# oracles and helpers
# ----------------------------------------------------------------------


def rng(seed=0):
    return np.random.default_rng(seed)


def random_covariance(n, seed=1, corr=0.5):
    g = rng(seed)
    A = g.normal(size=(n, n))
    S = A @ A.T / n + np.eye(n)
    S = corr * S + (1 - corr) * np.diag(np.diag(S))
    return 0.5 * (S + S.T)


def oracle_whiten(d, S, kept):
    """``(L^-1 d, L^-1)`` on the kept rows with ``S = L L^T`` (NumPy)."""
    kept = np.asarray(kept)
    Sk = S[np.ix_(kept, kept)] if S.ndim == 2 else np.diag(S[kept])
    L = np.linalg.cholesky(Sk)
    Linv = np.linalg.inv(L)
    return Linv @ d[kept], Linv


def make(n_ch=3, seed=0, **kw):
    g = rng(seed)
    stokes = g.normal(size=(n_ch, 4))
    var = g.uniform(0.5, 2.0, size=4 * n_ch)
    return stokes, var, StokesData(stokes, var, **kw)


# ----------------------------------------------------------------------
# construction and layout
# ----------------------------------------------------------------------


def test_defaults_n_data_is_four_n_ch_and_all_rows_kept():
    stokes, var, data = make(n_ch=3)
    assert data.n_data() == 12
    assert data.n_ch == 3
    assert data.n_kept() == 12
    assert data.kept_rows() == tuple(range(12))
    assert data.mask is None and data.response is None
    assert_allclose(np.asarray(data.data_vector()), stokes.reshape(-1))


def test_data_vector_is_channel_major_stokes_order():
    stokes = np.arange(8.0).reshape(2, 4)
    data = StokesData(stokes, np.ones(8))
    # row j*4 + s holds Stokes s of channel j
    assert_allclose(np.asarray(data.data_vector()), [0, 1, 2, 3, 4, 5, 6, 7])


def test_single_channel_extreme():
    stokes, var, data = make(n_ch=1, seed=3)
    assert data.n_data() == 4
    wd, Linv = data.whitened()
    od, oL = oracle_whiten(stokes.reshape(-1), var, range(4))
    assert_allclose(np.asarray(wd), od, rtol=1e-13)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-13)


@pytest.mark.parametrize(
    "bad_stokes",
    [np.ones((3, 3)), np.ones(12), np.ones((3, 4, 1)), np.ones((3, 4)) + 0j],
)
def test_stokes_shape_and_dtype_errors_are_value_errors(bad_stokes):
    with pytest.raises(ValueError):
        StokesData(bad_stokes, np.ones(12))


@pytest.mark.parametrize(
    "bad_noise",
    [np.ones(11), np.ones((12, 11)), np.ones((12, 12, 1)), np.ones(12) + 0j],
)
def test_noise_shape_and_dtype_errors_are_value_errors(bad_noise):
    with pytest.raises(ValueError):
        StokesData(np.ones((3, 4)), bad_noise)


def test_nonfinite_stokes_and_nonpositive_variances_are_runtime_errors():
    stokes = np.ones((2, 4))
    with_nan = stokes.copy()
    with_nan[1, 3] = np.nan
    with pytest.raises(Exception):
        StokesData(with_nan, np.ones(8))
    with pytest.raises(Exception):
        StokesData(stokes, np.r_[np.ones(7), 0.0])
    with pytest.raises(Exception):
        StokesData(stokes, np.r_[np.ones(7), -1.0])
    with pytest.raises(Exception):
        StokesData(stokes, np.r_[np.ones(7), np.inf])


# ----------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------


def test_mask_removes_rows_from_whitened_and_dof_variances():
    stokes, var, _ = make(n_ch=3, seed=5)
    mask = np.ones((3, 4), dtype=bool)
    mask[1, 2] = False  # U of channel 1
    mask[2, 3] = False  # V of channel 2
    data = StokesData(stokes, var, mask=mask)
    kept = [i for i in range(12) if i not in (6, 11)]
    assert data.kept_rows() == tuple(kept)
    assert data.n_kept() == 10
    assert data.n_data() == 12  # the unmasked count is unchanged
    wd, Linv = data.whitened()
    assert wd.shape == (10,) and Linv.shape == (10, 10)
    od, oL = oracle_whiten(stokes.reshape(-1), var, kept)
    assert_allclose(np.asarray(wd), od, rtol=1e-13)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-13, atol=1e-15)
    assert data.dof(3) == 7
    assert StokesData(stokes, var).dof(3) == 9


def test_mask_removes_rows_from_whitened_covariance():
    stokes, _, _ = make(n_ch=2, seed=7)
    S = random_covariance(8, seed=2)
    mask = np.array([[True, True, False, True], [True, False, True, False]])
    data = StokesData(stokes, S, mask=mask)
    kept = np.flatnonzero(mask.reshape(-1))
    wd, Linv = data.whitened()
    od, oL = oracle_whiten(stokes.reshape(-1), S, kept)
    assert_allclose(np.asarray(wd), od, rtol=1e-12, atol=1e-14)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-12, atol=1e-14)
    # the masked covariance is the sub-block, and Linv^T Linv is its inverse
    Sk = S[np.ix_(kept, kept)]
    assert_allclose(np.asarray(data.covariance()), Sk, rtol=1e-14)
    assert_allclose(np.asarray(Linv.T @ Linv), np.linalg.inv(Sk), rtol=1e-9)


def test_stokes_mask_broadcasts_from_four_to_n_ch_by_four():
    stokes, var, _ = make(n_ch=4, seed=9)
    per_stokes = np.array([True, True, True, False])  # drop V everywhere
    a = StokesData(stokes, var, mask=per_stokes)
    b = StokesData(stokes, var, mask=np.tile(per_stokes, (4, 1)))
    assert a.kept_rows() == b.kept_rows() == tuple(i for i in range(16) if i % 4 != 3)
    assert a.n_kept() == 12
    wa, La = a.whitened()
    wb, Lb = b.whitened()
    assert_allclose(np.asarray(wa), np.asarray(wb))
    assert_allclose(np.asarray(La), np.asarray(Lb))
    assert a.mask.shape == (4, 4)


def test_mask_accepts_zero_one_integers_but_not_other_values():
    stokes, var, _ = make(n_ch=2)
    data = StokesData(stokes, var, mask=np.array([1, 1, 0, 1]))
    assert data.n_kept() == 6
    with pytest.raises(ValueError):
        StokesData(stokes, var, mask=np.array([1, 2, 0, 1]))
    with pytest.raises(ValueError):
        StokesData(stokes, var, mask=np.array([1.0, 0.5, 0.0, 1.0]))


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 4, 1), (3, 4), (5,)])
def test_mask_wrong_shape_is_value_error(shape):
    stokes, var, _ = make(n_ch=2)
    with pytest.raises(ValueError):
        StokesData(stokes, var, mask=np.ones(shape, dtype=bool))


def test_all_masked_is_value_error():
    stokes, var, _ = make(n_ch=2)
    with pytest.raises(ValueError):
        StokesData(stokes, var, mask=np.zeros(4, dtype=bool))


def test_mask_must_be_concrete():
    stokes, var, _ = make(n_ch=2)

    def build(m):
        return StokesData(stokes, var, mask=m).n_kept()

    with pytest.raises(ValueError):
        jax.jit(build)(jnp.ones(4, dtype=bool))


def test_one_row_kept_boundary():
    stokes, var, _ = make(n_ch=2, seed=11)
    mask = np.zeros(8, dtype=bool).reshape(2, 4)
    mask[1, 0] = True
    data = StokesData(stokes, var, mask=mask)
    assert data.kept_rows() == (4,)
    wd, Linv = data.whitened()
    assert_allclose(np.asarray(wd), [stokes[1, 0] / np.sqrt(var[4])], rtol=1e-14)
    assert_allclose(np.asarray(Linv), [[1 / np.sqrt(var[4])]], rtol=1e-14)
    assert data.dof(1) == 0
    assert data.dof(2) == -1  # signed; the fit modules use the rank


# ----------------------------------------------------------------------
# variances vs covariance
# ----------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True])
def test_variances_and_diagonal_covariance_agree(masked):
    stokes, var, _ = make(n_ch=3, seed=13)
    mask = None
    if masked:
        mask = np.array([True, False, True, True])
    a = StokesData(stokes, var, mask=mask)
    b = StokesData(stokes, np.diag(var), mask=mask)
    wa, La = a.whitened()
    wb, Lb = b.whitened()
    assert_allclose(np.asarray(wa), np.asarray(wb), rtol=1e-13)
    assert_allclose(np.asarray(La), np.asarray(Lb), rtol=1e-13, atol=1e-16)
    assert_allclose(np.asarray(a.covariance()), np.asarray(b.covariance()))
    assert a.dof(2) == b.dof(2)


def test_whitening_is_exact_for_extreme_variances():
    stokes = np.array([[1.0, -2.0, 3.0, 4e100]])
    var = np.array([1e-300, 1e300, 1.0, 1e200])
    wd, Linv = StokesData(stokes, var).whitened()
    assert np.all(np.isfinite(np.asarray(wd)))
    assert_allclose(np.asarray(wd), stokes[0] / np.sqrt(var), rtol=1e-14)
    assert_allclose(np.diag(np.asarray(Linv)), 1 / np.sqrt(var), rtol=1e-14)


def test_strongly_correlated_covariance_matches_oracle():
    n = 8
    rho = 0.999999
    S = rho * np.ones((n, n)) + (1 - rho) * np.eye(n)
    stokes = rng(17).normal(size=(2, 4))
    wd, Linv = StokesData(stokes, S).whitened()
    od, oL = oracle_whiten(stokes.reshape(-1), S, range(n))
    assert_allclose(np.asarray(wd), od, rtol=1e-8)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-8, atol=1e-8)


def test_asymmetric_or_indefinite_covariance_is_rejected():
    stokes = np.ones((1, 4))
    S = random_covariance(4, seed=3)
    # symmetric to roundoff passes
    jitter = S + 1e-15 * np.triu(np.ones((4, 4)), 1)
    StokesData(stokes, jitter).whitened()
    # asymmetric by 1e-6 relative fails (error_if)
    bad = S.copy()
    bad[0, 1] += 1e-6 * np.abs(S).max()
    with pytest.raises(Exception):
        StokesData(stokes, bad).whitened()
    # indefinite: symmetric but with a negative eigenvalue
    indefinite = np.diag([1.0, 1.0, -1.0, 1.0])
    with pytest.raises(Exception):
        StokesData(stokes, indefinite).whitened()
    # singular: the masked block is rank deficient
    singular = np.ones((4, 4))
    with pytest.raises(Exception):
        StokesData(stokes, singular).whitened()


def test_masked_block_can_be_pd_even_when_full_matrix_is_not():
    # rows 2 and 3 are identical (singular full matrix); keeping I,Q only is fine
    S = np.array(
        [
            [2.0, 0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    stokes = np.array([[1.0, 2.0, 3.0, 4.0]])
    data = StokesData(stokes, S, mask=np.array([True, True, False, False]))
    wd, Linv = data.whitened()
    od, oL = oracle_whiten(stokes[0], S, [0, 1])
    assert_allclose(np.asarray(wd), od, rtol=1e-14)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-14)


# ----------------------------------------------------------------------
# response handling
# ----------------------------------------------------------------------


def test_response_defines_n_data_and_design():
    n_ch, n_data = 2, 5
    g = rng(21)
    R = g.normal(size=(n_data, 4 * n_ch))
    d = g.normal(size=n_data)
    var = g.uniform(1, 2, size=n_data)
    data = StokesData(d, var, response=R)
    assert data.n_data() == 5 and data.n_ch == 2 and data.n_kept() == 5
    assert_allclose(np.asarray(data.data_vector()), d)
    C = g.normal(size=(4 * n_ch, 3))
    assert_allclose(np.asarray(data.design(C)), R @ C, rtol=1e-14)
    # without a response the design is C itself
    plain = StokesData(g.normal(size=(n_ch, 4)), np.ones(8))
    assert_allclose(np.asarray(plain.design(C)), C)
    # whiten(x) applies L^-1 to the kept rows of any (n_data, ...) array
    wd, Linv = data.whitened()
    assert_allclose(
        np.asarray(data.whiten(R @ C)), np.asarray(Linv) @ (R @ C), rtol=1e-13
    )
    assert_allclose(np.asarray(data.whiten(d)), np.asarray(wd), rtol=1e-13)


def test_response_with_two_dimensional_stokes_requires_matching_size():
    R = np.eye(8)
    d = rng(1).normal(size=(2, 4))
    data = StokesData(d, np.ones(8), response=R)
    assert data.n_data() == 8
    assert_allclose(np.asarray(data.data_vector()), d.reshape(-1))
    with pytest.raises(ValueError):
        StokesData(d, np.ones(8), response=np.eye(8)[:, :4])  # 4 columns: n_ch = 1
    with pytest.raises(ValueError):
        StokesData(d, np.ones(5), response=np.ones((5, 8)))  # (2,4) data vs 5 rows


@pytest.mark.parametrize(
    "R",
    [np.ones((5, 7)), np.ones((5, 8, 1)), np.ones(8), np.ones((5, 8)) + 0j],
)
def test_response_shape_errors(R):
    with pytest.raises(ValueError):
        StokesData(np.ones(5), np.ones(5), response=R)


def test_response_noise_must_match_n_data():
    with pytest.raises(ValueError):
        StokesData(np.ones(5), np.ones(8), response=np.ones((5, 8)))
    with pytest.raises(ValueError):
        StokesData(np.ones(5), np.ones((8, 8)), response=np.ones((5, 8)))


def test_response_uncertainty_shape_rules():
    R = np.ones((5, 8))
    with pytest.raises(ValueError):
        StokesData(np.ones((2, 4)), np.ones(8), response_uncertainty=np.ones((8, 8)))
    with pytest.raises(ValueError):
        StokesData(
            np.ones(5), np.ones(5), response=R, response_uncertainty=np.ones((5, 7))
        )
    data = StokesData(np.ones(5), np.ones(5), response=R, response_uncertainty=0.1 * R)
    assert data.response_uncertainty.shape == (5, 8)
    with pytest.raises(Exception):  # negative entries are value errors
        StokesData(np.ones(5), np.ones(5), response=R, response_uncertainty=-R)


def test_mask_with_response_is_over_data_rows():
    n_ch, n_data = 2, 5
    g = rng(23)
    R = g.normal(size=(n_data, 4 * n_ch))
    d = g.normal(size=n_data)
    S = random_covariance(n_data, seed=5)
    mask = np.array([True, False, True, True, False])
    data = StokesData(d, S, response=R, mask=mask)
    assert data.kept_rows() == (0, 2, 3)
    wd, Linv = data.whitened()
    od, oL = oracle_whiten(d, S, [0, 2, 3])
    assert_allclose(np.asarray(wd), od, rtol=1e-12)
    assert_allclose(np.asarray(Linv), oL, rtol=1e-12, atol=1e-14)
    C = g.normal(size=(8, 3))
    assert_allclose(np.asarray(data.design(C)), (R @ C)[[0, 2, 3]], rtol=1e-14)
    assert data.dof(2) == 1
    # a Stokes-layout mask is only meaningful when n_data == 4 n_ch
    with pytest.raises(ValueError):
        StokesData(d, S, response=R, mask=np.ones((n_ch, 4), dtype=bool))
    with pytest.raises(ValueError):
        StokesData(d, S, response=R, mask=np.ones(4, dtype=bool))


def test_mask_with_square_response_accepts_stokes_layout():
    n_ch = 2
    R = np.diag(np.arange(1.0, 9.0))
    d = rng(3).normal(size=8)
    data = StokesData(
        d, np.ones(8), response=R, mask=np.array([True, True, False, True])
    )
    assert data.kept_rows() == (0, 1, 3, 4, 5, 7)
    # and with n_ch = 1 the (4,) mask is both a Stokes and a data-row mask
    one = StokesData(d[:4], np.ones(4), response=np.eye(4), mask=np.array([1, 0, 1, 1]))
    assert one.kept_rows() == (0, 2, 3)
    assert one.n_ch == 1 and n_ch == 2


# ----------------------------------------------------------------------
# discrepancy
# ----------------------------------------------------------------------


def test_discrepancy_vector_propagates_and_masks():
    n_ch = 2
    g = rng(31)
    stokes = g.normal(size=(n_ch, 4))
    var = np.ones(8)
    delta = g.uniform(0, 1, size=(n_ch, 4))
    mask = np.array([True, False, True, True])
    data = StokesData(stokes, var, mask=mask, discrepancy=ErrorTerm(delta, "bound"))
    kept = np.flatnonzero(np.tile(mask, (n_ch, 1)).reshape(-1))
    assert_allclose(np.asarray(data.discrepancy_vector()), delta.reshape(-1)[kept])
    # (4,) and scalar values broadcast over channels
    data4 = StokesData(
        stokes, var, discrepancy=ErrorTerm(np.array([1.0, 2, 3, 4]), "bound")
    )
    assert_allclose(
        np.asarray(data4.discrepancy_vector()), np.tile([1.0, 2, 3, 4], n_ch)
    )
    data0 = StokesData(stokes, var, discrepancy=ErrorTerm(0.5, "estimate"))
    assert_allclose(np.asarray(data0.discrepancy_vector()), 0.5 * np.ones(8))
    # unbounded and not_applicable carry no vector
    assert (
        StokesData(
            stokes, var, discrepancy=ErrorTerm.unbounded("x")
        ).discrepancy_vector()
        is None
    )
    assert StokesData(stokes, var).discrepancy_vector() is None
    with pytest.raises(ValueError):
        StokesData(stokes, var, discrepancy=ErrorTerm(np.ones(5), "bound"))
    with pytest.raises(ValueError):
        StokesData(stokes, var, discrepancy="not a term")


def test_discrepancy_in_stokes_space_goes_through_abs_response():
    n_ch, n_data = 2, 3
    g = rng(33)
    R = g.normal(size=(n_data, 4 * n_ch))
    d = g.normal(size=n_data)
    delta = g.uniform(0, 1, size=(n_ch, 4))
    data = StokesData(
        d, np.ones(n_data), response=R, discrepancy=ErrorTerm(delta, "bound")
    )
    assert_allclose(
        np.asarray(data.discrepancy_vector()), np.abs(R) @ delta.reshape(-1), rtol=1e-14
    )
    # a data-space discrepancy (n_data,) is used as is
    direct = StokesData(
        d, np.ones(n_data), response=R, discrepancy=ErrorTerm(np.ones(3), "bound")
    )
    assert_allclose(np.asarray(direct.discrepancy_vector()), np.ones(3))
    masked = StokesData(
        d,
        np.ones(n_data),
        response=R,
        mask=np.array([1, 0, 1]),
        discrepancy=ErrorTerm(delta, "bound"),
    )
    assert_allclose(
        np.asarray(masked.discrepancy_vector()), (np.abs(R) @ delta.reshape(-1))[[0, 2]]
    )


# ----------------------------------------------------------------------
# dof
# ----------------------------------------------------------------------


def test_dof_validation():
    stokes, var, data = make(n_ch=2)
    assert data.dof(0) == 8
    assert data.dof(8) == 0
    with pytest.raises(ValueError):
        data.dof(-1)
    with pytest.raises(ValueError):
        data.dof(1.5)
    with pytest.raises(ValueError):
        data.dof(True)


# ----------------------------------------------------------------------
# jit / autodiff / immutability
# ----------------------------------------------------------------------


@pytest.mark.parametrize("covariance", [False, True])
def test_whitened_under_jit_matches_eager(covariance):
    stokes, var, _ = make(n_ch=3, seed=41)
    noise = random_covariance(12, seed=4) if covariance else var
    mask = np.array([True, True, False, True])
    data = StokesData(stokes, noise, mask=mask)
    wd, Linv = data.whitened()

    @jax.jit
    def run(d):
        return d.whitened()

    jd, jL = run(data)
    assert_allclose(np.asarray(jd), np.asarray(wd), rtol=1e-13)
    assert_allclose(np.asarray(jL), np.asarray(Linv), rtol=1e-13, atol=1e-15)
    fd, fL = eqx.filter_jit(lambda d: d.whitened())(data)
    assert_allclose(np.asarray(fd), np.asarray(wd), rtol=1e-13)
    # jit over the traced leaves only (stokes as an argument)
    jd2 = jax.jit(lambda s: StokesData(s, noise, mask=mask).whitened()[0])(
        jnp.asarray(stokes)
    )
    assert_allclose(np.asarray(jd2), np.asarray(wd), rtol=1e-13)


def test_whitened_gradient_is_the_whitening_matrix():
    stokes, var, _ = make(n_ch=2, seed=43)
    S = random_covariance(8, seed=6)
    mask = np.array([True, False, True, True])

    def f(s):
        return StokesData(s, S, mask=mask).whitened()[0]

    J = jax.jacfwd(f)(jnp.asarray(stokes)).reshape(6, 8)
    kept = np.flatnonzero(np.tile(mask, (2, 1)).reshape(-1))
    _, oL = oracle_whiten(stokes.reshape(-1), S, kept)
    expected = np.zeros((6, 8))
    expected[:, kept] = oL
    assert_allclose(np.asarray(J), expected, rtol=1e-12, atol=1e-14)
    assert np.all(np.isfinite(np.asarray(jax.jacrev(f)(jnp.asarray(stokes)))))


def test_gradient_through_noise_is_finite():
    stokes, var, _ = make(n_ch=1, seed=45)

    def chi2(v):
        wd, _ = StokesData(stokes, v).whitened()
        return jnp.sum(wd**2)

    g = jax.grad(chi2)(jnp.asarray(var))
    assert_allclose(np.asarray(g), -(stokes.reshape(-1) ** 2) / var**2, rtol=1e-12)


def test_module_is_immutable_and_hashable_statics():
    stokes, var, data = make(n_ch=2)
    with pytest.raises(Exception):
        data.stokes = jnp.zeros((2, 4))
    other = StokesData(stokes, var, mask=np.array([1, 1, 1, 0]))
    assert data.kept_rows() != other.kept_rows()
    # one compile per static configuration: different kept rows retrace
    calls = []

    @jax.jit
    def run(d):
        calls.append(1)
        return d.n_kept()

    run(data)
    run(data)
    run(other)
    assert len(calls) == 2

"""Per-variable caps of ``Truncation`` and the lower-set Taylor remainder.

Oracles: brute-force enumeration of the retained ``(r, s, b)`` rows (total
cutoff AND caps), the minimal elements of the complement computed by
definition, exact NumPy polynomial algebra, and closed-form derivatives of
non-separable analytic test functions (``exp((a + i b).z)`` and
``1/(1 - c.z)``), whose sup along a path segment is taken on a dense grid
including both ends.

The bound under test (``syncmoments.model.index.lower_set_margin``) is

    |f(z) - T_Lambda f(z)| <= sum_{beta in S(Lambda)} |z^beta| / beta!
                               sup_{t in [0, 1]} |d^beta f(p_k(t))|,

with ``k`` the first nonzero coordinate of ``beta`` and ``p_k(t) =
(z_1, ..., z_{k-1}, t z_k, 0, ..., 0)``. ``S(Lambda)`` contains the minimal
elements of the complement and can contain more (see
``test_minimal_elements_alone_do_not_bound_a_tensor_product_set``).
"""

import itertools
import math

import numpy as np
import pytest

from syncmoments.model.index import MomentIndex, Truncation, lower_set_margin

CAPS = (None, 0, 1, 2)


def brute_rows(N, caps, depth_degree=None):
    """Retained ``(r, s, b)`` of the ``P`` block by definition."""
    cg, cB, cd = (10**6 if c is None else c for c in caps)
    out = []
    for r, s in itertools.product(range(N + 1), repeat=2):
        bmax = N - r - s if depth_degree is None else depth_degree
        for b in range(max(bmax, -1) + 1):
            if r + s <= N and r <= cg and s <= cB and b <= cd:
                out.append((r, s, b))
    return sorted(out)


def minimal_complement(lam):
    """Minimal elements of the complement of a lower set, by definition."""
    lam = set(lam)
    d = len(next(iter(lam)))
    top = max(max(b) for b in lam) + 2
    out = []
    for beta in itertools.product(range(top + 1), repeat=d):
        if beta in lam:
            continue
        preds = [beta[:i] + (beta[i] - 1,) + beta[i + 1 :] for i in range(d)]
        if all(beta[i] == 0 or preds[i] in lam for i in range(d)):
            out.append(beta)
    return sorted(out)


def is_lower(lam):
    lam = set(lam)
    return all(
        b[:i] + (b[i] - 1,) + b[i + 1 :] in lam
        for b in lam
        for i in range(len(b))
        if b[i] > 0
    )


# -- Truncation and MomentIndex --------------------------------------------------


def test_default_and_nonbinding_caps_are_identical_to_the_uncapped_truncation():
    base = Truncation(2, 2, 2)
    assert base.max_orders is None and not base.is_capped()
    for caps in [(None, None, None), (2, 2, 2), (5, None, 9)]:
        t = Truncation(2, 2, 2, max_orders=caps)
        assert t == base and hash(t) == hash(base) and t.max_orders is None
        assert MomentIndex.build(t) == MomentIndex.build(base)
    depth = Truncation(1, 1, 2, depth_degree=1, max_orders=(None, 7, 1))
    assert depth == Truncation(1, 1, 2, depth_degree=1)


@pytest.mark.parametrize("N", [0, 1, 2, 3])
@pytest.mark.parametrize("caps", list(itertools.product(CAPS, repeat=3)))
def test_retained_rows_are_the_total_cutoff_and_the_caps(N, caps):
    t = Truncation(1, 2, N, max_orders=caps)
    index = MomentIndex.build(t)
    rsb = brute_rows(N, caps)
    assert is_lower(rsb)
    got = sorted({row[2:] for row in index.h2})
    assert got == rsb
    rs = sorted({row[2:4] for row in index.h0})
    assert rs == sorted({(r, s) for r, s, _ in rsb})
    assert {row[4] for row in index.h0} == {0}
    assert t.max_b() == max(b for _, _, b in rsb)
    assert index.n0 == 2 * 3 * len(rs)
    assert t.is_capped() == (rsb != brute_rows(N, (None, None, None)))


def test_depth_layout_caps_act_on_r_and_s_and_refuse_a_binding_depth_cap():
    t = Truncation(2, 2, 3, depth_degree=2, max_orders=(1, 2, None))
    index = MomentIndex.build(t)
    assert sorted({row[2:] for row in index.h2}) == brute_rows(3, (1, 2, None), 2)
    assert t.max_b() == 2 and t.is_capped()
    with pytest.raises(ValueError, match="depth_degree"):
        Truncation(2, 2, 3, depth_degree=2, max_orders=(None, None, 1))


@pytest.mark.parametrize(
    "caps",
    [(1, 1), (1, 1, 1, 1), (-1, None, None), (True, None, None), (1.0, None, None)],
)
def test_invalid_caps_raise(caps):
    with pytest.raises(ValueError):
        Truncation(2, 2, 2, max_orders=caps)


def test_depth_cap_sets_the_phase_degree_of_the_main_layout():
    t = Truncation(2, 2, 3, max_orders=(None, None, 1))
    assert t.max_b() == 1 and t.is_capped()
    assert {row[4] for row in MomentIndex.build(t).h2} == {0, 1}


# -- the margin set S(Lambda) ------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("N", [0, 1, 2, 4])
def test_margin_of_a_total_degree_set_is_the_order_n_plus_one_shell(d, N):
    lam = [b for b in itertools.product(range(N + 1), repeat=d) if sum(b) <= N]
    shell = sorted(
        b for b in itertools.product(range(N + 2), repeat=d) if sum(b) == N + 1
    )
    assert sorted(lower_set_margin(lam)) == shell == minimal_complement(lam)


@pytest.mark.parametrize("N", [0, 1, 2, 3])
@pytest.mark.parametrize("caps", list(itertools.product(CAPS, repeat=3)))
def test_margin_contains_the_minimal_complement_and_lies_on_the_boundary(N, caps):
    lam = brute_rows(N, caps)
    S = lower_set_margin(lam)
    assert len(set(S)) == len(S) and not set(S) & set(lam)
    assert set(minimal_complement(lam)) <= set(S)
    for beta in S:  # every element has a predecessor in Lambda
        assert any(
            beta[i] > 0 and beta[:i] + (beta[i] - 1,) + beta[i + 1 :] in set(lam)
            for i in range(3)
        )
        assert sum(beta) <= N + 1
    t = Truncation(1, 1, N, max_orders=caps)
    assert tuple(t.margin_rows()) == tuple(S)


def test_margin_rows_of_the_depth_layout_are_the_two_dimensional_margin():
    t = Truncation(1, 1, 2, depth_degree=1, max_orders=(1, 1, None))
    S2 = lower_set_margin([(0, 0), (1, 0), (0, 1), (1, 1)])
    assert sorted(S2) == [(0, 2), (2, 0), (2, 1)]
    assert tuple(t.margin_rows()) == tuple(r + (0,) for r in S2)


def test_lower_set_margin_refuses_non_lower_sets():
    with pytest.raises(ValueError):
        lower_set_margin([(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        lower_set_margin([])


# -- numerical checks of the bound ---------------------------------------------------


def path_points(z, k, t):
    """``p_k(t)`` for an array of ``t``: coordinates ``< k`` at ``z``, ``k`` at ``t z_k``."""
    pts = np.zeros((t.size, z.size), dtype=z.dtype)
    pts[:, :k] = z[:k]
    pts[:, k] = t * z[k]
    return pts


def taylor(deriv0, lam, z):
    """``T_Lambda f(z)`` from the derivatives at the origin."""
    return sum(deriv0(b) * np.prod(z ** np.asarray(b)) / factorial(b) for b in lam)


def factorial(beta):
    return math.prod(math.factorial(int(b)) for b in beta)


def margin_bound(deriv, S, z, n_t=2001):
    """Right-hand side with the sup on a dense grid of each path segment."""
    t = np.linspace(0.0, 1.0, n_t)
    total = 0.0
    for beta in S:
        k = next(i for i, b in enumerate(beta) if b)
        sup = np.max(np.abs(deriv(beta, path_points(z, k, t))))
        total += sup * abs(np.prod(z ** np.asarray(beta))) / factorial(beta)
    return total


def exp_family(a):
    """``f = exp(a.z)`` (complex ``a``): ``d^beta f = a^beta f``."""
    a = np.asarray(a)

    def deriv(beta, pts):
        return np.prod(a ** np.asarray(beta)) * np.exp(pts @ a)

    return deriv


def pole_family(c):
    """``f = 1/(1 - c.z)``: ``d^beta f = |beta|! c^beta / (1 - c.z)^(|beta|+1)``."""
    c = np.asarray(c)

    def deriv(beta, pts):
        q = sum(beta)
        return (
            math.factorial(q)
            * np.prod(c ** np.asarray(beta))
            / (1 - pts @ c) ** (q + 1)
        )

    return deriv


LAMBDAS = [
    brute_rows(3, (1, 1, None)),
    brute_rows(3, (None, 0, 2)),
    brute_rows(2, (1, 1, 1)),
    brute_rows(4, (2, None, 1)),
    brute_rows(2, (None, None, None)),
    brute_rows(3, (1, 2, None), depth_degree=2),
]


@pytest.mark.parametrize("lam", LAMBDAS, ids=lambda l: f"n{len(l)}")
def test_polynomials_in_lambda_have_zero_remainder_and_zero_bound(lam):
    rng = np.random.default_rng(len(lam))
    coeff = {b: rng.standard_normal() for b in lam}
    S = lower_set_margin(lam)

    def deriv(beta, pts):  # d^beta of sum_alpha c_alpha z^alpha
        out = np.zeros(pts.shape[0])
        for alpha, c in coeff.items():
            if all(a >= b for a, b in zip(alpha, beta)):
                fall = math.prod(math.perm(a, b) for a, b in zip(alpha, beta))
                out += c * fall * np.prod(pts ** (np.asarray(alpha) - beta), axis=1)
        return out

    for z in rng.uniform(-1.5, 1.5, (20, 3)):
        value = deriv((0, 0, 0), z[None])[0]
        assert value == pytest.approx(
            taylor(lambda b: coeff.get(b, 0.0) * factorial(b), lam, z), abs=1e-12
        )
        assert margin_bound(deriv, S, z, n_t=5) == 0.0


@pytest.mark.parametrize("lam", LAMBDAS, ids=lambda l: f"n{len(l)}")
def test_monomials_on_the_margin_attain_or_are_covered(lam):
    S = lower_set_margin(lam)
    minimal = set(minimal_complement(lam))
    rng = np.random.default_rng(7)
    for gamma in S:

        def deriv(beta, pts, gamma=gamma):
            if not all(g >= b for g, b in zip(gamma, beta)):
                return np.zeros(pts.shape[0])
            fall = math.prod(math.perm(g, b) for g, b in zip(gamma, beta))
            return fall * np.prod(pts ** (np.asarray(gamma) - beta), axis=1)

        for z in rng.uniform(-1.2, 1.2, (5, 3)):
            remainder = abs(np.prod(z ** np.asarray(gamma)))  # T_Lambda z^gamma = 0
            bound = margin_bound(deriv, S, z, n_t=3)
            if gamma in minimal:
                assert bound == pytest.approx(remainder, rel=1e-13, abs=1e-300)
            else:
                assert bound >= remainder * (1 - 1e-13)


@pytest.mark.parametrize("lam", LAMBDAS, ids=lambda l: f"n{len(l)}")
@pytest.mark.parametrize("family", ["exp", "pole"])
def test_smooth_functions_are_covered_at_random_points(lam, family):
    rng = np.random.default_rng(11 + len(lam))
    S = lower_set_margin(lam)
    worst = 0.0
    for _ in range(12):
        if family == "exp":
            deriv = exp_family(rng.uniform(-2, 2, 3) + 1j * rng.uniform(-3, 3, 3))
            z = rng.uniform(-1.0, 1.0, 3)
        else:
            c = rng.uniform(-0.9, 0.9, 3)
            deriv = pole_family(c)
            z = rng.uniform(-1.0, 1.0, 3) * 0.3
        exact = deriv((0, 0, 0), z[None])[0]
        remainder = abs(exact - taylor(lambda b: deriv(b, np.zeros((1, 3)))[0], lam, z))
        bound = margin_bound(deriv, S, z)
        assert remainder <= bound * (1 + 1e-9) + 1e-14
        worst = max(worst, remainder / bound)
    assert worst > 1e-3  # the check is not vacuous


def test_margin_sum_is_below_the_frobenius_operator_form_on_total_degree_sets():
    rng = np.random.default_rng(3)
    for N in (0, 1, 2, 3):
        shell = [
            b for b in itertools.product(range(N + 2), repeat=3) if sum(b) == N + 1
        ]
        T = rng.standard_normal((3,) * (N + 1))
        T = sum(np.transpose(T, p) for p in itertools.permutations(range(N + 1)))
        z = rng.uniform(-1, 1, 3)
        entry = {b: T[tuple(i for i in range(3) for _ in range(b[i]))] for b in shell}
        margin = sum(
            abs(entry[b]) * abs(np.prod(z ** np.asarray(b))) / factorial(b)
            for b in shell
        )
        frobenius = (
            np.sqrt(np.sum(T**2)) * np.linalg.norm(z) ** (N + 1) / math.factorial(N + 1)
        )
        assert margin <= frobenius * (1 + 1e-12)


def test_minimal_elements_alone_do_not_bound_a_tensor_product_set():
    """``Lambda = {0,1}^2``: ``f = (x - x^2)(y - y^2)`` at ``(1, 1)`` has remainder 1
    but the sum over the minimal elements ``(2,0), (0,2)`` with the box sup is 1/2.
    The telescoped margin adds ``(2, 1)`` and covers it."""
    lam = [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert minimal_complement(lam) == [(0, 2), (2, 0)]
    S = lower_set_margin(lam)
    assert sorted(S) == [(0, 2), (2, 0), (2, 1)]
    g = np.polynomial.Polynomial([0, 1, -1])

    def deriv(beta, pts):
        gx, gy = g.deriv(beta[0]) if beta[0] else g, g.deriv(beta[1]) if beta[1] else g
        return gx(pts[:, 0]) * gy(pts[:, 1])

    z = np.array([1.0, 1.0])
    remainder = abs(
        deriv((0, 0), z[None])[0]
        - taylor(lambda b: deriv(b, np.zeros((1, 2)))[0], lam, z)
    )
    assert remainder == pytest.approx(1.0)
    grid = np.stack(
        np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101)), -1
    ).reshape(-1, 2)
    box_minimal = sum(
        np.max(np.abs(deriv(b, grid))) * abs(np.prod(z ** np.asarray(b))) / factorial(b)
        for b in minimal_complement(lam)
    )
    assert box_minimal == pytest.approx(0.5)
    assert margin_bound(deriv, S, z) >= remainder

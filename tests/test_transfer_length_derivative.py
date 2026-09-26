"""The derivative in the slab length, ``d out / d ds = Phi (eps - K S)``.

``out = Phi S + G eps`` with ``Phi = e^{-K ds}`` and ``dG / d ds = Phi``; ``Phi``
commutes with ``K``, so ``d out / d ds = -K Phi S + Phi eps = Phi (eps - K S)``.
Up to 0.3.0 the tangent came from the forward derivative of the 8x8 exponential
in ``ds``, i.e. ``-K Phi S`` and ``Phi eps`` from separately rounded blocks whose
errors scale with ``|G eps|``, not with ``|Phi|``. In thick or near-steady-state
slabs, where the derivative is far smaller than those terms, the relative error
reached 0.96, 7.5e-5, 2.1e20 and 5.1e2 on slabs 4, 53, 54 and 107 of the round-5
random set (``default_rng(7)``; truths below). Since 0.4.0 the rule computes
``Phi r``, ``r = eps - K S``, from the inputs.

Error bound (first order in ``u = 2^-53``, componentwise, ``gamma_n = n u``):

    |computed - Phi r| <= |Phi| gamma_5 (|eps| + |K| |S|) + gamma_4 |Phi| |r|
                          + |Phi_hat - Phi| |r|,

the first term from rounding ``eps - K S`` (four products, three sums, one
subtraction), the second from the product with ``Phi``, the third from the
error of the computed propagator (``syncmoments._expm.expm_pade13``). That last
error is not bounded rigorously; it is modelled as ``64 u max(|K ds|_1, 1)
max|Phi|`` per entry. Measured against mpmath: at most ``37 u max(|K ds|_1, 1)``
over the 119 finite round-5 slabs (slab 103, an absorbing ``K``: for a decaying
exponential the Pade quotient cancels by up to ``e^theta_13 = 215`` per
``theta_13 = 5.37`` of norm), 24 on an absorbing sweep, 3.5 on the slabs of
the random test below. The tests assert the measured error below this model
(with ``gamma_6``, ``gamma_5`` for margin). Powers of two ``2^-i`` scale ``eps`` and ``S`` before
the subtraction when ``|eps|`` or ``|K| |S|`` exceed ``2^512``, so ``K S`` does
not overflow where the derivative is finite; this adds nothing to the bound
except that entries below ``2^(i-1022)`` flush to zero on XLA CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.transfer import mueller_matrix, transfer_slab

U = 2.0**-53


def _mueller(diag, off):
    """Symmetric-absorption Mueller matrix from its diagonal and six coefficients."""
    return np.asarray(mueller_matrix(diag, *off))


# slab: (S, eps, K (flat), ds, mpmath truth of d out / d ds at 40 digits)
# fmt: off
Q8 = {
    4: ([-9.147016184036081e-07, 9.882670386888706e-05, -0.00021868115065146063, 5.872863017239018e-05],
        [1.3956310565412513e-11, -5.15941958776038e-13, 1.4283107263665253e-11, -7.2629130597429355e-12],
        [2.309479352642923, -0.016159537666630093, 0.018762081885890947, -0.006426475436557755, -0.016159537666630093, 2.309479352642923, -0.000613084031360577, 0.05928209608742515, 0.018762081885890947, 0.000613084031360577, 2.309479352642923, -0.010497951250691962, -0.006426475436557755, -0.05928209608742515, 0.010497951250691962, 2.309479352642923],
        24.267152754042023,
        [-1.328193874822929e-28, 6.312841541373767e-29, 2.288755103917901e-28, -1.713377242929349e-28]),
    53: ([0.06099443946483777, -0.03931711461806792, -0.05021847264774322, -0.005778430007329979],
         [1.0571315218032339e+19, 4.103659113715011e+19, 2.5764119269985817e+19, -2.5283958909955957e+19],
         [1.261488986134792, 0.0013951592532775465, -0.0006031682553913524, -0.0018035659914598385, 0.0013951592532775465, 1.261488986134792, -0.0035990675838701866, 0.0011278547902452683, -0.0006031682553913524, 0.0035990675838701866, 1.261488986134792, -0.002500589143227545, -0.0018035659914598385, -0.0011278547902452683, 0.002500589143227545, 1.261488986134792],
         23.038987091788403,
         [2027078.8853761924, 10342750.990434512, 4999860.155491557, -6001075.519407951]),
    54: ([-5.374910650647382, 6.646523894092945, 2.9271381438905575, -5.704470381799596],
         [-1.0823059899855888e-16, -4.701115889999866e-13, 8.717808636436099e-14, 1.4523372816819652e-13],
         [4.273397219004518, -0.002464112432413832, -0.0006001395206909817, -0.0005055695480802942, -0.002464112432413832, 4.273397219004518, -0.0004755696550864097, -0.0006043325905836596, -0.0006001395206909817, 0.0004755696550864097, 4.273397219004518, 0.0003698463116013371, -0.0005055695480802942, 0.0006043325905836596, -0.0003698463116013371, 4.273397219004518],
         27.72237358026865,
         [7.532960680621475e-51, -9.451389229286447e-51, -4.267373438248806e-51, 8.870064652871608e-51]),
    107: ([-3153.4917381962164, -2077.087374081205, -229.88527727558412, 5152.850689129403],
          [-1304527360112.436, 388045531648.22815, 1622514700462.1172, 998790583093.9158],
          [3.187215030303768, -0.009154873844589335, 0.001316718627595826, -0.004432077488529419, -0.009154873844589335, 3.187215030303768, 0.022725817828610585, -0.011379717612488408, 0.001316718627595826, -0.022725817828610585, 3.187215030303768, 0.006193716932533984, -0.004432077488529419, 0.011379717612488408, -0.006193716932533984, 3.187215030303768],
          14.2509478415021,
          [-2.3585754665269733e-08, -2.6298427233415224e-09, 3.0003728791083186e-08, 1.9578763325621524e-08]),
}
# fmt: on
MODES = {
    "jacfwd": jax.jacfwd,
    "jacrev": jax.jacrev,
    "jit jacrev": lambda f: jax.jit(jax.jacrev(f)),
}


def _bound(S, eps, K, ds):
    """The error bound above, per entry, with SciPy's ``Phi``."""
    Phi = np.abs(sla.expm(-K * ds))
    r = np.abs(eps - K @ S)
    e_phi = 64 * U * max(np.max(np.sum(np.abs(K * ds), axis=0)), 1.0) * np.max(Phi)
    rounding = Phi @ (6 * U * (np.abs(eps) + np.abs(K) @ np.abs(S))) + 5 * U * Phi @ r
    return rounding + e_phi * np.sum(r)


def _derivative(mode, S, eps, K, ds):
    f = lambda q: transfer_slab(S, eps, K, q)  # noqa: E731
    return np.asarray(MODES[mode](f)(jnp.asarray(ds)))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("slab", Q8)
def test_named_ill_conditioned_slabs_meet_the_bound(slab, mode):
    S, eps, K, ds, truth = Q8[slab]
    S, eps, K = np.asarray(S), np.asarray(eps), np.asarray(K).reshape(4, 4)
    got = _derivative(mode, S, eps, K, ds)
    assert np.all(np.abs(got - np.asarray(truth)) <= _bound(S, eps, K, ds))
    # The bound is far below the 0.3.0 errors (relative 7.5e-5 or more).
    assert np.max(_bound(S, eps, K, ds)) < 1e-9 * np.max(np.abs(truth))


def _ill_slabs(n, seed):
    """Near-steady-state slabs: ``S = K^-1 eps (1 + delta)``, ``|delta|`` from
    1e-12 to 1e-3, absorption 0.1 to 30, ``ds`` 0.1 to 20."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = 10.0 ** rng.uniform(-1, 1.5)
        K = _mueller(a, (*(a * rng.uniform(-0.9, 0.9, 3)), *rng.uniform(-3, 3, 3)))
        eps = rng.normal(size=4) * 10.0 ** rng.uniform(-20, 20)
        delta = 10.0 ** rng.uniform(-12, -3) * rng.normal(size=4)
        S = np.linalg.solve(K, eps) * (1 + delta)
        yield S, eps, K, float(10 ** rng.uniform(-1, 1.3))


def _mp_truth(S, eps, K, ds):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 40
    Km = mp.matrix([[mp.mpf(float(x)) for x in row] for row in K])
    E = mp.expm(-Km * mp.mpf(ds))
    r = [mp.mpf(float(eps[i])) - mp.fsum(Km[i, c] * mp.mpf(float(S[c])) for c in range(4)) for i in range(4)]  # fmt: skip
    return np.array(
        [float(mp.fsum(E[i, c] * r[c] for c in range(4))) for i in range(4)]
    )


def test_random_near_steady_state_slabs_meet_the_bound():
    """40 slabs (needs mpmath). Under 0.3.0, 10 of them (9 in jax 0.10.2)
    exceeded the bound, by factors up to 1.2e22."""
    pytest.importorskip("mpmath")
    for i, (S, eps, K, ds) in enumerate(_ill_slabs(40, seed=3)):
        truth = _mp_truth(S, eps, K, ds)
        for mode in ("jacfwd", "jacrev"):
            got = _derivative(mode, S, eps, K, ds)
            assert np.all(np.abs(got - truth) <= _bound(S, eps, K, ds)), (i, mode)


def _scaled_oracle(S, eps, K, ds, i):
    """``Phi (eps - K S)`` in NumPy with ``eps`` and ``S`` scaled by ``2^-i``."""
    r = np.ldexp(eps, -i) - K @ np.ldexp(S, -i)
    return np.ldexp(sla.expm(-K * ds) @ r, i)


# (S, eps, K, ds): K S beyond float64 unless scaled; sources near float64 max.
EXTREME = {
    "K S 1e310": (np.array([1e300, -3e299, 2e299, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), _mueller(1e10, (0, 0, 0, 3e9, 0, 1e9)), 2e-9),  # fmt: skip
    "eps 1.7e308": (np.zeros(4), np.array([1.7e308, 1.6e308, -1.5e308, 1e308]), _mueller(0.0, (0, 0, 0, 0.5, 0.2, 2.0)), 1e-3),  # fmt: skip
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", EXTREME)
def test_length_derivative_does_not_overflow_at_extreme_scale(case, mode):
    S, eps, K, ds = EXTREME[case]
    want = _scaled_oracle(S, eps, K, ds, 600)
    assert np.all(np.isfinite(want))
    got = _derivative(mode, S, eps, K, ds)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))


def test_batched_lengths_in_reverse_mode():
    """``grad`` of a sum over ``vmap`` in a batched ``ds`` (the rule stays linear
    in the tangent) equals the per-member derivatives and the identity."""
    S, eps, K, _, _ = Q8[107]
    S, eps, K = np.asarray(S), np.asarray(eps), np.asarray(K).reshape(4, 4)
    w = jnp.array([1.0, -2.0, 0.5, 0.25])
    lengths = jnp.array([0.5, 3.0, 14.2509478415021])

    def total(ds):
        return jnp.sum(jax.vmap(lambda d: w @ transfer_slab(S, eps, K, d))(ds))

    grads = np.asarray(jax.jit(jax.grad(total))(lengths))
    for i, d in enumerate(np.asarray(lengths)):
        want = float(np.asarray(w) @ (sla.expm(-K * d) @ (eps - K @ S)))
        size = float(np.abs(np.asarray(w)) @ _bound(S, eps, K, d))
        assert abs(grads[i] - want) <= 2 * size  # both sides meet the bound


@pytest.mark.parametrize("shift", (-8, -1, 0, 1, 8, 400, 500))
def test_scaled_rate_equals_the_unscaled_one_across_the_threshold(shift):
    """Boundary validation of the scaling in ``_length_rate``: ``i > 0`` from
    ``e(|K|) + e(|S|) + 2 > 512`` on. Both methods are evaluated directly (the
    unscaled one on inputs pre-scaled by ``2^-i``, which gives ``i = 0``) and
    agree bit for bit on both sides and far above: powers of two commute with
    rounding when nothing under- or overflows."""
    from syncmoments._expm import propagators
    from syncmoments.transfer import _length_rate

    K = jnp.asarray(_mueller(1.5, (0.2, -0.1, 0.05, 0.3, -0.2, 1.1)))  # e(|K|) = 1
    S = jnp.array([0.75, -0.5, 0.25, 0.6]) * 2.0 ** (
        509 + shift
    )  # e(|S|) = 509 + shift
    eps = jnp.array([1.0, 0.5, -0.25, 0.1]) * 2.0 ** (510 + shift)
    Phi = propagators(K, jnp.asarray(0.3), 32)[0]
    rate, i = _length_rate(Phi, S, eps, K)
    assert int(i) == max(0, shift)
    down = 2.0 ** -int(i)
    unscaled, zero = _length_rate(Phi, S * down, eps * down, K)
    assert int(zero) == 0
    np.testing.assert_array_equal(np.asarray(rate), np.asarray(unscaled))
    assert np.all(np.isfinite(np.asarray(rate) * 2.0 ** int(i)))

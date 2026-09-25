"""Second derivatives in the weights (T-004 round 5, items F2 and F4).

``_normalised_weights`` gives ``w / sum(w)`` a closed-form second-order tangent
(``syncmoments.rm._sum_tangent``). Round 4 refused every second derivative for
``max(w) < 2^-958``; that refused representable results such as Hessians in
log-weights ``theta`` (``w = exp(theta)``), which are softmax Hessians of order 1.
Checked here against the exact rational oracle of
``test_weight_normalisation_oracle.py`` for ``mixed_moments``, the RM variance
and ``PopulationSamples`` moments, in five second-order modes:

1. Log-weight Hessians are exact at every shift down to ``max(w) ~ 1e-307``.
2. Hessians in ``w`` for ``max(w) >= 2^-958`` (no headroom shift) are exact
   where representable and infinite with the exact sign where not, in every
   mode except forward-over-forward. "Exact" means within the float64 rounding
   bound ``64 eps sum|terms|``; where that bound itself exceeds float64 (an
   entry that cancels to a small value), an infinity of either sign passes.
3. Forward-over-forward materialises the second tangent of ``p`` itself. Its
   entries ``(2 p_i - delta_ia - delta_ib) / W^2`` overflow once
   ``W < 2^-512``; they sum to zero, so infinities of both signs occur and a
   consumer's contraction ``sum_i a_i d2p_i`` is NaN. That is not decidable
   from primal values (log-weight Hessians at the same weights are exact), so
   it is not refused; the result is never finite there, and the second
   tangent of ``p`` itself is exact or infinite with the exact sign.
4. For ``max(w) < 2^-958`` every exact entry of these Hessians in ``w`` along
   unit directions exceeds float64; no mode returns a finite entry (entries
   are NaN or infinite; reverse-over-reverse can give the wrong sign).
5. The RM variance near float64 max (``n = 16``, ``max(w) = 2^-511``, exact
   entries up to ``2^1022``): forward-over-forward overflows in the variance's
   own products of first derivatives (``4 dmu_a dmu_b``); those entries are
   not finite. Every other mode, and ``p @ x`` in every mode, is exact.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_weight_normalisation_oracle import (
    BIG,
    MODES,
    PATTERN,
    TARGETS,
    classify,
    failures,
    hessian_theta,
    hessian_w,
    linear,
    variance,
)
from syncmoments.rm import _normalised_weights, gaussian_rm_cumulants

WRAPS = {"eager": lambda f: f, "jit": jax.jit}
SHIFTS = [0.0, -300.0, -690.0, -700.0, -708.0]
NORMAL_SCALES = [1.0, 1e-100, 2.0**-511, 1e-160, 1e-200]
HEADROOM_SCALES = [1e-300, 2.0**-1022]
NOT_FWD_FWD = [m for m in MODES if m != "fwd-fwd"]
UNIT = [linear(np.eye(4)[i]) for i in range(4)]  # F(p) = p_i
N16, X16 = np.linspace(0.1, 1.0, 16), np.arange(16.0)


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("target", TARGETS)
def test_log_weight_hessian_is_the_softmax_hessian(target, shift, mode, wrap):
    f, derivatives = TARGETS[target]
    theta = jnp.log(jnp.asarray(PATTERN * 4)) + shift
    got = WRAPS[wrap](MODES[mode](lambda t: f(jnp.exp(t))))(theta)
    oracle = hessian_theta(np.exp(np.asarray(theta)), derivatives)
    assert failures(got, oracle) == {}


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("mode", NOT_FWD_FWD)
@pytest.mark.parametrize("scale", NORMAL_SCALES)
@pytest.mark.parametrize("target", TARGETS)
def test_weight_hessian_exact_or_signed_inf_above_the_headroom(
    target, scale, mode, wrap
):
    f, derivatives = TARGETS[target]
    w = jnp.asarray(PATTERN * scale)
    got = WRAPS[wrap](MODES[mode](f))(w)
    assert failures(got, hessian_w(np.asarray(w), derivatives)) == {}


@pytest.mark.parametrize("wrap", WRAPS)
@pytest.mark.parametrize("scale", NORMAL_SCALES)
@pytest.mark.parametrize("target", TARGETS)
def test_forward_over_forward_exact_where_the_p_tangent_is_representable(
    target, scale, wrap
):
    f, derivatives = TARGETS[target]
    w = jnp.asarray(PATTERN * scale)
    got = np.asarray(WRAPS[wrap](MODES["fwd-fwd"](f))(w))
    if float(jnp.sum(w)) >= 2.0**-512:
        assert failures(got, hessian_w(np.asarray(w), derivatives)) == {}
        return
    assert not np.any(np.isfinite(got))
    second = WRAPS[wrap](jax.jacfwd(jax.jacfwd(_normalised_weights)))(w)
    for i in range(4):  # the Hessian of p_i: exact or signed inf, never NaN
        assert failures(second[i], hessian_w(np.asarray(w), UNIT[i])) == {}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("scale", HEADROOM_SCALES)
@pytest.mark.parametrize("target", TARGETS)
def test_headroom_unit_direction_hessians_are_never_finite(target, scale, mode):
    f, derivatives = TARGETS[target]
    w = jnp.asarray(PATTERN * scale)
    H, _ = hessian_w(np.asarray(w), derivatives)
    assert all(abs(h) > BIG for row in H for h in row)
    assert not np.any(np.isfinite(np.asarray(MODES[mode](f)(w))))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("log2_max", [-511, -509])
def test_sixteen_samples_near_float64_max(log2_max, mode):
    w = jnp.asarray(N16 * 2.0**log2_max)
    x = jnp.asarray(X16)
    got_p = MODES[mode](lambda w: _normalised_weights(w) @ x)(w)
    assert failures(got_p, hessian_w(np.asarray(w), linear(X16))) == {}
    got = np.asarray(MODES[mode](lambda w: gaussian_rm_cumulants(x, w)[1])(w))
    H, S = hessian_w(np.asarray(w), variance(X16))
    labels = [
        [classify(got[a, b], H[a][b], S[a][b]) for b in range(16)] for a in range(16)
    ]
    if mode == "fwd-fwd" and log2_max == -511:
        assert all(
            np.isinf(got[a, b]) or np.isnan(got[a, b]) or labels[a][b] == "ok"
            for a in range(16)
            for b in range(16)
        )
    else:
        assert all(label == "ok" for row in labels for label in row)


def test_oracle_agrees_with_plain_division_at_unit_scale():
    """Independent check of the oracle itself: plain ``w / sum(w)`` and
    ``jax.nn.softmax`` in float64 at unit scale."""
    w = jnp.asarray(PATTERN)
    x = jnp.asarray(np.arange(1.0, 5.0))

    def var(p):
        return p @ (x - p @ x) ** 2

    got = jax.hessian(lambda w: var(w / jnp.sum(w)))(w)
    assert failures(got, hessian_w(np.asarray(w), variance(np.arange(1.0, 5.0)))) == {}
    got = jax.hessian(lambda t: var(jax.nn.softmax(t)))(jnp.log(w))
    assert (
        failures(got, hessian_theta(np.asarray(w), variance(np.arange(1.0, 5.0)))) == {}
    )

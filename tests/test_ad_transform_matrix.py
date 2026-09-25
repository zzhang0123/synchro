"""AD-transform matrix of every entry point with a custom derivative rule.

Entry points and references: ``tests/_ad_matrix_cases.py``; transforms and
the comparison: ``tests/_ad_matrix.py``. Each entry point is split into its
differentiable argument groups (one group per argument or argument family,
plus ``all``), because a ``custom_jvp`` with ``symbolic_zeros`` takes a
different path for each set of nonzero tangents, and ``vmap`` of a group
batches only that group. The transforms are ``jit``; ``vmap`` and nested
``vmap`` of the group; ``grad`` of a sum over ``vmap``, over nested ``vmap``,
under ``jit`` and under ``jax.checkpoint``; ``vmap(grad)``; ``jacfwd``;
``jacrev``; ``hessian`` (forward over reverse, also under ``jit``, under
``vmap`` and of a sum over ``vmap``); reverse over reverse; forward over
forward; reverse over forward; ``linearize`` and ``vjp`` with their primal
differentiated forward and the forward derivative of the ``vjp`` (HVP);
``jax.checkpoint``; ``lax.scan`` over the function and ``vmap`` of that scan.
Tolerance: 1e-12 of the largest entry of each result block.

The fast tier runs every transform on one group per custom-rule family
(``FAST``); the complete matrix (every group) is marked ``slow``. The
regression for the failure this matrix found, reverse mode through nested
``vmap`` of the transfer functions, is ``tests/test_ad_transform_matrix_nested.py``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import pytest

import syncmoments  # noqa: F401  (enables x64)
from _ad_matrix import TRANSFORMS, Expected, Group, check, mismatch, transforms
from _ad_matrix_cases import all_groups

GROUPS = {g.name: g for g in all_groups()}
BATCHED_REVERSE = (
    "grad_sum_vmap",
    "grad_nested_vmap",
    "checkpoint_grad_sum_vmap",
    "vmap_scan",
    "hessian_sum_vmap",
    "jit_hessian",
)
FAST = {
    "transfer_slab[K]": TRANSFORMS,  # tangent rule with symbolic-zero (S, eps)
    "transfer_slab[S]": TRANSFORMS,  # linear branch of the tangent rule
    "transfer_los[all]": BATCHED_REVERSE,  # scan of slabs, route cond per path
    "moment_driven_slab[all]": ("grad_nested_vmap", "hessian_sum_vmap"),
    "mixed_moments[weights]": TRANSFORMS,  # rm._normalised_weights rules
    "PopulationSamples.normalised_weights[weights]": TRANSFORMS,
    "bessel_band[x]": TRANSFORMS,  # model._bessel_recurrence rule
    "harmonic_lines[all]": ("grad_sum_vmap", "hessian", "fwd_fwd"),  # its consumer
}
FAST_CELLS = [(g, t) for g, names in FAST.items() for t in names]
SLOW_CELLS = [
    (g, t) for g in GROUPS for t in TRANSFORMS if (g, t) not in set(FAST_CELLS)
]


@functools.lru_cache(maxsize=None)
def _expected(name):
    return Expected(GROUPS[name])


def _run(name, transform):
    message = check(GROUPS[name], transform, _expected(name))
    assert message is None, f"{name} {transform}: {message}"


def test_fast_groups_exist():
    assert set(FAST) <= set(GROUPS)
    families = {name.split("[")[0] for name in GROUPS}
    assert len(GROUPS) >= 60 and len(families) == 19


@pytest.mark.parametrize("name, transform", FAST_CELLS, ids=map("-".join, FAST_CELLS))
def test_transform_matrix_fast(name, transform):
    _run(name, transform)


@pytest.mark.slow
@pytest.mark.parametrize("name, transform", SLOW_CELLS, ids=map("-".join, SLOW_CELLS))
def test_transform_matrix_full(name, transform):
    _run(name, transform)


# -- the harness detects what it is meant to detect --------------------------------


@jax.custom_jvp
def _sin_frozen_slope(x):
    return jnp.sin(x)


@_sin_frozen_slope.defjvp
def _sin_frozen_slope_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sin(x), jax.lax.stop_gradient(jnp.cos(x)) * dx  # wrong at 2nd order


@jax.custom_jvp
def _sin_nonlinear_rule(x):
    return jnp.sin(x)


@_sin_nonlinear_rule.defjvp
def _sin_nonlinear_rule_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sin(x), jnp.cos(x) * jax.lax.stop_gradient(dx) + (dx - dx)


def _synthetic(f):
    x = {"x": jnp.array([0.3, 1.1, -0.4])}
    return Group("synthetic", ("x",), lambda p: f(p["x"]), lambda p: jnp.sin(p["x"]), x)


FIRST_ORDER = ("eager", "jit", "vmap", "jacfwd", "jacrev", "vjp", "linearize")
SECOND_ORDER = ("hessian", "rev_rev", "fwd_fwd", "rev_fwd", "jvp_of_vjp")


def test_harness_flags_a_rule_that_is_wrong_only_at_second_order():
    group = _synthetic(_sin_frozen_slope)
    e = Expected(group)
    assert all(check(group, t, e) is None for t in FIRST_ORDER)
    assert all(check(group, t, e) is not None for t in SECOND_ORDER)


def test_harness_flags_a_tangent_that_reverse_mode_cannot_transpose():
    """``stop_gradient`` of a tangent: forward mode is right, reverse raises."""
    group = _synthetic(_sin_nonlinear_rule)
    e = Expected(group)
    assert check(group, "jacfwd", e) is None
    for name in ("jacrev", "grad_sum_vmap", "vjp"):
        with pytest.raises(Exception, match="stop_gradient|linear|transpose"):
            check(group, name, e)


def test_mismatch_scales_per_block_and_requires_finite_values():
    want = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([1e-8, 0.0])}
    assert mismatch({"a": want["a"] * (1 + 1e-13), "b": want["b"]}, want) is None
    assert mismatch({"a": want["a"], "b": want["b"] * (1 + 1e-10)}, want) is not None
    assert "non-finite" in mismatch({"a": want["a"] * jnp.nan, "b": want["b"]}, want)
    zero = {"a": jnp.zeros(2)}
    assert mismatch({"a": jnp.array([0.0, 1e-300])}, zero).startswith("expected")
    group = _synthetic(jnp.sin)
    assert tuple(transforms(group, Expected(group))) == TRANSFORMS

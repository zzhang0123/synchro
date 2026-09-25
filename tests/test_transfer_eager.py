"""Eager calls of the transfer functions: compiled once, refused quietly.

In round 4 of 0.3.0 every eager ``transfer_slab`` and ``transfer_los`` call
built new branch closures for a ``lax.cond``, so JAX compiled a new ``cond``
per call (critic q13): eager ``transfer_slab`` took 114 ms against 1.4 ms in
v0.2.0, and ``moment_driven_slab(_cgs)`` inherited the cost. An eager
``transfer_los`` refusal raised ``JaxRuntimeError`` and printed a callback
traceback to stderr, while the CHANGELOG promised ``EquinoxRuntimeError``
(critic item 5). The public cores are now compiled with ``equinox.filter_jit``
(``max_squarings`` static).
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import syncmoments  # noqa: F401  (enables x64)
from syncmoments.los_moments import moment_driven_slab, moment_driven_slab_cgs
from syncmoments.transfer import mueller_matrix, transfer_los, transfer_slab

THICK = mueller_matrix(5.0, 1.0, -0.5, 0.2, 4.0, -2.0, 1.0)
WEAK = mueller_matrix(0.3, 0.1, -0.05, 0.02, 0.4, -0.2, 0.1)
S0 = jnp.array([1.0, 0.2, -0.1, 0.05])
D = jnp.array([1.0, 0.3, -0.2, 0.1])
KL = jnp.stack([THICK, WEAK, THICK])
OVERFLOW = jnp.array([4e307, 1.2e307, -8e306, 4e306])


def _first_of_slab(K):
    return transfer_slab(S0, D, K, 1.0)[0]


# Each entry takes a call index; calls differ in values, not in shapes.
ENTRIES = {
    "transfer_slab": lambda i: transfer_slab(S0, D * (i + 2), THICK, 1.0),
    "transfer_slab extreme": lambda i: transfer_slab(
        S0, D * 1e306 * (i + 1), WEAK, 1.0
    ),
    "transfer_slab max_squarings": lambda i: transfer_slab(
        S0, D, THICK * (i + 1), 1.0, max_squarings=40
    ),
    "transfer_los": lambda i: transfer_los(
        S0, jnp.stack([D, D, -D]) * (i + 2), KL, 1.0
    ),
    "transfer_los per-slab ds": lambda i: transfer_los(
        S0, jnp.stack([D, D, -D]), KL, jnp.array([1.0, 0.5, 2.0]) * (i + 1)
    ),
    "grad in K": lambda i: jax.grad(_first_of_slab)(THICK * (1 + i / 10)),
    "moment_driven_slab": lambda i: moment_driven_slab(
        1.0 + i, 3.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0
    ),
    "moment_driven_slab_cgs": lambda i: moment_driven_slab_cgs(
        1e8 * (i + 1), 2500.0, 5e-6, 1.0, 1.0, 1.0, 0.5, 1e10, n_e=0.03
    ),
}


def _compiles_during(call, caplog):
    """Messages of the compilations JAX logs while ``call`` runs."""
    caplog.clear()
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.WARNING, logger="jax"):
            jax.block_until_ready(call())
    finally:
        jax.config.update("jax_log_compiles", False)
    return [
        r.getMessage()[:120] for r in caplog.records if "Compiling" in r.getMessage()
    ]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_second_eager_call_with_the_same_shapes_compiles_nothing(entry, caplog):
    jax.block_until_ready(ENTRIES[entry](0))
    assert _compiles_during(lambda: ENTRIES[entry](1), caplog) == []


REFUSALS = {
    "transfer_slab": lambda: transfer_slab(jnp.zeros(4), OVERFLOW, THICK, 50.0),
    "transfer_los": lambda: transfer_los(
        jnp.zeros(4), jnp.stack([D, OVERFLOW, D]), KL, 50.0
    ),
    "transfer_los per-slab ds": lambda: transfer_los(
        jnp.zeros(4), jnp.stack([D, OVERFLOW, D]), KL, jnp.array([1.0, 50.0, 1.0])
    ),
    "grad in eps": lambda: jax.grad(
        lambda e: transfer_slab(jnp.zeros(4), e, THICK, 50.0)[0]
    )(OVERFLOW),
    "jacrev of transfer_los": lambda: jax.jacrev(
        lambda e: transfer_los(jnp.zeros(4), e, KL, 50.0)
    )(jnp.stack([D, OVERFLOW, D])),
}


@pytest.mark.parametrize("entry", REFUSALS)
def test_eager_refusal_is_an_equinox_error_with_nothing_on_stderr(entry, capfd, caplog):
    capfd.readouterr()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(
            eqx.EquinoxRuntimeError, match=r"eps \* ds overflows"
        ) as info:
            jax.block_until_ready(REFUSALS[entry]())
    assert not isinstance(info.value, jax.errors.JaxRuntimeError)
    captured = capfd.readouterr()
    assert captured.err == "", captured.err[:400]
    assert "callback" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
